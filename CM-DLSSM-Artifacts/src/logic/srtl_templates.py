"""
==============================================================================
CM-DLSSM Artifact: Standard Rule Template Library (SRTL) Definitions
Path: src/logic/srtl_templates.py
==============================================================================
Reference: Appendix B (SRTL Definitions)
           Section 5.2 (Compliance Gating)

Description:
    This module manages the knowledge base of security axioms. It bridges the
    gap between human-readable security policies (e.g., "SQL Injection Logic")
    and the mathematical tensors required by the CAVI Engine.

    Key Features:
    1. Template Enforcement: Only allows T1, T2, T3 forms (Horn Clauses).
    2. Compliance Box: Clamps rule weights w_r to [w_min, w_max] to prevent
       "Rule Suppression" attacks during neural training.
    3. Tensor Compilation: Batch-prepares rules for vectorized scatter_add.

    Supported Templates:
    - T1: p_a -> p_c             (Simple Propagation / Aliasing)
    - T2: p_a ^ p_b -> p_c       (Conjunction / Chained Logic)
    - T3: p_a ^ !p_b -> p_c      (Negative Constraint / Vulnerability)
==============================================================================
"""

import torch
import yaml
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ==============================================================================
# 1. Definitions
# ==============================================================================

class RuleType(Enum):
    T1_PROPAGATION = "T1"  # a -> c
    T2_CONJUNCTION = "T2"  # a ^ b -> c
    T3_CONSTRAINT  = "T3"  # a ^ !b -> c (The Vuln Template)

@dataclass
class SecurityRule:
    rule_id: str
    template: RuleType
    variables: List[int]   # [a, c] for T1; [a, b, c] for T2/T3
    weight: float
    description: str = ""
    cwe_id: str = "CWE-UNKNOWN"

    def __post_init__(self):
        # Validate Arity
        if self.template == RuleType.T1_PROPAGATION and len(self.variables) != 2:
            raise ValueError(f"Rule {self.rule_id} (T1) requires 2 variables [a, c]")
        if self.template in [RuleType.T2_CONJUNCTION, RuleType.T3_CONSTRAINT] and len(self.variables) != 3:
            raise ValueError(f"Rule {self.rule_id} ({self.template.value}) requires 3 variables [a, b, c]")

# ==============================================================================
# 2. SRTL Manager (The Rule Compiler)
# ==============================================================================

class SRTLManager:
    def __init__(self, compliance_config: Optional[Dict] = None, device: str = "cpu"):
        """
        Args:
            compliance_config: Dict with 'w_min' and 'w_max'.
            device: torch device for output tensors.
        """
        self.device = device
        self.rules_registry: List[SecurityRule] = []
        
        # Default Compliance Box (Section 5.2)
        # Prevents model from setting w=0 to ignore rules.
        if compliance_config is None:
            self.w_min = 1.0
            self.w_max = 20.0
        else:
            self.w_min = compliance_config.get('w_min', 1.0)
            self.w_max = compliance_config.get('w_max', 20.0)

    def add_rule(self, rule_id: str, template: str, variables: List[int], weight: float, desc: str = ""):
        """
        Register a new rule programmatically.
        """
        # 1. Enforce Compliance Box
        safe_weight = self._enforce_compliance(weight, rule_id)
        
        # 2. Create Object
        rule_type = RuleType(template)
        new_rule = SecurityRule(
            rule_id=rule_id,
            template=rule_type,
            variables=variables,
            weight=safe_weight,
            description=desc
        )
        self.rules_registry.append(new_rule)

    def load_from_yaml(self, file_path: str, predicate_map: Dict[str, int]):
        """
        Load rules from a human-readable YAML config.
        
        Args:
            file_path: Path to rules.yaml
            predicate_map: Mapping from string names ('is_tainted') to int indices (0).
        """
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        for r in data['rules']:
            # Map string variable names to integers
            try:
                var_indices = [predicate_map[v] for v in r['vars']]
            except KeyError as e:
                raise ValueError(f"Rule {r['id']} uses unknown predicate: {e}")

            self.add_rule(
                rule_id=r['id'],
                template=r['template'],
                variables=var_indices,
                weight=r['weight'],
                desc=r.get('description', "")
            )
        print(f"[SRTL] Loaded {len(data['rules'])} rules from {file_path}")

    def compile(self) -> Tuple[Dict, Dict, Dict]:
        """
        Compiles the registry into PyTorch tensors for the CAVIEngine.
        
        Returns:
            (rules_t1, rules_t2, rules_t3) - each a dict with 'indices' and 'weights'
        """
        t1_list, t2_list, t3_list = [], [], []
        w1_list, w2_list, w3_list = [], [], []

        for r in self.rules_registry:
            if r.template == RuleType.T1_PROPAGATION:
                t1_list.append(r.variables)
                w1_list.append(r.weight)
            elif r.template == RuleType.T2_CONJUNCTION:
                t2_list.append(r.variables)
                w2_list.append(r.weight)
            elif r.template == RuleType.T3_CONSTRAINT:
                t3_list.append(r.variables)
                w3_list.append(r.weight)

        def make_dict(idx_list, w_list):
            if not idx_list:
                return None
            return {
                'indices': torch.tensor(idx_list, dtype=torch.long, device=self.device),
                'weights': torch.tensor(w_list, dtype=torch.float32, device=self.device)
            }

        return make_dict(t1_list, w1_list), make_dict(t2_list, w2_list), make_dict(t3_list, w3_list)

    def _enforce_compliance(self, weight: float, rule_id: str) -> float:
        """
        Ensures weight stays within the 'Compliance Box'.
        If a weight is too low, it means the system is ignoring the rule.
        """
        if weight < self.w_min:
            print(f"[!] WARNING: Rule {rule_id} weight {weight} < {self.w_min}. Clamping to min (Compliance Enforcement).")
            return self.w_min
        if weight > self.w_max:
            return self.w_max
        return weight

    def export_summary(self) -> str:
        """Generates a summary string for the VAA Metadata."""
        counts = {t: 0 for t in RuleType}
        for r in self.rules_registry:
            counts[r.template] += 1
        return f"SRTL_v3.3|T1:{counts[RuleType.T1_PROPAGATION]}|T2:{counts[RuleType.T2_CONJUNCTION]}|T3:{counts[RuleType.T3_CONSTRAINT]}"

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing SRTL Manager...")
    
    # 1. Setup Manager
    manager = SRTLManager(compliance_config={'w_min': 2.0, 'w_max': 50.0})
    
    # 2. Define a Predicate Map (Name -> ID)
    pred_map = {
        "is_source": 0,
        "is_sanitized": 1,
        "is_sink": 2,
        "is_alias": 3,
        "is_vulnerable": 4
    }
    
    # 3. Add Rules Manually
    # T3: Source ^ !Sanitized -> Vulnerable
    manager.add_rule("R101", "T3", [0, 1, 4], weight=10.0, desc="SQL Injection Logic")
    
    # T1: Source -> Alias (Weight 0.5 -> Should trigger Compliance Clamp)
    manager.add_rule("R102", "T1", [0, 3], weight=0.5, desc="Taint Propagation")
    
    # 4. Compile
    r1, r2, r3 = manager.compile()
    
    # 5. Assertions
    print("\n--- Compiled Tensors ---")
    if r3:
        print(f"T3 Indices: {r3['indices'].shape} (Expected [1, 3])")
        print(f"T3 Weights: {r3['weights']}")
    
    if r1:
        print(f"T1 Indices: {r1['indices'].shape} (Expected [1, 2])")
        # Check Compliance Clamping (0.5 -> 2.0)
        clamped_w = r1['weights'][0].item()
        print(f"T1 Weight: {clamped_w}")
        assert clamped_w == 2.0, "Compliance Box failed to clamp low weight!"
    
    print("\n[+] SRTL Test Passed.")