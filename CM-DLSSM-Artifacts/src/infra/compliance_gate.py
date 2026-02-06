"""
==============================================================================
CM-DLSSM Artifact: Compliance Gating & Release Protocols
Path: src/infra/compliance_gate.py
==============================================================================
Reference: Section 5.2 (Compliance Gating) - Eq (18)
           Appendix C (VAA Schema - "gating" field)

Description:
    This module implements the "Quality Control" layer of the system.
    It decides whether a specific finding is mathematically robust enough to be
    issued as a Verified Audit Artifact (VAA).

    The "Abstain" Mechanism:
    Unlike standard classifiers that forcedly output 0 or 1, CM-DLSSM can
    "Abstain" (Refuse to predict) if the internal consistency checks fail.
    This effectively converts "False Positives" into "System Warnings",
    reducing alert fatigue.

    Gate States:
    1. PASS:    High confidence, Consistent logic, Rule attached. (Issue Alert)
    2. FAIL:    Low confidence (Safe).
    3. ABSTAIN: Inconsistent logic (Flip Rate high) OR Low Rule Coverage.
                (Do NOT issue VAA, flag for manual review).
==============================================================================
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, Tuple, Optional

class GateStatus(Enum):
    PASS = "PASS"         # Vulnerable & Verifiable
    FAIL = "FAIL"         # Safe / Not Vulnerable
    ABSTAIN = "ABSTAIN"   # Internal Consistency Error (Don't trust result)

class ComplianceGate:
    def __init__(
        self,
        flip_threshold_09: float = 0.0001, # 0.01% (The Red Line)
        min_attachment_rate: float = 0.98, # 98% Rule Coverage needed
        decision_threshold: float = 0.5    # Prob > 0.5 is "Vulnerable"
    ):
        """
        Args:
            flip_threshold_09: Max allowed flip rate at tau=0.9.
            min_attachment_rate: Min % of SRVS sinks that must trigger rules.
            decision_threshold: Cutoff for the final posterior q.
        """
        self.flip_limit = flip_threshold_09
        self.cov_limit = min_attachment_rate
        self.dec_thresh = decision_threshold
        
        # Audit Logs
        self.last_flip_matrix = {}
        self.last_status = None

    def compute_flip_matrix(self, q_soft: torch.Tensor, q_hard: torch.Tensor) -> Dict[str, float]:
        """
        Calculates eta(tau) for tau in {0.5, 0.7, 0.9}.
        Equation (18).
        """
        # q_soft: (B, N) - Differentiable posterior
        # q_hard: (B, N) - Deterministic posterior (Logic Fixed Point)
        
        thresholds = [0.5, 0.7, 0.9]
        matrix = {}
        
        with torch.no_grad():
            for tau in thresholds:
                # sgn(q - tau)
                # We implement as: (q > tau)
                dec_soft = (q_soft > tau).float()
                dec_hard = (q_hard > tau).float()
                
                # Mismatch count
                mismatches = torch.abs(dec_soft - dec_hard).sum().item()
                total = q_soft.numel()
                
                eta = mismatches / (total + 1e-9)
                matrix[f"eta_{str(tau).replace('.', '')}"] = eta
                
        return matrix

    def evaluate(
        self, 
        q_final: torch.Tensor, 
        q_soft_check: torch.Tensor, 
        attachment_rate: float
    ) -> Tuple[GateStatus, Dict]:
        """
        Main decision function.
        
        Args:
            q_final: The final calibrated probability (for decision).
            q_soft_check: The soft probability (for consistency check).
            attachment_rate: From SRVSGate stats.
            
        Returns:
            (Status, Audit_Dict)
        """
        # 1. Compute Consistency (Flip Matrix)
        # We compare the final output vs the soft training signal
        flip_matrix = self.compute_flip_matrix(q_soft_check, q_final)
        self.last_flip_matrix = flip_matrix
        
        eta_09 = flip_matrix.get("eta_09", 1.0)
        
        # 2. Check Criteria 1: Consistency Gate
        if eta_09 > self.flip_limit:
            # The model is "guessing" at high confidence (Soft/Hard disagree).
            return GateStatus.ABSTAIN, self._make_report("Flip Rate Violation", flip_matrix)

        # 3. Check Criteria 2: Attachment Gate
        # Only relevant if we found something (q > threshold)
        # If model says Safe, we don't care if rules didn't attach.
        is_positive = (q_final.max() > self.dec_thresh).item()
        
        if is_positive and attachment_rate < self.cov_limit:
            # We found a 'bug' but no rules were attached to verify it.
            # Likely a hallucination or uncovered code path.
            return GateStatus.ABSTAIN, self._make_report(f"Low Attachment ({attachment_rate:.2f})", flip_matrix)

        # 4. Final Decision
        if is_positive:
            return GateStatus.PASS, self._make_report("VERIFIED", flip_matrix)
        else:
            return GateStatus.FAIL, self._make_report("SAFE", flip_matrix)

    def _make_report(self, msg, matrix):
        return {
            "status_msg": msg,
            "flip_matrix": matrix,
            "thresholds": {"eta_09": self.flip_limit}
        }

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing Compliance Gate...")
    
    gate = ComplianceGate(flip_threshold_09=0.01) # 1% limit for test
    
    # Case 1: Consistent & Vulnerable -> PASS
    q_hard = torch.tensor([[0.95]])
    q_soft = torch.tensor([[0.94]]) # Consistent > 0.9
    att_rate = 1.0
    
    status, report = gate.evaluate(q_hard, q_soft, att_rate)
    print(f"Case 1 (Ideal): {status.value} | Eta_09: {report['flip_matrix']['eta_09']:.4f}")
    assert status == GateStatus.PASS
    
    # Case 2: Inconsistent (Flip) -> ABSTAIN
    q_hard = torch.tensor([[0.95]]) # Logic says Vulnerable
    q_soft = torch.tensor([[0.80]]) # Neural says Unsure (< 0.9)
    # At tau=0.9, Hard=1, Soft=0 -> Flip!
    
    status, report = gate.evaluate(q_hard, q_soft, att_rate)
    print(f"Case 2 (Flip):  {status.value} | Eta_09: {report['flip_matrix']['eta_09']:.4f}")
    assert status == GateStatus.ABSTAIN
    
    # Case 3: Low Attachment -> ABSTAIN
    att_rate_low = 0.5
    status, report = gate.evaluate(q_hard, q_hard, att_rate_low) # Consistent but ungoverned
    print(f"Case 3 (Miss):  {status.value} | Reason: {report['status_msg']}")
    assert status == GateStatus.ABSTAIN
    
    print("[+] Gate Test Passed.")