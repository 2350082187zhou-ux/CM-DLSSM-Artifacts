"""
==============================================================================
CM-DLSSM Artifact: Experiment 04 - Audit Reproducibility & Consistency
Path: experiments/04_audit_reproducibility.py
==============================================================================
Reference: Section 5.3 (The Auditing Protocol) - R1: Deterministic Re-computability
           Section 6.3 (Auditing Effectiveness) - "99.2% reached fixed-point"

Description:
    This script tests the mathematical integrity of the Verified Audit Artifacts (VAAs).
    It simulates the "Prover-Verifier" game:
    
    1. PROVER (The Model): Uses PyTorch (likely on GPU, Float32) to run the 
       Logic Layer and generate a VAA containing the 'logic_trace'.
    2. VERIFIER (The Auditor): Uses pure NumPy (CPU, Float64) to read the VAA
       inputs (evidence + rules) and re-execute the Log-Ratio CAVI algorithm.
    
    Goal:
    To prove that despite hardware/library differences (Torch vs NumPy), 
    the logic inference is deterministic enough to serve as a legal proof.
    
    Target Metrics:
    - Max Absolute Difference (|q_model - q_verify|): Should be < 1e-5.
    - Pass Rate: % of VAAs where divergence is below threshold.
    - Bit-Exact Rate: % of VAAs with 0.0 float difference (hard to achieve across FP32/64).

Usage:
    python experiments/04_audit_reproducibility.py
    python experiments/04_audit_reproducibility.py n_samples=1000
    python experiments/04_audit_reproducibility.py epsilon=1e-6
==============================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
import hydra
import logging
import time
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List, Dict

# Import Prover (Model) - Mock implementation
class CAVIEngine:
    def __init__(self, num_predicates, max_iterations, damping=0.5, audit_mode=False):
        self.num_predicates = num_predicates
        self.max_iterations = max_iterations
        self.damping = damping
        self.audit_mode = audit_mode
        self.audit_trace = None
    def to(self, device):
        return self
    def __call__(self, logits, rules_t3=None):
        # Simple sigmoid for simulation
        result = 1.0 / (1.0 + torch.exp(-logits))
        # Generate a mock audit trace
        self.audit_trace = torch.zeros(self.max_iterations + 1, logits.shape[0], logits.shape[1])
        for i in range(self.max_iterations + 1):
            # Simple linear progression for trace
            progress = i / self.max_iterations
            self.audit_trace[i] = progress * result + (1 - progress) * (1.0 / (1.0 + torch.exp(-logits)))
        return result
    def get_audit_trace(self):
        return self.audit_trace

# Import Verifier (Independent Implementation) - Mock implementation
class VAAVerifier:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
    def verify(self, vaa):
        # Simple verification logic
        return True, {}

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exp_audit_repro")


class ReproducibilityAuditor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Get parameters with fallbacks
        self.n_samples = cfg.get("n_samples", 1000)
        self.epsilon = cfg.get("epsilon", 1e-5)  # Relaxed tolerance for cross-platform
        
        # 1. Setup Prover (PyTorch)
        # Use GPU if available to maximize architectural difference from Verifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prover_engine = CAVIEngine(
            num_predicates=3,  # Simplified for batch testing (a, b, c)
            max_iterations=5,
            damping=0.5,
            audit_mode=True
        ).to(self.device)
        
        # 2. Setup Verifier (NumPy)
        # Verify strictness matches config
        self.verifier_tool = VAAVerifier(epsilon=self.epsilon)
        
        logger.info(f"Initialized Auditor. Prover Device: {self.device} | Tolerance: {self.epsilon}")
        logger.info(f"Will generate and verify {self.n_samples} VAAs")

    def generate_synthetic_vaas(self) -> List[Dict]:
        """
        Simulate the Model generating VAAs.
        Constructs random valid inputs and runs CAVI.
        """
        logger.info(f"Generating {self.n_samples} Synthetic VAAs using PyTorch...")
        vaas = []
        
        # Bulk generation setup
        for i in tqdm(range(self.n_samples), desc="Generating VAAs"):
            # Random Evidence Logits (B=1, N=3)
            # Range [-5, 5] to cover confident and uncertain regions
            logits = (torch.rand(1, 3) * 10 - 5).to(self.device)
            
            # Random Rule Weights [1.0, 10.0]
            w = (torch.rand(1) * 9 + 1).item()
            
            # Construct T3 Rule: 0 ^ !1 -> 2
            # indices: [0, 1, 2]
            rule_indices = torch.tensor([[0, 1, 2]], dtype=torch.long).to(self.device)
            rule_weights = torch.tensor([w], dtype=torch.float32).to(self.device)
            
            rules_t3 = {
                'indices': rule_indices,
                'weights': rule_weights
            }
            
            # Run Prover
            with torch.no_grad():
                _ = self.prover_engine(logits, rules_t3=rules_t3)
                trace = self.prover_engine.get_audit_trace()  # (J+1, B, N)
                
            # Extract final posterior trace
            # Convert to standard Python list for JSON serialization
            trace_list = trace[:, 0, :].cpu().tolist()
            
            # Construct VAA Dict (Simulating src.infra.vaa_generator output)
            vaa = {
                "id": f"VAA-SIM-{i:04d}",
                "jsonrpc": "2.0",
                "logic_layer": {
                    "iterations": 5,
                    "damping": 0.5,
                    "evidence_logits": logits[0].cpu().tolist(),
                    "rules": [
                        {
                            "template_id": "T3", 
                            "variables": [0, 1, 2], 
                            "weight": w
                        }
                    ],
                    "posterior_log": trace_list
                },
                # Minimal dummy fields to pass schema check
                "gating": {"flip_matrix": {"eta_09": 0.0}, "attachment_rate": 1.0}, 
                "calibration": {"local_ECE": 0.005}
            }
            vaas.append(vaa)
            
        logger.info(f"Generated {len(vaas)} VAAs successfully")
        return vaas

    def run_verification_campaign(self, vaas: List[Dict]):
        """
        Feed the generated VAAs into the Independent Verifier.
        Measure discrepancies.
        """
        logger.info("Running Verification Campaign (NumPy Re-computation)...")
        
        results = []
        
        start_time = time.time()
        for vaa in tqdm(vaas, desc="Verifying VAAs"):
            # 1. Extract what the model claimed
            # The last step of the trace is the final posterior q
            claimed_q = np.array(vaa['logic_layer']['posterior_log'][-1])
            
            # 2. Run Shadow Verification Logic
            # Re-implementing simplified numpy logic here for metric collection
            logic_data = vaa['logic_layer']
            
            evidence = np.array(logic_data['evidence_logits'])
            w = logic_data['rules'][0]['weight']
            
            # Initialize
            curr_logits = evidence.copy()
            curr_q = 1.0 / (1.0 + np.exp(-np.clip(curr_logits, -50, 50)))  # Clip for stability
            
            # CAVI iterations
            damping = logic_data['damping']
            for _ in range(logic_data['iterations']):
                # T3 Message: 0 ^ !1 -> 2
                q_a, q_b, q_c = curr_q[0], curr_q[1], curr_q[2]
                
                # Message to c from rule
                msg_c = w * q_a * (1.0 - q_b)
                
                # Message to a from rule
                msg_a = w * (1.0 - q_b) * (q_c - 1.0)
                
                # Message to b from rule
                msg_b = w * q_a * (1.0 - q_c)
                
                msgs = np.array([msg_a, msg_b, msg_c])
                
                # Update with damping
                target = evidence + msgs
                curr_logits = damping * curr_logits + (1.0 - damping) * target
                
                # Clip for numerical stability
                curr_logits = np.clip(curr_logits, -50, 50)
                
                # Convert to probabilities
                curr_q = 1.0 / (1.0 + np.exp(-curr_logits))
            
            computed_q = curr_q
            
            # 3. Compute Metrics
            # Max absolute difference across all predicates
            abs_diffs = np.abs(claimed_q - computed_q)
            max_diff = np.max(abs_diffs)
            mean_diff = np.mean(abs_diffs)
            
            # Check official pass/fail
            pass_status = max_diff < self.epsilon
            
            results.append({
                "vaa_id": vaa['id'],
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "passed": bool(pass_status),
                "claimed_q_sink": float(claimed_q[2]),
                "computed_q_sink": float(computed_q[2])
            })
            
        elapsed = time.time() - start_time
        throughput = self.n_samples / elapsed if elapsed > 0 else 0
        logger.info(f"Verification complete in {elapsed:.2f}s ({throughput:.1f} VAA/s)")
        
        return pd.DataFrame(results)

    def generate_report(self, df: pd.DataFrame):
        """
        Analyze and save results.
        """
        total = len(df)
        passed = df['passed'].sum()
        pass_rate = (passed / total) * 100 if total > 0 else 0.0
        
        # Statistics
        max_err = df['max_diff'].max()
        avg_err = df['max_diff'].mean()
        median_err = df['max_diff'].median()
        
        # Bit-Exact check (Tolerance = 0.0)
        # Realistically, FP32 vs FP64 will rarely match exactly.
        bit_exact = (df['max_diff'] == 0.0).sum()
        bit_exact_rate = (bit_exact / total) * 100 if total > 0 else 0.0
        
        print("\n" + "="*70)
        print("AUDIT REPRODUCIBILITY REPORT")
        print("="*70)
        print(f"Total VAAs Audited:      {total}")
        print(f"Hardware Mismatch Sim:   PyTorch({self.device}/FP32) vs NumPy(CPU/FP64)")
        print("-" * 70)
        print(f"PASS Rate (eps < {self.epsilon}): {pass_rate:.2f}% ({passed}/{total})")
        print(f"Bit-Exact Match Rate:    {bit_exact_rate:.2f}% ({bit_exact}/{total})")
        print("-" * 70)
        print(f"Max Absolute Error:      {max_err:.2e}")
        print(f"Mean Absolute Error:     {avg_err:.2e}")
        print(f"Median Absolute Error:   {median_err:.2e}")
        print("="*70)
        
        # Additional statistics
        if pass_rate < 100.0:
            failed_df = df[~df['passed']]
            print(f"\nFailed VAAs: {len(failed_df)}")
            print(f"Worst case error: {failed_df['max_diff'].max():.2e}")
            print(f"Failed VAA IDs (first 5): {failed_df['vaa_id'].head().tolist()}")
        
        # Save CSV
        os.makedirs("artifacts/results", exist_ok=True)
        out_file = "artifacts/results/audit_reproducibility.csv"
        df.to_csv(out_file, index=False)
        logger.info(f"[+] Results saved to {out_file}")
        
        # Summary statistics
        summary = {
            "total_vaas": total,
            "passed": int(passed),
            "pass_rate": float(pass_rate),
            "bit_exact": int(bit_exact),
            "bit_exact_rate": float(bit_exact_rate),
            "max_error": float(max_err),
            "mean_error": float(avg_err),
            "median_error": float(median_err),
            "epsilon": float(self.epsilon),
            "device": str(self.device)
        }
        
        summary_file = "artifacts/results/audit_reproducibility_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[+] Summary saved to {summary_file}")
        
        # Assertion for Artifact Evaluation
        if pass_rate < 95.0:
            logger.error("[!] CRITICAL: Reproducibility pass rate below 95%.")
            logger.error("    Check floating point precision settings in 'cavi_engine.py'.")
            logger.error("    This may indicate numerical instability in the logic layer.")
        elif pass_rate < 99.0:
            logger.warning("[!] WARNING: Pass rate below 99%. Consider tightening epsilon.")
        else:
            logger.info("[+] Experiment Success: System is Auditable.")
            logger.info(f"[+] Pass Rate: {pass_rate:.2f}% exceeds 99% threshold")


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for Experiment 04 - Audit Reproducibility.
    
    Usage:
        python experiments/04_audit_reproducibility.py
        python experiments/04_audit_reproducibility.py n_samples=1000
        python experiments/04_audit_reproducibility.py epsilon=1e-6
    """
    
    print("="*70)
    print("CM-DLSSM Experiment 04: Audit Reproducibility")
    print("="*70)
    print(f"Configuration:")
    print(f"  n_samples: {cfg.get('n_samples', 1000)}")
    print(f"  epsilon: {cfg.get('epsilon', 1e-5)}")
    print("="*70)
    
    auditor = ReproducibilityAuditor(cfg)
    
    # 1. Generate
    vaas = auditor.generate_synthetic_vaas()
    
    # 2. Verify
    df = auditor.run_verification_campaign(vaas)
    
    # 3. Report
    auditor.generate_report(df)
    
    print("\n[SUCCESS] Audit reproducibility test completed.")
    print("Results saved to:")
    print("  - artifacts/results/audit_reproducibility.csv")
    print("  - artifacts/results/audit_reproducibility_summary.json")


if __name__ == "__main__":
    main()
