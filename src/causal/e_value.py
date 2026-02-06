"""
==============================================================================
CM-DLSSM Artifact: E-Value Sensitivity Analysis
Path: src/causal/e_value.py
==============================================================================
Reference: Section 4.4 (Causal Inference Module) - Sensitivity Analysis
           Section 6.4 (Case Studies) - "E-Value of 5.4"
           Appendix C (Causal Trace)

Description:
    This module implements the E-Value metric (VanderWeele & Ding, 2017).
    
    The "Doubly Robust Estimator" assumes "No Unmeasured Confounders". 
    In real-world audit logs, this is a strong assumption. There might be a 
    hidden variable (e.g., an off-chain private key state) that we didn't model.

    The E-Value quantifies the robustness of the causal conclusion.
    It represents the minimum strength of association that an unmeasured
    confounder would need to have with both the Treatment (A) and the Outcome (S)
    to explain away the estimated causal effect.

    Formula: E-Value = RR + sqrt(RR * (RR - 1))
    Where RR is the Risk Ratio.

    Audit Usage:
    - E-Value = 1.0: No robustness (Correlation could easily be spurious).
    - E-Value > 3.0: Strong evidence (Requires a very powerful hidden cause).
==============================================================================
"""

import numpy as np
from typing import Dict, Tuple, Union

class EValueAnalyzer:
    def __init__(self):
        pass

    def compute_risk_ratio(self, ate: float, baseline_risk: float) -> float:
        """
        Converts Average Treatment Effect (Risk Difference) to Risk Ratio.
        
        Args:
            ate: The estimated causal effect (e.g., 0.82 from DR estimator).
            baseline_risk: The probability of outcome in the control group P(S=1|A=0).
                           This is usually estimated by the Outcome Model mu0.
        
        Returns:
            RR: Risk Ratio = P(S=1|A=1) / P(S=1|A=0)
        """
        # P(Treatment) = Baseline + Effect
        treated_risk = baseline_risk + ate
        
        # Clip to valid probability range [0, 1]
        treated_risk = np.clip(treated_risk, 0.0, 1.0)
        baseline_risk = np.clip(baseline_risk, 1e-6, 1.0) # Avoid div/0
        
        rr = treated_risk / baseline_risk
        return float(rr)

    def calculate_e_value(self, risk_ratio: float) -> float:
        """
        Computes E-Value for a Risk Ratio > 1.
        Formula: RR + sqrt( RR * (RR - 1) )
        """
        if risk_ratio <= 1.0:
            # If RR < 1, the action reduces risk (or no effect).
            # For vulnerability detection, we usually care about RR > 1 (Increased Risk).
            # However, if we are evaluating a "Patch", we might care about inverse.
            # Here we return 1.0 for "No evidence of increased vulnerability".
            return 1.0
            
        e_val = risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1))
        return float(e_val)

    def run_analysis(self, dr_estimate: float, control_baseline: float) -> Dict[str, Union[float, str]]:
        """
        Full pipeline to generate the VAA Causal Trace entry.
        
        Args:
            dr_estimate: The ATE from DoublyRobustEstimator.
            control_baseline: The average outcome for the control group.
        """
        # 1. Convert ATE -> RR
        rr = self.compute_risk_ratio(dr_estimate, control_baseline)
        
        # 2. Compute E-Value
        e_value = self.calculate_e_value(rr)
        
        # 3. Interpret
        interpretation = self._describe_strength(e_value)
        
        return {
            "metric": "Risk Ratio (converted from ATE)",
            "risk_ratio": round(rr, 4),
            "e_value": round(e_value, 4),
            "interpretation": interpretation,
            "robustness_check": "PASS" if e_value > 1.5 else "FAIL"
        }

    def _describe_strength(self, e_val: float) -> str:
        """Qualitative interpretation for the Audit Report."""
        if e_val == 1.0:
            return "None (Null Effect)"
        elif e_val < 1.5:
            return "Weak (Sensitive to Confounding)"
        elif e_val < 3.0:
            return "Moderate (Plausible Causal Link)"
        elif e_val < 5.0:
            return "Strong (Likely Causal)"
        else:
            return "Very Strong (Robust Evidence)"

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing E-Value Sensitivity Analyzer...")
    
    analyzer = EValueAnalyzer()
    
    # Case 1: The DeFi Case Study (Section 6.4)
    # DR Estimate was 0.82 (High Risk increase)
    # Assume baseline risk of getting a reward is low (e.g., 0.05)
    ate = 0.82
    baseline = 0.05
    
    print(f"\nScenario: DeFi Oracle Manipulation")
    print(f"  ATE (DR Est):   {ate}")
    print(f"  Baseline Risk:  {baseline}")
    
    report = analyzer.run_analysis(ate, baseline)
    print("  -> Audit Report:", report)
    
    # Validation logic
    # Treated Risk = 0.87. RR = 0.87 / 0.05 = 17.4
    # E-Value approx 17.4 + sqrt(17.4 * 16.4) approx 17.4 + 16.9 = 34.3
    # Note: If baseline is extremely low, RR explodes, which is correct for security 
    # (moving from impossible to probable is a massive causal jump).
    
    # Case 2: Weak Signal
    # ATE = 0.02, Baseline = 0.10
    print(f"\nScenario: Noisy Logs")
    report_weak = analyzer.run_analysis(0.02, 0.10)
    print("  -> Audit Report:", report_weak)
    
    assert report['e_value'] > 10.0, "E-Value calculation under-estimated risk."
    assert report_weak['interpretation'].startswith("Weak"), "Interpretation failed."
    
    print("\n[+] E-Value Test Passed.")