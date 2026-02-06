"""
==============================================================================
CM-DLSSM Artifact: Business Logic Vulnerability (BLV) Synthetic Generator
Path: data/generators/blv_synth.py
==============================================================================
Reference: Section 4.4 (Causal Inference Module) & Section 6.4 (Case Studies)

Description:
    This script generates synthetic execution logs that simulate a "Business Logic
    Vulnerability" scenario. It constructs a Structural Causal Model (SCM)
    where an Action (A) produces a dangerous Outcome (S) only under specific
    Contexts (U).

    The generator introduces "Confounding Bias" (U influences both A and S),
    which makes naive statistical correlation (Correlation != Causation) fail.
    This dataset allows us to verify if the CM-DLSSM DR Estimator can recover
    the true Ground Truth Causal Effect.

Scenario: "The Coupon Logic Flaw"
    - Context (U): User attributes (e.g., 'is_vip', 'account_age', 'geo_score').
    - Action (A): Applying a specific discount code (Treatment).
    - Outcome (S): Final transaction amount (The lower, the worse for the system).
    - Vulnerability: A logical flaw allows 'new_users' (U) to stack the coupon (A)
      multiple times, causing massive revenue loss (S), but 'vip_users' cannot.
      However, 'vip_users' naturally apply coupons more often (Confounding).

Output:
    - data/processed/blv_simulation_train.csv
    - data/processed/blv_simulation_test.csv
    - blv_ground_truth_report.txt
==============================================================================
"""

import numpy as np
import pandas as pd
import os
import argparse
from scipy.special import expit as sigmoid

class BLVDataGenerator:
    def __init__(self, seed=42, n_samples=10000, confounding_strength=3.0):
        """
        Args:
            seed: Random seed for reproducibility (Audit Requirement R1).
            n_samples: Number of transaction logs to generate.
            confounding_strength: How strongly Context(U) affects Action(A).
                                  Higher = Harder for naive estimators.
        """
        np.random.seed(seed)
        self.n = n_samples
        self.gamma = confounding_strength
        
        # Dimensions of the latent context U
        self.u_dim = 5 

    def generate(self):
        """Generates the observational data and the counterfactual ground truth."""
        
        # ----------------------------------------------------------------------
        # 1. Context Generation (U) - The User State
        # ----------------------------------------------------------------------
        # U ~ N(0, 1)
        # Represents latent features: [VIP_Level, Account_Age, Risk_Score, ...]
        U = np.random.normal(0, 1, size=(self.n, self.u_dim))
        
        # ----------------------------------------------------------------------
        # 2. Propensity Score (P(A=1|U)) - The Behavior Policy
        # ----------------------------------------------------------------------
        # Simulates that certain users (e.g., VIPs) are more likely to perform 
        # the Action (apply coupon), regardless of the vulnerability.
        # This creates the "Confounding" bias.
        
        # Random weights for how U affects A
        w_a = np.random.uniform(-1, 1, size=(self.u_dim,))
        
        # Logits: We scale by gamma to increase confounding
        logits_a = np.dot(U, w_a) * self.gamma
        
        # Propensity e(x)
        propensity = sigmoid(logits_a)
        
        # Clip propensity to avoid positivity violation (0 < e < 1), 
        # unless we want to test the "Overlap" assumption failure.
        propensity = np.clip(propensity, 0.05, 0.95)
        
        # ----------------------------------------------------------------------
        # 3. Action Assignment (A) - The Intervention
        # ----------------------------------------------------------------------
        # A ~ Bernoulli(propensity)
        # A=1: Trigger Potential Exploit (e.g., Apply Coupon)
        # A=0: Normal Behavior
        A = np.random.binomial(1, propensity)
        
        # ----------------------------------------------------------------------
        # 4. Outcome Generation (S) - The Structural Equation
        # ----------------------------------------------------------------------
        # We define the Outcome S (e.g., Revenue Loss) as:
        # S = f(U) + Average_Effect * A + Interaction(U, A) + Noise
        
        # Base outcome (baseline revenue depends on user type)
        w_s = np.random.uniform(0, 1, size=(self.u_dim,))
        base_S = np.dot(U, w_s) + 10.0
        
        # The Vulnerability Logic:
        # Let's say users with U[0] < -1 (New Users) trigger the bug.
        # The bug causes a massive drop in S (Revenue) when A=1.
        
        # Define the logic condition (Hidden from the naive observer)
        is_vulnerable_context = (U[:, 0] < -1.0).astype(float)
        
        # True Causal Effects:
        # - normal_effect: Standard discount (-2.0)
        # - exploit_effect: Logic flaw double-dip (-10.0)
        treatment_effect = -2.0 + (-8.0 * is_vulnerable_context)
        
        # S = Base + (Effect * A) + Noise
        noise = np.random.normal(0, 0.5, size=self.n)
        S = base_S + (treatment_effect * A) + noise
        
        # ----------------------------------------------------------------------
        # 5. Counterfactual Ground Truth (For Validation)
        # ----------------------------------------------------------------------
        # To audit the DR estimator, we must know what WOULD have happened.
        # Potential Outcomes: S(1) and S(0)
        S_1 = base_S + treatment_effect + noise
        S_0 = base_S + noise
        
        # The True Average Treatment Effect (ATE)
        true_ate = np.mean(S_1 - S_0)
        
        # The True Average Treatment Effect on the Treated (ATT)
        # (This is often what matters in security: Impact on *actual* attacks)
        true_att = np.mean((S_1 - S_0)[A == 1])

        # ----------------------------------------------------------------------
        # 6. Packaging
        # ----------------------------------------------------------------------
        df = pd.DataFrame(U, columns=[f"u_ctx_{i}" for i in range(self.u_dim)])
        df['action_A'] = A
        df['outcome_S'] = S
        df['true_propensity'] = propensity # Saved for debugging/oracle comparison
        
        # Hidden columns (Ground Truth - NOT available to the model during training)
        df['__oracle_S1'] = S_1
        df['__oracle_S0'] = S_0
        df['__oracle_effect'] = treatment_effect
        
        return df, true_ate, true_att

def save_manifest(df, output_path, mode='train'):
    """Saves the dataset in a format simulating system logs."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_path = os.path.join(output_path, f"blv_simulation_{mode}.csv")
    df.to_csv(save_path, index=False)
    print(f"[+] Saved {mode} dataset to: {save_path} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic BLV Data")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--out_dir", type=str, default="data/processed/causal")
    args = parser.parse_args()

    print("="*60)
    print("CM-DLSSM: Generating Synthetic Business Logic Vulnerabilities")
    print("="*60)

    # 1. Generate Train Set
    print(f"[*] Generating TRAIN set (N={args.samples})...")
    gen_train = BLVDataGenerator(seed=2026, n_samples=args.samples)
    df_train, ate_train, att_train = gen_train.generate()
    save_manifest(df_train, args.out_dir, 'train')

    # 2. Generate Test Set
    print(f"[*] Generating TEST set (N={args.samples//5})...")
    gen_test = BLVDataGenerator(seed=9999, n_samples=args.samples//5)
    df_test, ate_test, att_test = gen_test.generate()
    save_manifest(df_test, args.out_dir, 'test')

    # 3. Naive Estimation (To prove the problem exists)
    # Naive ATE = E[S|A=1] - E[S|A=0]
    naive_treated = df_test[df_test['action_A'] == 1]['outcome_S'].mean()
    naive_control = df_test[df_test['action_A'] == 0]['outcome_S'].mean()
    naive_ate = naive_treated - naive_control
    
    bias = naive_ate - ate_test

    # 4. Generate Ground Truth Report
    report_path = os.path.join(args.out_dir, "blv_ground_truth.txt")
    with open(report_path, "w") as f:
        f.write("=== CM-DLSSM Causal Validation Report ===\n")
        f.write(f"Scenario: Confounded Logic Flaw (gamma={gen_train.gamma})\n\n")
        f.write(f"[Ground Truth] True ATE (Risk Effect): {ate_test:.4f}\n")
        f.write(f"[Naive Obs.]   Naive Diff (Correlation): {naive_ate:.4f}\n")
        f.write(f"[Bias]         Confounding Error:        {bias:.4f}\n\n")
        f.write("NOTE: The DR Estimator in 'src/causal/dr_estimator.py' must\n")
        f.write("      recover the 'True ATE' closer than the 'Naive Diff'.\n")
    
    print("-" * 60)
    print(f"True Causal Risk (ATE): {ate_test:.4f}")
    print(f"Naive Correlation:      {naive_ate:.4f}")
    print(f"Confounding Bias:       {bias:.4f}")
    print(f"[!] If Bias is close to 0, increase 'confounding_strength'.")
    print("="*60)

if __name__ == "__main__":
    main()