"""
==============================================================================
CM-DLSSM Diagnostic Script: Causal Inference Quality Check
Path: diagnose_causal.py
==============================================================================
Purpose: 
    Diagnose why the Doubly Robust Estimator performed worse than naive methods.
    This script checks:
    1. Propensity Score Overlap (Positivity Assumption)
    2. Covariate Balance
    3. Extreme Weights (IPW Diagnostics)
    4. Outcome Model Fit Quality
    5. True vs Estimated ATE Comparison

Usage:
    python3 diagnose_causal.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CM-DLSSM: Causal Inference Diagnostic Report")
print("="*70)

# ==============================================================================
# 1. Load Data
# ==============================================================================
print("\n[STEP 1] Loading Test Data...")
df_test = pd.read_csv('data/processed/causal/blv_simulation_test.csv')

# Extract features
u_cols = [c for c in df_test.columns if c.startswith('u_ctx_')]
U = df_test[u_cols].values
A = df_test['action_A'].values
S = df_test['outcome_S'].values

# Get ground truth
true_ate = df_test['__oracle_effect'].mean()

print(f"  ✓ Loaded {len(df_test)} samples")
print(f"  ✓ Features: {len(u_cols)} confounders")
print(f"  ✓ Ground Truth ATE: {true_ate:.4f}")

# ==============================================================================
# 2. Basic Statistics
# ==============================================================================
print("\n[STEP 2] Basic Statistics")
print("-"*70)

attack_rate = A.mean()
print(f"Attack Rate (Treatment Prevalence): {attack_rate:.2%}")

if attack_rate < 0.05 or attack_rate > 0.95:
    print("  ⚠️  WARNING: Extreme treatment prevalence may cause instability")

print(f"\nOutcome Distribution:")
print(f"  Overall Mean: {S.mean():.3f} ± {S.std():.3f}")
print(f"  Attack Group (A=1): {S[A==1].mean():.3f} ± {S[A==1].std():.3f}")
print(f"  Control Group (A=0): {S[A==0].mean():.3f} ± {S[A==0].std():.3f}")
print(f"  Naive ATE (Observed Diff): {S[A==1].mean() - S[A==0].mean():.3f}")

# ==============================================================================
# 3. Covariate Balance Check
# ==============================================================================
print("\n[STEP 3] Covariate Balance (Before Adjustment)")
print("-"*70)

# Standardized Mean Difference (SMD)
mean_treated = df_test[df_test['action_A']==1][u_cols].mean()
mean_control = df_test[df_test['action_A']==0][u_cols].mean()
std_pooled = np.sqrt(
    (df_test[df_test['action_A']==1][u_cols].var() + 
     df_test[df_test['action_A']==0][u_cols].var()) / 2
)
smd = (mean_treated - mean_control) / std_pooled

print("\nStandardized Mean Differences (SMD):")
print("  Feature | SMD    | Interpretation")
print("  --------|--------|------------------")
for i, col in enumerate(u_cols):
    status = "✓ Good" if abs(smd.iloc[i]) < 0.1 else ("⚠ Moderate" if abs(smd.iloc[i]) < 0.25 else "✗ Poor")
    print(f"  {col:8s}| {smd.iloc[i]:6.3f} | {status}")

avg_smd = abs(smd).mean()
print(f"\nAverage |SMD|: {avg_smd:.3f}")
if avg_smd < 0.1:
    print("  ✓ Excellent balance (< 0.1)")
elif avg_smd < 0.25:
    print("  ⚠ Acceptable balance (0.1 - 0.25)")
else:
    print("  ✗ Poor balance (> 0.25) - Confounding is severe!")

# ==============================================================================
# 4. Propensity Score Model
# ==============================================================================
print("\n[STEP 4] Propensity Score Diagnostics")
print("-"*70)

ps_model = LogisticRegression(max_iter=1000, solver='lbfgs')
ps_model.fit(U, A)
ps_scores = ps_model.predict_proba(U)[:, 1]

print(f"Propensity Score Model: Logistic Regression")
print(f"  Accuracy: {ps_model.score(U, A):.3f}")

print(f"\nPropensity Score Distribution:")
print(f"  Attack Group (A=1):")
print(f"    Mean: {ps_scores[A==1].mean():.3f}")
print(f"    Std:  {ps_scores[A==1].std():.3f}")
print(f"    Min:  {ps_scores[A==1].min():.3f}")
print(f"    Max:  {ps_scores[A==1].max():.3f}")

print(f"  Control Group (A=0):")
print(f"    Mean: {ps_scores[A==0].mean():.3f}")
print(f"    Std:  {ps_scores[A==0].std():.3f}")
print(f"    Min:  {ps_scores[A==0].min():.3f}")
print(f"    Max:  {ps_scores[A==0].max():.3f}")

# Overlap Check
overlap_min = max(ps_scores[A==1].min(), ps_scores[A==0].min())
overlap_max = min(ps_scores[A==1].max(), ps_scores[A==0].max())
overlap_range = overlap_max - overlap_min

print(f"\nOverlap Region: [{overlap_min:.3f}, {overlap_max:.3f}]")
print(f"  Overlap Width: {overlap_range:.3f}")

if overlap_range < 0:
    print("  ✗ CRITICAL: No overlap! Positivity assumption violated!")
elif overlap_range < 0.3:
    print("  ⚠ WARNING: Limited overlap. Estimates may be unstable.")
else:
    print("  ✓ Good overlap.")

# ==============================================================================
# 5. IPW Weights Diagnostics
# ==============================================================================
print("\n[STEP 5] Inverse Propensity Weighting (IPW) Diagnostics")
print("-"*70)

# Clip propensity scores to avoid extreme weights
ps_clipped = np.clip(ps_scores, 0.05, 0.95)

# Calculate IPW weights
weights = np.where(A == 1, 1 / ps_clipped, 1 / (1 - ps_clipped))

print(f"IPW Weight Statistics:")
print(f"  Mean: {weights.mean():.2f}")
print(f"  Std:  {weights.std():.2f}")
print(f"  Min:  {weights.min():.2f}")
print(f"  Max:  {weights.max():.2f}")
print(f"  Median: {np.median(weights):.2f}")

# Check for extreme weights
extreme_threshold = 10
n_extreme = np.sum(weights > extreme_threshold)
print(f"\nExtreme Weights (> {extreme_threshold}): {n_extreme} ({n_extreme/len(weights)*100:.1f}%)")

if weights.max() > 100:
    print("  ✗ CRITICAL: Extreme weights detected (> 100)!")
    print("    This indicates severe positivity violations.")
elif weights.max() > 20:
    print("  ⚠ WARNING: Large weights detected (> 20).")
    print("    Consider more flexible propensity models or trimming.")
else:
    print("  ✓ Weights are reasonable.")

# ==============================================================================
# 6. Outcome Model Diagnostics
# ==============================================================================
print("\n[STEP 6] Outcome Model Diagnostics")
print("-"*70)

# Test different outcome models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest (d=8)": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
    "Random Forest (d=15)": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
}

X_full = np.column_stack([U, A])

print("Outcome Model Performance (R² and MAE):")
print("  Model                    | R²     | MAE   ")
print("  -------------------------|--------|-------")

for name, model in models.items():
    model.fit(X_full, S)
    S_pred = model.predict(X_full)
    r2 = r2_score(S, S_pred)
    mae = mean_absolute_error(S, S_pred)
    print(f"  {name:24s} | {r2:6.3f} | {mae:5.2f}")

# ==============================================================================
# 7. Manual ATE Estimation
# ==============================================================================
print("\n[STEP 7] Manual ATE Estimation (Verification)")
print("-"*70)

# Naive
ate_naive = S[A==1].mean() - S[A==0].mean()

# IPW
ate_ipw = np.mean(weights[A==1] * S[A==1]) / np.mean(weights[A==1]) - \
          np.mean(weights[A==0] * S[A==0]) / np.mean(weights[A==0])

# Regression (Linear)
reg_model = LinearRegression()
reg_model.fit(X_full, S)
X_1 = np.column_stack([U, np.ones(len(U))])
X_0 = np.column_stack([U, np.zeros(len(U))])
ate_reg = np.mean(reg_model.predict(X_1) - reg_model.predict(X_0))

# Doubly Robust (Simplified)
mu_1 = reg_model.predict(X_1)
mu_0 = reg_model.predict(X_0)
dr_term_1 = (A * S) / ps_clipped - ((A - ps_clipped) / ps_clipped) * mu_1
dr_term_0 = ((1-A) * S) / (1 - ps_clipped) + ((A - ps_clipped) / (1 - ps_clipped)) * mu_0
ate_dr = np.mean(dr_term_1 - dr_term_0)

print("Method                    | Estimate | Bias   | Abs Error")
print("--------------------------|----------|--------|----------")
print(f"Ground Truth              | {true_ate:8.4f} |   -    |    -")
print(f"Naive (Observed Diff)     | {ate_naive:8.4f} | {ate_naive - true_ate:6.3f} | {abs(ate_naive - true_ate):6.3f}")
print(f"IPW                       | {ate_ipw:8.4f} | {ate_ipw - true_ate:6.3f} | {abs(ate_ipw - true_ate):6.3f}")
print(f"Regression (Linear)       | {ate_reg:8.4f} | {ate_reg - true_ate:6.3f} | {abs(ate_reg - true_ate):6.3f}")
print(f"Doubly Robust (Manual)    | {ate_dr:8.4f} | {ate_dr - true_ate:6.3f} | {abs(ate_dr - true_ate):6.3f}")

# ==============================================================================
# 8. Visualization
# ==============================================================================
print("\n[STEP 8] Generating Diagnostic Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Propensity Score Distribution
ax = axes[0, 0]
ax.hist(ps_scores[A==0], bins=30, alpha=0.5, label='Control (A=0)', density=True, color='blue')
ax.hist(ps_scores[A==1], bins=30, alpha=0.5, label='Attack (A=1)', density=True, color='red')
ax.axvline(0.05, color='black', linestyle='--', linewidth=1, label='Clip Bounds')
ax.axvline(0.95, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Propensity Score P(A=1|U)')
ax.set_ylabel('Density')
ax.set_title('Propensity Score Overlap')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: IPW Weights Distribution
ax = axes[0, 1]
ax.hist(weights, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {weights.mean():.2f}')
ax.axvline(np.median(weights), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(weights):.2f}')
ax.set_xlabel('IPW Weight')
ax.set_ylabel('Frequency')
ax.set_title('IPW Weight Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Covariate Balance (Love Plot)
ax = axes[1, 0]
ax.barh(range(len(u_cols)), abs(smd.values), color=['green' if abs(x) < 0.1 else 'orange' if abs(x) < 0.25 else 'red' for x in smd.values])
ax.axvline(0.1, color='green', linestyle='--', linewidth=1, label='Good (< 0.1)')
ax.axvline(0.25, color='orange', linestyle='--', linewidth=1, label='Acceptable (< 0.25)')
ax.set_yticks(range(len(u_cols)))
ax.set_yticklabels(u_cols)
ax.set_xlabel('Absolute Standardized Mean Difference')
ax.set_title('Covariate Balance (Love Plot)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: ATE Estimates Comparison
ax = axes[1, 1]
methods = ['True ATE', 'Naive', 'IPW', 'Regression', 'DR (Manual)']
estimates = [true_ate, ate_naive, ate_ipw, ate_reg, ate_dr]
colors = ['black', 'blue', 'orange', 'green', 'red']
bars = ax.barh(methods, estimates, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(true_ate, color='black', linestyle='--', linewidth=2, label='Ground Truth')
ax.set_xlabel('Estimated ATE')
ax.set_title('ATE Estimation Comparison')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('artifacts/results/causal/diagnostic_report.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved diagnostic plots to: artifacts/results/causal/diagnostic_report.png")

# ==============================================================================
# 9. Summary and Recommendations
# ==============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

issues = []
recommendations = []

# Check 1: Covariate Balance
if avg_smd > 0.25:
    issues.append("Poor covariate balance (SMD > 0.25)")
    recommendations.append("Use more flexible propensity models (e.g., Gradient Boosting)")

# Check 2: Overlap
if overlap_range < 0.3:
    issues.append("Limited propensity score overlap")
    recommendations.append("Consider trimming extreme propensity scores or using matching")

# Check 3: Extreme Weights
if weights.max() > 20:
    issues.append(f"Extreme IPW weights detected (max: {weights.max():.1f})")
    recommendations.append("Use weight trimming or stabilized weights")

# Check 4: Outcome Model Fit
if r2 < 0.5:
    issues.append("Poor outcome model fit (R² < 0.5)")
    recommendations.append("Try non-linear models or add interaction terms")

# Check 5: DR Performance
if abs(ate_dr - true_ate) > abs(ate_naive - true_ate):
    issues.append("DR estimator performs worse than naive method")
    recommendations.append("Both propensity and outcome models may be misspecified")

if len(issues) == 0:
    print("\n✓ No major issues detected. The causal inference setup appears sound.")
else:
    print(f"\n⚠ Found {len(issues)} potential issues:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

print("\n" + "="*70)
print("Diagnostic report complete. Review plots for visual inspection.")
print("="*70)
