"""
==============================================================================
CM-DLSSM Artifact: Doubly Robust (DR) Causal Estimator
Path: src/causal/dr_estimator.py
==============================================================================
Reference: Section 4.4 (Causal Inference Module) - Equation (15)
           Section 6.4 (Case Studies)
           Appendix C (Causal Trace)

Description:
    This module implements the Doubly Robust Estimator for calculating the
    Average Treatment Effect (ATE) of a potential logic vulnerability.

    Why DR?
    In security logs, "Attacks" (A=1) are rare and highly confounded by user
    context (U). Naive comparisons (Average(S|A=1) - Average(S|A=0)) are biased.
    DR combines:
    1. Propensity Score Model: P(A=1|U) - Models the "Attacker Profile".
    2. Outcome Regression Model: E[S|A,U] - Models the "System Logic".

    Key Features:
    - Implements the DR equation for ATE.
    - Uses Propensity Score CLIPPING (not trimming) to avoid sample loss.
    - Computes Confidence Intervals using "Cluster Bootstrap" (by Session ID)
      to account for within-session correlations (LSI Assumption).

Changelog:
    [2026-01-18] Fixed critical bug: Changed from sample trimming to weight 
                 clipping to preserve all observations and reduce bias.
==============================================================================
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from joblib import Parallel, delayed

# Configure Logging for Audit Trace
logger = logging.getLogger("cm_dlssm.causal")

class DoublyRobustEstimator:
    def __init__(
        self,
        propensity_model: BaseEstimator = None,
        outcome_model: BaseEstimator = None,
        clip_min: float = 0.01,  # Changed from 0.05 to 0.01 (less aggressive)
        clip_max: float = 0.99,  # Changed from 0.95 to 0.99
        n_bootstrap: int = 1000,
        n_jobs: int = -1
    ):
        """
        Args:
            propensity_model: Classifier for P(A=1|U). Default: LogisticRegression.
            outcome_model: Regressor for E[S|A,U]. Default: RandomForestRegressor.
            clip_min/max: Positivity thresholds for propensity score clipping.
                         Scores are clipped (not trimmed) to avoid extreme weights.
            n_bootstrap: Number of resamples for CI calculation.
            n_jobs: Parallel jobs for bootstrapping.
        """
        # Fixed: Use 'is not None' instead of truthiness check
        self.propensity_model = propensity_model if propensity_model is not None else \
            LogisticRegression(solver='lbfgs', max_iter=1000)
        
        self.outcome_model = outcome_model if outcome_model is not None else \
            RandomForestRegressor(
                n_estimators=200,      # Increased from 100
                max_depth=15,          # Increased from 10
                min_samples_leaf=10,   # Added regularization
                random_state=42,       # Fixed seed for reproducibility
                n_jobs=-1
            )
        
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs
        
        # State
        self.is_fitted = False
        self._last_audit_trace = {}

    def fit(self, U: np.ndarray, A: np.ndarray, S: np.ndarray):
        """
        Train the nuisance models (Propensity & Outcome).
        
        Args:
            U: Context features (Confounders), shape (N, d_u)
            A: Action (Treatment), binary 0/1, shape (N,)
            S: Outcome (Metric), shape (N,)
        """
        # 1. Train Propensity Model: U -> A
        self.propensity_model.fit(U, A)
        
        # 2. Train Outcome Model: (U, A) -> S
        # We concatenate U and A to learn E[S|U,A]
        X_outcome = np.column_stack([U, A])
        self.outcome_model.fit(X_outcome, S)
        
        self.is_fitted = True
        logger.info("[DR Estimator] Nuisance models fitted successfully.")

    def estimate_ate(self, U: np.ndarray, A: np.ndarray, S: np.ndarray) -> float:
        """
        Calculate the point estimate of ATE using the DR formula.
        
        Returns:
            ate: The estimated causal risk difference.
        
        Note:
            This implementation uses WEIGHT CLIPPING instead of SAMPLE TRIMMING.
            All samples are retained, but extreme propensity scores are clipped
            to [clip_min, clip_max] to stabilize IPW weights.
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before estimation.")

        N = len(U)

        # 1. Predict Propensity Scores: e(x) = P(A=1|U)
        e_scores = self.propensity_model.predict_proba(U)[:, 1]
        
        # --- Audit Check: Positivity ---
        # CRITICAL FIX: Use weight clipping instead of sample trimming
        # This preserves all observations and reduces bias
        e_clipped = np.clip(e_scores, self.clip_min, self.clip_max)
        n_clipped = np.sum((e_scores < self.clip_min) | (e_scores > self.clip_max))
        
        if n_clipped > 0:
            logger.warning(
                f"[Positivity] Clipped {n_clipped}/{N} ({n_clipped/N*100:.1f}%) "
                f"propensity scores to [{self.clip_min}, {self.clip_max}]"
            )
        
        # Use ALL samples (no trimming)
        U_use = U
        A_use = A
        S_use = S
        e_use = e_clipped
        
        # 2. Predict Counterfactual Outcomes using Regression
        # mu1 = E[S | U, A=1]
        X_1 = np.column_stack([U_use, np.ones_like(A_use)])
        mu_1 = self.outcome_model.predict(X_1)
        
        # mu0 = E[S | U, A=0]
        X_0 = np.column_stack([U_use, np.zeros_like(A_use)])
        mu_0 = self.outcome_model.predict(X_0)

        # 3. Apply Doubly Robust Formula (Eq 15)
        # DR_1 = (A*S)/e - ((A-e)/e)*mu1
        term_1 = (A_use * S_use) / e_use - ((A_use - e_use) / e_use) * mu_1
        
        # DR_0 = ((1-A)*S)/(1-e) + ((A-e)/(1-e))*mu0
        term_0 = ((1 - A_use) * S_use) / (1 - e_use) + ((A_use - e_use) / (1 - e_use)) * mu_0
        
        dr_ate = np.mean(term_1 - term_0)
        
        # Enhanced audit trace with additional diagnostics
        self._last_audit_trace = {
            "n_samples": int(N),
            "n_clipped": int(n_clipped),
            "clip_rate": float(n_clipped / N),
            "propensity_mean": float(e_scores.mean()),
            "propensity_std": float(e_scores.std()),
            "propensity_min": float(e_scores.min()),
            "propensity_max": float(e_scores.max()),
            "dr_estimate": float(dr_ate)
        }
        
        # Debug logging
        logger.debug(f"[DR Debug] ATE={dr_ate:.4f}, E[e(X)]={e_scores.mean():.3f}, "
                    f"Var[e(X)]={e_scores.var():.3f}")
        
        return dr_ate

    def bootstrap_confidence_interval(
        self, 
        U: np.ndarray, 
        A: np.ndarray, 
        S: np.ndarray, 
        session_ids: np.ndarray = None,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Compute Confidence Interval using Cluster Bootstrap (LSI Assumption).
        
        Args:
            session_ids: Array of Session IDs. If provided, we resample SESSIONS,
                         not individual rows. This handles SUTVA violations within sessions.
            alpha: Significance level (default: 0.05 for 95% CI)
        
        Returns:
            (ci_lower, ci_upper): Percentile bootstrap confidence interval
        """
        if session_ids is None:
            # Fallback to standard i.i.d bootstrap (Risk of underestimating variance)
            logger.info("No session_ids provided. Using I.I.D Bootstrap.")
            indices = np.arange(len(U))
            resample_strategy = lambda: np.random.choice(indices, size=len(indices), replace=True)
        else:
            # Cluster/Session Bootstrap
            unique_sessions = np.unique(session_ids)
            logger.info(f"Using Session Bootstrap over {len(unique_sessions)} unique sessions.")
            
            def resample_strategy():
                # 1. Sample sessions with replacement
                sampled_sessions = np.random.choice(unique_sessions, size=len(unique_sessions), replace=True)
                # 2. Reconstruct indices
                # Note: This is a simplified/slow implementation. Pre-grouped indices are faster.
                mask_indices = []
                for sess in sampled_sessions:
                    # Find original indices for this session
                    # (In production, use a pre-built dict: sess -> [indices])
                    mask_indices.extend(np.where(session_ids == sess)[0])
                return np.array(mask_indices)

        def bootstrap_step(seed):
            np.random.seed(seed)
            idx = resample_strategy()
            if len(idx) == 0: 
                return np.nan
            
            # Refit models on resampled data for full uncertainty quantification
            # (This is computationally expensive but mathematically rigorous)
            U_b, A_b, S_b = U[idx], A[idx], S[idx]
            
            # Create a fresh estimator clone
            est_b = DoublyRobustEstimator(
                clone(self.propensity_model),
                clone(self.outcome_model),
                self.clip_min, 
                self.clip_max,
                n_bootstrap=0  # Disable nested bootstrap
            )
            
            try:
                est_b.fit(U_b, A_b, S_b)
                return est_b.estimate_ate(U_b, A_b, S_b)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {seed} failed: {e}")
                return np.nan

        # Run Parallel Bootstrap
        logger.info(f"Running {self.n_bootstrap} bootstrap iterations...")
        estimates = Parallel(n_jobs=self.n_jobs)(
            delayed(bootstrap_step)(i) for i in range(self.n_bootstrap)
        )
        
        # Filter out failed iterations
        estimates = np.array([e for e in estimates if not np.isnan(e)])
        
        if len(estimates) < self.n_bootstrap * 0.9:
            logger.warning(
                f"Only {len(estimates)}/{self.n_bootstrap} bootstrap iterations succeeded. "
                f"CI may be unreliable."
            )
        
        # Compute percentile confidence interval
        lower = np.percentile(estimates, 100 * (alpha / 2))
        upper = np.percentile(estimates, 100 * (1 - alpha / 2))
        
        logger.info(f"Bootstrap CI: [{lower:.4f}, {upper:.4f}] from {len(estimates)} estimates")
        
        return lower, upper

    def get_causal_trace(self) -> Dict[str, Any]:
        """Returns the JSON-serializable evidence for the VAA."""
        return self._last_audit_trace

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("CM-DLSSM: Testing Doubly Robust Estimator")
    print("="*70)
    
    # 1. Create Dummy Data (Confounded)
    # U determines A, U determines S. A adds effect.
    print("\n[Step 1] Generating synthetic confounded data...")
    N = 1000
    np.random.seed(42)
    U = np.random.normal(0, 1, (N, 2))
    
    # Propensity: Sigmoid(U0)
    prob_A = 1 / (1 + np.exp(-U[:, 0]))
    A = np.random.binomial(1, prob_A)
    
    # Outcome: S = U0 + U1 + 2*A + Noise (True Effect = 2.0)
    true_ate = 2.0
    S = U[:, 0] + U[:, 1] + true_ate * A + np.random.normal(0, 0.1, N)
    
    # Session IDs (10 rows per session)
    sessions = np.repeat(np.arange(N // 10), 10)
    
    print(f"  ✓ Generated {N} samples with true ATE = {true_ate:.4f}")
    print(f"  ✓ Treatment rate: {A.mean():.2%}")
    print(f"  ✓ Naive ATE: {S[A==1].mean() - S[A==0].mean():.4f}")

    # 2. Init Estimator
    print("\n[Step 2] Initializing Doubly Robust Estimator...")
    dr = DoublyRobustEstimator(
        n_bootstrap=100,  # Reduced for speed in testing
        n_jobs=1
    )
    
    # 3. Fit & Estimate
    print("\n[Step 3] Fitting nuisance models and estimating ATE...")
    dr.fit(U, A, S)
    ate = dr.estimate_ate(U, A, S)
    
    print(f"\n  Estimated ATE: {ate:.4f}")
    print(f"  True ATE:      {true_ate:.4f}")
    print(f"  Absolute Error: {abs(ate - true_ate):.4f}")
    print(f"  Relative Error: {abs(ate - true_ate) / true_ate * 100:.1f}%")
    
    # 4. Bootstrap CI
    print("\n[Step 4] Computing bootstrap confidence interval...")
    ci_lower, ci_upper = dr.bootstrap_confidence_interval(U, A, S, session_ids=sessions)
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI Width: {ci_upper - ci_lower:.4f}")
    print(f"  Coverage: {'✓ PASS' if ci_lower <= true_ate <= ci_upper else '✗ FAIL'}")
    
    # 5. Check Trace
    print("\n[Step 5] Audit trace:")
    trace = dr.get_causal_trace()
    for key, value in trace.items():
        print(f"  {key}: {value}")
    
    # 6. Assertions for smoke test
    print("\n[Step 6] Running validation checks...")
    try:
        assert 1.5 < ate < 2.5, f"ATE sanity check failed: {ate} (expected ~{true_ate})"
        print("  ✓ ATE is within reasonable range")
        
        assert ci_lower < ate < ci_upper, "Point estimate should be within CI"
        print("  ✓ Point estimate is within confidence interval")
        
        assert ci_upper - ci_lower < 1.5, "CI too wide, check bootstrap"
        print("  ✓ Confidence interval width is reasonable")
        
        assert abs(ate - true_ate) < 0.3, f"Bias too large: {abs(ate - true_ate):.4f}"
        print("  ✓ Estimation bias is acceptable")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED! DR Estimator is working correctly.")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("="*70)
        raise
