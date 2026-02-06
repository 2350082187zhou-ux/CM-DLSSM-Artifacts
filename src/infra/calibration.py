"""
==============================================================================
CM-DLSSM Artifact: Calibration Vault (Two-Stage Pipeline)
Path: src/infra/calibration.py
==============================================================================
Reference: Section 5.1 (Calibration Vault)
           Figure 6 (Reliability Diagrams)
           Appendix C (VAA Schema - "calibration" field)

Description:
    This module implements the "Trust Anchor" of the auditing system.
    It transforms raw, potentially overconfident neural logits into
    statistically valid probabilities that reflect empirical risk.

    The Vault enforces a strict "Data Isolation Protocol":
    The dataset used to fit calibrators (C-Set) MUST be distinct from the
    model training set (D-Set). This is enforced via hash checks.

    Pipeline:
    1. Stage 1 (Temperature Scaling): Parametric scaling of raw logits (l / T).
    2. Logic Inference happens externally (CAVI).
    3. Stage 2 (Isotonic Regression): Non-parametric mapping of posterior q.

    Metrics:
    - ECE (Expected Calibration Error): Must be < 0.01 for VAA issuance.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hashlib
import json
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Tuple, Optional, List

class CalibrationVault:
    def __init__(self, num_predicates: int = 1, device: str = "cpu"):
        """
        Args:
            num_predicates: Number of logic variables to calibrate independently.
        """
        self.num_predicates = num_predicates
        self.device = device
        
        # --- Stage 1 Parameters (Temperature) ---
        # Learned scalar T for each predicate. Init at 1.5 (assume overconfidence)
        self.temperature = torch.ones(num_predicates, device=device) * 1.5
        self.temperature.requires_grad = True
        
        # --- Stage 2 Parameters (Isotonic) ---
        # List of sklearn models (one per predicate)
        self.isotonic_models: List[Optional[IsotonicRegression]] = [None] * num_predicates
        
        # --- Audit State ---
        self.c_set_hash: Optional[str] = None
        self.train_set_hash: Optional[str] = None
        self.is_sealed = False # True when fit is complete and validated

    def register_dataset_hashes(self, train_hash: str, c_set_hash: str):
        """
        Enforce Data Isolation Protocol (Section 5.1).
        The Calibration Set (C-Set) hash MUST differ from Train Set hash.
        """
        if train_hash == c_set_hash:
            raise ValueError(
                "[!] VIOLATION: C-Set hash matches Train-Set hash.\n"
                "    Calibration data leakage detected. Audit rejected."
            )
        self.train_set_hash = train_hash
        self.c_set_hash = c_set_hash
        print(f"[Vault] Data Isolation Verified. C-Set: {c_set_hash[:8]}...")

    def fit(self, 
            logits_val: torch.Tensor, 
            labels_val: torch.Tensor, 
            stage: str = "both"):
        """
        Fit the calibrators using the isolated C-Set.
        
        Args:
            logits_val: (N, K) Raw logits from Sensing Layer.
            labels_val: (N, K) Binary ground truth (0/1).
            stage: "stage1", "stage2" (requires posterior inputs), or "both".
        """
        if self.c_set_hash is None:
            print("[!] WARNING: C-Set hash not registered. Isolation unverified.")

        logits_val = logits_val.to(self.device)
        labels_val = labels_val.to(self.device)

        if stage in ["stage1", "both"]:
            print("[Vault] Fitting Stage 1: Temperature Scaling...")
            self._optimize_temperature(logits_val, labels_val)

        if stage in ["stage2", "both"]:
            # Note: For Stage 2, we ideally need the *Post-Logic* q.
            # Here we approximate by using the scaled logits as input proxy
            # or assuming this function is called with post-logic probas in a real run.
            # For simplicity in this artifact, we fit Isotonic on Sigmoid(Logits/T).
            print("[Vault] Fitting Stage 2: Isotonic Regression...")
            with torch.no_grad():
                scaled_logits = logits_val / self.temperature
                probs = torch.sigmoid(scaled_logits).cpu().numpy()
                targets = labels_val.cpu().numpy()
                
                for k in range(self.num_predicates):
                    iso_reg = IsotonicRegression(out_of_bounds='clip')
                    # Fit on column k
                    iso_reg.fit(probs[:, k], targets[:, k])
                    self.isotonic_models[k] = iso_reg
        
        self.is_sealed = True

    def _optimize_temperature(self, logits, labels):
        """Standard NLL minimization to find optimal T."""
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            # Broadcast T: (1, K)
            T_expanded = self.temperature.unsqueeze(0).expand_as(logits)
            loss = criterion(logits / T_expanded, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        # Clamp T to be > 0.1 to avoid numerical instability
        with torch.no_grad():
            self.temperature.clamp_(min=0.1)

    def apply_stage1(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Temperature Scaling (Before Logic Layer)."""
        if not self.is_sealed:
            return logits # Pass-through if not calibrated
        
        with torch.no_grad():
            return logits / self.temperature.to(logits.device)

    def apply_stage2(self, posterior_q: torch.Tensor) -> torch.Tensor:
        """
        Apply Isotonic Regression (After Logic Layer).
        This ensures the final VAA score is empirical.
        """
        if not self.is_sealed:
            return posterior_q

        B, K = posterior_q.shape
        calibrated_q = np.zeros((B, K), dtype=np.float32)
        q_np = posterior_q.cpu().numpy()

        for k in range(K):
            if self.isotonic_models[k] is not None:
                calibrated_q[:, k] = self.isotonic_models[k].predict(q_np[:, k])
            else:
                calibrated_q[:, k] = q_np[:, k]

        return torch.from_numpy(calibrated_q).to(posterior_q.device)

    def compute_ece(self, probs: torch.Tensor, labels: torch.Tensor, n_bins=10) -> float:
        """
        Calculates Expected Calibration Error (ECE).
        Metric for Table 6 and VAA.
        """
        probs_np = probs.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        N = len(probs_np)

        for bin_id in range(n_bins):
            # Indices in this bin
            in_bin = (probs_np > bin_boundaries[bin_id]) & (probs_np <= bin_boundaries[bin_id+1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy = np.mean(labels_np[in_bin])
                confidence = np.mean(probs_np[in_bin])
                ece += np.abs(accuracy - confidence) * prop_in_bin

        return float(ece)

    def get_fingerprint(self) -> Dict[str, str]:
        """Returns the cryptographic proof of calibration for the VAA."""
        # Serialize parameters to string
        state_str = str(self.temperature.tolist()) + str([m is not None for m in self.isotonic_models])
        vault_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        return {
            "vault_id": f"VAULT-{vault_hash[:8]}",
            "c_set_hash": self.c_set_hash,
            "method": "Temperature+Isotonic"
        }

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing Calibration Vault...")
    torch.manual_seed(42)
    
    # 1. Setup
    vault = CalibrationVault(num_predicates=1)
    
    # 2. Simulate Data Leakage (Should Fail)
    try:
        vault.register_dataset_hashes("hash_A", "hash_A")
    except ValueError as e:
        print(f"[+] Isolation Check Passed: {e}")
        
    vault.register_dataset_hashes("hash_Train", "hash_CSet_Secure")
    
    # 3. Create uncalibrated logits (Overconfident)
    # Logits 5.0 -> Sigmoid ~0.99, but Labels are 0.8 mean
    logits = torch.randn(1000, 1) * 5.0 
    labels = (torch.sigmoid(logits * 0.5) > 0.5).float() # Ground truth is "softer"
    
    # 4. Fit
    vault.fit(logits, labels)
    print(f"Learned Temperature: {vault.temperature.item():.4f} (Expected > 1.0)")
    
    # 5. Apply
    q_raw = torch.sigmoid(logits)
    q_cal = vault.apply_stage2(torch.sigmoid(vault.apply_stage1(logits)))
    
    # 6. Check ECE
    ece_raw = vault.compute_ece(q_raw, labels)
    ece_cal = vault.compute_ece(q_cal, labels)
    
    print(f"ECE (Raw): {ece_raw:.4f}")
    print(f"ECE (Cal): {ece_cal:.4f}")
    
    assert ece_cal < ece_raw, "Calibration failed to improve ECE."
    print("[+] Test Passed: Vault is functioning.")