"""
==============================================================================
CM-DLSSM Artifact: Log-Ratio CAVI Inference Engine
Path: src/logic/cavi_engine.py
==============================================================================
Reference: Section 4.2 (Symbolic Reasoning Layer)
           Appendix A (Mathematical Proofs)
           Appendix B (SRTL Definitions)

Description:
    This module implements the Coordinate Ascent Mean-Field Inference (CAVI)
    algorithm in the Log-Odds (Logit) space.

    It serves as the "Reasoning Core" of the framework. Unlike a black-box MLP,
    this engine enforces logical consistency defined by the rule set.

    Mathematical Update Operator (Eq. 10):
        logit(q_i^{t+1}) = (1-alpha)*logit(q_i^t) + alpha * (evidence + messages)

    Key Features:
    1. Deterministic: Given inputs, the output trace is bit-exact.
    2. Differentiable: Allows end-to-end training of rule weights w_r.
    3. Audit-Ready: Exports the full convergence trace for the VAA.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CAVIEngine(nn.Module):
    def __init__(
        self,
        num_predicates: int,
        max_iterations: int = 5,
        damping: float = 0.5,
        audit_mode: bool = True
    ):
        """
        Args:
            num_predicates: Total number of unique logic variables (nodes).
            max_iterations: Number of mean-field steps (J). Paper uses J=5.
            damping: Alpha parameter (0 < alpha <= 1). 
                     Lower = More stable, Slower convergence.
            audit_mode: If True, stores the trajectory for VAA generation.
        """
        super().__init__()
        self.num_predicates = num_predicates
        self.max_iterations = max_iterations
        self.damping = damping
        self.audit_mode = audit_mode
        
        # Buffer to store the audit trace (not a parameter)
        self.trace_buffer = []

    def forward(self, evidence_logits, rules_t1=None, rules_t2=None, rules_t3=None):
        """
        Execute the Fixed-Point Iteration.

        Args:
            evidence_logits: (B, N) - Raw neural evidence from Tier 1.
            rules_tX: Dictionaries containing rule indices and weights.
                      Format: {
                          'indices': (Num_Rules, Arity), # Variable IDs
                          'weights': (Num_Rules,)        # Rule Strengths
                      }
        
        Returns:
            final_posterior: (B, N) - Refined probabilities q
        """
        B, N = evidence_logits.shape
        
        # 1. Initialize Beliefs q^(0)
        # q = sigma(evidence)
        curr_logits = evidence_logits.clone()
        curr_q = torch.sigmoid(curr_logits)
        
        # Reset trace
        if self.audit_mode:
            self.trace_buffer = [curr_q.detach().cpu()]

        # 2. Iterative Update Loop
        for j in range(self.max_iterations):
            # Accumulator for incoming logical messages
            # Delta_L (B, N)
            total_messages = torch.zeros_like(evidence_logits)

            # --- Template T1: Simple Propagation (a -> c) ---
            if rules_t1 is not None:
                msg = self._compute_t1_messages(curr_q, rules_t1)
                total_messages += msg

            # --- Template T2: Conjunction (a ^ b -> c) ---
            if rules_t2 is not None:
                msg = self._compute_t2_messages(curr_q, rules_t2)
                total_messages += msg

            # --- Template T3: Vulnerability (a ^ !b -> c) ---
            # This is the most critical template for security.
            if rules_t3 is not None:
                msg = self._compute_t3_messages(curr_q, rules_t3)
                total_messages += msg

            # 3. Log-Ratio Update Step (Equation 10)
            # logit_new = (1-alpha)*logit_old + alpha*(evidence + messages)
            # Note: The 'evidence' term is constant throughout iterations (Prior)
            target_logits = evidence_logits + total_messages
            
            new_logits = (1 - self.damping) * curr_logits + self.damping * target_logits
            
            # Update q for next step
            curr_q = torch.sigmoid(new_logits)
            curr_logits = new_logits

            # Log trace
            if self.audit_mode:
                self.trace_buffer.append(curr_q.detach().cpu())

        return curr_q

    # ==========================================================================
    # Message Calculation Kernels (Appendix B)
    # ==========================================================================
    
    def _compute_t3_messages(self, q, rules):
        """
        Template T3: p_a ^ !p_b -> p_c  (Source ^ !Sanitizer -> Vulnerability)
        
        Messages (Appendix B.3):
            To c:  w * q_a * (1 - q_b)
            To a:  w * (1 - q_b) * (q_c - 1)
            To b:  w * q_a * (1 - q_c)  <-- Note: Encourages b if c is false
        """
        indices = rules['indices'] # (R, 3) -> [a, b, c]
        weights = rules['weights'] # (R,)

        # Gather probabilities for a, b, c
        # q: (B, N) -> q_a: (B, R)
        q_a = q[:, indices[:, 0]]
        q_b = q[:, indices[:, 1]]
        q_c = q[:, indices[:, 2]]
        
        # Expand weights for batch: (1, R)
        w = weights.unsqueeze(0)

        # 1. Message to Conclusion (c): Increase risk
        # msg = w * q_a * (1 - q_b)
        msg_to_c = w * q_a * (1.0 - q_b)

        # 2. Message to Source (a): Backpropagate risk context
        # msg = w * (1-b) * (c-1)  (Negative potential gradient)
        msg_to_a = w * (1.0 - q_b) * (q_c - 1.0)

        # 3. Message to Sanitizer (b): Explain away
        msg_to_b = w * q_a * (1.0 - q_c)

        # Scatter Add to Global Message Tensor
        B, N = q.shape
        delta_L = torch.zeros(B, N, device=q.device)
        
        # We add messages to the specific variable indices
        # scatter_add_(dim, index, src)
        # We need to repeat indices for the batch dimension
        
        # Helper for vectorized scatter
        def scatter_msg(col_idx, msg_tensor):
            # idx: (R,) -> (B, R)
            idx_expanded = indices[:, col_idx].unsqueeze(0).expand(B, -1)
            delta_L.scatter_add_(1, idx_expanded, msg_tensor)

        scatter_msg(2, msg_to_c) # Add to c
        scatter_msg(0, msg_to_a) # Add to a
        scatter_msg(1, msg_to_b) # Add to b

        return delta_L

    def _compute_t1_messages(self, q, rules):
        """Template T1: p_a -> p_c"""
        indices = rules['indices'] # [a, c]
        weights = rules['weights']
        
        q_a = q[:, indices[:, 0]]
        q_c = q[:, indices[:, 1]]
        w = weights.unsqueeze(0)

        msg_to_c = w * q_a
        msg_to_a = w * (q_c - 1.0)

        B, N = q.shape
        delta_L = torch.zeros(B, N, device=q.device)
        
        idx_a = indices[:, 0].unsqueeze(0).expand(B, -1)
        idx_c = indices[:, 1].unsqueeze(0).expand(B, -1)
        
        delta_L.scatter_add_(1, idx_c, msg_to_c)
        delta_L.scatter_add_(1, idx_a, msg_to_a)
        
        return delta_L

    def _compute_t2_messages(self, q, rules):
        """Template T2: p_a ^ p_b -> p_c"""
        indices = rules['indices'] # [a, b, c]
        weights = rules['weights']
        
        q_a = q[:, indices[:, 0]]
        q_b = q[:, indices[:, 1]]
        q_c = q[:, indices[:, 2]]
        w = weights.unsqueeze(0)

        msg_to_c = w * q_a * q_b
        msg_to_a = w * q_b * (q_c - 1.0)
        msg_to_b = w * q_a * (q_c - 1.0)

        B, N = q.shape
        delta_L = torch.zeros(B, N, device=q.device)
        
        idx_a = indices[:, 0].unsqueeze(0).expand(B, -1)
        idx_b = indices[:, 1].unsqueeze(0).expand(B, -1)
        idx_c = indices[:, 2].unsqueeze(0).expand(B, -1)
        
        delta_L.scatter_add_(1, idx_c, msg_to_c)
        delta_L.scatter_add_(1, idx_a, msg_to_a)
        delta_L.scatter_add_(1, idx_b, msg_to_b)
        
        return delta_L

    def get_audit_trace(self):
        """Returns the convergence log for VAA generation."""
        # Convert list of tensors to list of numpy arrays/lists
        # Shape: (Iterations+1, Batch, N)
        return torch.stack(self.trace_buffer).permute(1, 0, 2)

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing CAVI Engine...")
    torch.manual_seed(42)
    
    # Setup
    B, N = 1, 3 # 3 Predicates: a, b, c
    engine = CAVIEngine(num_predicates=N, max_iterations=5, damping=0.5)
    
    # 1. Evidence: a=High, b=Low, c=Unsure
    # Corresponds to: Taint=Yes, Check=No => Vuln=?
    logits = torch.tensor([[2.0, -2.0, 0.0]]) 
    
    # 2. Rule T3: a ^ !b -> c (Weight 5.0)
    rules_t3 = {
        'indices': torch.tensor([[0, 1, 2]], dtype=torch.long),
        'weights': torch.tensor([5.0])
    }
    
    # 3. Forward
    final_q = engine(logits, rules_t3=rules_t3)
    trace = engine.get_audit_trace()
    
    print(f"Initial q: {torch.sigmoid(logits).numpy()}")
    print(f"Final q:   {final_q.numpy()}")
    print(f"Trace (q_c): {trace[0, :, 2].numpy()}")
    
    # Assertion: q_c should increase significantly
    assert final_q[0, 2] > 0.9, "Logic failed to propagate risk!"
    print("[+] Test Passed: Risk propagated successfully.")