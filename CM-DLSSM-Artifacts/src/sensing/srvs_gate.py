"""
==============================================================================
CM-DLSSM Artifact: SRVS Gating Module (Static Risk Vulnerability Sinks)
Path: src/sensing/srvs_gate.py
==============================================================================
Reference: Section 4.1 (Neural Sensing Layer) - Equation (7)
           Section 5.2 (Compliance Gating) - Attachment Rate

Description:
    This module implements the "Attention Focusing" mechanism.
    Unlike standard attention which learns purely from data, SRVS Gating
    injects HARD architectural priors based on known vulnerability sinks.

    Functionality:
    1. Embeds Sink Types: Maps detected sink IDs (e.g., MEMCPY_ID) to vectors.
    2. Generates Gating Signal M_t: Computes the modulation strength.
    3. Audit Logging: Tracks 'Activation Rate' to prove the model is focusing
       on the right parts (Sparse Activation).

    Math:
        Gate_t = sigmoid( W_g * h_t + b_g + Embedding(Sink_t) )
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SRVSGate(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_sink_types: int = 128,  # Size of the SRVS Taxonomy
        gate_dim: int = 64,         # Dimension of the gating projection
        dropout: float = 0.1,
        audit_mode: bool = True     # If True, tracks statistics for VAA
    ):
        super().__init__()
        self.d_model = d_model
        self.audit_mode = audit_mode

        # ------------------------------------------------------------------
        # 1. Learnable Projections
        # ------------------------------------------------------------------
        # Project input hidden state to gate space
        self.proj = nn.Linear(d_model, gate_dim)
        
        # Embedding for Sink Types (e.g., 0=None, 1=strcpy, 2=malloc...)
        # Padding index 0 ensures "No Sink" adds a zero vector (initially)
        self.sink_embedding = nn.Embedding(num_sink_types, gate_dim, padding_idx=0)
        
        # The final decision layer for the scalar Mask M_t
        # Maps gate_dim -> 1 scalar value per token
        self.gate_head = nn.Linear(gate_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # ------------------------------------------------------------------
        # 2. Audit Statistics (Buffers)
        # ------------------------------------------------------------------
        # These are not model parameters, but persistent state for the VAA
        self.register_buffer("total_tokens_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_gates_active", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_batch_sparsity", torch.tensor(0.0))

    def forward(self, hidden_states, sink_ids=None):
        """
        Args:
            hidden_states: (B, L, D) - Contextual embeddings from SSM backbone
            sink_ids:      (B, L)    - Integer IDs of detected sinks (from Static Pre-scan).
                                       0 means "No Sink".
        Returns:
            gating_signal: (B, L, 1) - The scalar modulation signal M_t in [0, 1]
        """
        B, L, D = hidden_states.shape

        # 1. Project Hidden State
        # h_proj: (B, L, gate_dim)
        h_proj = self.proj(hidden_states)

        # 2. Inject Static Sink Knowledge (if available)
        if sink_ids is not None:
            # sink_emb: (B, L, gate_dim)
            sink_emb = self.sink_embedding(sink_ids)
            # Fuse: Neural Context + Static Type
            fusion = h_proj + sink_emb
        else:
            # Blind mode (pure neural gating)
            fusion = h_proj

        # 3. Activation
        fusion = self.act(fusion)
        fusion = self.dropout(fusion)

        # 4. Generate Scalar Gate M_t
        # logits: (B, L, 1)
        gate_logits = self.gate_head(fusion)
        
        # M_t in (0, 1)
        gating_signal = torch.sigmoid(gate_logits)

        # ------------------------------------------------------------------
        # 5. Audit Logging (For VAA generation)
        # ------------------------------------------------------------------
        if self.audit_mode and self.training is False:
            self._update_stats(gating_signal, sink_ids)

        return gating_signal

    def _update_stats(self, gating_signal, sink_ids):
        """
        Updates internal counters for 'Attachment Rate' and 'Sparsity'.
        Reference: Section 6.3 Auditing Effectiveness
        
        FIX: Properly handle type conversion for buffer updates
        """
        with torch.no_grad():
            # Threshold for "Active": > 0.5
            active_mask = (gating_signal > 0.5).float()
            
            num_tokens = gating_signal.numel()
            num_active = int(active_mask.sum().item())  # FIX: Convert to Python int

            # FIX: Ensure type compatibility with Long buffers
            self.total_tokens_seen += num_tokens
            self.total_gates_active += num_active
            
            # Calculate sparsity for this batch
            # FIX: Use fill_ to update tensor buffer in-place
            sparsity_value = float(num_active) / (num_tokens + 1e-9)
            self.last_batch_sparsity.fill_(sparsity_value)

    def get_audit_report(self):
        """
        Returns the data needed for the VAA 'Gating Trace'.
        Used by: src/infra/vaa_generator.py
        """
        global_sparsity = self.total_gates_active.float() / (self.total_tokens_seen.float() + 1e-9)
        
        return {
            "module": "SRVS_Gate",
            "global_activation_rate": global_sparsity.item(),
            "last_batch_sparsity": self.last_batch_sparsity.item(),
            "status": "PASS" if global_sparsity < 0.1 else "WARNING_DENSE" # Gate shouldn't be always on
        }

    def reset_stats(self):
        """Call at start of a new Audit Session."""
        self.total_tokens_seen.zero_()
        self.total_gates_active.zero_()
        self.last_batch_sparsity.fill_(0.0)  # FIX: Use fill_ for tensor buffer

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing SRVSGate Module...")
    
    # 1. Setup
    model = SRVSGate(d_model=1024, gate_dim=64)
    model.eval() # Enable audit logging
    
    # 2. Dummy Inputs
    B, L, D = 2, 1024, 1024
    hidden = torch.randn(B, L, D)
    
    # Simulate sinks: mostly 0 (none), sparse 1s (memcpy), rare 2s (system)
    sinks = torch.zeros(B, L, dtype=torch.long)
    sinks[:, 50] = 1 # Sink at pos 50
    sinks[:, 500] = 2 # Sink at pos 500
    
    # 3. Forward
    gate_out = model(hidden, sinks)
    
    # 4. Check Stats
    report = model.get_audit_report()
    
    print(f"Input Shape: {hidden.shape}")
    print(f"Gate Output: {gate_out.shape}")
    print(f"Audit Report: {report}")
    
    assert gate_out.min() >= 0 and gate_out.max() <= 1, "Gate output must be [0,1]"
    print("[+] Test Passed.")
