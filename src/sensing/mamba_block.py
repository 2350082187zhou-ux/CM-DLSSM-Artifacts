"""
==============================================================================
CM-DLSSM Artifact: Hardware-Aware Selective SSM Block (Mamba-2)
Path: src/sensing/mamba_block.py
==============================================================================
Reference: Section 4.1 (Neural Sensing Layer) & Section 6.2 (Efficiency)

Description:
    This module implements the core "Selective State Space Model" block.
    It overcomes the O(L^2) bottleneck of Transformers by using a linear O(L)
    scan operation.

    Key Features for 128k Context:
    1. Hardware-Aware IO: Minimizes HBM reads/writes via kernel fusion structure.
    2. Numerical Stability: Forces float32 for the recurrent state h_t.
    3. SRVS Gating: Injects the static vulnerability priors into the dynamics.

Hardware Note:
    In production (USENIX Eval), this Python 'forward' is replaced by a
    custom CUDA kernel (src/ops/selective_scan_cuda.cu) for maximum throughput.
    This file provides the rigorous mathematical definition (Golden Reference).
==============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Optional: Try importing optimized CUDA kernels
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False
    print("[!] Warning: Mamba CUDA kernel not found. Falling back to slow PyTorch reference.")

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,    # N: State dimension (High capacity for Taint tracking)
        d_conv: int = 4,       # Local convolution width
        expand: int = 2,       # Expansion factor (E)
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True, # Enable CUDA optimization
        layer_idx: int = None,      # For debugging/logging
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # ----------------------------------------------------------------------
        # 1. Input Projection (Expansion)
        # ----------------------------------------------------------------------
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # ----------------------------------------------------------------------
        # 2. Convolution (Short-term memory)
        # ----------------------------------------------------------------------
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        # ----------------------------------------------------------------------
        # 3. State Space Dynamics (The Selection Mechanism)
        # ----------------------------------------------------------------------
        # Projection to generate dynamic B, C, and dt from input x
        # x_proj takes x -> (dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        
        # dt_proj projects dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special parameters for A (Recurrence weight)
        # S4D / Mamba initialization strategy: Real Diagonal
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A)) # Learnable log(A)
        self.D = nn.Parameter(torch.ones(self.d_inner)) # Skip connection
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # ----------------------------------------------------------------------
        # 4. Output Projection
        # ----------------------------------------------------------------------
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, srvs_mask=None, inference_params=None):
        """
        Args:
            hidden_states: (B, L, D) - The input sequence (Code tokens)
            srvs_mask: (B, L, 1) - [Optional] Static Risk mask (0 or 1).
                                   Used for SRVS Gating (Equation 7).
        """
        batch, seqlen, dim = hidden_states.shape

        # 1. Project inputs: (B, L, D) -> (B, L, 2*E*D)
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1) # x: signal, z: gating branch

        # 2. Convolution (1D)
        # Rearrange for Conv1d: (B, D, L)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen] # Causal padding
        x = x.transpose(1, 2)
        
        x = self.act(x)

        # 3. Selective SSM Scan
        # This is the O(L) magic step.
        
        # Linear projection to get dynamic params
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        
        # Parameter A (Discretized dynamics will happen inside ssm_scan)
        A = -torch.exp(self.A_log.float()) # Ensure negativity for stability

        # --- SRVS Gating Injection (Section 4.1, Eq 7) ---
        # If SRVS mask is present, we modulate the dynamics.
        # "Selectivity" means dt is input-dependent.
        # We augment dt to "slow down" forgetting at critical sinks.
        if srvs_mask is not None:
            # Logic: If srvs_mask=1 (Critical Sink), boost dt to capture more state.
            # Or modulate B to accept more input.
            # Implementation: Boost dt bias effectively.
            # Note: This is a differentiable soft-injection.
            dt = dt + (srvs_mask * 2.0) 

        # 4. Run the Scan (Hardware-Aware)
        if self.use_fast_path and HAS_CUDA_KERNEL:
            # The Optimized CUDA Kernel
            # Note: This kernel fuses discretization + recurrence + output
            # Precision: Inputs are bf16, but internal state h is float32.
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=None, # Bias merged above
                delta_softplus=True,
                return_last_state=False,
            )
        else:
            # The PyTorch Reference (Slow but readable/auditable)
            # Useful for verification or CPU inference
            y = self.selective_scan_ref(x, dt, A, B, C, self.D, z, delta_softplus=True)

        # 5. Output Projection
        out = self.out_proj(y)
        return out

    def selective_scan_ref(self, u, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        """
        PyTorch reference implementation of the selective scan.
        Used for audit verification (Artifact Requirement R1).
        
        Math:
            h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t
            y_t = C_t h_t
        """
        # 1. Discretize
        dt = F.softplus(dt + (delta_bias if delta_bias is not None else 0.0)) if delta_softplus else dt
        dA = torch.exp(torch.einsum("b l d, d n -> b l d n", dt, A)) # (B, L, D, N)
        dB = torch.einsum("b l d, b l n -> b l d n", dt, B)         # (B, L, D, N)
        
        # 2. Scan (Sequential loop - O(L))
        # Note: In Python this is slow. 
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[-1]
        
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=torch.float32) # State is float32
        ys = []
        
        for t in range(seqlen):
            # h_t = dA * h_{t-1} + dB * u_t
            # Note: u[:, t] needs to be broadcasted to (B, D, N) or similar
            ut = u[:, t].unsqueeze(-1) # (B, D, 1)
            
            # Recurrence
            h = dA[:, t] * h + dB[:, t] * ut
            
            # Output map
            y = torch.einsum("b d n, b n -> b d", h, C[:, t])
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        
        # 3. Add skip connection and gate with z
        if D is not None:
            y = y + u * D
        if z is not None:
            y = y * F.silu(z)
            
        return y

if __name__ == "__main__":
    # Artifact Smoke Test
    print("Running MambaBlock Smoke Test...")
    model = MambaBlock(d_model=64, d_state=16).cuda()
    x = torch.randn(2, 1024, 64).cuda() # 1k tokens
    mask = torch.randint(0, 2, (2, 1024, 1)).float().cuda() # Dummy SRVS mask
    
    y = model(x, srvs_mask=mask)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print("Test Passed.")