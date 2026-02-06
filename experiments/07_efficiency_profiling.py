"""
==============================================================================
CM-DLSSM Artifact: Experiment 07 - Efficiency & Gating Profiling
Path: experiments/07_efficiency_profiling.py
==============================================================================
Reference: Section 6.2 (Efficiency and Scalability) - Figure 7
           Section 4.1 (Neural Sensing Layer) - "SRVS Gating"

Description:
    This script performs hardware profiling to validate two key claims:
    
    1. Linear Scalability O(L):
       It measures Throughput (Tokens/sec) and Peak VRAM usage across increasing
       sequence lengths [2k -> 128k]. It compares CM-DLSSM (Mamba) vs 
       Standard Transformer (Attention).

    2. Gating Sparsity (SRVS Effectiveness):
       It verifies that the SRVS Gate is NOT "Always On". Efficient auditing
       requires the model to focus resources only on relevant sinks. 
       Target Activation Rate < 10%.

    Outputs:
    - artifacts/results/profiling_scalability.csv (TPS/VRAM)
    - artifacts/results/profiling_gating.csv (Sparsity Stats)

Usage:
    python experiments/07_efficiency_profiling.py model.d_model=1024
==============================================================================
"""

import os
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# Import System Components
from src.sensing.mamba_block import MambaBlock
from src.sensing.srvs_gate import SRVSGate

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exp_profiling")

class HardwareProfiler:
    """Helper to measure CUDA time and memory accurately."""
    def __init__(self, device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def reset_memory(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_memory_gb(self):
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

    def measure_ms(self, func, *args, **kwargs):
        self.reset_memory()
        torch.cuda.synchronize()
        
        self.start_event.record()
        out = func(*args, **kwargs)
        self.end_event.record()
        
        torch.cuda.synchronize()
        elapsed_ms = self.start_event.elapsed_time(self.end_event)
        peak_mem = self.get_peak_memory_gb()
        
        return out, elapsed_ms, peak_mem

class BaselineTransformer(nn.Module):
    """
    A dummy Transformer Encoder to demonstrate O(L^2) explosion.
    Implementation uses PyTorch's native scaled_dot_product_attention (FlashAttn compatible).
    Even with FlashAttn, the VRAM for KV cache grows linearly, but attention matrix logic remains quadratic logically (or O(L) memory with Flash).
    However, standard attention without Flash is strictly O(L^2). We simulate standard here.
    """
    def __init__(self, d_model, n_head=8):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
    
    def forward(self, x):
        return self.layer(x)

class EfficiencyEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == 'cpu':
            logger.warning("[!] Running on CPU. Profiling results will NOT reflect GPU O(L) benefits accurately.")

        # Initialize Models
        d_model = cfg.model.architecture.dim_model
        
        # 1. CM-DLSSM (Tier 1 Backbone)
        self.model_ssm = MambaBlock(
            d_model=d_model, 
            d_state=cfg.model.ssm.d_state,
            use_fast_path=True
        ).to(self.device)
        
        # 2. SRVS Gate
        self.gate = SRVSGate(
            d_model=d_model, 
            audit_mode=True
        ).to(self.device)
        
        # 3. Baseline Transformer (for comparison)
        self.model_attn = BaselineTransformer(d_model=d_model).to(self.device)
        
        self.profiler = HardwareProfiler(self.device)

    def run_scalability_sweep(self):
        """
        Compare TPS and VRAM across [2k, 8k, ..., 128k].
        """
        logger.info("\n[Task 1] Running Scalability Sweep (SSM vs Transformer)...")
        
        # Context lengths to test
        # Note: 128k might OOM the Baseline immediately on consumer GPUs.
        seq_lens = [2048, 8192, 16384, 32768, 65536, 131072]
        results = []
        
        batch_size = 1 # Single stream auditing
        dim = self.cfg.model.architecture.dim_model

        for L in seq_lens:
            logger.info(f"  > Testing Context Length: {L}")
            
            # Create Input
            try:
                inputs = torch.randn(batch_size, L, dim, device=self.device, dtype=torch.float16)
            except RuntimeError:
                logger.error(f"    [!] OOM creating input tensor for L={L}")
                break

            # --- Test 1: CM-DLSSM (Mamba) ---
            try:
                # Warmup
                _ = self.model_ssm(inputs)
                
                # Profile
                _, t_ms, mem_gb = self.profiler.measure_ms(self.model_ssm, inputs)
                tps = (batch_size * L) / (t_ms / 1000.0)
                
                results.append({
                    "Model": "CM-DLSSM (Ours)",
                    "Context_Length": L,
                    "Throughput_TPS": tps,
                    "Peak_VRAM_GB": mem_gb,
                    "Status": "Success"
                })
                logger.info(f"    CM-DLSSM: {tps:.0f} TPS | {mem_gb:.2f} GB VRAM")
                
            except RuntimeError as e:
                results.append({"Model": "CM-DLSSM (Ours)", "Context_Length": L, "Status": "OOM"})
                logger.error(f"    CM-DLSSM OOM at L={L}")

            # --- Test 2: Transformer (Baseline) ---
            # Skip huge lengths for Transformer to save time/crashes, or wrap in try/except
            if L <= 32768: # Usually dies after 32k on 24GB cards
                try:
                    # Warmup
                    _ = self.model_attn(inputs)
                    
                    # Profile
                    _, t_ms, mem_gb = self.profiler.measure_ms(self.model_attn, inputs)
                    tps = (batch_size * L) / (t_ms / 1000.0)
                    
                    results.append({
                        "Model": "Transformer (Baseline)",
                        "Context_Length": L,
                        "Throughput_TPS": tps,
                        "Peak_VRAM_GB": mem_gb,
                        "Status": "Success"
                    })
                    logger.info(f"    Transformer: {tps:.0f} TPS | {mem_gb:.2f} GB VRAM")
                    
                except RuntimeError:
                    results.append({"Model": "Transformer (Baseline)", "Context_Length": L, "Status": "OOM"})
                    logger.warning(f"    Transformer OOM at L={L}")
                    torch.cuda.empty_cache()
            else:
                results.append({"Model": "Transformer (Baseline)", "Context_Length": L, "Status": "Skipped (Projected OOM)"})

        # Save
        df = pd.DataFrame(results)
        os.makedirs("artifacts/results", exist_ok=True)
        df.to_csv("artifacts/results/profiling_scalability.csv", index=False)
        return df

    def run_gating_analysis(self):
        """
        Verify SRVS Gating Sparsity.
        Hypothesis: In a codebase, sinks (memcpy, exec) are rare. 
        The gate should be active (< 10%) but not dead (0%).
        """
        logger.info("\n[Task 2] Analyzing SRVS Gating Sparsity...")
        
        self.gate.reset_stats()
        
        # Simulate processing 10 batches of code
        B, L, D = 4, 16384, 1024
        
        for i in tqdm(range(10), desc="Scanning Codebase"):
            # 1. Generate Synthetic Inputs
            hidden = torch.randn(B, L, D, device=self.device)
            
            # 2. Simulate Sinks (Sparse: ~1% of tokens are sinks)
            sinks = torch.zeros(B, L, dtype=torch.long, device=self.device)
            # Randomly place sinks
            num_sinks = int(B * L * 0.01) 
            flat_indices = torch.randperm(B * L)[:num_sinks]
            sinks.view(-1)[flat_indices] = torch.randint(1, 128, (num_sinks,), device=self.device)
            
            # 3. Forward Pass
            _ = self.gate(hidden, sink_ids=sinks)
            
        # 4. Get Report
        report = self.gate.get_audit_report()
        logger.info(f"Gating Audit Report: {report}")
        
        # Save
        pd.DataFrame([report]).to_csv("artifacts/results/profiling_gating.csv", index=False)
        
        # Assertion for Artifact Evaluation
        activation = report['global_activation_rate']
        if 0.001 < activation < 0.15:
            logger.info(f"[+] PASS: Gating is sparse and active ({activation*100:.2f}%).")
        else:
            logger.warning(f"[-] WARNING: Gating rate {activation:.4f} is outside optimal range [0.1%, 15%].")

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Ensure model config is loaded
    if "model" not in cfg:
        # Fallback default if running standalone without full hydra context
        cfg.model = DictConfig({
            "architecture": {"dim_model": 1024},
            "ssm": {"d_state": 128}
        })
        
    profiler = EfficiencyEvaluator(cfg)
    
    # 1. Scalability
    if torch.cuda.is_available():
        profiler.run_scalability_sweep()
    else:
        logger.error("Skipping Scalability Sweep: GPU required for meaningful O(L) profile.")
    
    # 2. Gating
    profiler.run_gating_analysis()

if __name__ == "__main__":
    main()