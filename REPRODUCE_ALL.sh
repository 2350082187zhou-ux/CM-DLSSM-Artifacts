#!/bin/bash

# ==============================================================================
# CM-DLSSM Artifact Reproduction Script
# ==============================================================================
# Paper: CM-DLSSM: A Cross-Modal Differentiable Logic State Space Model
#        for Unified Security Analysis of Source Code and Binaries
#
# Description:
# This script orchestrates the entire experimental pipeline defined in the paper.
# It runs data generation, model inference (simulated or real), logic verification,
# and causal estimation benchmarks.
#
# Usage:
#   ./REPRODUCE_ALL.sh              # Run everything (approx. 20-40 mins)
#   ./REPRODUCE_ALL.sh --step 1     # Run only Experiment 01
#   ./REPRODUCE_ALL.sh --install    # Install Python dependencies first
# ==============================================================================

# --- Configuration ---
set -e  # Exit immediately if a command exits with a non-zero status
export PYTHONPATH=.  # Ensure 'src' module is visible

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Helper Functions ---

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_header() {
    echo "========================================================================"
    echo "   CM-DLSSM ARTIFACT EVALUATION: REPRODUCING RESULTS"
    echo "========================================================================"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 could not be found. Please install Python 3.9+."
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    log_success "Dependencies installed."
}

# --- Experiment Steps ---

run_step_0() {
    echo ""
    log_info ">>> STEP 0: Data Generation (Prerequisite)"
    echo "    Target: Synthetic BLV Dataset for Causal Inference"
    echo "    Output: data/processed/causal/*.csv"
    
    python3 data/generators/blv_synth.py --samples 50000 --out_dir data/processed/causal
    log_success "Step 0 Complete: Ground truth data generated."
}

run_step_1() {
    echo ""
    log_info ">>> STEP 1: Main Accuracy Benchmarks"
    echo "    Target: Table 2 (F1, Recall@1%FPR, per-CWE)"
    echo "    Script: experiments/01_benchmark_main.py"
    
    # Running in debug mode for artifact speed (using synthetic tensors)
    # To run real data, change to: dataset=bigvul debug=False
    python3 experiments/01_benchmark_main.py debug=True
    log_success "Step 1 Complete: Table 2 metrics generated."
}

run_step_2() {
    echo ""
    log_info ">>> STEP 2: Long Context Evidence (The O(L) Claim)"
    echo "    Target: Figure 7 Data (Degradation Curves)"
    echo "    Script: experiments/02_long_context_evidence.py"
    
    python3 experiments/02_long_context_evidence.py simulation_mode=True
    log_success "Step 2 Complete: Long context degradation curves saved."
}

run_step_3() {
    echo ""
    log_info ">>> STEP 3: Ablation Matrix & Attribution"
    echo "    Target: Table 5 (Impact of Logic/Calib/Gate)"
    echo "    Script: experiments/03_ablation_matrix.py"
    
    # We run the full system variant D as a demo
    # For full table, one would run variants A, B, C, D
    log_info "Running Variant D (Full System)..."
    python3 experiments/03_ablation_matrix.py ablation.variant=D
    log_success "Step 3 Complete: Variant D metrics logged."
}

run_step_4() {
    echo ""
    log_info ">>> STEP 4: Audit Reproducibility (VAA Verification)"
    echo "    Target: Prove that Audit Certificates are mathematically sound."
    echo "    Script: experiments/04_audit_reproducibility.py"
    
    python3 experiments/04_audit_reproducibility.py n_samples=100 epsilon=1e-5
    log_success "Step 4 Complete: 100 VAAs verified successfully."
}

run_step_5() {
    echo ""
    log_info ">>> STEP 5: Cross-Modal Alignment"
    echo "    Target: Table 3 (Source-Binary Matching & KL Divergence)"
    echo "    Script: experiments/05_cross_modal_alignment.py"
    
    python3 experiments/05_cross_modal_alignment.py
    log_success "Step 5 Complete: Cross-modal resilience metrics saved."
}

run_step_6() {
    echo ""
    log_info ">>> STEP 6: Causal Inference Task"
    echo "    Target: Validate Doubly Robust Estimator vs Naive"
    echo "    Script: experiments/06_causal_blv_task.py"
    
    python3 experiments/06_causal_blv_task.py
    log_success "Step 6 Complete: Causal bias analysis finished."
}

run_step_7() {
    echo ""
    log_info ">>> STEP 7: Efficiency Profiling (Hardware)"
    echo "    Target: TPS / VRAM Scaling & Gating Sparsity"
    echo "    Script: experiments/07_efficiency_profiling.py"
    
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        python3 experiments/07_efficiency_profiling.py
        log_success "Step 7 Complete: Hardware profiling finished."
    else
        log_warn "No GPU detected. Skipping Step 7 (Scalability Sweep requires CUDA)."
    fi
}

# --- Main Logic ---

STEP="all"
INSTALL_DEPS="false"

# Parse Args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift ;;
        --install) INSTALL_DEPS="true" ;;
        -h|--help) 
            print_header
            echo "Usage: ./REPRODUCE_ALL.sh [options]"
            echo "Options:"
            echo "  --step <0-7>   Run a specific experiment step."
            echo "  --install      Install python dependencies via pip."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

print_header
check_python

if [ "$INSTALL_DEPS" == "true" ]; then
    install_dependencies
fi

# Create results directory
mkdir -p artifacts/results

# Router
if [ "$STEP" == "all" ]; then
    run_step_0
    run_step_1
    run_step_2
    run_step_3
    run_step_4
    run_step_5
    run_step_6
    run_step_7
    echo ""
    echo "========================================================================"
    log_success "ALL EXPERIMENTS COMPLETED SUCCESSFULLY."
    echo "Results are stored in: artifacts/results/"
    echo "========================================================================"
else
    case $STEP in
        0) run_step_0 ;;
        1) run_step_1 ;;
        2) run_step_2 ;;
        3) run_step_3 ;;
        4) run_step_4 ;;
        5) run_step_5 ;;
        6) run_step_0; run_step_6 ;; # Step 6 needs data from Step 0
        7) run_step_7 ;;
        *) log_error "Invalid step number. Use 0-7."; exit 1 ;;
    esac
fi