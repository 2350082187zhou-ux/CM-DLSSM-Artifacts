
# CM-DLSSM: Cross-Modal Differentiable Logic State Space Model

[![Code License: Apache 2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Artifacts License: CC BY 4.0](https://img.shields.io/badge/Artifacts%20License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch (Container) 2.4+](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)

> **CM-DLSSM**: A Cross-Modal Differentiable Logic State Space Model for Unified Security Analysis of Source Code and Binaries  
> This repository provides the implementation + scripts to reproduce the main paper results, including **Verified Audit Artifacts (VAA)**.

---

## 📌 Paper ↔ Repo Consistency (What matches the paper)
This artifact package is aligned with the paper (main.pdf) on the following points:

- **Long-context scalability**: up to **128,000 tokens** with **O(L)** selective SSM backbone
- **Tokenizer**: **SentencePiece, vocab size |V| = 32,000** (paper setup)
- **Auditable reasoning**: deterministic **Log-Ratio CAVI** traces included in VAAs
- **Cross-modal alignment**: InfoNCE latent alignment + logic fingerprinting via **KL divergence**
- **VAA verification tolerance**: default **ε = 1e-6** for independent verification
- **Reference throughput (H100)**: approx. **3,650–4,120 tokens/sec** depending on context/settings

> Note: Some configurations use power-of-two max lengths internally (e.g., 131,072) while the paper reports 128k as the headline context.

---

## 🧱 Repository Structure
A quick map of where things live:

```
CM-DLSSM-Artifacts/
├── src/                    # Core implementation (SSM + logic + VAA)
├── configs/                # Experiment configurations (paper-aligned)
├── data/                   # Preprocessing + synthetic generators (if provided)
├── experiments/             # Scripts to reproduce key tables/figures
├── outputs/                # Generated results (created by runs)
├── artifacts/              # Paper artifacts / saved reports (if provided)
└── requirements.txt
```

---

## 🚀 Quick Start (Paper Reproduction)
The paper reproduction environment is provided via Docker to lock dependencies and ensure determinism.

### 1) Install
```bash
git clone <ANON_REPO_URL>
cd CM-DLSSM-Artifacts
```

### 2) Build / Run the container (recommended)
```bash
# Example (adjust to your provided Dockerfile / scripts)
docker build -t cm-dlssm:paper .
docker run --gpus all -it --rm -v $PWD:/work cm-dlssm:paper
```

The paper environment targets:
- CUDA **12.x**
- PyTorch **2.4**
- Triton (for fused kernels / scan ops)

### 3) Reproduce all results
```bash
./REPRODUCE_ALL.sh
```

Outputs will be written to `outputs/`.

---

## 🔬 Key Experiments
These scripts correspond to the main claims in the paper.

### Main benchmark
```bash
python experiments/01_benchmark_main.py --config configs/benchmark_config.yaml
```

### Long-context evaluation (128k)
```bash
python experiments/02_long_context_evidence.py --context_length 128000
```

### Ablation study
```bash
python experiments/03_ablation_matrix.py --config configs/ablation/
```

### Audit reproducibility / VAA verification
```bash
python experiments/04_audit_reproducibility.py
```

---

## ⚙️ Configuration (Paper-Aligned Defaults)
This repo follows the paper setup unless a config explicitly states otherwise.

- **Tokenizer**: SentencePiece, **vocab_size = 32000**
- **Max context**: up to **128k** (internally may use 131072 in some configs)
- **VAA verification tolerance**: **ε = 1e-6**
- **Determinism**:
  - Fixed random seed (default: 42)
  - CUDA deterministic flags enabled where applicable

> If you previously used a 65,536 vocab in an older config, please update configs to 32,000 to match the paper, and regenerate any incompatible checkpoints.

---

## ✅ Verified Audit Artifacts (VAA)
CM-DLSSM outputs a schema-enforced VAA (JSON-RPC 2.0 style) containing:
- target hash binding
- evidence logits
- deterministic CAVI trace
- compliance gate (flip matrix + ECE)
- digital signature field (if enabled)

Independent verification recomputes the CAVI fixed-point iteration and accepts the artifact if:
- ||Δq|| < **ε** (default **1e-6**)

---

## 📄 Licensing
To match the paper’s Open Science statement:

- **Code**: Apache License 2.0 (see `LICENSE`)
- **SRTL + synthetic datasets + non-code artifacts**: CC BY 4.0 (see `ARTIFACTS_LICENSE` or `LICENSE-ARTIFACTS`)

If any third-party datasets are referenced, their original licenses apply.

---

## 🔐 Responsible Use
This project supports defensive security research and compliance auditing. Do not use it to target systems you do not own or lack explicit authorization to test.
