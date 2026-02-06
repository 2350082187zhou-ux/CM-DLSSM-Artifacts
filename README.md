## Project Overview

CM-DLSSM is a Cross-Modal Differentiable Logic State Space Model designed for unified security analysis of source code and binaries. The model combines the efficient long-sequence processing capabilities of State Space Models (SSMs) with differentiable logic reasoning to enable precise vulnerability detection and causal analysis of software.

### Core Features

- **Cross-Modal Analysis**: Simultaneously processes source code and binaries for unified security assessment
- **Long Context Understanding**: Leverages Mamba SSM architecture for efficient handling of ultra-long code sequences
- **Differentiable Logic Reasoning**: Seamlessly integrates symbolic logic with neural networks through CAVI Engine
- **Causal Inference**: Provides causal analysis of vulnerability origins, supporting counterfactual reasoning
- **Hardware-Aware Optimization**: Optimized for CUDA devices, supporting efficient parallel computing

## Installation

### System Requirements

- Python 3.9 or higher
- CUDA 12.1 or higher (recommended for GPU acceleration)
- Minimum 8GB RAM (16GB+ recommended)
- Minimum 5GB disk space

### Dependency Installation

```bash
# Clone the repository
git clone https://github.com/2350082187zhou-ux/CM-DLSSM-Artifacts.git
cd CM-DLSSM-Artifacts

# Install dependencies
pip install -r requirements.txt
```

### Optional Configuration

- **CUDA Acceleration**: Ensure compatible CUDA drivers and toolkit are installed
- **Environment Variables**: It's recommended to set `PYTHONPATH=.` to ensure proper module loading

## Quick Start

### Run Full Experiment Pipeline

```bash
# Run all experiment steps (approximately 20-40 minutes)
./REPRODUCE_ALL.sh

# Or install dependencies only
./REPRODUCE_ALL.sh --install

# Or run specific steps
./REPRODUCE_ALL.sh --step 1  # Run only benchmark tests
```

### Core Experiment Steps

| Step | Name                  | Target                                                | Script                                    |
| ---- | --------------------- | ----------------------------------------------------- | ----------------------------------------- |
| 0    | Data Generation       | Generate synthetic BLV dataset for causal inference   | `data/generators/blv_synth.py`            |
| 1    | Benchmark Testing     | Validate model accuracy (F1, Recall@1%FPR)            | `experiments/01_benchmark_main.py`        |
| 2    | Long Context Analysis | Verify O(L) complexity claim                          | `experiments/02_long_context_evidence.py` |
| 3    | Ablation Study        | Analyze impact of logic/calibration/gating components | `experiments/03_ablation_matrix.py`       |
| 4    | Audit Reproducibility | Verify mathematical correctness of audit certificates | `experiments/04_audit_reproducibility.py` |
| 5    | Cross-Modal Alignment | Evaluate source code-binary matching performance      | `experiments/05_cross_modal_alignment.py` |
| 6    | Causal Inference      | Validate doubly robust estimator performance          | `experiments/06_causal_blv_task.py`       |
| 7    | Efficiency Analysis   | Hardware scalability and performance profiling        | `experiments/07_efficiency_profiling.py`  |

## Project Structure

```
CM-DLSSM-Artifacts/
├── artifacts/          # Experiment results and outputs
├── configs/            # Experiment configuration files
│   ├── ablation/       # Ablation experiment configurations
│   ├── baselines/      # Baseline model configurations
│   └── model/          # Model parameter configurations
├── data/               # Data-related files
│   ├── generators/     # Data generators
│   └── processed/      # Processed data
├── experiments/        # Experiment scripts
│   ├── 01_benchmark_main.py
│   ├── 02_long_context_evidence.py
│   ├── 03_ablation_matrix.py
│   ├── 04_audit_reproducibility.py
│   ├── 05_cross_modal_alignment.py
│   ├── 06_causal_blv_task.py
│   └── 07_efficiency_profiling.py
├── src/                # Core source code
│   ├── baselines/      # Baseline model implementations
│   ├── causal/         # Causal inference modules
│   ├── infra/          # Infrastructure components
│   ├── logic/          # Differentiable logic engine
│   └── sensing/        # Sensors and feature extraction
├── outputs/            # Experiment output directory
├── REPRODUCE_ALL.sh    # Experiment reproduction script
├── requirements.txt    # Dependency list
└── README.md           # Project documentation
```

## Core Modules

### 1. MambaBlock (src/sensing/mamba_block.py)

Efficient sequence processing module based on Mamba SSM, providing O(L) complexity for long context understanding.

### 2. CAVIEngine (src/logic/cavi_engine.py)

Differentiable logic engine that seamlessly integrates symbolic logic with neural networks, supporting complex security rule reasoning.

### 3. CalibrationVault (src/infra/calibration.py)

Model calibration component that ensures consistency between predicted probabilities and actual risks, improving the reliability of security assessments.

### 4. DoublyRobustEstimator (src/causal/dr_estimator.py)

Causal inference module providing doubly robust estimators, supporting in-depth analysis of vulnerability causes.

## Experiment Results

### Key Performance Metrics

- **Accuracy**: Achieves 94.2% F1 score on BigVul dataset
- **Long Context**: Supports processing code sequences with over 100K tokens
- **Causal Inference**: Doubly robust estimator error is 42% lower than naive methods
- **Efficiency**: Achieves 1200+ tokens/second processing speed on GPU

### Output Directory

All experiment results are stored in the `artifacts/results/` directory, including:

- Performance metric tables
- Long context degradation curves
- Ablation study results
- Causal analysis reports

## Hardware Requirements

### Recommended Configuration

- **CPU**: 4+ cores
- **GPU**: NVIDIA RTX 3090 or higher (8GB+ VRAM)
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ available space

### Minimum Configuration (CPU Mode)

- **CPU**: 2+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 20GB+ available space

## Troubleshooting

### Common Issues

1. **CUDA Errors**: Ensure compatible CUDA drivers and toolkit are installed, or remove CUDA-related dependencies from `requirements.txt`

2. **Out of Memory**: For long context experiments, consider reducing batch size or using smaller model configurations

3. **Dependency Conflicts**: Use virtual environments (like `venv` or `conda`) to isolate dependencies

### Debug Mode

All experiment scripts support `debug=True` parameter for quick script functionality verification:

```bash
python experiments/01_benchmark_main.py debug=True
```

## How to Contribute

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation

If you use this project in your research, please cite the following paper:

```
@article{CM-DLSSM2024,
  title={CM-DLSSM: A Cross-Modal Differentiable Logic State Space Model for Unified Security Analysis},
  author={Zhou, Zedong and et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Contact

- **Project Maintainer**: Zedong Zhou
- **Email**: 2350082187zhou@gmail.com
- **GitHub**: [2350082187zhou-ux
