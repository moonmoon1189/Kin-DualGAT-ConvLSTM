# Kin-DualGAT-ConvLSTM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper:  
**"A Basketball Defensive Footwork Quality Assessment Method Based on Offensive and Defensive Distance Fields and Postural Attention Mechanisms"**.

## 📖 Abstract
Current basketball defensive footwork assessments face bottlenecks due to a lack of spatial interaction potential energy and decoupling from microscopic dynamics. This paper proposes a dual-stream network, **Kin-DualGAT-ConvLSTM** (Kinematic-constrained Dual Graph Attention and Convolutional Long Short-Term Memory Network), which integrates kinematic constraint regularization. 
- **Macro level**: Integrates relative motion velocity and defensive boundary to construct a dynamic potential energy field.
- **Micro level**: Utilizes structural tensor decomposition of the local covariance matrix to adaptively aggregate temporal characteristics of force-generating joints.
- **Kinematic Penalty**: Imposes soft regularization on the evolution of hidden states based on biomechanical plausibility thresholds (acceleration and joint angles).

Tested on the highly-adversarial **CB-FQA** dataset, the proposed model achieves an $R^2$ of 0.924 and demonstrates significant robustness under missing-data (20%) and noisy conditions.

## 📁 Repository Structure
```text
Kin-DualGAT-ConvLSTM/
├── configs/            # YAML configuration files (main.yaml)
├── data/               # Dataset loaders and preparation scripts
├── models/             # Core architecture (Kin-DualGAT-ConvLSTM, modules)
├── baselines/          # Benchmark models (ST-GCN, CTR-GCN, ICP, etc.)
├── checkpoints/        # Saved model weights (Generated during training)
├── losses.py           # Kinematic constraint loss & total loss
├── metrics.py          # R2, RMSE, MAE, PLCC, SRCC, ICC computations
├── train.py            # Main training pipeline
├── evaluate.py         # Inference and evaluation script
├── run_experiments.py  # Automation for ablations & robustness tests
└── validate_dataset.py # Automated integrity checker for CB-FQA
