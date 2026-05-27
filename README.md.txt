# DeepSphere-ResNet Atmospheric Super-Resolution

Spherical graph neural super-resolution for ERA5 atmospheric humidity
fields using DeepSphere-ResNet, conservation correction and
cross-resolution transfer.

## Overview

This repository contains the implementation accompanying the paper:

> **Spherical Graph Neural Super-Resolution for Atmospheric Moisture Fields with Conservation Correction and Cross-Resolution Transfer**

The project investigates atmospheric super-resolution on spherical climate domains using graph neural networks. Atmospheric humidity
fields from ERA5 reanalysis are represented as multi-resolution spherical graphs and reconstructed using spectral graph filtering, directional stencil convolutions and residual learning.

The framework evaluates not only pointwise reconstruction accuracy, but also:

- spectral realism,
- physical consistency,
- seam stability,
- cross-region generalization,
- zero-shot cross-resolution transfer.

---

## Main Features

- Multi-resolution spherical graph construction
- Spectral graph convolutions using Chebyshev filtering
- Directional local stencil aggregation
- Residual graph learning for atmospheric super-resolution
- Conservation-aware moisture correction
- Zero-shot LR→MR→HR transfer evaluation
- Fourier spectral diagnostics
- Physical consistency analysis
- Cross-region climate evaluation

---

## Repository Structure

```text
deepsphere-atmospheric-super-resolution/
│
├── src/                 # Core model and graph utilities
├── notebooks/           # Experimental notebooks
├── figures/             # Figures used in manuscript
├── results/             # Evaluation outputs
├── environment/         # Environment and dependency files
├── data/                # Processed metadata/configuration
│
├── requirements.txt
├── LICENSE
└── README.md