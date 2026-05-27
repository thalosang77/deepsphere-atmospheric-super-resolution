# DeepSphere Atmospheric Super-Resolution

Spherical graph neural super-resolution for atmospheric moisturefields using DeepSphere-ResNet, conservation-aware correction, and cross-resolution transfer.

This repository accompanies the manuscript:

> **Spherical Graph Neural Super-Resolution for Atmospheric Moisture Fields with Conservation Correction and Cross-Resolution Transfer**  
> Submitted to *Environmental Modelling & Software*.

---

# Overview

This project investigates spherical graph neural networks for atmospheric super-resolution using ERA5 reanalysis humidity data.

The framework combines:

- spherical spectral graph convolutions,
- residual super-resolution learning,
- directional stencil filtering,
- conservation-aware correction,
- seam-stability diagnostics,
- spectral realism evaluation,
- zero-shot cross-resolution transfer.

The methodology is designed for climate and environmental applications where preserving physically meaningful atmospheric structure is important beyond pointwise RMSE optimization.

---

# Main Features

- DeepSphere-ResNet atmospheric super-resolution
- Physics-aware spherical graph construction
- ERA5 humidity preprocessing pipeline
- Bilinear and CNN baseline comparisons
- Spectral power and spectral slope diagnostics
- Gradient/front preservation analysis
- Extreme-value reconstruction evaluation
- Longitude seam consistency analysis
- Multi-region transfer experiments
- Zero-shot LR→MR→HR transfer evaluation
- Conservation-aware moisture correction

---

# Repository Structure

```text
datasets/
    ERA5 downloading and preprocessing

models/
    DeepSphere-ResNet architectures

src/
    Training, evaluation, diagnostics, and analysis scripts
