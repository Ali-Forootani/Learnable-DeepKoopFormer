# Learnable-DeepKoopFormer

**A Learnable Koopman-Enhanced Transformer Framework with Spectral Control for Multivariate Time Series Forecasting**

---

## Overview

**Learnable-DeepKoopFormer** is a research-grade forecasting framework that unifies
**Transformer-based sequence models** with **learnable Koopman operators**
to deliver **stable, interpretable, and scalable multivariate time-series forecasting**.

Unlike classical Koopman approaches with fixed or overly constrained operators,
Learnable-DeepKoopFormer introduces a **family of learnable Koopman parameterizations**
that explicitly control **spectral radius, stability, rank, and latent time scales**,
while remaining fully compatible with modern Transformer backbones.

The framework supports **PatchTST, Autoformer, and Informer**, alongside strong
baselines (LSTM, DLinear, SSM), with **reproducible benchmarking, spectral diagnostics,
and large-scale HPC experimentation**.

---

## Key Contributions

- **Learnable Koopman Operators with Spectral Control**
  - Scalar-gated Koopman operators
  - Per-mode (dimension-wise) gated Koopman operators
  - MLP-shaped spectral mappings
  - Low-rank Koopman operators
  - Optional unconstrained Koopman baselines for ablation

- **Orthogonal–Diagonal–Orthogonal (ODO) Parameterization**
  - Explicit spectral radius control: ρ(K) < ρₘₐₓ
  - Normal, well-conditioned latent dynamics
  - Guaranteed exponential stability when desired

- **Lyapunov-Regularized Training**
  - Penalizes latent energy growth
  - Encourages contractive and invertible latent evolution
  - Improves robustness for long-horizon forecasting

- **Transformer-Agnostic Design**
  - Drop-in Koopman modules for:
    - PatchTST
    - Autoformer
    - Informer
  - Channel-independent latent modeling for high-dimensional signals

- **Full Spectral Diagnostics**
  - Eigenvalue / singular-value logging
  - Spectral radius tracking
  - Stability envelope visualization
  - Bias–variance and expressiveness analysis

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/learnabledeepkoopformer.git
cd learnabledeepkoopformer



## Lineage and Origin

**Learnable-DeepKoopFormer** originates directly from the original **DeepKoopFormer** framework, which introduced the integration of spectrally constrained Koopman operators with Transformer-based time-series forecasting architectures.

The original **DeepKoopFormer** established the following core ideas:
- Koopman-enhanced encoder–propagator–decoder architectures,
- Orthogonal–Diagonal–Orthogonal (ODO) parameterization of the Koopman operator,
- Explicit spectral-radius control and Lyapunov-based stability regularization,
- Stable and interpretable long-horizon forecasting with PatchTST, Autoformer, and Informer backbones.

**Learnable-DeepKoopFormer extends this foundation** by introducing *learnable* Koopman operator families that generalize the original strictly constrained formulation. These extensions enable adaptive spectral shaping, anisotropic temporal dynamics, and low-rank latent evolution, while preserving the stability and interpretability guarantees introduced in DeepKoopFormer.

The original DeepKoopFormer implementation is publicly available at:  
https://github.com/Ali-Forootani/deepkoopformer


## Simulation Results and Reproducibility

All simulation results, trained models, metrics, and figures associated with **Learnable-DeepKoopFormer** are publicly available for full reproducibility.

The archived results include:
- Forecasting metrics (MSE, MAE) across all patch lengths and horizons
- Spectral diagnostics of learned Koopman operators
- Stability and robustness analyses
- Results for climate (CMIP6, ERA5), energy systems, and cryptocurrency datasets

The complete simulation and dataset archive is hosted on and can be accessed at:

https://zenodo.org/records/17988424

https://doi.org/10.5281/zenodo.18115612

## Example Figures

### Learnable DeepKoopFormer Energy systems dataset (6 Channels)

![Energy Systems Dataset](./pictures/koopformer_pro_error_dist_energy_systems_violin_panel.png)

