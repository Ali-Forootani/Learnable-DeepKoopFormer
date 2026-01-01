#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 20:11:19 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:10:24 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:04:59 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer-PRO benchmark on Wind-Speed (or similar) data (HPC-ready)

Sweeps
  • patch lengths     [via patch_lens]
  • forecast horizons [via horizons]

Koopman variants:
  · StrictStableKoopmanOperator      – constrained (spectral radius < ρ_max)
  · Learnable Koopman family:
      - LearnableKoopmanOperatorScalar   – global α, β
      - LearnableKoopmanOperatorPerMode  – per-mode α_i, β_i
      - LearnableKoopmanOperatorMLP      – MLP-based squashing
      - LearnableKoopmanOperatorLowRank  – low-rank K (configurable rank)
  · UnconstrainedKoopmanOperator     – free dense Koopman matrix

Backbones:
  · Koopformer_PatchTST              – constrained / learnable / unconstrained
  · Koopformer_Autoformer            – constrained / learnable / unconstrained
  · Koopformer_Informer              – constrained / learnable / unconstrained

Baselines:
  · SimpleLSTMForecaster
  · DLinearForecaster                – linear temporal mapping (per-channel)
  · SimpleSSMForecaster              – linear state-space model (discrete-time)

Extended with:
  - Train/Test split (train_frac)
  - Metrics CSV with Set ∈ {Train, Test}
  - Saving train/test predictions & errors
  - Per-family Lyapunov weights:
      * lyap_weight_constr
      * lyap_weight_learn
      * lyap_weight_unconstr
  - Full eigenspectrum logging for all Koopman variants
  - Learnable Koopman family loop:
      learnable_kinds = ["scalar", "permode", "mlp", "lowrank16"]

2025-05-20 – Ali Forootani
Extended with constrained vs learnable vs unconstrained Koopman

2025-11-29 – Extended with DLinear + SSM baselines
"""

# --------------------------------------------------------------------------- #
# 0)  HPC-safe backend & imports                                              #
# --------------------------------------------------------------------------- #
import os
import argparse
from pathlib import Path

import matplotlib
if os.getenv("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------------------------------- #
# 1)  Reproducibility                                                         #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(7)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 2)  Koopman operators                                                       #
# --------------------------------------------------------------------------- #
def _orth(w: torch.Tensor) -> torch.Tensor:
    # QR-based orthonormalisation
    return torch.linalg.qr(w)[0]


class StrictStableKoopmanOperator(nn.Module):
    """
    ODO-style Koopman parameterisation with spectral radius < ρ_max.

    K = U diag(Σ) V^T,
      U, V orthonormal, Σ ∈ (0, ρ_max) via a squashing nonlinearity.
    """
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        Σ_unit = torch.sigmoid(self.S_raw)
        Σ = Σ_unit * self.ρ_max
        return Σ

    def forward(self, z: torch.Tensor):
        # Orthonormal factors
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        z_next = z @ K.T
        return z_next, K, Σ


# ---- Learnable Koopman base + variants ------------------------------------ #
class LearnableKoopmanBase(nn.Module):
    """
    Base class for learnable Koopman parameterisations (ODO-style).
    Subclasses must implement:
      - _sigma(): returns Σ ∈ (0, ρ_max)
    """
    def _sigma(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _sigma().")

    def forward(self, z: torch.Tensor):
        raise NotImplementedError("Subclasses must implement forward().")


class LearnableKoopmanOperatorScalar(LearnableKoopmanBase):
    """
    ODO-style Koopman with global spectral squashing:
      Σ_i = ρ_max * sigmoid(α * S_i + β),  α, β scalars.
    """
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        Σ_unit = torch.sigmoid(self.alpha * self.S_raw + self.beta)
        Σ = Σ_unit * self.ρ_max
        return Σ

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        z_next = z @ K.T
        return z_next, K, Σ


class LearnableKoopmanOperatorPerMode(LearnableKoopmanBase):
    """
    ODO-style Koopman with per-mode spectral squashing:
      Σ_i = ρ_max * sigmoid(α_i * S_i + β_i)
    """
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.ones(latent_dim))   # vector
        self.beta  = nn.Parameter(torch.zeros(latent_dim))  # vector
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        Σ_unit = torch.sigmoid(self.alpha * self.S_raw + self.beta)
        Σ = Σ_unit * self.ρ_max
        return Σ

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        z_next = z @ K.T
        return z_next, K, Σ


class LearnableKoopmanOperatorMLP(LearnableKoopmanBase):
    """
    ODO-style Koopman with MLP-based spectral squashing:
      Σ_i = ρ_max * sigmoid( g(S_i) ),
    where g is a small 1D MLP shared across modes.
    """
    def __init__(self, latent_dim: int, ρ_max: float = 0.99, hidden_dim: int = 16):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.ρ_max = ρ_max
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _sigma(self) -> torch.Tensor:
        s = self.S_raw.view(-1, 1)
        g_s = self.mlp(s).view(-1)
        Σ_unit = torch.sigmoid(g_s)
        Σ = Σ_unit * self.ρ_max
        return Σ

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        z_next = z @ K.T
        return z_next, K, Σ


class LearnableKoopmanOperatorLowRank(LearnableKoopmanBase):
    """
    Low-rank ODO Koopman:
      K = U_r diag(Σ_r) V_r^T,   U_r, V_r ∈ R^{d×r}, r < d.

    NOTE: K is still (d x d) but with rank at most r.
    """
    def __init__(self, latent_dim: int, rank: int, ρ_max: float = 0.99):
        super().__init__()
        self.d = latent_dim
        self.r = rank
        self.U_raw = nn.Parameter(torch.randn(latent_dim, rank))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, rank))
        self.S_raw = nn.Parameter(torch.randn(rank))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        Σ_unit = torch.sigmoid(self.alpha * self.S_raw + self.beta)
        Σ = Σ_unit * self.ρ_max
        return Σ

    def forward(self, z: torch.Tensor):
        # Orthonormalise columns (R^d×r)
        U, _ = torch.linalg.qr(self.U_raw, mode="reduced")
        V, _ = torch.linalg.qr(self.V_raw, mode="reduced")
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T  # (d, d) low-rank
        z_next = z @ K.T
        return z_next, K, Σ


class UnconstrainedKoopmanOperator(nn.Module):
    """
    Linear latent propagator with no spectral-radius constraint.

    K is a free dense matrix in R^{d x d}. The forward API matches
    StrictStableKoopmanOperator / LearnableKoopmanBase:
        forward(z) -> (z_next, K, Σ)
    where Σ is None.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.K_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

    def forward(self, z: torch.Tensor):
        K = self.K_raw          # (d, d)
        z_next = z @ K.T        # (B, d)
        Σ = None
        return z_next, K, Σ


# ---- Factory for learnable Koopman ---------------------------------------- #
def make_learnable_koopman(kind: str, latent_dim: int, ρ_max: float = 0.99):
    """
    Construct a learnable Koopman operator given a string key.

    kind ∈ {"scalar", "permode", "mlp", "lowrank", "lowrank16", ...}

    - "scalar":   LearnableKoopmanOperatorScalar
    - "permode":  LearnableKoopmanOperatorPerMode
    - "mlp":      LearnableKoopmanOperatorMLP
    - "lowrank":  default rank = latent_dim // 2
    - "lowrankK": where K is an integer, rank = K
    """
    key = kind.lower()

    if key == "scalar":
        return LearnableKoopmanOperatorScalar(latent_dim, ρ_max=ρ_max)

    if key == "permode":
        return LearnableKoopmanOperatorPerMode(latent_dim, ρ_max=ρ_max)

    if key == "mlp":
        return LearnableKoopmanOperatorMLP(latent_dim, ρ_max=ρ_max)

    if key.startswith("lowrank"):
        # parse rank from string if possible, e.g. "lowrank16"
        r = latent_dim // 2
        suffix = key.replace("lowrank", "")
        if suffix.strip():
            try:
                r = int(suffix)
            except ValueError:
                pass
        r = max(1, min(latent_dim, r))
        return LearnableKoopmanOperatorLowRank(latent_dim, rank=r, ρ_max=ρ_max)

    raise ValueError(f"Unknown learnable Koopman kind: {kind}")


# --------------------------------------------------------------------------- #
# 3)  Positional encodings & patch embed                                      #
# --------------------------------------------------------------------------- #
class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return x + self.pe[: x.size(1)]


class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int,
                 patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model,
                              kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.conv(x)           # (B, d, T_p)
        return x.permute(0, 2, 1)  # (B, P, d)


# --------------------------------------------------------------------------- #
# 4)  PatchTST backbone & Koopformer variants                                 #
# --------------------------------------------------------------------------- #
class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int,
                 patch_len: int, d_model: int = 64,
                 num_layers: int = 3, num_heads: int = 4,
                 dim_ff: int = 96):
        super().__init__()
        self.patch = PatchEmbed1D(input_dim, d_model,
                                  patch_len, patch_len)
        n_patches = int(np.ceil(seq_len / patch_len))
        self.pos  = SinCosPosEnc(d_model, max_len=n_patches + 1)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x   = self.patch(x)                  # (B, P, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)     # (B, P+1, d)
        x   = self.encoder(self.pos(x))
        return x[:, 0]                       # (B, d)


class Koopformer_PatchTST_Constrained(nn.Module):
    """
    PatchTST backbone + strictly stable Koopman operator (ρ(K) < ρ_max).
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64, ρ_max: float = 0.99):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len,
                                          patch_len=patch_len,
                                          d_model=d_model)
        self.koop = StrictStableKoopmanOperator(d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class Koopformer_PatchTST_Learnable(nn.Module):
    """
    PatchTST backbone + learnable Koopman (scalar/permode/mlp/lowrank).
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64, ρ_max: float = 0.99,
                 koop_kind: str = "scalar"):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len,
                                          patch_len=patch_len,
                                          d_model=d_model)
        self.koop = make_learnable_koopman(koop_kind, d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class Koopformer_PatchTST_Unconstrained(nn.Module):
    """
    PatchTST backbone + unconstrained Koopman operator.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len,
                                          patch_len=patch_len,
                                          d_model=d_model)
        self.koop = UnconstrainedKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# 5)  Autoformer backbone & Koopformer variants                               #
# --------------------------------------------------------------------------- #
class SeriesDecomp(nn.Module):
    """
    Simple moving-average decomposition: x = trend + seasonal.
    """
    def __init__(self, k: int = 3):
        super().__init__()
        self.avg = nn.AvgPool1d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        # Ensure same length as x
        if trend.size(1) != x.size(1):
            trend = trend[:, : x.size(1)]
        seas = x - trend
        return trend, seas


class SimpleAutoformer(nn.Module):
    def __init__(self, input_len, horizon, input_dim,
                 patch_len,  # used as MA window
                 d_model=64, num_heads=4,
                 dim_ff=64, num_layers=3):
        super().__init__()
        self.dec   = SeriesDecomp(k=patch_len)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos   = SinCosPosEnc(d_model, max_len=input_len)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fc_seas  = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(input_len * input_dim, horizon)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        trend, seas = self.dec(x)
        seas   = self.encoder(self.pos(self.embed(seas)))
        seas_o = self.fc_seas(seas.mean(1))
        trend_o= self.fc_trend(trend.reshape(trend.size(0), -1))
        return seas_o + trend_o          # (B, H)


class Koopformer_Autoformer_Constrained(nn.Module):
    """
    Autoformer-style backbone + strictly stable Koopman in horizon-space.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=16, ρ_max: float = 0.99):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim,
                                         patch_len=patch_len,
                                         d_model=d_model)
        # Latent dimension = horizon (backbone output size)
        self.koop = StrictStableKoopmanOperator(horizon, ρ_max=ρ_max)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)          # (B, H)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class Koopformer_Autoformer_Learnable(nn.Module):
    """
    Autoformer-style backbone + learnable Koopman in horizon-space.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=16, ρ_max: float = 0.99,
                 koop_kind: str = "scalar"):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim,
                                         patch_len=patch_len,
                                         d_model=d_model)
        self.koop = make_learnable_koopman(koop_kind, horizon, ρ_max=ρ_max)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)          # (B, H)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class Koopformer_Autoformer_Unconstrained(nn.Module):
    """
    Autoformer-style backbone + unconstrained Koopman in horizon-space.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=16):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim,
                                         patch_len=patch_len,
                                         d_model=d_model)
        self.koop = UnconstrainedKoopmanOperator(horizon)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)          # (B, H)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# 6)  Informer backbone & Koopformer variants                                 #
# --------------------------------------------------------------------------- #
class InformerSparse(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4,
                 dim_ff=96, num_layers=3,
                 seq_len=120, patch_len=1):
        super().__init__()
        self.use_patch = patch_len > 1
        if self.use_patch:
            self.patch = PatchEmbed1D(input_dim, d_model,
                                      patch_len, patch_len)
            n_tokens = int(np.ceil(seq_len / patch_len))
        else:
            self.embed = nn.Linear(input_dim, d_model)
            n_tokens = seq_len
        self.pos  = SinCosPosEnc(d_model, max_len=n_tokens)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_patch:
            x = self.patch(x)          # (B, P, d_model)
        else:
            x = self.embed(x)          # (B, T, d_model)
        x = self.encoder(self.pos(x))
        return x.mean(dim=1)           # (B, d_model)


class Koopformer_Informer_Constrained(nn.Module):
    """
    Informer-style backbone + strictly stable Koopman.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64, ρ_max: float = 0.99):
        super().__init__()
        self.backbone = InformerSparse(input_dim,
                                       d_model=d_model,
                                       seq_len=seq_len,
                                       patch_len=patch_len)
        self.koop = StrictStableKoopmanOperator(d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class Koopformer_Informer_Learnable(nn.Module):
    """
    Informer-style backbone + learnable Koopman.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64, ρ_max: float = 0.99,
                 koop_kind: str = "scalar"):
        super().__init__()
        self.backbone = InformerSparse(input_dim,
                                       d_model=d_model,
                                       seq_len=seq_len,
                                       patch_len=patch_len)
        self.koop = make_learnable_koopman(koop_kind, d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class Koopformer_Informer_Unconstrained(nn.Module):
    """
    Informer-style backbone + unconstrained Koopman.
    """
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64):
        super().__init__()
        self.backbone = InformerSparse(input_dim,
                                       d_model=d_model,
                                       seq_len=seq_len,
                                       patch_len=patch_len)
        self.koop = UnconstrainedKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# 7)  Baselines: LSTM, DLinear, SSM                                           #
# --------------------------------------------------------------------------- #
class SimpleLSTMForecaster(nn.Module):
    """
    Standard LSTM-based forecaster baseline.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x, return_latents=False):
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]               # (B, hidden_dim)
        out = self.fc(z)          # (B, horizon * F)
        if return_latents:
            # treat z_next = z (no separate propagator)
            return out, z, z
        return out


class DLinearForecaster(nn.Module):
    """
    DLinear-style baseline:
      - purely linear temporal mapping, applied per feature.
      - channel-independent linear projection from past window (seq_len)
        to horizon for each feature.

    For input x ∈ R^{B×T×F}:
      y_hat[:, f, :] = W @ x[:, :, f] + b,  with shared W across f.
    """
    def __init__(self, input_dim: int, seq_len: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len   = seq_len
        self.horizon   = horizon
        self.linear = nn.Linear(seq_len, horizon)  # shared across channels

    def forward(self, x, return_latents: bool = False):
        # x: (B, T, F)
        B, T, F = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"

        # (B, T, F) -> (B, F, T)
        x_perm = x.permute(0, 2, 1)
        # collapse batch & feature dims: (B*F, T)
        x_flat = x_perm.reshape(B * F, T)
        # linear temporal map: (B*F, H)
        z = self.linear(x_flat)
        # reshape back: (B, F, H)
        y = z.reshape(B, F, self.horizon)
        # flatten horizon × features: (B, F*H)
        out = y.reshape(B, F * self.horizon)

        if return_latents:
            # treat z as latent, z_next=z (no Koopman)
            # latent representation shape: (B, F*H)
            z_lat = out
            return out, z_lat, z_lat
        return out


class SimpleSSMForecaster(nn.Module):
    """
    Simple linear state-space model (SSM) baseline.

    h_{t+1} = h_t A^T + x_t B^T
    y       = h_T C^T

    where:
      - h_t ∈ R^{d_h}
      - x_t ∈ R^{F}
      - y   ∈ R^{F * horizon}
    """
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        # A: (d_h, d_h), B: (d_h, F), C: (F * H, d_h)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.C = nn.Linear(hidden_dim, input_dim * horizon)

    def forward(self, x, return_latents: bool = False):
        # x: (B, T, F)
        B, T, F = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(T):
            x_t = x[:, t, :]                          # (B, F)
            h = h @ self.A.T + x_t @ self.B.T        # (B, d_h)
        z = h                                        # final state
        out = self.C(z)                              # (B, F*H)
        if return_latents:
            # no separate propagator, so z_next = z
            return out, z, z
        return out


# --------------------------------------------------------------------------- #
# 8)  Dataset util                                                            #
# --------------------------------------------------------------------------- #
def build_dataset(data: np.ndarray, indices: list[int],
                  seq_len: int, horizon: int):
    """
    Build sliding-window dataset.

    data: (T, D) raw time series
    indices: selected feature indices
    seq_len: input window length
    horizon: forecast horizon
    """
    sub = data[0:2500, indices]
    scaler = MinMaxScaler().fit(sub)
    sub = scaler.transform(sub)
    X, Y = [], []
    for i in range(len(sub) - seq_len - horizon):
        X.append(sub[i: i + seq_len])
        y = sub[i + seq_len: i + seq_len + horizon].T.reshape(-1)
        Y.append(y)
    x_in  = torch.tensor(np.stack(X), dtype=torch.float32)
    x_out = torch.tensor(np.stack(Y), dtype=torch.float32)
    return x_in, x_out, scaler


# --------------------------------------------------------------------------- #
# 9)  Training helper & spectra                                               #
# --------------------------------------------------------------------------- #
def _nested_getattr(obj, path: str):
    for p in path.split('.'):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def get_koopman_spectrum(model: nn.Module, koop_attr: str = "koop"):
    """
    Extract full eigenspectrum for the Koopman layer of the given model.

    Returns:
        np.ndarray of eigenvalues / singular values, or None if no Koopman.
        - For StrictStableKoopmanOperator:
            returns diagonal Σ (bounded singular values).
        - For LearnableKoopmanBase variants:
            returns Σ via _sigma().
        - For UnconstrainedKoopmanOperator:
            returns eigenvalues of K_raw (complex in general).
    """
    obj = _nested_getattr(model, koop_attr) if koop_attr else None
    if isinstance(obj, StrictStableKoopmanOperator):
        with torch.no_grad():
            Σ = obj._sigma()
            return Σ.detach().cpu().numpy()
    if isinstance(obj, LearnableKoopmanBase):
        with torch.no_grad():
            Σ = obj._sigma()
            return Σ.detach().cpu().numpy()
    if isinstance(obj, UnconstrainedKoopmanOperator):
        with torch.no_grad():
            vals = torch.linalg.eigvals(obj.K_raw)
            return vals.detach().cpu().numpy()
    return None


def train(model, x_in, x_out, epochs=4000, lr=3e-4,
          koop_attr="koop", lyap_weight=0.1):
    """
    Train model with MSE + Lyapunov-style term (optional).

    koop_attr: attribute path to Koopman layer, e.g. "koop" or None

    Returns:
        losses: np.ndarray of shape (epochs,)
        eigs:   np.ndarray of shape (epochs,) – max spectral proxy per epoch
    """
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses, eigs = [], []

    # size for Lyapunov (latent dim)
    with torch.no_grad():
        try:
            _, z, z_next = model(x_in[:1], return_latents=True)
        except ValueError:
            z = z_next = torch.zeros(
                1, getattr(model, "fc").in_features, device=DEV
            )
    P = torch.eye(z.shape[1], device=DEV)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred, z, z_next = model(x_in, return_latents=True)
        pred_loss = mse(pred, x_out)

        if lyap_weight:
            zp  = torch.einsum("bi,ij->bj", z, P)
            znp = torch.einsum("bi,ij->bj", z_next, P)
            lyap = torch.relu((znp * z_next).sum(1)
                              - (zp * z).sum(1)).mean()
        else:
            lyap = torch.zeros(1, device=DEV)

        loss = pred_loss + lyap_weight * lyap
        loss.backward()
        opt.step()
        losses.append(loss.item())

        # Log spectral radius / max singular value proxy
        obj = _nested_getattr(model, koop_attr) if koop_attr else None
        if isinstance(obj, StrictStableKoopmanOperator):
            with torch.no_grad():
                Σ = obj._sigma()
                eigs.append(Σ.abs().max().item())
        elif isinstance(obj, LearnableKoopmanBase):
            with torch.no_grad():
                Σ = obj._sigma()
                eigs.append(Σ.abs().max().item())
        elif isinstance(obj, UnconstrainedKoopmanOperator):
            with torch.no_grad():
                try:
                    vals = torch.linalg.eigvals(obj.K_raw)
                    eigs.append(vals.abs().max().item())
                except RuntimeError:
                    eigs.append(0.0)
        else:
            eigs.append(0.0)

        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"Epoch {ep:>4}/{epochs}  "
                  f"MSE {pred_loss.item():.5f}  "
                  f"Lyap {(lyap_weight * lyap).item():.5f}")

    return np.array(losses), np.array(eigs)


# --------------------------------------------------------------------------- #
# 10)  Plot helpers                                                           #
# --------------------------------------------------------------------------- #
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]
_STYLES = ["-", "--", "-.", ":"]


def _save(fig, base):
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight",
                transparent=True)
    fig.savefig(f"{base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_training(res_loss, path=None):
    if not res_loss:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (n, r) in enumerate(res_loss.items()):
        ax1.plot(r["loss"], label=n,
                 color=_COLORS[i % 10], linestyle=_STYLES[i % 4], lw=3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Loss")
    ax1.grid(True, ls="--", alpha=.6)
    ax1.legend()

    k = 0
    for n, r in res_loss.items():
        if np.allclose(r["eig"], 0):
            continue
        ax2.plot(r["eig"], label=n,
                 color=_COLORS[k % 10], linestyle=_STYLES[k % 4], lw=3)
        k += 1
    ax2.set_title("Koopman spectrum proxy (max)")
    ax2.grid(True, ls="--", alpha=.6)
    ax2.legend()
    plt.tight_layout()
    if path:
        _save(fig, path)


def plot_per_feature(x_out: np.ndarray, preds: dict,
                     F: int, H: int, save_path: str | None = None):
    """
    x_out: ground truth (N, F*H)
    preds: dict[name -> (N, F*H)]
    """
    N = x_out.shape[0]
    gt = x_out.reshape(N, F, H)[:, :, 0]
    preds_plot = {n: p.reshape(N, F, H)[:, :, 0] for n, p in preds.items()}
    t = np.arange(N)
    fig, axes = plt.subplots(F, 1, figsize=(10, 3 * F), sharex=True)
    axes = [axes] if F == 1 else axes
    for f in range(F):
        ax = axes[f]
        ax.plot(t, gt[:, f], lw=3, label="Ground Truth", color="black")
        for i, (n, p) in enumerate(preds_plot.items()):
            ax.plot(t, p[:, f], label=n,
                    color=_COLORS[i % len(_COLORS)],
                    linestyle=_STYLES[i % len(_STYLES)],
                    linewidth=3)
        ax.set_ylabel("Value")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if f == 0:
            ax.legend(ncol=3)
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)


# --------------------------------------------------------------------------- #
# 11)  Saving                                                                 #
# --------------------------------------------------------------------------- #
def save_results(pred: np.ndarray, x_out: np.ndarray,
                 model_name: str, prefix: Path, metrics_file: Path,
                 patch_len: int, horizon: int, set_name: str):
    """
    Save predictions/errors and append metrics for a given set (Train/Test).

    Files:
      prefix_{set}_predictions.npy
      prefix_{set}_errors.npy
    """
    err = x_out - pred
    set_tag = set_name.lower()  # "train" or "test"

    np.save(f"{prefix}_{set_tag}_predictions.npy", pred)
    np.save(f"{prefix}_{set_tag}_errors.npy", err)

    mse = float((err ** 2).mean())
    mae = float(np.abs(err).mean())
    pd.DataFrame(
        {
            "Model": [model_name],
            "PatchLen": [patch_len],
            "Horizon": [horizon],
            "Set": [set_name],
            "MSE": [mse],
            "MAE": [mae],
        }
    ).to_csv(metrics_file, mode="a", header=False, index=False)


# --------------------------------------------------------------------------- #
# 12)  Main loop                                                              #
# --------------------------------------------------------------------------- #
def main(file="bitstamp_windows.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results", train_frac: float = 0.8,
         lyap_weight_constr: float = 0.1,
         lyap_weight_learn: float = 0.1,
         lyap_weight_unconstr: float = 0.1,
         learnable_kinds=None):

    patch_lens = patch_lens or [80, 100, 120, 140]
    horizons   = horizons   or [4, 8, 12, 16]
    indices    = indices    or [0, 1, 2, 3, 4, 5]
    learnable_kinds = learnable_kinds or ["scalar", "permode", "mlp", "lowrank16"]

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "Set", "MSE", "MAE"]
                 ).to_csv(metrics_file, index=False)

    raw = np.load(file)

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")

            # NOTE: we use seq_len = patch_len here for the sliding window
            x_all, y_all, _ = build_dataset(raw, indices,
                                            patch_len, horizon)
            N = x_all.shape[0]
            n_train = int(N * train_frac)
            x_train, y_train = x_all[:n_train], y_all[:n_train]
            x_test,  y_test  = x_all[n_train:], y_all[n_train:]

            F, H = len(indices), horizon
            y_train_np = y_train.numpy()
            y_test_np  = y_test.numpy()

            # for plotting: training curves and test predictions
            res_loss = {}
            test_preds = {}

            # ----- LSTM baseline -----
            print("== LSTM baseline ==")
            lstm = SimpleLSTMForecaster(F, hidden_dim=96,
                                        num_layers=2, horizon=horizon)
            loss_l, eig_l = train(lstm, x_train, y_train,
                                  epochs=epochs,
                                  koop_attr=None,
                                  lyap_weight=0.0)
            res_loss["LSTM"] = {"loss": loss_l, "eig": eig_l}

            lstm.to(DEV).eval()
            with torch.no_grad():
                train_pred_l = lstm(x_train.to(DEV)).cpu().numpy()
                test_pred_l  = lstm(x_test.to(DEV)).cpu().numpy()
            test_preds["LSTM"] = test_pred_l

            prefix_lstm = save_dir / f"LSTM_{patch_len}_{horizon}"
            save_results(train_pred_l, y_train_np, "LSTM",
                         prefix_lstm, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_l, y_test_np, "LSTM",
                         prefix_lstm, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(lstm.cpu().state_dict(), f"{prefix_lstm}.pt")

            # ----- DLinear baseline -----
            print("== DLinear baseline ==")
            dlinear = DLinearForecaster(F, seq_len=patch_len, horizon=horizon)
            loss_dl, eig_dl = train(dlinear, x_train, y_train,
                                    epochs=epochs,
                                    koop_attr=None,
                                    lyap_weight=0.0)
            res_loss["DLinear"] = {"loss": loss_dl, "eig": eig_dl}

            dlinear.to(DEV).eval()
            with torch.no_grad():
                train_pred_dl = dlinear(x_train.to(DEV)).cpu().numpy()
                test_pred_dl  = dlinear(x_test.to(DEV)).cpu().numpy()
            test_preds["DLinear"] = test_pred_dl

            prefix_dl = save_dir / f"DLinear_{patch_len}_{horizon}"
            save_results(train_pred_dl, y_train_np, "DLinear",
                         prefix_dl, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_dl, y_test_np, "DLinear",
                         prefix_dl, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(dlinear.cpu().state_dict(), f"{prefix_dl}.pt")

            # ----- SSM baseline -----
            print("== SSM baseline ==")
            ssm = SimpleSSMForecaster(F, hidden_dim=96, horizon=horizon)
            loss_ssm, eig_ssm = train(ssm, x_train, y_train,
                                      epochs=epochs,
                                      koop_attr=None,
                                      lyap_weight=0.0)
            res_loss["SSM"] = {"loss": loss_ssm, "eig": eig_ssm}

            ssm.to(DEV).eval()
            with torch.no_grad():
                train_pred_ssm = ssm(x_train.to(DEV)).cpu().numpy()
                test_pred_ssm  = ssm(x_test.to(DEV)).cpu().numpy()
            test_preds["SSM"] = test_pred_ssm

            prefix_ssm = save_dir / f"SSM_{patch_len}_{horizon}"
            save_results(train_pred_ssm, y_train_np, "SSM",
                         prefix_ssm, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_ssm, y_test_np, "SSM",
                         prefix_ssm, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(ssm.cpu().state_dict(), f"{prefix_ssm}.pt")
            
            # OPTIONAL: save singular values of A for offline analysis
            with torch.no_grad():
                A_cpu = ssm.A.detach().cpu()
                svals_ssm = torch.linalg.svdvals(A_cpu).numpy()
            np.save(f"{prefix_ssm}_spectrum.npy", svals_ssm)

            # ----- PatchTST: constrained & unconstrained -----
            print("== PatchTST – constrained (ρ(K) < ρ_max) ==")
            m_pc = Koopformer_PatchTST_Constrained(
                F, patch_len, horizon,
                patch_len=patch_len, d_model=96, ρ_max=0.99
            )
            loss_pc, eig_pc = train(
                m_pc, x_train, y_train,
                epochs=epochs, koop_attr="koop",
                lyap_weight=lyap_weight_constr
            )
            res_loss["Koop-PatchTST (constr.)"] = {
                "loss": loss_pc, "eig": eig_pc
            }

            m_pc.to(DEV).eval()
            with torch.no_grad():
                train_pred_pc = m_pc(x_train.to(DEV)).cpu().numpy()
                test_pred_pc  = m_pc(x_test.to(DEV)).cpu().numpy()
            test_preds["Koop-PatchTST (constr.)"] = test_pred_pc

            prefix_pc = save_dir / f"PatchTST_constr_{patch_len}_{horizon}"
            save_results(train_pred_pc, y_train_np, "Koop-PatchTST (constr.)",
                         prefix_pc, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_pc, y_test_np, "Koop-PatchTST (constr.)",
                         prefix_pc, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(m_pc.cpu().state_dict(), f"{prefix_pc}.pt")
            spec_pc = get_koopman_spectrum(m_pc, "koop")
            if spec_pc is not None:
                np.save(f"{prefix_pc}_spectrum.npy", spec_pc)

            print("== PatchTST – unconstrained ==")
            m_pu = Koopformer_PatchTST_Unconstrained(
                F, patch_len, horizon,
                patch_len=patch_len, d_model=96
            )
            loss_pu, eig_pu = train(
                m_pu, x_train, y_train,
                epochs=epochs, koop_attr="koop",
                lyap_weight=lyap_weight_unconstr
            )
            res_loss["Koop-PatchTST (unconstr.)"] = {
                "loss": loss_pu, "eig": eig_pu
            }

            m_pu.to(DEV).eval()
            with torch.no_grad():
                train_pred_pu = m_pu(x_train.to(DEV)).cpu().numpy()
                test_pred_pu  = m_pu(x_test.to(DEV)).cpu().numpy()
            test_preds["Koop-PatchTST (unconstr.)"] = test_pred_pu

            prefix_pu = save_dir / f"PatchTST_unconstr_{patch_len}_{horizon}"
            save_results(train_pred_pu, y_train_np, "Koop-PatchTST (unconstr.)",
                         prefix_pu, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_pu, y_test_np, "Koop-PatchTST (unconstr.)",
                         prefix_pu, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(m_pu.cpu().state_dict(), f"{prefix_pu}.pt")
            spec_pu = get_koopman_spectrum(m_pu, "koop")
            if spec_pu is not None:
                np.save(f"{prefix_pu}_spectrum.npy", spec_pu)

            # ----- PatchTST: learnable family -----
            for kind in learnable_kinds:
                print(f"== PatchTST – learnable Koopman ({kind}) ==")
                m_pl = Koopformer_PatchTST_Learnable(
                    F, patch_len, horizon,
                    patch_len=patch_len, d_model=96, ρ_max=0.99,
                    koop_kind=kind
                )
                loss_pl, eig_pl = train(
                    m_pl, x_train, y_train,
                    epochs=epochs, koop_attr="koop",
                    lyap_weight=lyap_weight_learn
                )
                key_name = f"Koop-PatchTST (learn:{kind})"
                res_loss[key_name] = {
                    "loss": loss_pl, "eig": eig_pl
                }

                m_pl.to(DEV).eval()
                with torch.no_grad():
                    train_pred_pl = m_pl(x_train.to(DEV)).cpu().numpy()
                    test_pred_pl  = m_pl(x_test.to(DEV)).cpu().numpy()
                test_preds[key_name] = test_pred_pl

                prefix_pl = save_dir / f"PatchTST_learn-{kind}_{patch_len}_{horizon}"
                save_results(train_pred_pl, y_train_np, key_name,
                             prefix_pl, metrics_file,
                             patch_len, horizon, set_name="Train")
                save_results(test_pred_pl, y_test_np, key_name,
                             prefix_pl, metrics_file,
                             patch_len, horizon, set_name="Test")
                torch.save(m_pl.cpu().state_dict(), f"{prefix_pl}.pt")
                spec_pl = get_koopman_spectrum(m_pl, "koop")
                if spec_pl is not None:
                    np.save(f"{prefix_pl}_spectrum.npy", spec_pl)

            # ----- Autoformer: constrained & unconstrained -----
            print("== Autoformer – constrained (ρ(K) < ρ_max) ==")
            m_ac = Koopformer_Autoformer_Constrained(
                F, patch_len, horizon,
                patch_len=patch_len, d_model=96, ρ_max=0.99
            )
            loss_ac, eig_ac = train(
                m_ac, x_train, y_train,
                epochs=epochs, koop_attr="koop",
                lyap_weight=lyap_weight_constr
            )
            res_loss["Koop-Autoformer (constr.)"] = {
                "loss": loss_ac, "eig": eig_ac
            }

            m_ac.to(DEV).eval()
            with torch.no_grad():
                train_pred_ac = m_ac(x_train.to(DEV)).cpu().numpy()
                test_pred_ac  = m_ac(x_test.to(DEV)).cpu().numpy()
            test_preds["Koop-Autoformer (constr.)"] = test_pred_ac

            prefix_ac = save_dir / f"Autoformer_constr_{patch_len}_{horizon}"
            save_results(train_pred_ac, y_train_np, "Koop-Autoformer (constr.)",
                         prefix_ac, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_ac, y_test_np, "Koop-Autoformer (constr.)",
                         prefix_ac, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(m_ac.cpu().state_dict(), f"{prefix_ac}.pt")
            spec_ac = get_koopman_spectrum(m_ac, "koop")
            if spec_ac is not None:
                np.save(f"{prefix_ac}_spectrum.npy", spec_ac)

            print("== Autoformer – unconstrained ==")
            m_au = Koopformer_Autoformer_Unconstrained(
                F, patch_len, horizon,
                patch_len=patch_len, d_model=96
            )
            loss_au, eig_au = train(
                m_au, x_train, y_train,
                epochs=epochs, koop_attr="koop",
                lyap_weight=lyap_weight_unconstr
            )
            res_loss["Koop-Autoformer (unconstr.)"] = {
                "loss": loss_au, "eig": eig_au
            }

            m_au.to(DEV).eval()
            with torch.no_grad():
                train_pred_au = m_au(x_train.to(DEV)).cpu().numpy()
                test_pred_au  = m_au(x_test.to(DEV)).cpu().numpy()
            test_preds["Koop-Autoformer (unconstr.)"] = test_pred_au

            prefix_au = save_dir / f"Autoformer_unconstr_{patch_len}_{horizon}"
            save_results(train_pred_au, y_train_np, "Koop-Autoformer (unconstr.)",
                         prefix_au, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_au, y_test_np, "Koop-Autoformer (unconstr.)",
                         prefix_au, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(m_au.cpu().state_dict(), f"{prefix_au}.pt")
            spec_au = get_koopman_spectrum(m_au, "koop")
            if spec_au is not None:
                np.save(f"{prefix_au}_spectrum.npy", spec_au)

            # ----- Autoformer: learnable family -----
            for kind in learnable_kinds:
                print(f"== Autoformer – learnable Koopman ({kind}) ==")
                m_al = Koopformer_Autoformer_Learnable(
                    F, patch_len, horizon,
                    patch_len=patch_len, d_model=96, ρ_max=0.99,
                    koop_kind=kind
                )
                loss_al, eig_al = train(
                    m_al, x_train, y_train,
                    epochs=epochs, koop_attr="koop",
                    lyap_weight=lyap_weight_learn
                )
                key_name = f"Koop-Autoformer (learn:{kind})"
                res_loss[key_name] = {
                    "loss": loss_al, "eig": eig_al
                }

                m_al.to(DEV).eval()
                with torch.no_grad():
                    train_pred_al = m_al(x_train.to(DEV)).cpu().numpy()
                    test_pred_al  = m_al(x_test.to(DEV)).cpu().numpy()
                test_preds[key_name] = test_pred_al

                prefix_al = save_dir / f"Autoformer_learn-{kind}_{patch_len}_{horizon}"
                save_results(train_pred_al, y_train_np, key_name,
                             prefix_al, metrics_file,
                             patch_len, horizon, set_name="Train")
                save_results(test_pred_al, y_test_np, key_name,
                             prefix_al, metrics_file,
                             patch_len, horizon, set_name="Test")
                torch.save(m_al.cpu().state_dict(), f"{prefix_al}.pt")
                spec_al = get_koopman_spectrum(m_al, "koop")
                if spec_al is not None:
                    np.save(f"{prefix_al}_spectrum.npy", spec_al)

            # ----- Informer: constrained & unconstrained -----
            print("== Informer – constrained (ρ(K) < ρ_max) ==")
            m_ic = Koopformer_Informer_Constrained(
                F, patch_len, horizon,
                patch_len=patch_len, d_model=96, ρ_max=0.99
            )
            loss_ic, eig_ic = train(
                m_ic, x_train, y_train,
                epochs=epochs, koop_attr="koop",
                lyap_weight=lyap_weight_constr
            )
            res_loss["Koop-Informer (constr.)"] = {
                "loss": loss_ic, "eig": eig_ic
            }

            m_ic.to(DEV).eval()
            with torch.no_grad():
                train_pred_ic = m_ic(x_train.to(DEV)).cpu().numpy()
                test_pred_ic  = m_ic(x_test.to(DEV)).cpu().numpy()
            test_preds["Koop-Informer (constr.)"] = test_pred_ic

            prefix_ic = save_dir / f"Informer_constr_{patch_len}_{horizon}"
            save_results(train_pred_ic, y_train_np, "Koop-Informer (constr.)",
                         prefix_ic, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_ic, y_test_np, "Koop-Informer (constr.)",
                         prefix_ic, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(m_ic.cpu().state_dict(), f"{prefix_ic}.pt")
            spec_ic = get_koopman_spectrum(m_ic, "koop")
            if spec_ic is not None:
                np.save(f"{prefix_ic}_spectrum.npy", spec_ic)

            print("== Informer – unconstrained ==")
            m_iu = Koopformer_Informer_Unconstrained(
                F, patch_len, horizon,
                patch_len=patch_len, d_model=96
            )
            loss_iu, eig_iu = train(
                m_iu, x_train, y_train,
                epochs=epochs, koop_attr="koop",
                lyap_weight=lyap_weight_unconstr
            )
            res_loss["Koop-Informer (unconstr.)"] = {
                "loss": loss_iu, "eig": eig_iu
            }

            m_iu.to(DEV).eval()
            with torch.no_grad():
                train_pred_iu = m_iu(x_train.to(DEV)).cpu().numpy()
                test_pred_iu  = m_iu(x_test.to(DEV)).cpu().numpy()
            test_preds["Koop-Informer (unconstr.)"] = test_pred_iu

            prefix_iu = save_dir / f"Informer_unconstr_{patch_len}_{horizon}"
            save_results(train_pred_iu, y_train_np, "Koop-Informer (unconstr.)",
                         prefix_iu, metrics_file,
                         patch_len, horizon, set_name="Train")
            save_results(test_pred_iu, y_test_np, "Koop-Informer (unconstr.)",
                         prefix_iu, metrics_file,
                         patch_len, horizon, set_name="Test")
            torch.save(m_iu.cpu().state_dict(), f"{prefix_iu}.pt")
            spec_iu = get_koopman_spectrum(m_iu, "koop")
            if spec_iu is not None:
                np.save(f"{prefix_iu}_spectrum.npy", spec_iu)

            # ----- Informer: learnable family -----
            for kind in learnable_kinds:
                print(f"== Informer – learnable Koopman ({kind}) ==")
                m_il = Koopformer_Informer_Learnable(
                    F, patch_len, horizon,
                    patch_len=patch_len, d_model=96, ρ_max=0.99,
                    koop_kind=kind
                )
                loss_il, eig_il = train(
                    m_il, x_train, y_train,
                    epochs=epochs, koop_attr="koop",
                    lyap_weight=lyap_weight_learn
                )
                key_name = f"Koop-Informer (learn:{kind})"
                res_loss[key_name] = {
                    "loss": loss_il, "eig": eig_il
                }

                m_il.to(DEV).eval()
                with torch.no_grad():
                    train_pred_il = m_il(x_train.to(DEV)).cpu().numpy()
                    test_pred_il  = m_il(x_test.to(DEV)).cpu().numpy()
                test_preds[key_name] = test_pred_il

                prefix_il = save_dir / f"Informer_learn-{kind}_{patch_len}_{horizon}"
                save_results(train_pred_il, y_train_np, key_name,
                             prefix_il, metrics_file,
                             patch_len, horizon, set_name="Train")
                save_results(test_pred_il, y_test_np, key_name,
                             prefix_il, metrics_file,
                             patch_len, horizon, set_name="Test")
                torch.save(m_il.cpu().state_dict(), f"{prefix_il}.pt")
                spec_il = get_koopman_spectrum(m_il, "koop")
                if spec_il is not None:
                    np.save(f"{prefix_il}_spectrum.npy", spec_il)

            # ----- Figures -----
            plot_training(res_loss,
                          save_dir / f"train_patch{patch_len}_h{horizon}")
            plot_per_feature(y_test_np,
                             test_preds,
                             F, H, save_dir / f"feat_patch{patch_len}_h{horizon}")

            # Console summary (TEST set)
            print("\nFinal TEST metrics:")
            for n, p in test_preds.items():
                e = y_test_np - p
                print(f"{n:<45s} MSE {(e**2).mean():.4e} "
                      f"MAE {np.abs(e).mean():.4e}")


# --------------------------------------------------------------------------- #
# 13)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Koopformer-PRO benchmark "
            "(constrained vs learnable family vs unconstrained Koopman, "
            "with scalar/permode/mlp/lowrank learnable variants, "
            "plus LSTM / DLinear / SSM baselines, HPC-ready)"
        )
    )
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str,
                   default="results/results_dkf_lean_var_ful_others_energy")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--lyap_weight_constr", type=float, default=0.1,
                   help="Lyapunov weight for constrained Koopman variants (0 to turn off).")
    p.add_argument("--lyap_weight_learn", type=float, default=0.1,
                   help="Lyapunov weight for learnable Koopman variants (0 to turn off).")
    p.add_argument("--lyap_weight_unconstr", type=float, default=0.1,
                   help="Lyapunov weight for unconstrained Koopman variants (0 to turn off).")
    p.add_argument(
        "--learnable_kinds",
        type=str,
        default="scalar,permode,mlp,lowrank16",
        help="Comma-separated learnable Koopman variants: "
             "e.g. scalar,permode,mlp,lowrank16"
    )
    args = p.parse_args()

    if args.learnable_kinds:
        kinds_list = [s.strip() for s in args.learnable_kinds.split(",") if s.strip()]
    else:
        kinds_list = None

    main(file=args.file,
         epochs=args.epochs,
         seq_len=args.seq_len,
         save_dir=args.save_dir,
         train_frac=args.train_frac,
         lyap_weight_constr=args.lyap_weight_constr,
         lyap_weight_learn=args.lyap_weight_learn,
         lyap_weight_unconstr=args.lyap_weight_unconstr,
         learnable_kinds=kinds_list)
