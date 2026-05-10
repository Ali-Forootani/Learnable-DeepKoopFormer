#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:29:35 2026

@author: forootani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer-PRO benchmark with PatchTST, Autoformer, Informer,
iTransformer, TimesNet, SSM, and DLinear backbones.

Key addition in this version
----------------------------
SSM and DLinear are available in two ways:

1) Original baselines:
   - SimpleLSTMForecaster
   - DLinearForecaster
   - SimpleSSMForecaster

2) Koopman backbones:
   - DLinear_Backbone
   - SSM_Backbone

So you can now run:
   backbone -> Koopman operator -> forecast head

for:
   patchtst, autoformer, informer, itransformer, timesnet, dlinear, ssm

Koopman variants:
  - StrictStableKoopmanOperator      : constrained spectral proxy < rho_max
  - Learnable Koopman family         : scalar, permode, mlp, lowrankK
  - UnconstrainedKoopmanOperator     : free dense Koopman matrix

Outputs:
  - metrics.csv with Set in {Train, Test}
  - train/test predictions and errors as .npy
  - model checkpoints as .pt
  - Koopman/SSM spectra as .npy
  - training and per-feature plots as PNG/PDF
"""

# --------------------------------------------------------------------------- #
# 0) HPC-safe backend & imports                                               #
# --------------------------------------------------------------------------- #
import os
import argparse
from pathlib import Path
from typing import Optional, Sequence

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
# 1) Reproducibility                                                          #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(7)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 2) Koopman operators                                                        #
# --------------------------------------------------------------------------- #
def _orth(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(w)[0]


class StrictStableKoopmanOperator(nn.Module):
    """
    ODO-style Koopman parameterisation.

    K = U diag(Sigma) V^T,
    where U,V are orthonormal and Sigma in (0, rho_max).

    Note: For non-normal matrices, bounding singular values bounds the spectral
    radius, but the logged Sigma values are singular-value proxies rather than
    full eigenvalues.
    """
    def __init__(self, latent_dim: int, rho_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.rho_max = rho_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.S_raw) * self.rho_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Sigma = self._sigma()
        K = U @ torch.diag(Sigma) @ V.T
        z_next = z @ K.T
        return z_next, K, Sigma


class LearnableKoopmanBase(nn.Module):
    def _sigma(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor):
        raise NotImplementedError


class LearnableKoopmanOperatorScalar(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, rho_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.rho_max = rho_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.rho_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Sigma = self._sigma()
        K = U @ torch.diag(Sigma) @ V.T
        z_next = z @ K.T
        return z_next, K, Sigma


class LearnableKoopmanOperatorPerMode(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, rho_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.ones(latent_dim))
        self.beta = nn.Parameter(torch.zeros(latent_dim))
        self.rho_max = rho_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.rho_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Sigma = self._sigma()
        K = U @ torch.diag(Sigma) @ V.T
        z_next = z @ K.T
        return z_next, K, Sigma


class LearnableKoopmanOperatorMLP(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, rho_max: float = 0.99, hidden_dim: int = 16):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.rho_max = rho_max
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _sigma(self) -> torch.Tensor:
        s = self.S_raw.view(-1, 1)
        g_s = self.mlp(s).view(-1)
        return torch.sigmoid(g_s) * self.rho_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Sigma = self._sigma()
        K = U @ torch.diag(Sigma) @ V.T
        z_next = z @ K.T
        return z_next, K, Sigma


class LearnableKoopmanOperatorLowRank(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, rank: int, rho_max: float = 0.99):
        super().__init__()
        self.d = latent_dim
        self.r = rank
        self.U_raw = nn.Parameter(torch.randn(latent_dim, rank))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, rank))
        self.S_raw = nn.Parameter(torch.randn(rank))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.rho_max = rho_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.rho_max

    def forward(self, z: torch.Tensor):
        U, _ = torch.linalg.qr(self.U_raw, mode="reduced")
        V, _ = torch.linalg.qr(self.V_raw, mode="reduced")
        Sigma = self._sigma()
        K = U @ torch.diag(Sigma) @ V.T
        z_next = z @ K.T
        return z_next, K, Sigma


class UnconstrainedKoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.K_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

    def forward(self, z: torch.Tensor):
        K = self.K_raw
        z_next = z @ K.T
        Sigma = None
        return z_next, K, Sigma


def make_learnable_koopman(kind: str, latent_dim: int, rho_max: float = 0.99):
    key = kind.lower()

    if key == "scalar":
        return LearnableKoopmanOperatorScalar(latent_dim, rho_max=rho_max)
    if key == "permode":
        return LearnableKoopmanOperatorPerMode(latent_dim, rho_max=rho_max)
    if key == "mlp":
        return LearnableKoopmanOperatorMLP(latent_dim, rho_max=rho_max)
    if key.startswith("lowrank"):
        r = latent_dim // 2
        suffix = key.replace("lowrank", "")
        if suffix.strip():
            try:
                r = int(suffix)
            except ValueError:
                pass
        r = max(1, min(latent_dim, r))
        return LearnableKoopmanOperatorLowRank(latent_dim, rank=r, rho_max=rho_max)

    raise ValueError(f"Unknown learnable Koopman kind: {kind}")


# --------------------------------------------------------------------------- #
# 3) Positional encodings & patch embedding                                   #
# --------------------------------------------------------------------------- #
class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)


# --------------------------------------------------------------------------- #
# 4) Transformer / TimesNet-style backbones                                   #
# --------------------------------------------------------------------------- #
class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 3,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.patch = PatchEmbed1D(input_dim, d_model, patch_len, patch_len)
        n_patches = max(1, int(np.floor((seq_len - patch_len) / patch_len) + 1))
        self.pos = SinCosPosEnc(d_model, max_len=n_patches + 1)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(self.pos(x))
        return x[:, 0]


class SeriesDecomp(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        if k % 2 == 0:
            k += 1
        self.avg = nn.AvgPool1d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor):
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        trend = trend[:, : x.size(1)]
        seas = x - trend
        return trend, seas


class SimpleAutoformer(nn.Module):
    def __init__(self, input_len: int, horizon: int, input_dim: int,
                 patch_len: int, d_model: int = 64, num_heads: int = 4,
                 dim_ff: int = 64, num_layers: int = 3):
        super().__init__()
        k = max(1, min(patch_len, input_len))
        self.dec = SeriesDecomp(k=k)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = SinCosPosEnc(d_model, max_len=input_len)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fc_seas = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(input_len * input_dim, horizon)

    def forward(self, x: torch.Tensor):
        trend, seas = self.dec(x)
        seas = self.encoder(self.pos(self.embed(seas)))
        seas_o = self.fc_seas(seas.mean(1))
        trend_o = self.fc_trend(trend.reshape(trend.size(0), -1))
        return seas_o + trend_o


class InformerSparse(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4,
                 dim_ff: int = 96, num_layers: int = 3,
                 seq_len: int = 120, patch_len: int = 1):
        super().__init__()
        self.use_patch = patch_len > 1
        if self.use_patch:
            self.patch = PatchEmbed1D(input_dim, d_model, patch_len, patch_len)
            n_tokens = max(1, int(np.floor((seq_len - patch_len) / patch_len) + 1))
        else:
            self.embed = nn.Linear(input_dim, d_model)
            n_tokens = seq_len
        self.pos = SinCosPosEnc(d_model, max_len=n_tokens)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_patch:
            x = self.patch(x)
        else:
            x = self.embed(x)
        x = self.encoder(self.pos(x))
        return x.mean(dim=1)


class ITransformer_Backbone(nn.Module):
    """
    Lightweight iTransformer-style backbone compatible with Koopman latent interface.
    Input:  [B, L, F]
    Output: [B, D]
    """
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int, num_heads: int = 4, num_layers: int = 1,
                 dim_ff: int = 96):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, d_model)
        self.pos = SinCosPosEnc(d_model, max_len=seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.value_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = z.transpose(1, 2)
        z = self.pool(z).squeeze(-1)
        return z


class TimesBlockSimple(nn.Module):
    """Lightweight temporal convolution block inspired by TimesNet."""
    def __init__(self, d_model: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.norm(out)
        return self.act(x + out)


class TimesNet_Backbone(nn.Module):
    """
    Lightweight TimesNet-style backbone compatible with Koopman latent interface.
    Input:  [B, L, F]
    Output: [B, D]
    """
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int, num_heads: int = 4, num_layers: int = 1,
                 dim_ff: int = 96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TimesBlockSimple(d_model) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = z.transpose(1, 2)
        for blk in self.blocks:
            z = blk(z)
        z = self.pool(z).squeeze(-1)
        return z


# --------------------------------------------------------------------------- #
# 5) New Koopman-compatible DLinear and SSM backbones                         #
# --------------------------------------------------------------------------- #
class DLinear_Backbone(nn.Module):
    """
    DLinear as a Koopman backbone.

    Instead of directly forecasting the horizon, this maps the full input window
    to a latent vector z. Then the common Koopman layer evolves z -> z_next,
    and the shared forecast head maps z_next to horizon * input_dim.

    Input:  [B, L, F]
    Output: [B, D]
    """
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int, num_heads: int = 4, num_layers: int = 1,
                 dim_ff: int = 96):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.temporal = nn.Linear(seq_len, d_model)
        self.feature_mixer = nn.Sequential(
            nn.Linear(input_dim * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        x = x.permute(0, 2, 1)          # [B, F, L]
        z = self.temporal(x)            # [B, F, D]
        z = z.reshape(B, F * z.size(-1))
        z = self.feature_mixer(z)        # [B, D]
        return self.norm(z)


class SSM_Backbone(nn.Module):
    """
    Simple state-space model as a Koopman backbone.

    This SSM produces a latent state h_T. The Koopman operator then acts on
    this latent state, so the whole model becomes:

        x_{1:L} -> SSM hidden state z -> Koopman(z) -> forecast head

    Input:  [B, L, F]
    Output: [B, D]
    """
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int, num_heads: int = 4, num_layers: int = 1,
                 dim_ff: int = 96):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = d_model
        self.seq_len = seq_len
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, T, F = x.shape
        h = torch.zeros(Bsz, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(T):
            x_t = x[:, t, :]
            h = h @ self.A.T + x_t @ self.B.T + self.bias
            h = self.act(h)
        return self.norm(h)


class GatedSSM_Backbone(nn.Module):
    """
    Optional stronger SSM backbone with input gate.

    Use it by passing --backbones gatedssm.
    """
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int, num_heads: int = 4, num_layers: int = 1,
                 dim_ff: int = 96):
        super().__init__()
        self.hidden_dim = d_model
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Linear(input_dim, d_model)
        self.gate = nn.Linear(input_dim + d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, T, _ = x.shape
        h = torch.zeros(Bsz, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(T):
            x_t = x[:, t, :]
            proposal = torch.tanh(h @ self.A.T + self.B(x_t))
            g = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=-1)))
            h = g * proposal + (1.0 - g) * h
        return self.norm(h)


# --------------------------------------------------------------------------- #
# 6) Backbone factory                                                         #
# --------------------------------------------------------------------------- #
def make_backbone(backbone_type: str, input_dim: int, seq_len: int,
                  patch_len: int, horizon: int, d_model: int,
                  num_heads: int = 4, num_layers: int = 3,
                  dim_ff: int = 96):
    key = backbone_type.lower()

    if key == "patchtst":
        return PatchTST_Backbone(input_dim, seq_len, patch_len, d_model,
                                 num_layers, num_heads, dim_ff), d_model

    if key == "autoformer":
        # Autoformer backbone returns horizon-dimensional latent.
        return SimpleAutoformer(seq_len, horizon, input_dim, patch_len,
                                d_model=d_model, num_heads=num_heads,
                                dim_ff=dim_ff, num_layers=num_layers), horizon

    if key == "informer":
        return InformerSparse(input_dim, d_model=d_model, num_heads=num_heads,
                              dim_ff=dim_ff, num_layers=num_layers,
                              seq_len=seq_len, patch_len=patch_len), d_model

    if key == "itransformer":
        return ITransformer_Backbone(input_dim, seq_len, patch_len, d_model,
                                     num_heads=num_heads, num_layers=num_layers,
                                     dim_ff=dim_ff), d_model

    if key == "timesnet":
        return TimesNet_Backbone(input_dim, seq_len, patch_len, d_model,
                                 num_heads=num_heads, num_layers=num_layers,
                                 dim_ff=dim_ff), d_model

    if key == "dlinear":
        return DLinear_Backbone(input_dim, seq_len, patch_len, d_model,
                                num_heads=num_heads, num_layers=num_layers,
                                dim_ff=dim_ff), d_model

    if key == "ssm":
        return SSM_Backbone(input_dim, seq_len, patch_len, d_model,
                            num_heads=num_heads, num_layers=num_layers,
                            dim_ff=dim_ff), d_model

    if key == "gatedssm":
        return GatedSSM_Backbone(input_dim, seq_len, patch_len, d_model,
                                 num_heads=num_heads, num_layers=num_layers,
                                 dim_ff=dim_ff), d_model

    raise ValueError(f"Unknown backbone_type: {backbone_type}")


# --------------------------------------------------------------------------- #
# 7) Generic Koopformer wrapper                                               #
# --------------------------------------------------------------------------- #
class KoopformerGeneric(nn.Module):
    """
    Generic wrapper:

        backbone -> Koopman operator -> linear forecast head
    """
    def __init__(self, input_dim: int, seq_len: int, horizon: int,
                 patch_len: int, backbone_type: str,
                 koopman_type: str,
                 d_model: int = 96, rho_max: float = 0.99,
                 koop_kind: str = "scalar",
                 num_heads: int = 4, num_layers: int = 3,
                 dim_ff: int = 96):
        super().__init__()
        self.backbone_type = backbone_type
        self.koopman_type = koopman_type

        self.backbone, latent_dim = make_backbone(
            backbone_type=backbone_type,
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            horizon=horizon,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
        )

        ktype = koopman_type.lower()
        if ktype == "constr":
            self.koop = StrictStableKoopmanOperator(latent_dim, rho_max=rho_max)
        elif ktype == "learn":
            self.koop = make_learnable_koopman(koop_kind, latent_dim, rho_max=rho_max)
        elif ktype == "unconstr":
            self.koop = UnconstrainedKoopmanOperator(latent_dim)
        else:
            raise ValueError(f"Unknown koopman_type: {koopman_type}")

        self.fc = nn.Linear(latent_dim, horizon * input_dim)

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


# --------------------------------------------------------------------------- #
# 8) Original baselines: LSTM, DLinear, SSM                                   #
# --------------------------------------------------------------------------- #
class SimpleLSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]
        out = self.fc(z)
        if return_latents:
            return out, z, z
        return out


class DLinearForecaster(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.linear = nn.Linear(seq_len, horizon)

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        B, T, F = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        x_perm = x.permute(0, 2, 1)
        x_flat = x_perm.reshape(B * F, T)
        z = self.linear(x_flat)
        y = z.reshape(B, F, self.horizon)
        out = y.reshape(B, F * self.horizon)
        if return_latents:
            return out, out, out
        return out


class SimpleSSMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.C = nn.Linear(hidden_dim, input_dim * horizon)

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        B, T, F = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(T):
            x_t = x[:, t, :]
            h = h @ self.A.T + x_t @ self.B.T
        z = h
        out = self.C(z)
        if return_latents:
            return out, z, z
        return out


# --------------------------------------------------------------------------- #
# 9) Dataset util                                                             #
# --------------------------------------------------------------------------- #
def build_dataset(data: np.ndarray, indices: Sequence[int],
                  seq_len: int, horizon: int, max_rows: int = 2500):
    sub = data[:max_rows, indices]
    scaler = MinMaxScaler().fit(sub)
    sub = scaler.transform(sub)

    X, Y = [], []
    for i in range(len(sub) - seq_len - horizon):
        X.append(sub[i: i + seq_len])
        y = sub[i + seq_len: i + seq_len + horizon].T.reshape(-1)
        Y.append(y)

    if not X:
        raise ValueError(
            f"Not enough rows for seq_len={seq_len}, horizon={horizon}, max_rows={max_rows}."
        )

    x_in = torch.tensor(np.stack(X), dtype=torch.float32)
    x_out = torch.tensor(np.stack(Y), dtype=torch.float32)
    return x_in, x_out, scaler


# --------------------------------------------------------------------------- #
# 10) Training helper & spectra                                               #
# --------------------------------------------------------------------------- #
def _nested_getattr(obj, path: Optional[str]):
    if not path:
        return None
    for p in path.split("."):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def get_koopman_spectrum(model: nn.Module, koop_attr: str = "koop"):
    obj = _nested_getattr(model, koop_attr)
    if isinstance(obj, StrictStableKoopmanOperator):
        with torch.no_grad():
            return obj._sigma().detach().cpu().numpy()
    if isinstance(obj, LearnableKoopmanBase):
        with torch.no_grad():
            return obj._sigma().detach().cpu().numpy()
    if isinstance(obj, UnconstrainedKoopmanOperator):
        with torch.no_grad():
            vals = torch.linalg.eigvals(obj.K_raw)
            return vals.detach().cpu().numpy()
    return None


def get_ssm_backbone_spectrum(model: nn.Module):
    """
    Returns singular values of an SSM backbone A matrix when present.
    Works for KoopformerGeneric(backbone_type='ssm' or 'gatedssm').
    """
    backbone = getattr(model, "backbone", None)
    A = getattr(backbone, "A", None)
    if A is None:
        return None
    with torch.no_grad():
        return torch.linalg.svdvals(A.detach().cpu()).numpy()


def train(model: nn.Module, x_in: torch.Tensor, x_out: torch.Tensor,
          epochs: int = 4000, lr: float = 3e-4,
          koop_attr: Optional[str] = "koop", lyap_weight: float = 0.1):
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses, eigs = [], []

    with torch.no_grad():
        try:
            _, z, z_next = model(x_in[:1], return_latents=True)
        except Exception:
            z = z_next = torch.zeros(1, getattr(model, "fc").in_features, device=DEV)
    P = torch.eye(z.shape[1], device=DEV)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred, z, z_next = model(x_in, return_latents=True)
        pred_loss = mse(pred, x_out)

        if lyap_weight:
            zp = torch.einsum("bi,ij->bj", z, P)
            znp = torch.einsum("bi,ij->bj", z_next, P)
            lyap = torch.relu((znp * z_next).sum(1) - (zp * z).sum(1)).mean()
        else:
            lyap = torch.zeros(1, device=DEV)

        loss = pred_loss + lyap_weight * lyap
        loss.backward()
        opt.step()
        losses.append(loss.item())

        obj = _nested_getattr(model, koop_attr)
        if isinstance(obj, (StrictStableKoopmanOperator, LearnableKoopmanBase)):
            with torch.no_grad():
                eigs.append(obj._sigma().abs().max().item())
        elif isinstance(obj, UnconstrainedKoopmanOperator):
            with torch.no_grad():
                try:
                    eigs.append(torch.linalg.eigvals(obj.K_raw).abs().max().item())
                except RuntimeError:
                    eigs.append(0.0)
        else:
            eigs.append(0.0)

        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(
                f"Epoch {ep:>4}/{epochs}  "
                f"MSE {pred_loss.item():.5f}  "
                f"Lyap {(lyap_weight * lyap).item():.5f}"
            )

    return np.array(losses), np.array(eigs)



# --------------------------------------------------------------------------- #
# 10.5) Koopman perturbation sensitivity                                      #
# --------------------------------------------------------------------------- #
def _get_koopman_matrix(model: nn.Module):
    koop = getattr(model, "koop", None)
    if koop is None:
        return None

    if isinstance(koop, UnconstrainedKoopmanOperator):
        return koop.K_raw

    if isinstance(koop, (StrictStableKoopmanOperator, LearnableKoopmanBase)):
        U, V = _orth(koop.U_raw), _orth(koop.V_raw)
        Sigma = koop._sigma()
        return U @ torch.diag(Sigma) @ V.T

    return None


def forward_with_perturbed_koopman(
    model,
    x,
    eps,
    mode="additive",
    noise_type="gaussian",
    normalize=True,
):
    z = model.backbone(x)
    K = _get_koopman_matrix(model)

    if K is None:
        return model(x), None

    if noise_type == "gaussian":
        Delta = torch.randn_like(K)
    else:
        Delta = 2 * torch.rand_like(K) - 1

    if normalize:
        Delta = Delta / (torch.norm(Delta) + 1e-12) * (torch.norm(K) + 1e-12)

    if mode == "additive":
        Kp = K + eps * Delta
    elif mode == "multiplicative":
        Kp = K * (1 + eps * Delta)
    else:
        raise ValueError("Unknown mode")

    z_next = z @ Kp.T
    pred = model.fc(z_next)
    return pred, Kp


def evaluate_koopman_sensitivity(
    model,
    x,
    y,
    eps_list,
    n_trials=10,
    mode="additive",
):
    model.eval()
    x, y = x.to(DEV), y.to(DEV)

    rows = []

    with torch.no_grad():
        base_pred = model(x)
        base_mse = ((y - base_pred) ** 2).mean().item()

        for eps in eps_list:
            for t in range(n_trials):

                pred, Kp = forward_with_perturbed_koopman(
                    model, x, eps, mode=mode
                )

                err = y - pred
                mse = (err ** 2).mean().item()
                mae = err.abs().mean().item()

                try:
                    rho = torch.linalg.eigvals(Kp).abs().max().item()
                except:
                    rho = np.nan

                rows.append({
                    "eps": eps,
                    "trial": t,
                    "MSE": mse,
                    "MAE": mae,
                    "rel_MSE": mse / max(base_mse, 1e-12),
                    "rho": rho,
                })

    return pd.DataFrame(rows)


def plot_koopman_sensitivity(df, path=None):
    g = df.groupby("eps").agg(
        mean=("rel_MSE", "mean"),
        std=("rel_MSE", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g["eps"], g["mean"], yerr=g["std"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Relative MSE")
    ax.set_title("Koopman Sensitivity")
    ax.grid(True, ls="--", alpha=0.6)

    if path:
        _save(fig, path)
    else:
        plt.show()





# --------------------------------------------------------------------------- #
# 11) Plot helpers                                                            #
# --------------------------------------------------------------------------- #
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]
_STYLES = ["-", "--", "-.", ":"]


def _save(fig, base):
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(f"{base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_training(res_loss: dict, path=None):
    if not res_loss:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (n, r) in enumerate(res_loss.items()):
        ax1.plot(r["loss"], label=n,
                 color=_COLORS[i % len(_COLORS)],
                 linestyle=_STYLES[i % len(_STYLES)], lw=2.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Loss")
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend(fontsize=7)

    k = 0
    for n, r in res_loss.items():
        if np.allclose(r["eig"], 0):
            continue
        ax2.plot(r["eig"], label=n,
                 color=_COLORS[k % len(_COLORS)],
                 linestyle=_STYLES[k % len(_STYLES)], lw=2.5)
        k += 1
    ax2.set_title("Koopman spectrum proxy / radius")
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend(fontsize=7)
    plt.tight_layout()
    if path:
        _save(fig, path)


def plot_per_feature(x_out: np.ndarray, preds: dict,
                     F: int, H: int, save_path: Optional[str] = None):
    N = x_out.shape[0]
    gt = x_out.reshape(N, F, H)[:, :, 0]
    preds_plot = {n: p.reshape(N, F, H)[:, :, 0] for n, p in preds.items()}
    t = np.arange(N)

    fig, axes = plt.subplots(F, 1, figsize=(12, 3 * F), sharex=True)
    axes = [axes] if F == 1 else axes
    for f in range(F):
        ax = axes[f]
        ax.plot(t, gt[:, f], lw=3, label="Ground Truth", color="black")
        for i, (n, p) in enumerate(preds_plot.items()):
            ax.plot(t, p[:, f], label=n,
                    color=_COLORS[i % len(_COLORS)],
                    linestyle=_STYLES[i % len(_STYLES)],
                    linewidth=2)
        ax.set_ylabel("Value")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if f == 0:
            ax.legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)


# --------------------------------------------------------------------------- #
# 12) Saving                                                                  #
# --------------------------------------------------------------------------- #
def save_results(pred: np.ndarray, x_out: np.ndarray,
                 model_name: str, prefix: Path, metrics_file: Path,
                 patch_len: int, horizon: int, set_name: str):
    err = x_out - pred
    set_tag = set_name.lower()

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


def eval_and_save_model(model: nn.Module, model_name: str, prefix: Path,
                        x_train: torch.Tensor, y_train_np: np.ndarray,
                        x_test: torch.Tensor, y_test_np: np.ndarray,
                        metrics_file: Path, patch_len: int, horizon: int,
                        test_preds: dict):
    model.to(DEV).eval()
    with torch.no_grad():
        train_pred = model(x_train.to(DEV)).cpu().numpy()
        test_pred = model(x_test.to(DEV)).cpu().numpy()

    test_preds[model_name] = test_pred
    save_results(train_pred, y_train_np, model_name, prefix, metrics_file,
                 patch_len, horizon, set_name="Train")
    save_results(test_pred, y_test_np, model_name, prefix, metrics_file,
                 patch_len, horizon, set_name="Test")
    torch.save(model.cpu().state_dict(), f"{prefix}.pt")

    spec = get_koopman_spectrum(model, "koop")
    if spec is not None:
        np.save(f"{prefix}_koopman_spectrum.npy", spec)

    ssm_spec = get_ssm_backbone_spectrum(model)
    if ssm_spec is not None:
        np.save(f"{prefix}_ssm_backbone_spectrum.npy", ssm_spec)


# --------------------------------------------------------------------------- #
# 13) Main loop                                                               #
# --------------------------------------------------------------------------- #
def main(file: str = "df_cleaned_numeric_2.npy",
         epochs: int = 200,
         seq_len: int = 120,
         patch_lens: Optional[Sequence[int]] = None,
         horizons: Optional[Sequence[int]] = None,
         indices: Optional[Sequence[int]] = None,
         save_dir: str = "results",
         train_frac: float = 0.8,
         lyap_weight_constr: float = 0.1,
         lyap_weight_learn: float = 0.1,
         lyap_weight_unconstr: float = 0.1,
         learnable_kinds: Optional[Sequence[str]] = None,
         backbones: Optional[Sequence[str]] = None,
         d_model: int = 96,
         num_heads: int = 4,
         num_layers: int = 3,
         dim_ff: int = 96,
         max_rows: int = 2500,
         run_plain_baselines: bool = True):

    patch_lens = list(patch_lens or [80, 100, 120, 140])
    horizons = list(horizons or [4, 8, 12, 16])
    indices = list(indices or [0, 1, 2, 3, 4, 5])
    learnable_kinds = list(learnable_kinds or ["scalar", "permode", "mlp", "lowrank16"])

    # Important: dlinear and ssm are now included as Koopman backbones.
    backbones = list(backbones or [
        "patchtst",
        "autoformer",
        "informer",
        "itransformer",
        "timesnet",
        "dlinear",
        "ssm",
    ])

    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "Set", "MSE", "MAE"]).to_csv(
        metrics_file, index=False
    )

    raw = np.load(file)

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")

            # Preserves your original experimental design:
            # each patch_len is also the input sliding-window length.
            effective_seq_len = patch_len
            x_all, y_all, _ = build_dataset(raw, indices, effective_seq_len,
                                            horizon, max_rows=max_rows)
            N = x_all.shape[0]
            n_train = int(N * train_frac)
            x_train, y_train = x_all[:n_train], y_all[:n_train]
            x_test, y_test = x_all[n_train:], y_all[n_train:]

            F, H = len(indices), horizon
            y_train_np = y_train.numpy()
            y_test_np = y_test.numpy()

            res_loss = {}
            test_preds = {}

            if run_plain_baselines:
                # ----- LSTM baseline -----
                print("== LSTM baseline ==")
                lstm = SimpleLSTMForecaster(F, hidden_dim=96, num_layers=2, horizon=horizon)
                loss_l, eig_l = train(lstm, x_train, y_train, epochs=epochs,
                                      koop_attr=None, lyap_weight=0.0)
                res_loss["LSTM"] = {"loss": loss_l, "eig": eig_l}
                eval_and_save_model(
                    lstm, "LSTM", save_dir / f"LSTM_{patch_len}_{horizon}",
                    x_train, y_train_np, x_test, y_test_np,
                    metrics_file, patch_len, horizon, test_preds
                )
                
                
                
                

                # ----- Original DLinear baseline -----
                print("== DLinear baseline ==")
                dlinear = DLinearForecaster(F, seq_len=effective_seq_len, horizon=horizon)
                loss_dl, eig_dl = train(dlinear, x_train, y_train, epochs=epochs,
                                        koop_attr=None, lyap_weight=0.0)
                res_loss["DLinear"] = {"loss": loss_dl, "eig": eig_dl}
                eval_and_save_model(
                    dlinear, "DLinear", save_dir / f"DLinear_{patch_len}_{horizon}",
                    x_train, y_train_np, x_test, y_test_np,
                    metrics_file, patch_len, horizon, test_preds
                )

                # ----- Original SSM baseline -----
                print("== SSM baseline ==")
                ssm = SimpleSSMForecaster(F, hidden_dim=96, horizon=horizon)
                loss_ssm, eig_ssm = train(ssm, x_train, y_train, epochs=epochs,
                                          koop_attr=None, lyap_weight=0.0)
                res_loss["SSM"] = {"loss": loss_ssm, "eig": eig_ssm}
                prefix_ssm = save_dir / f"SSM_{patch_len}_{horizon}"
                eval_and_save_model(
                    ssm, "SSM", prefix_ssm,
                    x_train, y_train_np, x_test, y_test_np,
                    metrics_file, patch_len, horizon, test_preds
                )
                with torch.no_grad():
                    A_cpu = ssm.A.detach().cpu()
                    svals_ssm = torch.linalg.svdvals(A_cpu).numpy()
                np.save(f"{prefix_ssm}_spectrum.npy", svals_ssm)
                
                
                

            # ----- Koopformer families over all requested backbones -----
            for backbone in backbones:
                pretty_backbone = {
                    "patchtst": "PatchTST",
                    "autoformer": "Autoformer",
                    "informer": "Informer",
                    "itransformer": "iTransformer",
                    "timesnet": "TimesNet",
                    "dlinear": "DLinear",
                    "ssm": "SSM",
                    "gatedssm": "GatedSSM",
                }.get(backbone.lower(), backbone)

                # Constrained
                print(f"== {pretty_backbone} – Koopman constrained (rho(K) < rho_max proxy) ==")
                m_c = KoopformerGeneric(
                    F, effective_seq_len, horizon,
                    patch_len=patch_len,
                    backbone_type=backbone,
                    koopman_type="constr",
                    d_model=d_model,
                    rho_max=0.99,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dim_ff=dim_ff,
                )
                loss_c, eig_c = train(m_c, x_train, y_train, epochs=epochs,
                                      koop_attr="koop", lyap_weight=lyap_weight_constr)
                name_c = f"Koop-{pretty_backbone} (constr.)"
                res_loss[name_c] = {"loss": loss_c, "eig": eig_c}
                prefix_c = save_dir / f"{pretty_backbone}_koop_constr_{patch_len}_{horizon}"
                eval_and_save_model(
                    m_c, name_c, prefix_c,
                    x_train, y_train_np, x_test, y_test_np,
                    metrics_file, patch_len, horizon, test_preds
                )
                
                
                
                # ----- Sensitivity analysis (AFTER eval_and_save_model) -----
                sens_df = evaluate_koopman_sensitivity(
                    model=m_c,
                    x=x_test,
                    y=y_test,
                    eps_list=[0, 1e-4, 1e-3, 1e-2, 1e-1],
                    n_trials=10,
                )
                
                sens_path = save_dir / f"{pretty_backbone}_koop_constr_{patch_len}_{horizon}_sens"
                sens_df.to_csv(f"{sens_path}.csv", index=False)
                
                plot_koopman_sensitivity(sens_df, sens_path)
                

                # Unconstrained
                print(f"== {pretty_backbone} – Koopman unconstrained ==")
                m_u = KoopformerGeneric(
                    F, effective_seq_len, horizon,
                    patch_len=patch_len,
                    backbone_type=backbone,
                    koopman_type="unconstr",
                    d_model=d_model,
                    rho_max=0.99,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dim_ff=dim_ff,
                )
                loss_u, eig_u = train(m_u, x_train, y_train, epochs=epochs,
                                      koop_attr="koop", lyap_weight=lyap_weight_unconstr)
                name_u = f"Koop-{pretty_backbone} (unconstr.)"
                res_loss[name_u] = {"loss": loss_u, "eig": eig_u}
                prefix_u = save_dir / f"{pretty_backbone}_koop_unconstr_{patch_len}_{horizon}"
                eval_and_save_model(
                    m_u, name_u, prefix_u,
                    x_train, y_train_np, x_test, y_test_np,
                    metrics_file, patch_len, horizon, test_preds
                )
                
                
                # ----- Sensitivity analysis (AFTER eval_and_save_model) -----
                
                
                sens_df = evaluate_koopman_sensitivity(
                    model=m_u,
                    x=x_test,
                    y=y_test,
                    eps_list=[0, 1e-4, 1e-3, 1e-2, 1e-1],
                    n_trials=10,
                )
                
                sens_path = save_dir / f"{pretty_backbone}_koop_unconstr_{patch_len}_{horizon}_sens"
                sens_df.to_csv(f"{sens_path}.csv", index=False)
                
                plot_koopman_sensitivity(sens_df, sens_path)
                                
                
                

                # Learnable family
                for kind in learnable_kinds:
                    print(f"== {pretty_backbone} – learnable Koopman ({kind}) ==")
                    m_l = KoopformerGeneric(
                        F, effective_seq_len, horizon,
                        patch_len=patch_len,
                        backbone_type=backbone,
                        koopman_type="learn",
                        d_model=d_model,
                        rho_max=0.99,
                        koop_kind=kind,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dim_ff=dim_ff,
                    )
                    loss_lrn, eig_lrn = train(m_l, x_train, y_train, epochs=epochs,
                                              koop_attr="koop", lyap_weight=lyap_weight_learn)
                    name_l = f"Koop-{pretty_backbone} (learn:{kind})"
                    res_loss[name_l] = {"loss": loss_lrn, "eig": eig_lrn}
                    prefix_l = save_dir / f"{pretty_backbone}_koop_learn-{kind}_{patch_len}_{horizon}"
                    eval_and_save_model(
                        m_l, name_l, prefix_l,
                        x_train, y_train_np, x_test, y_test_np,
                        metrics_file, patch_len, horizon, test_preds
                    )
                    
                    
                    # ----- Sensitivity analysis (AFTER eval_and_save_model) -----
                    sens_df = evaluate_koopman_sensitivity(
                        model=m_l,
                        x=x_test,
                        y=y_test,
                        eps_list=[0, 1e-4, 1e-3, 1e-2, 1e-1],
                        n_trials=10,
                    )
                    
                    sens_path = save_dir / f"{pretty_backbone}_koop_learn-{kind}_{patch_len}_{horizon}_sens"
                    sens_df.to_csv(f"{sens_path}.csv", index=False)
                    
                    plot_koopman_sensitivity(sens_df, sens_path)
                    

            # ----- Figures -----
            plot_training(res_loss, save_dir / f"train_patch{patch_len}_h{horizon}")
            plot_per_feature(y_test_np, test_preds, F, H,
                             save_dir / f"feat_patch{patch_len}_h{horizon}")

            # Console summary: TEST set
            print("\nFinal TEST metrics:")
            for n, p in test_preds.items():
                e = y_test_np - p
                print(f"{n:<45s} MSE {(e ** 2).mean():.4e} "
                      f"MAE {np.abs(e).mean():.4e}")


# --------------------------------------------------------------------------- #
# 14) CLI                                                                     #
# --------------------------------------------------------------------------- #
def _parse_int_list(s: Optional[str]):
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: Optional[str]):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Koopformer-PRO benchmark: PatchTST / Autoformer / Informer / "
            "iTransformer / TimesNet / DLinear / SSM backbones with constrained, "
            "learnable, and unconstrained Koopman variants plus LSTM, DLinear, "
            "SSM plain baselines."
        )
    )
    p.add_argument("--file", type=str, default="./wind_speeds_2020.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120,
                   help="Kept for compatibility. Original loop uses patch_len as effective seq_len.")
    p.add_argument("--save_dir", type=str,
                   default="results/results_dkf_lean_var_ful_plus_wind")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--patch_lens", type=str, default="80,100,120,140",
                   help="Comma-separated patch/input lengths, e.g. 80,100,120,140")
    p.add_argument("--horizons", type=str, default="4,8,12,16",
                   help="Comma-separated horizons, e.g. 4,8,12,16")
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5",
                   help="Comma-separated feature indices")
    p.add_argument("--backbones", type=str,
                   default="patchtst,autoformer,informer,itransformer,timesnet,dlinear,ssm",
                   help=(
                       "Comma-separated backbones: patchtst,autoformer,informer,"
                       "itransformer,timesnet,dlinear,ssm,gatedssm"
                   ))
    p.add_argument("--learnable_kinds", type=str,
                   default="scalar,permode,mlp,lowrank16",
                   help="Comma-separated learnable Koopman variants")
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dim_ff", type=int, default=96)
    p.add_argument("--max_rows", type=int, default=2500)
    p.add_argument("--lyap_weight_constr", type=float, default=0.1,
                   help="Lyapunov weight for constrained Koopman variants")
    p.add_argument("--lyap_weight_learn", type=float, default=0.1,
                   help="Lyapunov weight for learnable Koopman variants")
    p.add_argument("--lyap_weight_unconstr", type=float, default=0.1,
                   help="Lyapunov weight for unconstrained Koopman variants")
    p.add_argument("--run_plain_baselines", type=str, default="true",
                   help="true/false. If false, only Koopman-backbone models are trained.")

    args = p.parse_args()

    main(
        file=args.file,
        epochs=args.epochs,
        seq_len=args.seq_len,
        patch_lens=_parse_int_list(args.patch_lens),
        horizons=_parse_int_list(args.horizons),
        indices=_parse_int_list(args.indices),
        save_dir=args.save_dir,
        train_frac=args.train_frac,
        lyap_weight_constr=args.lyap_weight_constr,
        lyap_weight_learn=args.lyap_weight_learn,
        lyap_weight_unconstr=args.lyap_weight_unconstr,
        learnable_kinds=_parse_str_list(args.learnable_kinds),
        backbones=_parse_str_list(args.backbones),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        max_rows=args.max_rows,
        run_plain_baselines=_parse_bool(args.run_plain_baselines),
    )
