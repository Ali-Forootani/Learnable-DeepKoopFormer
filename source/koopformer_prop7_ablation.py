#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer Proposition-7 Operational Ablation
=============================================

Purpose
-------
This script is designed to answer the Action Editor / reviewer concern:

    Does the theoretical stability mechanism, e.g. Proposition 7,
    actually play an operational role during training and forecasting?

It compares constrained, learnable, and unconstrained Koopman variants with
Lyapunov penalty ON/OFF, and logs:

    - Train/Test MSE and MAE
    - Koopman spectral proxy / spectral radius
    - Raw Lyapunov residual statistics
    - Lyapunov penalty activation frequency
    - Error growth over forecast horizon
    - Training curves and diagnostic plots

Main experiment
---------------
For each backbone, patch length, horizon, and seed, the script trains:

    1. constrained Koopman + Lyapunov penalty ON
    2. constrained Koopman + Lyapunov penalty OFF
    3. learnable Koopman + Lyapunov penalty ON
    4. learnable Koopman + Lyapunov penalty OFF
    5. unconstrained Koopman + Lyapunov penalty ON
    6. unconstrained Koopman + Lyapunov penalty OFF

Default backbone is DLinear because it is fast and clear for ablation.
You can also use ssm, patchtst, informer, autoformer, itransformer, timesnet.

Example
-------
python koopformer_prop7_ablation.py \
    --file ./wind_speeds_2020.npy \
    --save_dir results_prop7 \
    --backbones dlinear,ssm \
    --patch_lens 80 \
    --horizons 4,8,16 \
    --indices 0,1,2,3,4,5 \
    --epochs 1000 \
    --seeds 7,42

Author
------
Ali Forootani / generated revision helper
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Sequence, Dict, List, Tuple

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
# 0) Reproducibility and device                                                #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 1) Koopman operators                                                         #
# --------------------------------------------------------------------------- #
def _orth(w: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(w)
    return q


class StrictStableKoopmanOperator(nn.Module):
    """
    ODO-style stable Koopman parameterization.

    K = U diag(sigma) V^T,
    where U,V are orthonormal and sigma_i in (0, rho_max).

    This bounds singular values and therefore bounds spectral radius.
    The logged value is a singular-value stability proxy.
    """
    def __init__(self, latent_dim: int, rho_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.rho_max = rho_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.S_raw) * self.rho_max

    def matrix(self) -> torch.Tensor:
        U = _orth(self.U_raw)
        V = _orth(self.V_raw)
        sigma = self._sigma()
        return U @ torch.diag(sigma) @ V.T

    def forward(self, z: torch.Tensor):
        K = self.matrix()
        z_next = z @ K.T
        return z_next, K, self._sigma()


class LearnableKoopmanBase(nn.Module):
    def _sigma(self) -> torch.Tensor:
        raise NotImplementedError

    def matrix(self) -> torch.Tensor:
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

    def matrix(self) -> torch.Tensor:
        U = _orth(self.U_raw)
        V = _orth(self.V_raw)
        return U @ torch.diag(self._sigma()) @ V.T

    def forward(self, z: torch.Tensor):
        K = self.matrix()
        return z @ K.T, K, self._sigma()


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

    def matrix(self) -> torch.Tensor:
        U = _orth(self.U_raw)
        V = _orth(self.V_raw)
        return U @ torch.diag(self._sigma()) @ V.T

    def forward(self, z: torch.Tensor):
        K = self.matrix()
        return z @ K.T, K, self._sigma()


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
        g = self.mlp(s).view(-1)
        return torch.sigmoid(g) * self.rho_max

    def matrix(self) -> torch.Tensor:
        U = _orth(self.U_raw)
        V = _orth(self.V_raw)
        return U @ torch.diag(self._sigma()) @ V.T

    def forward(self, z: torch.Tensor):
        K = self.matrix()
        return z @ K.T, K, self._sigma()


class LearnableKoopmanOperatorLowRank(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, rank: int, rho_max: float = 0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = max(1, min(rank, latent_dim))
        self.U_raw = nn.Parameter(torch.randn(latent_dim, self.rank))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, self.rank))
        self.S_raw = nn.Parameter(torch.randn(self.rank))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.rho_max = rho_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.rho_max

    def matrix(self) -> torch.Tensor:
        U, _ = torch.linalg.qr(self.U_raw, mode="reduced")
        V, _ = torch.linalg.qr(self.V_raw, mode="reduced")
        return U @ torch.diag(self._sigma()) @ V.T

    def forward(self, z: torch.Tensor):
        K = self.matrix()
        return z @ K.T, K, self._sigma()


class UnconstrainedKoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.K_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

    def matrix(self) -> torch.Tensor:
        return self.K_raw

    def forward(self, z: torch.Tensor):
        K = self.K_raw
        return z @ K.T, K, None


def make_learnable_koopman(kind: str, latent_dim: int, rho_max: float = 0.99):
    key = kind.lower()
    if key == "scalar":
        return LearnableKoopmanOperatorScalar(latent_dim, rho_max)
    if key == "permode":
        return LearnableKoopmanOperatorPerMode(latent_dim, rho_max)
    if key == "mlp":
        return LearnableKoopmanOperatorMLP(latent_dim, rho_max)
    if key.startswith("lowrank"):
        suffix = key.replace("lowrank", "")
        rank = latent_dim // 2
        if suffix.strip():
            try:
                rank = int(suffix)
            except ValueError:
                rank = latent_dim // 2
        return LearnableKoopmanOperatorLowRank(latent_dim, rank, rho_max)
    raise ValueError(f"Unknown learnable Koopman kind: {kind}")


# --------------------------------------------------------------------------- #
# 2) Backbone modules                                                          #
# --------------------------------------------------------------------------- #
class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)


class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.patch = PatchEmbed1D(input_dim, d_model, patch_len, patch_len)
        n_patches = max(1, int(np.floor((seq_len - patch_len) / patch_len) + 1))
        self.pos = SinCosPosEnc(d_model, max_len=n_patches + 2)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
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
        trend = trend[:, :x.size(1)]
        seas = x - trend
        return trend, seas


class SimpleAutoformerBackbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        k = max(1, min(patch_len, seq_len))
        self.dec = SeriesDecomp(k=k)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = SinCosPosEnc(d_model, max_len=seq_len + 1)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.trend_fc = nn.Linear(seq_len * input_dim, d_model)
        self.mix = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor):
        trend, seas = self.dec(x)
        seas_z = self.encoder(self.pos(self.embed(seas))).mean(1)
        trend_z = self.trend_fc(trend.reshape(trend.size(0), -1))
        return self.mix(torch.cat([seas_z, trend_z], dim=-1))


class InformerSparse_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.use_patch = patch_len > 1
        if self.use_patch:
            self.patch = PatchEmbed1D(input_dim, d_model, patch_len, patch_len)
            n_tokens = max(1, int(np.floor((seq_len - patch_len) / patch_len) + 1))
        else:
            self.embed = nn.Linear(input_dim, d_model)
            n_tokens = seq_len
        self.pos = SinCosPosEnc(d_model, max_len=n_tokens + 1)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_patch:
            z = self.patch(x)
        else:
            z = self.embed(x)
        z = self.encoder(self.pos(z))
        return z.mean(1)


class ITransformer_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, d_model)
        self.pos = SinCosPosEnc(d_model, max_len=seq_len + 1)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.value_proj(x)
        z = self.encoder(self.pos(z))
        return self.pool(z.transpose(1, 2)).squeeze(-1)


class TimesBlockSimple(nn.Module):
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
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TimesBlockSimple(d_model) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x).transpose(1, 2)
        for blk in self.blocks:
            z = blk(z)
        return self.pool(z).squeeze(-1)


class DLinear_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.seq_len = seq_len
        self.temporal = nn.Linear(seq_len, d_model)
        self.feature_mixer = nn.Sequential(
            nn.Linear(input_dim * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")
        z = self.temporal(x.permute(0, 2, 1))
        return self.feature_mixer(z.reshape(B, -1))


class SSM_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
        super().__init__()
        self.hidden_dim = d_model
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, T, _ = x.shape
        h = torch.zeros(Bsz, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(T):
            h = self.act(h @ self.A.T + x[:, t, :] @ self.B.T + self.bias)
        return self.norm(h)


class GatedSSM_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, patch_len: int,
                 d_model: int = 64, num_layers: int = 1,
                 num_heads: int = 4, dim_ff: int = 96):
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


def make_backbone(backbone_type: str, input_dim: int, seq_len: int,
                  patch_len: int, d_model: int, num_heads: int,
                  num_layers: int, dim_ff: int):
    key = backbone_type.lower()
    if key == "patchtst":
        return PatchTST_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "autoformer":
        return SimpleAutoformerBackbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "informer":
        return InformerSparse_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "itransformer":
        return ITransformer_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "timesnet":
        return TimesNet_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "dlinear":
        return DLinear_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "ssm":
        return SSM_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    if key == "gatedssm":
        return GatedSSM_Backbone(input_dim, seq_len, patch_len, d_model, num_layers, num_heads, dim_ff), d_model
    raise ValueError(f"Unknown backbone: {backbone_type}")


# --------------------------------------------------------------------------- #
# 3) Generic Koopformer                                                        #
# --------------------------------------------------------------------------- #
class KoopformerGeneric(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, horizon: int,
                 patch_len: int, backbone_type: str, koopman_type: str,
                 d_model: int = 96, rho_max: float = 0.99,
                 koop_kind: str = "scalar", num_heads: int = 4,
                 num_layers: int = 1, dim_ff: int = 96):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.backbone_type = backbone_type
        self.koopman_type = koopman_type
        self.koop_kind = koop_kind

        self.backbone, latent_dim = make_backbone(
            backbone_type=backbone_type,
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
        )
        self.latent_dim = latent_dim

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
        z_next, K, sigma = self.koop(z)
        pred = self.fc(z_next)
        if return_latents:
            return pred, z, z_next, K, sigma
        return pred


# --------------------------------------------------------------------------- #
# 4) Dataset utilities                                                         #
# --------------------------------------------------------------------------- #
def build_dataset(data: np.ndarray, indices: Sequence[int],
                  seq_len: int, horizon: int, max_rows: int = 2500):
    sub = data[:max_rows, indices]
    scaler = MinMaxScaler().fit(sub)
    sub = scaler.transform(sub)

    X, Y = [], []
    for i in range(len(sub) - seq_len - horizon):
        X.append(sub[i:i + seq_len])
        y = sub[i + seq_len:i + seq_len + horizon].T.reshape(-1)
        Y.append(y)

    if not X:
        raise ValueError(
            f"Not enough rows for seq_len={seq_len}, horizon={horizon}, max_rows={max_rows}."
        )

    x_in = torch.tensor(np.stack(X), dtype=torch.float32)
    x_out = torch.tensor(np.stack(Y), dtype=torch.float32)
    return x_in, x_out, scaler


def split_train_test(x_all: torch.Tensor, y_all: torch.Tensor, train_frac: float):
    n_train = int(x_all.shape[0] * train_frac)
    return x_all[:n_train], y_all[:n_train], x_all[n_train:], y_all[n_train:]


# --------------------------------------------------------------------------- #
# 5) Diagnostics                                                               #
# --------------------------------------------------------------------------- #
def koopman_spectral_value(model: KoopformerGeneric) -> Tuple[float, str]:
    """Return spectral proxy/radius and its name."""
    koop = model.koop
    with torch.no_grad():
        if isinstance(koop, (StrictStableKoopmanOperator, LearnableKoopmanBase)):
            sigma = koop._sigma()
            return float(sigma.abs().max().detach().cpu().item()), "sigma_proxy_max"
        if isinstance(koop, UnconstrainedKoopmanOperator):
            eig = torch.linalg.eigvals(koop.K_raw)
            return float(eig.abs().max().detach().cpu().item()), "spectral_radius"
    return float("nan"), "unknown"


def koopman_singular_max(model: KoopformerGeneric) -> float:
    with torch.no_grad():
        K = model.koop.matrix()
        return float(torch.linalg.svdvals(K).max().detach().cpu().item())


def lyapunov_residual(z: torch.Tensor, z_next: torch.Tensor, P: torch.Tensor):
    """
    Raw residual: V(z_next) - V(z), with V(z)=z^T P z.

    Proposition-style contraction evidence expects this residual to be mostly <= 0.
    Positive residual means the Lyapunov penalty is active.
    """
    zp = torch.einsum("bi,ij->bj", z, P)
    znp = torch.einsum("bi,ij->bj", z_next, P)
    return (znp * z_next).sum(dim=1) - (zp * z).sum(dim=1)


def horizon_error_profile(pred: np.ndarray, target: np.ndarray, n_features: int, horizon: int):
    """MSE/MAE per forecast step, averaged over samples and features."""
    p = pred.reshape(pred.shape[0], n_features, horizon)
    y = target.reshape(target.shape[0], n_features, horizon)
    err = y - p
    mse_h = (err ** 2).mean(axis=(0, 1))
    mae_h = np.abs(err).mean(axis=(0, 1))
    return mse_h, mae_h


@torch.no_grad()
def evaluate_model(model: KoopformerGeneric, x: torch.Tensor, y: torch.Tensor,
                   device: torch.device, batch_size: int = 4096):
    model.eval().to(device)
    preds = []
    zs, zns = [], []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device)
        pred, z, z_next, _, _ = model(xb, return_latents=True)
        preds.append(pred.cpu())
        zs.append(z.cpu())
        zns.append(z_next.cpu())
    pred = torch.cat(preds, dim=0)
    z = torch.cat(zs, dim=0)
    z_next = torch.cat(zns, dim=0)
    err = y - pred
    mse = float((err ** 2).mean().item())
    mae = float(err.abs().mean().item())
    return pred.numpy(), mse, mae, z, z_next


# --------------------------------------------------------------------------- #
# 6) Training with operational Lyapunov logging                                #
# --------------------------------------------------------------------------- #
def train_model(model: KoopformerGeneric,
                x_train: torch.Tensor,
                y_train: torch.Tensor,
                device: torch.device,
                epochs: int = 1000,
                lr: float = 3e-4,
                lyap_weight: float = 0.1,
                batch_size: int = 0,
                print_every: Optional[int] = None):
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        pred0, z0, zn0, _, _ = model(x_train[:1], return_latents=True)
    P = torch.eye(z0.shape[1], device=device)

    logs = {
        "epoch": [],
        "loss_total": [],
        "loss_pred": [],
        "loss_lyap_weighted": [],
        "lyap_raw_mean": [],
        "lyap_raw_max": [],
        "lyap_raw_min": [],
        "lyap_active_rate": [],
        "spectral_value": [],
        "singular_max": [],
    }

    if print_every is None:
        print_every = max(1, epochs // 10)

    n = x_train.shape[0]
    use_minibatch = batch_size is not None and batch_size > 0 and batch_size < n

    for ep in range(epochs):
        model.train()
        if use_minibatch:
            perm = torch.randperm(n, device=device)
            xb_all = x_train[perm]
            yb_all = y_train[perm]
        else:
            xb_all = x_train
            yb_all = y_train

        epoch_total = []
        epoch_pred = []
        epoch_lyap = []
        epoch_raw_mean = []
        epoch_raw_max = []
        epoch_raw_min = []
        epoch_active = []

        step = batch_size if use_minibatch else n
        for start in range(0, n, step):
            xb = xb_all[start:start + step]
            yb = yb_all[start:start + step]

            opt.zero_grad()
            pred, z, z_next, _, _ = model(xb, return_latents=True)
            pred_loss = mse_loss(pred, yb)

            raw_resid = lyapunov_residual(z, z_next, P)
            lyap_penalty = torch.relu(raw_resid).mean()
            total_loss = pred_loss + lyap_weight * lyap_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_total.append(float(total_loss.detach().cpu().item()))
            epoch_pred.append(float(pred_loss.detach().cpu().item()))
            epoch_lyap.append(float((lyap_weight * lyap_penalty).detach().cpu().item()))
            epoch_raw_mean.append(float(raw_resid.mean().detach().cpu().item()))
            epoch_raw_max.append(float(raw_resid.max().detach().cpu().item()))
            epoch_raw_min.append(float(raw_resid.min().detach().cpu().item()))
            epoch_active.append(float((raw_resid > 0).float().mean().detach().cpu().item()))

        spec_val, _ = koopman_spectral_value(model)
        smax = koopman_singular_max(model)

        logs["epoch"].append(ep)
        logs["loss_total"].append(float(np.mean(epoch_total)))
        logs["loss_pred"].append(float(np.mean(epoch_pred)))
        logs["loss_lyap_weighted"].append(float(np.mean(epoch_lyap)))
        logs["lyap_raw_mean"].append(float(np.mean(epoch_raw_mean)))
        logs["lyap_raw_max"].append(float(np.max(epoch_raw_max)))
        logs["lyap_raw_min"].append(float(np.min(epoch_raw_min)))
        logs["lyap_active_rate"].append(float(np.mean(epoch_active)))
        logs["spectral_value"].append(spec_val)
        logs["singular_max"].append(smax)

        if ep % print_every == 0 or ep == epochs - 1:
            print(
                f"Epoch {ep:5d}/{epochs} | "
                f"MSE={logs['loss_pred'][-1]:.6e} | "
                f"LyapW={logs['loss_lyap_weighted'][-1]:.6e} | "
                f"ResidMean={logs['lyap_raw_mean'][-1]:+.3e} | "
                f"Active={100.0 * logs['lyap_active_rate'][-1]:5.1f}% | "
                f"Spec={logs['spectral_value'][-1]:.4f}"
            )

    return pd.DataFrame(logs)


# --------------------------------------------------------------------------- #
# 7) Plotting                                                                  #
# --------------------------------------------------------------------------- #
def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(logs_by_name: Dict[str, pd.DataFrame], out_base: Path):
    if not logs_by_name:
        return

    metrics = [
        ("loss_pred", "Prediction MSE"),
        ("lyap_active_rate", "Lyapunov activation rate"),
        ("lyap_raw_mean", "Mean raw Lyapunov residual"),
        ("spectral_value", "Koopman spectral proxy/radius"),
        ("singular_max", "Max singular value of K"),
    ]

    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, df in logs_by_name.items():
            ax.plot(df["epoch"], df[key], label=name, lw=1.8)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        if key == "loss_pred":
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save_fig(fig, out_base.parent / f"{out_base.name}_{key}")


def plot_horizon_profiles(profile_df: pd.DataFrame, out_base: Path):
    if profile_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, sub in profile_df.groupby("Model"):
        ax.plot(sub["Step"], sub["MSE"], marker="o", label=name, lw=1.8)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("MSE")
    ax.set_title("Error growth across forecast horizon")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_base)


def plot_prediction_example(y_true: np.ndarray, preds: Dict[str, np.ndarray],
                            n_features: int, horizon: int, out_base: Path,
                            max_points: int = 250):
    if not preds:
        return
    n = min(max_points, y_true.shape[0])
    gt = y_true[:n].reshape(n, n_features, horizon)[:, :, 0]

    for f in range(n_features):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gt[:, f], label="Ground truth", lw=2.5)
        for name, pred in preds.items():
            pp = pred[:n].reshape(n, n_features, horizon)[:, :, 0]
            ax.plot(pp[:, f], label=name, lw=1.5, alpha=0.85)
        ax.set_title(f"One-step prediction trace, feature {f}")
        ax.set_xlabel("Test sample")
        ax.set_ylabel("Scaled value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save_fig(fig, out_base.parent / f"{out_base.name}_feature{f}")


# --------------------------------------------------------------------------- #
# 8) Main experiment                                                           #
# --------------------------------------------------------------------------- #
def run_single_experiment(raw: np.ndarray,
                          save_dir: Path,
                          seed: int,
                          backbone: str,
                          patch_len: int,
                          horizon: int,
                          indices: Sequence[int],
                          train_frac: float,
                          max_rows: int,
                          epochs: int,
                          lr: float,
                          d_model: int,
                          num_heads: int,
                          num_layers: int,
                          dim_ff: int,
                          rho_max: float,
                          learnable_kind: str,
                          lyap_weight: float,
                          batch_size: int,
                          device: torch.device):
    set_seed(seed)
    effective_seq_len = patch_len
    n_features = len(indices)

    x_all, y_all, _ = build_dataset(raw, indices, effective_seq_len, horizon, max_rows=max_rows)
    x_train, y_train, x_test, y_test = split_train_test(x_all, y_all, train_frac)

    configs = [
        ("constr", "none", lyap_weight, "Constrained + Lyap"),
        ("constr", "none", 0.0, "Constrained no Lyap"),
        ("learn", learnable_kind, lyap_weight, f"Learnable-{learnable_kind} + Lyap"),
        ("learn", learnable_kind, 0.0, f"Learnable-{learnable_kind} no Lyap"),
        ("unconstr", "none", lyap_weight, "Unconstrained + Lyap"),
        ("unconstr", "none", 0.0, "Unconstrained no Lyap"),
    ]

    exp_tag = f"seed{seed}_{backbone}_P{patch_len}_H{horizon}"
    exp_dir = save_dir / exp_tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    horizon_rows = []
    logs_by_name = {}
    preds_for_plot = {}

    for koop_type, kind, lw, display_name in configs:
        print("\n" + "=" * 80)
        print(f"Experiment: {exp_tag} | {display_name} | lyap_weight={lw}")
        print("=" * 80)

        model = KoopformerGeneric(
            input_dim=n_features,
            seq_len=effective_seq_len,
            horizon=horizon,
            patch_len=patch_len,
            backbone_type=backbone,
            koopman_type=koop_type,
            d_model=d_model,
            rho_max=rho_max,
            koop_kind=kind,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
        )

        log_df = train_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            device=device,
            epochs=epochs,
            lr=lr,
            lyap_weight=lw,
            batch_size=batch_size,
        )

        safe_name = display_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
        log_df.to_csv(exp_dir / f"trainlog_{safe_name}.csv", index=False)
        logs_by_name[display_name] = log_df

        train_pred, train_mse, train_mae, train_z, train_zn = evaluate_model(model, x_train, y_train, device)
        test_pred, test_mse, test_mae, test_z, test_zn = evaluate_model(model, x_test, y_test, device)
        preds_for_plot[display_name] = test_pred

        P_cpu = torch.eye(train_z.shape[1])
        train_resid = lyapunov_residual(train_z, train_zn, P_cpu).numpy()
        test_resid = lyapunov_residual(test_z, test_zn, P_cpu).numpy()

        spec_val, spec_name = koopman_spectral_value(model)
        smax = koopman_singular_max(model)

        # Save artifacts
        np.save(exp_dir / f"{safe_name}_test_predictions.npy", test_pred)
        np.save(exp_dir / f"{safe_name}_test_errors.npy", y_test.numpy() - test_pred)
        torch.save(model.cpu().state_dict(), exp_dir / f"{safe_name}.pt")

        mse_h, mae_h = horizon_error_profile(test_pred, y_test.numpy(), n_features, horizon)
        for step in range(horizon):
            horizon_rows.append({
                "Seed": seed,
                "Backbone": backbone,
                "PatchLen": patch_len,
                "Horizon": horizon,
                "Model": display_name,
                "Step": step + 1,
                "MSE": float(mse_h[step]),
                "MAE": float(mae_h[step]),
            })

        summary_rows.append({
            "Seed": seed,
            "Backbone": backbone,
            "PatchLen": patch_len,
            "Horizon": horizon,
            "Model": display_name,
            "KoopmanType": koop_type,
            "LearnableKind": kind,
            "LyapWeight": lw,
            "TrainMSE": train_mse,
            "TrainMAE": train_mae,
            "TestMSE": test_mse,
            "TestMAE": test_mae,
            "SpectralMetric": spec_name,
            "SpectralValue": spec_val,
            "SingularMax": smax,
            "TrainLyapResidualMean": float(train_resid.mean()),
            "TrainLyapResidualMax": float(train_resid.max()),
            "TrainLyapActiveRate": float((train_resid > 0).mean()),
            "TestLyapResidualMean": float(test_resid.mean()),
            "TestLyapResidualMax": float(test_resid.max()),
            "TestLyapActiveRate": float((test_resid > 0).mean()),
            "FinalTrainLoss": float(log_df["loss_total"].iloc[-1]),
            "FinalPredLoss": float(log_df["loss_pred"].iloc[-1]),
            "FinalWeightedLyapLoss": float(log_df["loss_lyap_weighted"].iloc[-1]),
            "FinalLoggedActiveRate": float(log_df["lyap_active_rate"].iloc[-1]),
        })

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    summary_df.to_csv(exp_dir / "summary_metrics.csv", index=False)
    horizon_df.to_csv(exp_dir / "horizon_error_profile.csv", index=False)

    plot_diagnostics(logs_by_name, exp_dir / "diagnostics")
    plot_horizon_profiles(horizon_df, exp_dir / "horizon_error_growth")
    plot_prediction_example(y_test.numpy(), preds_for_plot, n_features, horizon, exp_dir / "prediction_trace")

    print("\nFinal summary:")
    print(summary_df[[
        "Model", "TestMSE", "TestMAE", "SpectralValue", "SingularMax",
        "TestLyapResidualMean", "TestLyapActiveRate"
    ]].to_string(index=False))

    return summary_df, horizon_df


def aggregate_results(save_dir: Path, all_summary: List[pd.DataFrame], all_horizon: List[pd.DataFrame]):
    if all_summary:
        df = pd.concat(all_summary, ignore_index=True)
        df.to_csv(save_dir / "ALL_summary_metrics.csv", index=False)

        group_cols = ["Backbone", "PatchLen", "Horizon", "Model"]
        metric_cols = [
            "TestMSE", "TestMAE", "SpectralValue", "SingularMax",
            "TestLyapResidualMean", "TestLyapResidualMax", "TestLyapActiveRate"
        ]
        agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
        agg.to_csv(save_dir / "ALL_summary_mean_std.csv", index=False)

    if all_horizon:
        hdf = pd.concat(all_horizon, ignore_index=True)
        hdf.to_csv(save_dir / "ALL_horizon_error_profile.csv", index=False)

        agg_h = hdf.groupby(["Backbone", "PatchLen", "Horizon", "Model", "Step"])[["MSE", "MAE"]].agg(["mean", "std"]).reset_index()
        agg_h.columns = ["_".join(c).strip("_") for c in agg_h.columns.to_flat_index()]
        agg_h.to_csv(save_dir / "ALL_horizon_error_profile_mean_std.csv", index=False)


def main(args):
    device = get_device(args.force_cpu)
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.file)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D numpy array [T,F], got shape {raw.shape}")

    seeds = parse_int_list(args.seeds)
    backbones = parse_str_list(args.backbones)
    patch_lens = parse_int_list(args.patch_lens)
    horizons = parse_int_list(args.horizons)
    indices = parse_int_list(args.indices)

    all_summary = []
    all_horizon = []

    for seed in seeds:
        for backbone in backbones:
            for patch_len in patch_lens:
                for horizon in horizons:
                    summary_df, horizon_df = run_single_experiment(
                        raw=raw,
                        save_dir=save_dir,
                        seed=seed,
                        backbone=backbone,
                        patch_len=patch_len,
                        horizon=horizon,
                        indices=indices,
                        train_frac=args.train_frac,
                        max_rows=args.max_rows,
                        epochs=args.epochs,
                        lr=args.lr,
                        d_model=args.d_model,
                        num_heads=args.num_heads,
                        num_layers=args.num_layers,
                        dim_ff=args.dim_ff,
                        rho_max=args.rho_max,
                        learnable_kind=args.learnable_kind,
                        lyap_weight=args.lyap_weight,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    all_summary.append(summary_df)
                    all_horizon.append(horizon_df)
                    aggregate_results(save_dir, all_summary, all_horizon)

    print("\nSaved aggregate files:")
    print(f"  {save_dir / 'ALL_summary_metrics.csv'}")
    print(f"  {save_dir / 'ALL_summary_mean_std.csv'}")
    print(f"  {save_dir / 'ALL_horizon_error_profile.csv'}")
    print(f"  {save_dir / 'ALL_horizon_error_profile_mean_std.csv'}")


# --------------------------------------------------------------------------- #
# 9) CLI                                                                       #
# --------------------------------------------------------------------------- #
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Operational ablation for Proposition-7-style Koopman/Lyapunov stability evidence."
    )
    p.add_argument("--file", type=str, default="./wind_speeds_2020.npy",
                   help="Path to .npy data array with shape [time, features].")
    p.add_argument("--save_dir", type=str, default="results_prop7_ablation")
    p.add_argument("--seeds", type=str, default="7,42")
    p.add_argument("--backbones", type=str, default="dlinear",
                   help="Comma-separated: dlinear,ssm,gatedssm,patchtst,autoformer,informer,itransformer,timesnet")
    p.add_argument("--patch_lens", type=str, default="80")
    p.add_argument("--horizons", type=str, default="4,8,16")
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--max_rows", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 means full-batch training. Use e.g. 256 for minibatches.")
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1,
                   help="Use 1 for fast ablation; increase to 3 for stronger models.")
    p.add_argument("--dim_ff", type=int, default=96)
    p.add_argument("--rho_max", type=float, default=0.99)
    p.add_argument("--learnable_kind", type=str, default="scalar",
                   help="scalar, permode, mlp, lowrank16, etc.")
    p.add_argument("--lyap_weight", type=float, default=0.1)
    p.add_argument("--force_cpu", action="store_true")

    main(p.parse_args())
