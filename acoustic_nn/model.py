"""几何参数到吸声谱、特征值间隔的代理网络。

网络把 32 维归一化几何编码成一个向量，再和频率的傅里叶特征拼在一起，
对每个频率输出三项：前向吸声、后向吸声、特征值间隔。
吸声用 sigmoid 限制在 (0, 1)，特征值间隔用 softplus 保持非负。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SpectralSurrogate(nn.Module):
    def __init__(
        self,
        n_geom: int = 32,
        n_freq: int = 401,
        geom_dim: int = 128,
        hidden: int = 256,
        n_fourier: int = 8,
    ) -> None:
        super().__init__()
        self.n_geom = n_geom
        self.n_freq = n_freq
        self.n_fourier = n_fourier
        self.encoder = nn.Sequential(
            nn.Linear(n_geom, geom_dim),
            nn.GELU(),
            nn.Linear(geom_dim, geom_dim),
            nn.GELU(),
        )
        frequency_dim = 1 + 2 * n_fourier
        self.decoder = nn.Sequential(
            nn.Linear(geom_dim + frequency_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        frequencies = torch.linspace(0.0, 1.0, n_freq)
        self.register_buffer("f_norm", frequencies)

    def frequency_features(self, f_norm: torch.Tensor) -> torch.Tensor:
        scales = 2.0 ** torch.arange(self.n_fourier, device=f_norm.device, dtype=f_norm.dtype)
        angle = 2.0 * math.pi * f_norm[:, None] * scales[None, :]
        return torch.cat([f_norm[:, None], torch.sin(angle), torch.cos(angle)], dim=-1)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        """``geometry`` 形状 ``(batch, 32)``，返回 ``(batch, 3, frequency)``。"""

        encoded = self.encoder(geometry)
        features = self.frequency_features(self.f_norm)
        combined = torch.cat(
            [
                encoded[:, None, :].expand(-1, features.shape[0], -1),
                features[None, :, :].expand(geometry.shape[0], -1, -1),
            ],
            dim=-1,
        )
        raw = self.decoder(combined)
        absorption_forward = torch.sigmoid(raw[..., 0])
        absorption_backward = torch.sigmoid(raw[..., 1])
        eigenvalue_gap = torch.nn.functional.softplus(raw[..., 2])
        return torch.stack(
            [absorption_forward, absorption_backward, eigenvalue_gap],
            dim=1,
        )
