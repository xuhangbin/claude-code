"""几何参数到吸声谱、特征值间隔的代理网络。

输入是名义设计附近的 32 维归一化几何。三层全连接网络直接输出每个频率上的
三项：前向吸声、后向吸声、特征值间隔。吸声经过 sigmoid，特征值间隔经过 softplus。
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralSurrogate(nn.Module):
    def __init__(
        self,
        n_geom: int = 32,
        n_freq: int = 401,
        hidden: int = 384,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.n_freq = n_freq
        layers: list[nn.Module] = []
        width = n_geom
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.GELU())
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_freq * 3)
        with torch.no_grad():
            self.head.bias.zero_()
            # 名义设计附近：前向吸声大约 0.5，后向吸声接近 0，特征值间隔大约 1.5。
            self.head.bias[:n_freq].fill_(0.15)
            self.head.bias[n_freq : 2 * n_freq].fill_(-5.5)
            self.head.bias[2 * n_freq :].fill_(1.2)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        """``geometry`` 形状 ``(batch, 32)``，返回 ``(batch, 3, frequency)``。"""

        raw = self.head(self.backbone(geometry)).reshape(geometry.shape[0], 3, self.n_freq)
        absorption_forward = torch.sigmoid(raw[:, 0])
        absorption_backward = torch.sigmoid(raw[:, 1])
        eigenvalue_gap = torch.nn.functional.softplus(raw[:, 2])
        return torch.stack(
            [absorption_forward, absorption_backward, eigenvalue_gap],
            dim=1,
        )
