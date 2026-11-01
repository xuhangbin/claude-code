"""Encoder (spectrum to geometry) and decoder (geometry to spectrum)."""

from __future__ import annotations

import torch
from torch import nn

from origami_mpp_tandem.physics import N_PARAMS


class ResidualMLP(nn.Module):
    """MLP with residual blocks. Sharp absorption peaks need the extra depth."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 512, blocks: int = 6):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                for _ in range(blocks)
            ]
        )
        self.output = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class ForwardNet(nn.Module):
    """Normalized geometry (20,) to absorption spectrum."""

    def __init__(self, n_freq: int, hidden: int = 512, blocks: int = 6):
        super().__init__()
        self.body = ResidualMLP(N_PARAMS, n_freq, hidden=hidden, blocks=blocks)

    def forward(self, unit_params: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(unit_params))


class InverseNet(nn.Module):
    """Absorption spectrum to normalized geometry in (0, 1)."""

    def __init__(self, n_freq: int, hidden: int = 512, blocks: int = 6):
        super().__init__()
        self.body = ResidualMLP(n_freq, N_PARAMS, hidden=hidden, blocks=blocks)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(spectrum))
