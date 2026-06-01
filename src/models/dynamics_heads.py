"""Residual MLP dynamics heads and a growable head library."""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn


class ResidualMLPDynamicsHead(nn.Module):
    """Residual latent dynamics ``F_i(z) = z + dt * f_i(z)``."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        input_dim = latent_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self.dt = float(dt)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Advance one latent step with a residual MLP update."""

        return z + self.dt * self.net(z)


class DynamicsHeadLibrary(nn.Module):
    """Maintain a growable library ``{F_phi^1, ..., F_phi^M}``."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dt: float = 1.0,
        initial_heads: int = 1,
    ) -> None:
        super().__init__()
        if initial_heads < 1:
            raise ValueError("initial_heads must be >= 1")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt
        self.heads = nn.ModuleList(
            [
                ResidualMLPDynamicsHead(latent_dim, hidden_dim, num_layers, dt)
                for _ in range(initial_heads)
            ]
        )

    def _new_head(self) -> ResidualMLPDynamicsHead:
        return ResidualMLPDynamicsHead(self.latent_dim, self.hidden_dim, self.num_layers, self.dt)

    def _library_device(self) -> torch.device:
        return next(self.parameters()).device

    def add_head(self, init_from: Optional[int] = None) -> int:
        """Add a new dynamics head and return its id.

        If ``init_from`` is provided, the new head is a deep copy of an
        existing head. This is the minimal hook for few-shot completion or
        OWL-style warm-start updates.
        """

        if init_from is None:
            head = self._new_head()
            head.to(self._library_device())
        else:
            if init_from < 0 or init_from >= self.num_heads():
                raise IndexError(f"Invalid init_from head id: {init_from}")
            head = copy.deepcopy(self.heads[init_from])
        self.heads.append(head)
        return self.num_heads() - 1

    def forward_head(self, head_id: int, z: torch.Tensor) -> torch.Tensor:
        """Run one selected head on latent states."""

        if head_id < 0 or head_id >= self.num_heads():
            raise IndexError(f"Invalid head id: {head_id}")
        return self.heads[head_id](z)

    def forward_all(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Run every head on the same latent states."""

        return [head(z) for head in self.heads]

    def num_heads(self) -> int:
        """Return the current number of dynamics heads."""

        return len(self.heads)
