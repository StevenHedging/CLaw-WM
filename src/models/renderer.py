"""Renderer ``R_psi`` from latent state to image frame."""

from __future__ import annotations

import torch
from torch import nn


class Renderer(nn.Module):
    """Small MLP decoder that renders ``z_{t+1}`` to ``I_{t+1}``."""

    def __init__(
        self,
        latent_dim: int,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        channels, height, width = image_shape
        self.output_dim = channels * height * width
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states of shape ``[B, latent_dim]`` to images."""

        decoded = self.decoder(z)
        return decoded.reshape(z.shape[0], *self.image_shape)
