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


class SpatialBroadcastRenderer(nn.Module):
    """Render images by broadcasting latent states over a coordinate grid.

    This decoder is a good fit for single-object physics sanity checks: the
    renderer sees both the latent state and explicit pixel coordinates, making
    it much easier to learn "draw a ball at this position" than with a dense
    MLP over all pixels.
    """

    def __init__(
        self,
        latent_dim: int,
        image_shape: tuple[int, int, int] = (3, 64, 64),
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        channels, height, width = image_shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        self.register_buffer("coord_grid", grid, persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim + 2, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states to images using spatial broadcast."""

        batch_size = z.shape[0]
        _, height, width = self.image_shape
        latent_planes = z[:, :, None, None].expand(batch_size, z.shape[1], height, width)
        grid = self.coord_grid.expand(batch_size, -1, -1, -1)
        return self.net(torch.cat([latent_planes, grid], dim=1))
