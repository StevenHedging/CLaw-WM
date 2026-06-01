"""Representation mapper ``E_theta`` for context windows."""

from __future__ import annotations

import torch
from torch import nn


class RepresentationMapper(nn.Module):
    """Map ``C_t = (I_{t-K+1}, ..., I_t)`` to a latent state ``z_t``.

    The first implementation is deliberately small: each frame is encoded by a
    shared CNN and the resulting per-frame features are averaged over time.
    This keeps the interface compatible with future ViT, slot, or
    object-centric encoders.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 32,
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Linear(base_channels * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Encode a context tensor of shape ``[B, K, C, H, W]``."""

        if context.ndim != 5:
            raise ValueError(f"Expected context shape [B, K, C, H, W], got {tuple(context.shape)}")
        batch_size, context_length, channels, height, width = context.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")

        frames = context.reshape(batch_size * context_length, channels, height, width)
        features = self.frame_encoder(frames).flatten(start_dim=1)
        features = features.reshape(batch_size, context_length, -1).mean(dim=1)
        return self.projector(features)
