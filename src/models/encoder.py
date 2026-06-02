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
        normalize_output: bool = True,
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
        projector_layers: list[nn.Module] = [nn.Linear(base_channels * 4, latent_dim)]
        if normalize_output:
            projector_layers.append(nn.LayerNorm(latent_dim))
        self.projector = nn.Sequential(*projector_layers)

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


class SpatialFlattenRepresentationMapper(nn.Module):
    """Coordinate-sensitive encoder for physical state extraction.

    Unlike ``RepresentationMapper``, this encoder does not globally average
    away the spatial feature map. It flattens the final CNN grid before the
    latent projection, which is much better suited to sanity checks where
    absolute object position is part of the state.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        image_size: int = 64,
        context_length: int = 3,
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        reduced = image_size // 8
        feature_dim = base_channels * 4 * reduced * reduced
        self.temporal_projector = nn.Sequential(
            nn.Linear(feature_dim * context_length, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 8, latent_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, K, C, H, W]`` while preserving spatial information."""

        if context.ndim != 5:
            raise ValueError(f"Expected context shape [B, K, C, H, W], got {tuple(context.shape)}")
        batch_size, context_length, channels, height, width = context.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")
        if context_length != self.context_length:
            raise ValueError(f"Expected context_length={self.context_length}, got {context_length}")
        frames = context.reshape(batch_size * context_length, channels, height, width)
        features = self.frame_encoder(frames).flatten(start_dim=1)
        features = features.reshape(batch_size, context_length * features.shape[-1])
        return self.temporal_projector(features)
