from __future__ import annotations

import torch

from src.models.encoder import RepresentationMapper


def test_encoder_context_to_latent_shape() -> None:
    encoder = RepresentationMapper(in_channels=3, latent_dim=16, base_channels=8)
    context = torch.randn(2, 4, 3, 32, 32)
    latent = encoder(context)
    assert latent.shape == (2, 16)
