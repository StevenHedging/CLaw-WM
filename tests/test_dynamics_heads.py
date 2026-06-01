from __future__ import annotations

import torch

from src.models.dynamics_heads import DynamicsHeadLibrary, ResidualMLPDynamicsHead


def test_residual_dynamics_head_preserves_shape() -> None:
    head = ResidualMLPDynamicsHead(latent_dim=16, hidden_dim=32)
    z = torch.randn(5, 16)
    z_next = head(z)
    assert z_next.shape == z.shape


def test_head_library_can_add_head() -> None:
    library = DynamicsHeadLibrary(latent_dim=16, hidden_dim=32, initial_heads=1)
    new_head_id = library.add_head(init_from=0)
    assert new_head_id == 1
    assert library.num_heads() == 2
