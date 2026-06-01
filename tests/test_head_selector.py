from __future__ import annotations

import torch

from src.models.dynamics_heads import DynamicsHeadLibrary
from src.models.head_selector import ErrorBasedHeadSelector
from src.models.renderer import Renderer


def test_error_based_selector_returns_legal_head_id() -> None:
    library = DynamicsHeadLibrary(latent_dim=8, hidden_dim=16, initial_heads=2)
    renderer = Renderer(latent_dim=8, image_shape=(3, 16, 16), hidden_dim=32)
    selector = ErrorBasedHeadSelector(select_every_m=1)
    z = torch.randn(4, 8)
    target = torch.zeros(4, 3, 16, 16)
    head_id, info = selector.select(z, target, library, renderer)
    assert 0 <= head_id < library.num_heads()
    assert info["scores"] is not None
    assert len(info["scores"]) == library.num_heads()
