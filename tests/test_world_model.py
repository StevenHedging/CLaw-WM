from __future__ import annotations

import torch

from src.models.dynamics_heads import DynamicsHeadLibrary
from src.models.encoder import RepresentationMapper
from src.models.head_selector import ErrorBasedHeadSelector
from src.models.renderer import Renderer
from src.models.world_model import WorldModel


def test_world_model_forward_outputs_prediction_frame() -> None:
    model = WorldModel(
        encoder=RepresentationMapper(in_channels=3, latent_dim=8, base_channels=8),
        head_library=DynamicsHeadLibrary(latent_dim=8, hidden_dim=16, initial_heads=1),
        renderer=Renderer(latent_dim=8, image_shape=(3, 32, 32), hidden_dim=32),
        selector=ErrorBasedHeadSelector(select_every_m=1),
    )
    context = torch.randn(2, 4, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    output = model(context, target_frame=target)
    assert output["prediction"].shape == (2, 3, 32, 32)
    assert output["head_id"] == 0
