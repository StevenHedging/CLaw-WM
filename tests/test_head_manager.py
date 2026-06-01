from __future__ import annotations

import torch

from src.continual.head_manager import HeadManager
from src.models.dynamics_heads import DynamicsHeadLibrary
from src.models.encoder import RepresentationMapper
from src.models.head_selector import ErrorBasedHeadSelector
from src.models.renderer import Renderer
from src.models.world_model import WorldModel


def _build_model() -> WorldModel:
    return WorldModel(
        encoder=RepresentationMapper(in_channels=3, latent_dim=8, base_channels=8),
        head_library=DynamicsHeadLibrary(latent_dim=8, hidden_dim=16, initial_heads=1),
        renderer=Renderer(latent_dim=8, image_shape=(3, 16, 16), hidden_dim=32),
        selector=ErrorBasedHeadSelector(select_every_m=1),
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "context": torch.randn(2, 4, 3, 16, 16),
        "target": torch.zeros(2, 3, 16, 16),
    }


def test_head_manager_spawns_when_error_exceeds_threshold() -> None:
    model = _build_model()
    manager = HeadManager(model, reuse_threshold=-1.0, max_heads=2)
    head_id, evaluation, spawned = manager.choose_head_for_batch(_batch())
    assert spawned is True
    assert head_id == 1
    assert model.head_library.num_heads() == 2
    assert evaluation.best_error > manager.reuse_threshold


def test_head_manager_reuses_when_error_below_threshold() -> None:
    model = _build_model()
    manager = HeadManager(model, reuse_threshold=10.0, max_heads=2)
    head_id, evaluation, spawned = manager.choose_head_for_batch(_batch())
    assert spawned is False
    assert head_id == evaluation.best_head_id
    assert model.head_library.num_heads() == 1
