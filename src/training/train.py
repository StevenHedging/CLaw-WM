"""Entrypoints and factories for toy continual training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.data.datasets import build_toy_dataloader
from src.models.dynamics_heads import DynamicsHeadLibrary
from src.models.encoder import RepresentationMapper
from src.models.head_selector import ErrorBasedHeadSelector
from src.models.renderer import Renderer
from src.models.world_model import WorldModel
from src.training.loops import run_continual_training
from src.utils.logging import MetricLogger
from src.utils.seed import set_seed


def load_config(config_path: str | Path) -> Any:
    """Load a YAML config with OmegaConf."""

    return OmegaConf.load(config_path)


def resolve_device(device_name: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_world_model(cfg: Any) -> WorldModel:
    """Construct the world model from config."""

    image_shape = (int(cfg.data.channels), int(cfg.data.image_size), int(cfg.data.image_size))
    encoder = RepresentationMapper(
        in_channels=int(cfg.data.channels),
        latent_dim=int(cfg.model.latent_dim),
        base_channels=int(cfg.model.encoder_channels),
    )
    head_library = DynamicsHeadLibrary(
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.head_hidden_dim),
        num_layers=int(cfg.model.head_num_layers),
        dt=float(cfg.model.dt),
        initial_heads=int(cfg.model.initial_heads),
    )
    renderer = Renderer(
        latent_dim=int(cfg.model.latent_dim),
        image_shape=image_shape,
        hidden_dim=int(cfg.model.renderer_hidden_dim),
    )
    selector = ErrorBasedHeadSelector(select_every_m=int(cfg.model.select_every_m))
    return WorldModel(encoder=encoder, head_library=head_library, renderer=renderer, selector=selector)


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    """Train the toy continual world model from a config path."""

    cfg = load_config(config_path)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model = build_world_model(cfg)
    train_loader = build_toy_dataloader(cfg.data, train=True)
    logger = MetricLogger(use_wandb=bool(cfg.use_wandb))
    try:
        result = run_continual_training(model, train_loader, cfg, device, logger)
    finally:
        logger.finish()
    print(f"Training finished: {result}")
    return result
