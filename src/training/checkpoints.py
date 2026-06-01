"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.models.world_model import WorldModel


def save_checkpoint(model: WorldModel, cfg: Any, path: str | Path, step: int) -> None:
    """Save model state, current head count, config, and step."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_heads": model.head_library.num_heads(),
            "step": int(step),
            "config": OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg,
        },
        checkpoint_path,
    )


def load_checkpoint(model: WorldModel, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint, growing the head library before restoring weights."""

    checkpoint = torch.load(Path(path), map_location=map_location)
    target_heads = int(checkpoint.get("num_heads", model.head_library.num_heads()))
    while model.head_library.num_heads() < target_heads:
        model.head_library.add_head(init_from=0)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
