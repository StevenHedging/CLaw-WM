"""Evaluation utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.continual.losses import prediction_loss
from src.models.world_model import WorldModel
from src.training.loops import move_batch_to_device


@torch.no_grad()
def evaluate_model(
    model: WorldModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate one-step prediction with dynamic head selection."""

    model.to(device)
    model.eval()
    model.selector.reset()
    total_loss = 0.0
    total_batches = 0
    head_counter: Counter[int] = Counter()
    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        output = model(batch["context"], target_frame=batch["target"], step=step)
        loss = prediction_loss(output["prediction"], batch["target"])
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        head_counter[int(output["head_id"])] += 1
    mean_loss = total_loss / max(total_batches, 1)
    return {
        "one_step_loss": mean_loss,
        "num_batches": total_batches,
        "num_heads": model.head_library.num_heads(),
        "selected_head_distribution": dict(head_counter),
    }
