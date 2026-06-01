"""Pure PyTorch training loops for continual toy experiments."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.continual.head_manager import HeadManager
from src.continual.losses import distillation_loss, prediction_loss, replay_loss
from src.continual.regularizers import l2_regularization
from src.continual.replay_buffer import ReplayBuffer
from src.models.world_model import WorldModel
from src.training.checkpoints import save_checkpoint
from src.utils.logging import MetricLogger


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch to a device."""

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def predict_grouped_by_head(
    model: WorldModel,
    context: torch.Tensor,
    head_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict samples grouped by stored head id.

    Returns predictions and a boolean mask marking samples whose head exists in
    the supplied model. The mask matters for distillation when the teacher was
    snapshotted before a new head was spawned.
    """

    predictions = torch.zeros(
        (context.shape[0], *model.renderer.image_shape),
        dtype=context.dtype,
        device=context.device,
    )
    valid_mask = torch.zeros(context.shape[0], dtype=torch.bool, device=context.device)
    for head_id_tensor in head_ids.unique(sorted=True):
        head_id = int(head_id_tensor.item())
        if head_id < 0 or head_id >= model.head_library.num_heads():
            continue
        mask = head_ids == head_id
        predictions[mask] = model.predict_with_head(context[mask], head_id)["prediction"]
        valid_mask[mask] = True
    return predictions, valid_mask


def train_one_batch(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    active_head_id: int,
    replay_buffer: ReplayBuffer,
    cfg: Any,
    device: torch.device,
    global_step: int,
    teacher_model: WorldModel | None = None,
) -> dict[str, float | int]:
    """Run one continual update step."""

    model.train()
    batch = move_batch_to_device(batch, device)
    context = batch["context"]
    target = batch["target"]
    force_head = active_head_id if bool(cfg.continual.force_manager_head_for_update) else None

    optimizer.zero_grad(set_to_none=True)
    output = model(context, target_frame=target, force_head_id=force_head, step=global_step)
    new_loss = prediction_loss(output["prediction"], target)

    replay_value = _zero(device)
    distill_value = _zero(device)
    if len(replay_buffer) > 0 and int(cfg.continual.replay_batch_size) > 0:
        replay_batch = move_batch_to_device(replay_buffer.sample(int(cfg.continual.replay_batch_size)), device)
        replay_context = replay_batch["context"]
        replay_target = replay_batch["target"]
        replay_head_ids = replay_batch["head_id"]

        replay_prediction, replay_mask = predict_grouped_by_head(model, replay_context, replay_head_ids)
        if replay_mask.any():
            replay_value = replay_loss(replay_prediction[replay_mask], replay_target[replay_mask])

        if teacher_model is not None:
            teacher_model.eval()
            with torch.no_grad():
                teacher_prediction, teacher_mask = predict_grouped_by_head(
                    teacher_model,
                    replay_context,
                    replay_head_ids,
                )
            distill_mask = replay_mask & teacher_mask
            if distill_mask.any():
                distill_value = distillation_loss(
                    replay_prediction[distill_mask],
                    teacher_prediction[distill_mask],
                )

    l2_value = l2_regularization(model.head_library.heads[active_head_id])
    total_loss = (
        new_loss
        + float(cfg.continual.lambda_replay) * replay_value
        + float(cfg.continual.lambda_distill) * distill_value
        + float(cfg.continual.lambda_l2) * l2_value
    )
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_new": float(new_loss.detach().cpu()),
        "loss_replay": float(replay_value.detach().cpu()),
        "loss_distill": float(distill_value.detach().cpu()),
        "loss_l2": float(l2_value.detach().cpu()),
        "selected_head": int(output["head_id"]),
    }


def clone_teacher(model: WorldModel, device: torch.device) -> WorldModel:
    """Create a frozen teacher snapshot for distillation."""

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def run_continual_training(
    model: WorldModel,
    train_loader: DataLoader,
    cfg: Any,
    device: torch.device,
    logger: MetricLogger,
) -> dict[str, Any]:
    """Train a world model on streaming toy samples."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    manager = HeadManager(
        model=model,
        reuse_threshold=float(cfg.continual.reuse_threshold),
        max_heads=int(cfg.continual.max_heads),
        device=device,
    )
    replay_buffer = ReplayBuffer(capacity=int(cfg.continual.replay_capacity))
    selected_counter: Counter[int] = Counter()
    teacher_model: WorldModel | None = None
    global_step = 0

    for _epoch in range(int(cfg.train.epochs)):
        for batch in train_loader:
            active_head_id, evaluation, spawned = manager.choose_head_for_batch(
                batch,
                init_from_best=bool(cfg.continual.spawn_init_from_best),
            )
            if spawned:
                optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

            metrics = train_one_batch(
                model=model,
                optimizer=optimizer,
                batch=batch,
                active_head_id=active_head_id,
                replay_buffer=replay_buffer,
                cfg=cfg,
                device=device,
                global_step=global_step,
                teacher_model=teacher_model,
            )
            replay_buffer.add_batch(batch, active_head_id)
            selected_counter[int(metrics["selected_head"])] += 1

            metrics.update(
                {
                    "num_heads": model.head_library.num_heads(),
                    "spawned": int(spawned),
                    "spawned_total": manager.spawned_count,
                    "manager_best_head": evaluation.best_head_id,
                    "manager_best_error": evaluation.best_error,
                }
            )
            if global_step % int(cfg.train.log_every) == 0:
                logger.log(metrics, step=global_step)

            if (global_step + 1) % int(cfg.train.checkpoint_every) == 0:
                save_checkpoint(model, cfg, cfg.checkpoint_path, step=global_step)

            teacher_model = clone_teacher(model, device)
            global_step += 1
            if int(cfg.train.max_steps) > 0 and global_step >= int(cfg.train.max_steps):
                save_checkpoint(model, cfg, cfg.checkpoint_path, step=global_step)
                return {
                    "steps": global_step,
                    "selected_head_distribution": dict(selected_counter),
                    "num_heads": model.head_library.num_heads(),
                    "spawned_total": manager.spawned_count,
                }

    save_checkpoint(model, cfg, cfg.checkpoint_path, step=global_step)
    return {
        "steps": global_step,
        "selected_head_distribution": dict(selected_counter),
        "num_heads": model.head_library.num_heads(),
        "spawned_total": manager.spawned_count,
        "checkpoint": str(Path(cfg.checkpoint_path)),
    }
