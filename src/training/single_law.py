"""Single-law world-model sanity check on PhyWorld uniform motion."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.phyworld import PhyWorldUniformMotionDataset
from src.models.dynamics_heads import DynamicsHeadLibrary
from src.models.encoder import SpatialFlattenRepresentationMapper
from src.models.head_selector import ErrorBasedHeadSelector
from src.models.renderer import SpatialBroadcastRenderer
from src.models.world_model import WorldModel
from src.training.train import resolve_device
from src.utils.logging import MetricLogger
from src.utils.seed import set_seed
from src.utils.visualization import save_rollout_comparison


def build_single_law_model(cfg: Any) -> WorldModel:
    """Build a one-head state-supervised world model."""

    image_shape = (3, int(cfg.data.image_size), int(cfg.data.image_size))
    encoder = SpatialFlattenRepresentationMapper(
        in_channels=3,
        latent_dim=int(cfg.model.latent_dim),
        image_size=int(cfg.data.image_size),
        context_length=int(cfg.data.context_length),
        base_channels=int(cfg.model.encoder_channels),
    )
    head_library = DynamicsHeadLibrary(
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.head_hidden_dim),
        num_layers=int(cfg.model.head_num_layers),
        dt=float(cfg.model.dt),
        initial_heads=1,
    )
    renderer = SpatialBroadcastRenderer(
        latent_dim=int(cfg.model.latent_dim),
        image_shape=image_shape,
        hidden_dim=int(cfg.model.renderer_hidden_dim),
    )
    selector = ErrorBasedHeadSelector(select_every_m=1)
    return WorldModel(encoder, head_library, renderer, selector)


def _as_path_list(value: Any) -> list[str]:
    """Normalize a config scalar/list path field."""

    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def build_phyworld_dataset(cfg: Any, train: bool) -> Dataset:
    """Build one or more PhyWorld HDF5 datasets for the configured split."""

    split_paths = cfg.data.train_paths if train and "train_paths" in cfg.data else None
    split_paths = cfg.data.eval_paths if (not train and "eval_paths" in cfg.data) else split_paths
    if split_paths is None:
        split_paths = cfg.data.train_path if train else cfg.data.eval_path
    paths = _as_path_list(split_paths)
    if not paths:
        raise ValueError("No PhyWorld HDF5 paths were configured")
    max_videos = _optional_int(cfg.data.max_train_videos if train else cfg.data.max_eval_videos)
    datasets = [
        PhyWorldUniformMotionDataset(
            hdf5_path=path,
            context_length=int(cfg.data.context_length),
            image_size=int(cfg.data.image_size),
            max_videos=max_videos,
            state_scale=float(cfg.data.state_scale),
            cache_size=int(cfg.data.cache_size),
        )
        for path in paths
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_phyworld_loader(cfg: Any, train: bool) -> DataLoader:
    """Build a PhyWorld dataloader for the configured split."""

    dataset = build_phyworld_dataset(cfg, train=train)
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=train,
        num_workers=int(cfg.data.num_workers),
        drop_last=False,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _foreground_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float,
) -> torch.Tensor:
    """Pixel MSE that upweights non-background ball pixels."""

    foreground = (target < 0.95).any(dim=1, keepdim=True).float()
    weights = 1.0 + float(foreground_weight) * foreground
    return ((prediction - target).pow(2) * weights).mean()


def _losses(
    model: WorldModel,
    batch: dict[str, Any],
    foreground_weight: float = 0.0,
    compute_state_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch["context"], target_frame=batch["target"], force_head_id=0)
    image_loss = _foreground_weighted_mse(output["prediction"], batch["target"], foreground_weight)
    image_mse = F.mse_loss(output["prediction"], batch["target"])
    losses = {
        "image_loss": image_loss,
        "image_mse": image_mse,
    }
    if compute_state_metrics and "state" in batch and "next_state" in batch and output["z_t"].shape[-1] == batch["state"].shape[-1]:
        state_loss = F.mse_loss(output["z_t"], batch["state"])
        dynamics_loss = F.mse_loss(output["z_next"], batch["next_state"])
        position_error = torch.linalg.vector_norm(output["z_next"][:, :2] - batch["next_state"][:, :2], dim=-1).mean()
        losses.update(
            {
                "state_loss": state_loss,
                "dynamics_loss": dynamics_loss,
                "position_error": position_error,
            }
        )
    return output["prediction"], losses


def _weighted_optional_loss(weight: float, value: torch.Tensor | None) -> torch.Tensor | float:
    """Return ``weight * value`` only when that term is intentionally enabled."""

    if weight == 0.0 or value is None:
        return 0.0
    return weight * value


def train_single_law(config_path: str | Path, max_steps: int | None = None) -> dict[str, float]:
    """Train the single-head sanity-check model and run a small evaluation."""

    cfg = OmegaConf.load(config_path)
    if max_steps is not None:
        cfg.train.max_steps = int(max_steps)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model = build_single_law_model(cfg).to(device)
    train_loader = build_phyworld_loader(cfg, train=True)
    eval_loader = build_phyworld_loader(cfg, train=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    logger = MetricLogger(use_wandb=False)

    global_step = 0
    model.train()
    for _epoch in range(int(cfg.train.epochs)):
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            use_auxiliary_state_loss = bool(cfg.train.get("use_auxiliary_state_loss", True))
            _, losses = _losses(
                model,
                batch,
                foreground_weight=float(cfg.train.foreground_loss_weight),
                compute_state_metrics=use_auxiliary_state_loss,
            )
            total_loss = (
                float(cfg.train.image_loss_weight) * losses["image_loss"]
                + _weighted_optional_loss(float(cfg.train.state_loss_weight), losses.get("state_loss"))
                + _weighted_optional_loss(float(cfg.train.dynamics_loss_weight), losses.get("dynamics_loss"))
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if global_step % int(cfg.train.log_every) == 0:
                logger.log(
                    {
                        "loss_total": float(total_loss.detach().cpu()),
                        **{key: float(value.detach().cpu()) for key, value in losses.items()},
                    },
                    step=global_step,
                )
            global_step += 1
            if int(cfg.train.max_steps) > 0 and global_step >= int(cfg.train.max_steps):
                break
        if int(cfg.train.max_steps) > 0 and global_step >= int(cfg.train.max_steps):
            break

    torch.save({"model_state": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}, cfg.checkpoint_path)
    metrics = evaluate_single_law(model, eval_loader, cfg, device)
    logger.log(metrics, step=global_step)
    save_first_rollout(model, eval_loader, cfg, device)
    return metrics


@torch.no_grad()
def evaluate_single_law(model: WorldModel, loader: DataLoader, cfg: Any, device: torch.device) -> dict[str, float]:
    """Evaluate one-step image/state/dynamics losses."""

    model.eval()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        if batches >= int(cfg.eval.max_batches):
            break
        batch = _move_batch(batch, device)
        _, losses = _losses(
            model,
            batch,
            foreground_weight=float(cfg.train.foreground_loss_weight),
            compute_state_metrics=bool(cfg.train.get("report_state_metrics", True)),
        )
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        batches += 1
    return {f"eval_{key}": value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def save_first_rollout(model: WorldModel, loader: DataLoader, cfg: Any, device: torch.device) -> None:
    """Save a latent-dynamics rollout comparison for the first eval sample."""

    model.eval()
    batch = next(iter(loader))
    context = batch["context"][:1].to(device)
    horizon = int(cfg.eval.rollout_horizon)
    z = model.encoder(context)
    predictions: list[torch.Tensor] = []
    for _ in range(horizon):
        z = model.head_library.forward_head(0, z)
        predictions.append(model.renderer(z))
    prediction_tensor = torch.cat(predictions, dim=0).detach().cpu()

    dataset = loader.dataset
    if not isinstance(dataset, PhyWorldUniformMotionDataset):
        return
    episode_id = int(batch["episode_id"][0])
    time = int(batch["time"][0])
    group_key, local_idx = dataset._video_refs[episode_id]
    frames = dataset._read_video(group_key, local_idx)
    targets = frames[time + 1 : time + 1 + horizon].detach().cpu()
    output_path = Path(cfg.output_dir) / "single_law_rollout.png"
    save_rollout_comparison(prediction_tensor[: targets.shape[0]], targets, output_path, max_frames=targets.shape[0])
