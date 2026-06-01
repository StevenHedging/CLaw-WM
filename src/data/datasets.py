"""Dataset and dataloader factories."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.data.toy_physics import ToyPhysicsDataset


def build_toy_dataset(cfg: Any, train: bool = True) -> ToyPhysicsDataset:
    """Build the toy physics dataset from an OmegaConf-like config object."""

    num_episodes = int(cfg.num_train_episodes if train else cfg.num_eval_episodes)
    seed = int(cfg.seed if train else cfg.seed + 10_000)
    return ToyPhysicsDataset(
        num_episodes=num_episodes,
        episode_length=int(cfg.episode_length),
        context_length=int(cfg.context_length),
        image_size=int(cfg.image_size),
        channels=int(cfg.channels),
        ball_radius=int(cfg.ball_radius),
        switch_time=int(cfg.switch_time),
        seed=seed,
    )


def build_toy_dataloader(cfg: Any, train: bool = True) -> DataLoader:
    """Build a deterministic dataloader for toy physics."""

    dataset = build_toy_dataset(cfg, train=train)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=train,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
