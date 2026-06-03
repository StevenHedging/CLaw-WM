"""Synthetic 2D ball dynamics for continual world-model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

DynamicsMode = Literal["velocity_flip", "gravity", "turn"]


@dataclass(frozen=True)
class ToyPhysicsEpisode:
    """A generated toy-physics episode."""

    frames: np.ndarray
    states: np.ndarray
    dynamics_ids: np.ndarray
    episode_id: int


def _render_ball(
    x: float,
    y: float,
    image_size: int,
    channels: int,
    radius: int,
    color: np.ndarray,
) -> np.ndarray:
    """Render a filled ball on a black canvas as CHW float32 image."""

    yy, xx = np.mgrid[0:image_size, 0:image_size]
    cx = x * (image_size - 1)
    cy = y * (image_size - 1)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    frame = np.zeros((channels, image_size, image_size), dtype=np.float32)
    for channel in range(channels):
        frame[channel, mask] = color[channel % len(color)]
    return frame


def _bounce_position(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ball in the unit square with elastic wall bounces."""

    for axis in range(2):
        if position[axis] < 0.05:
            position[axis] = 0.05
            velocity[axis] = abs(velocity[axis])
        elif position[axis] > 0.95:
            position[axis] = 0.95
            velocity[axis] = -abs(velocity[axis])
    return position, velocity


def simulate_episode(
    episode_id: int,
    episode_length: int,
    image_size: int = 32,
    channels: int = 3,
    ball_radius: int = 2,
    switch_time: int | None = None,
    mode: DynamicsMode = "velocity_flip",
    seed: int = 0,
) -> ToyPhysicsEpisode:
    """Simulate one episode with a sudden dynamics switch.

    The latent debug state is ``[x, y, vx, vy, ax, ay]``. Dynamics id ``0``
    denotes constant velocity before the switch; ids ``1..3`` denote the
    post-switch rules.
    """

    rng = np.random.default_rng(seed + episode_id * 9973)
    switch = switch_time if switch_time is not None else episode_length // 2
    switch = int(np.clip(switch, 1, episode_length - 1))

    position = rng.uniform(0.2, 0.8, size=2).astype(np.float32)
    velocity = rng.uniform(-0.035, 0.035, size=2).astype(np.float32)
    velocity += np.sign(velocity + 1e-6) * 0.015
    acceleration = np.zeros(2, dtype=np.float32)
    color = np.asarray([1.0, 0.35 + 0.3 * (episode_id % 2), 0.2], dtype=np.float32)

    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []
    dynamics_ids: list[int] = []
    flipped = False

    for time in range(episode_length):
        if time >= switch:
            if mode == "velocity_flip":
                dynamics_id = 1
                if not flipped:
                    velocity *= np.asarray([-1.0, 1.0], dtype=np.float32)
                    flipped = True
                acceleration = np.zeros(2, dtype=np.float32)
            elif mode == "gravity":
                dynamics_id = 2
                acceleration = np.asarray([0.0, 0.006], dtype=np.float32)
            elif mode == "turn":
                dynamics_id = 3
                if not flipped:
                    velocity = np.asarray([-velocity[1], velocity[0]], dtype=np.float32)
                    flipped = True
                acceleration = np.zeros(2, dtype=np.float32)
            else:
                raise ValueError(f"Unknown dynamics mode: {mode}")
        else:
            dynamics_id = 0
            acceleration = np.zeros(2, dtype=np.float32)

        frames.append(_render_ball(position[0], position[1], image_size, channels, ball_radius, color))
        states.append(np.asarray([position[0], position[1], velocity[0], velocity[1], acceleration[0], acceleration[1]], dtype=np.float32))
        dynamics_ids.append(dynamics_id)

        velocity = velocity + acceleration
        position = position + velocity
        position, velocity = _bounce_position(position, velocity)

    return ToyPhysicsEpisode(
        frames=np.stack(frames, axis=0).astype(np.float32),
        states=np.stack(states, axis=0).astype(np.float32),
        dynamics_ids=np.asarray(dynamics_ids, dtype=np.int64),
        episode_id=episode_id,
    )


class ToyPhysicsDataset(Dataset[dict[str, torch.Tensor | int]]):
    """Streaming-style dataset of context windows and next-frame targets.

    For a context length ``K``, the first supervised prediction is
    ``[I_0, ..., I_{K-1}] -> I_K``. In one-based frame numbering this is
    ``[I_1, ..., I_K] -> I_{K+1}``, so no loss is computed before a full
    velocity-informative context exists.
    """

    def __init__(
        self,
        num_episodes: int,
        episode_length: int,
        context_length: int,
        image_size: int = 32,
        channels: int = 3,
        ball_radius: int = 2,
        switch_time: int | None = None,
        seed: int = 0,
        modes: tuple[DynamicsMode, ...] = ("velocity_flip", "gravity", "turn"),
    ) -> None:
        if episode_length <= context_length:
            raise ValueError("episode_length must be larger than context_length")
        self.context_length = context_length
        self.first_target_index = context_length
        self.first_target_frame_number = context_length + 1
        self.episodes = [
            simulate_episode(
                episode_id=episode_id,
                episode_length=episode_length,
                image_size=image_size,
                channels=channels,
                ball_radius=ball_radius,
                switch_time=switch_time,
                mode=modes[episode_id % len(modes)],
                seed=seed,
            )
            for episode_id in range(num_episodes)
        ]
        self.index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            for time in range(context_length - 1, episode.frames.shape[0] - 1):
                self.index.append((episode_idx, time))

    def __len__(self) -> int:
        """Return the number of frame-wise training samples."""

        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        """Return ``C_t`` and ``I_{t+1}`` for one episode time step."""

        episode_idx, time = self.index[item]
        episode = self.episodes[episode_idx]
        start = time - self.context_length + 1
        context = episode.frames[start : time + 1]
        target = episode.frames[time + 1]
        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "state": torch.from_numpy(episode.states[time]),
            "episode_id": episode.episode_id,
            "time": time,
            "context_start": start,
            "context_end": time,
            "target_time": time + 1,
            "target_frame_number": time + 2,
            "dynamics_id": int(episode.dynamics_ids[time]),
        }
