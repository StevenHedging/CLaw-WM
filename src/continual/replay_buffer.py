"""Simple replay buffer for continual consolidation."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch


class ReplayBuffer:
    """Store frame-wise samples and the head id used to learn them."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add_batch(self, batch: dict[str, Any], head_id: int) -> None:
        """Add a mini-batch to the replay buffer."""

        batch_size = int(batch["context"].shape[0])
        for idx in range(batch_size):
            item: dict[str, Any] = {
                "context": batch["context"][idx].detach().cpu(),
                "target": batch["target"][idx].detach().cpu(),
                "head_id": int(head_id),
            }
            for key in ("state", "episode_id", "time", "dynamics_id"):
                if key in batch:
                    value = batch[key]
                    item[key] = value[idx].detach().cpu() if torch.is_tensor(value) else value
            self._items.append(item)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a mini-batch and collate tensors."""

        if len(self._items) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        size = min(batch_size, len(self._items))
        samples = random.sample(list(self._items), size)
        batch: dict[str, Any] = {
            "context": torch.stack([sample["context"] for sample in samples], dim=0),
            "target": torch.stack([sample["target"] for sample in samples], dim=0),
            "head_id": torch.as_tensor([sample["head_id"] for sample in samples], dtype=torch.long),
        }
        if "state" in samples[0]:
            batch["state"] = torch.stack([sample["state"] for sample in samples], dim=0)
        for key in ("episode_id", "time", "dynamics_id"):
            if key in samples[0]:
                batch[key] = torch.as_tensor([int(sample[key]) for sample in samples], dtype=torch.long)
        return batch
