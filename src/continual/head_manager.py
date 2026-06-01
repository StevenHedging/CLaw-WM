"""Head reuse, spawning, and update orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.continual.losses import prediction_loss
from src.models.world_model import WorldModel


@dataclass(frozen=True)
class HeadEvaluation:
    """Prediction-error summary for a data block against all heads."""

    errors: dict[int, float]
    best_head_id: int
    best_error: float


class HeadManager:
    """Decide whether to reuse an old dynamics head or spawn a new one."""

    def __init__(
        self,
        model: WorldModel,
        reuse_threshold: float,
        max_heads: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.reuse_threshold = float(reuse_threshold)
        self.max_heads = max_heads
        self.device = torch.device(device)
        self.spawned_count = 0

    @torch.no_grad()
    def evaluate_existing_heads(self, batch: dict[str, Any]) -> HeadEvaluation:
        """Compute ``E_i(D_n)`` for every existing head on a mini-batch."""

        self.model.eval()
        context = batch["context"].to(self.device)
        target = batch["target"].to(self.device)
        errors: dict[int, float] = {}
        for head_id in range(self.model.head_library.num_heads()):
            output = self.model.predict_with_head(context, head_id)
            errors[head_id] = float(prediction_loss(output["prediction"], target).detach().cpu())
        best_head_id = min(errors, key=errors.get)
        return HeadEvaluation(
            errors=errors,
            best_head_id=int(best_head_id),
            best_error=float(errors[best_head_id]),
        )

    def should_reuse_head(self, evaluation: HeadEvaluation) -> bool:
        """Return true when the best existing head is below threshold ``tau``."""

        return evaluation.best_error <= self.reuse_threshold

    def should_spawn_new_head(self, evaluation: HeadEvaluation) -> bool:
        """Return true when no current head is good enough and capacity remains."""

        has_capacity = self.max_heads is None or self.model.head_library.num_heads() < self.max_heads
        return evaluation.best_error > self.reuse_threshold and has_capacity

    def update_existing_head(self, evaluation: HeadEvaluation) -> int:
        """Return the old head id selected for few-shot/fine-tuning updates."""

        return int(evaluation.best_head_id)

    def spawn_new_head(self, init_from: int | None = None) -> int:
        """Add a new dynamics head and return its id."""

        if self.max_heads is not None and self.model.head_library.num_heads() >= self.max_heads:
            return self.model.head_library.num_heads() - 1
        head_id = self.model.head_library.add_head(init_from=init_from)
        self.spawned_count += 1
        return head_id

    def choose_head_for_batch(self, batch: dict[str, Any], init_from_best: bool = True) -> tuple[int, HeadEvaluation, bool]:
        """Evaluate a batch and either reuse the best head or spawn a new one."""

        evaluation = self.evaluate_existing_heads(batch)
        if self.should_spawn_new_head(evaluation):
            init_from = evaluation.best_head_id if init_from_best else None
            return self.spawn_new_head(init_from=init_from), evaluation, True
        return self.update_existing_head(evaluation), evaluation, False
