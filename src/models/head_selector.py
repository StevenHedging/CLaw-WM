"""Frame-wise or every-m-frames dynamics head selection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.dynamics_heads import DynamicsHeadLibrary


class ErrorBasedHeadSelector:
    """Hard-select the head with the lowest one-step prediction error."""

    def __init__(self, select_every_m: int = 1, loss_type: str = "mse") -> None:
        if select_every_m < 1:
            raise ValueError("select_every_m must be >= 1")
        self.select_every_m = select_every_m
        self.loss_type = loss_type
        self._cached_head_id: int | None = None

    def reset(self) -> None:
        """Forget the cached head used by every-m-frames selection."""

        self._cached_head_id = None

    def _error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target)
        if self.loss_type == "l1":
            return F.l1_loss(prediction, target)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def select(
        self,
        z_t: torch.Tensor,
        target_frame: torch.Tensor | None,
        head_library: DynamicsHeadLibrary,
        renderer: nn.Module,
        step: int | None = None,
        force_resample: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        """Select a head id and optionally return all error scores.

        When ``select_every_m > 1``, the selector reuses the cached head for
        intermediate frames and resamples only on frames divisible by ``m``.
        """

        if head_library.num_heads() == 0:
            raise ValueError("Cannot select from an empty head library")

        should_resample = force_resample or self._cached_head_id is None or (step is None and self.select_every_m == 1)
        if step is not None and step % self.select_every_m == 0:
            should_resample = True
        elif step is not None and step % self.select_every_m != 0:
            should_resample = False

        if not should_resample and self._cached_head_id is not None:
            return self._cached_head_id, {
                "scores": None,
                "selected_head_id": self._cached_head_id,
                "resampled": False,
            }

        if target_frame is None:
            selected = self._cached_head_id if self._cached_head_id is not None else 0
            self._cached_head_id = selected
            return selected, {"scores": None, "selected_head_id": selected, "resampled": False}

        scores: list[float] = []
        with torch.no_grad():
            for head_id in range(head_library.num_heads()):
                z_next = head_library.forward_head(head_id, z_t)
                prediction = renderer(z_next)
                scores.append(float(self._error(prediction, target_frame).detach().cpu()))

        selected = int(min(range(len(scores)), key=scores.__getitem__))
        self._cached_head_id = selected
        return selected, {
            "scores": scores,
            "selected_head_id": selected,
            "resampled": True,
        }
