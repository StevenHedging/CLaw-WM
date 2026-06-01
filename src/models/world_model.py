"""Composable continual multi-head world model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.dynamics_heads import DynamicsHeadLibrary
from src.models.encoder import RepresentationMapper
from src.models.head_selector import ErrorBasedHeadSelector
from src.models.renderer import Renderer


class WorldModel(nn.Module):
    """World model with explicit ``E -> pi -> F_i -> R`` computation graph."""

    def __init__(
        self,
        encoder: RepresentationMapper,
        head_library: DynamicsHeadLibrary,
        renderer: Renderer,
        selector: ErrorBasedHeadSelector,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head_library = head_library
        self.renderer = renderer
        self.selector = selector

    def predict_with_head(self, context: torch.Tensor, head_id: int) -> dict[str, torch.Tensor | int]:
        """Predict one step with a fixed dynamics head."""

        z_t = self.encoder(context)
        z_next = self.head_library.forward_head(head_id, z_t)
        frame_next = self.renderer(z_next)
        return {
            "z_t": z_t,
            "z_next": z_next,
            "prediction": frame_next,
            "head_id": head_id,
        }

    def predict_one_step(
        self,
        context: torch.Tensor,
        target_frame: torch.Tensor | None = None,
        force_head_id: int | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Run ``R(F^{pi(C_t, F_n)}(E(C_t)))`` for one next-frame prediction."""

        z_t = self.encoder(context)
        if force_head_id is None:
            head_id, selection_info = self.selector.select(
                z_t=z_t,
                target_frame=target_frame,
                head_library=self.head_library,
                renderer=self.renderer,
                step=step,
            )
        else:
            head_id = int(force_head_id)
            selection_info = {
                "scores": None,
                "selected_head_id": head_id,
                "resampled": False,
                "forced": True,
            }
        z_next = self.head_library.forward_head(head_id, z_t)
        frame_next = self.renderer(z_next)
        return {
            "z_t": z_t,
            "z_next": z_next,
            "prediction": frame_next,
            "head_id": head_id,
            "selection": selection_info,
        }

    def forward(
        self,
        context: torch.Tensor,
        target_frame: torch.Tensor | None = None,
        force_head_id: int | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Alias for ``predict_one_step`` used by training code."""

        return self.predict_one_step(context, target_frame, force_head_id, step)

    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,
        horizon: int,
        force_head_id: int | None = None,
    ) -> dict[str, torch.Tensor | list[int]]:
        """Autoregressively roll out predicted frames for ``horizon`` steps."""

        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        current_context = context
        predictions: list[torch.Tensor] = []
        head_ids: list[int] = []
        for step in range(horizon):
            output = self.predict_one_step(
                current_context,
                target_frame=None,
                force_head_id=force_head_id,
                step=step,
            )
            prediction = output["prediction"]
            predictions.append(prediction)
            head_ids.append(int(output["head_id"]))
            current_context = torch.cat([current_context[:, 1:], prediction.unsqueeze(1)], dim=1)
        return {
            "predictions": torch.stack(predictions, dim=1),
            "head_ids": head_ids,
        }
