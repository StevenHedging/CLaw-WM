"""Loss functions for prediction and continual consolidation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """Compute one-step image prediction loss."""

    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "l1":
        return F.l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def distillation_loss(student_prediction: torch.Tensor, teacher_prediction: torch.Tensor) -> torch.Tensor:
    """Match current predictions to frozen old predictions."""

    return F.mse_loss(student_prediction, teacher_prediction.detach())


def replay_loss(prediction: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """Prediction loss evaluated on replay samples."""

    return prediction_loss(prediction, target, loss_type=loss_type)
