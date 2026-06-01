"""Regularization placeholders for continual learning."""

from __future__ import annotations

import torch
from torch import nn


def l2_regularization(module: nn.Module) -> torch.Tensor:
    """Return a differentiable L2 penalty over all trainable parameters."""

    penalty: torch.Tensor | None = None
    for parameter in module.parameters():
        if parameter.requires_grad:
            value = parameter.pow(2).sum()
            penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.tensor(0.0)
    return penalty
