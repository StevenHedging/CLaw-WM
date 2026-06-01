"""Visualization helpers for rollout comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _to_hwc(frame: torch.Tensor) -> torch.Tensor:
    return frame.detach().cpu().clamp(0, 1).permute(1, 2, 0)


def save_rollout_comparison(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_path: str | Path,
    max_frames: int = 8,
) -> None:
    """Save a two-row target-vs-prediction rollout grid."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    horizon = min(predictions.shape[0], targets.shape[0], max_frames)
    fig, axes = plt.subplots(2, horizon, figsize=(1.7 * horizon, 3.4), squeeze=False)
    for idx in range(horizon):
        axes[0, idx].imshow(_to_hwc(targets[idx]))
        axes[0, idx].set_title(f"GT {idx + 1}", fontsize=9)
        axes[1, idx].imshow(_to_hwc(predictions[idx]))
        axes[1, idx].set_title(f"Pred {idx + 1}", fontsize=9)
        axes[0, idx].axis("off")
        axes[1, idx].axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
