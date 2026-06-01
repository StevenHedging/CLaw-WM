"""Small logging helpers with optional Weights & Biases integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricLogger:
    """Console-first metric logger with optional wandb backend."""

    use_wandb: bool = False
    project: str = "continual-multihead-wm"
    run_name: str | None = None
    _wandb_run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.use_wandb:
            import wandb

            self._wandb_run = wandb.init(project=self.project, name=self.run_name)

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log metrics to stdout and, optionally, wandb."""

        prefix = f"[step {step}] " if step is not None else ""
        pretty = ", ".join(f"{key}={value:.6g}" for key, value in metrics.items())
        print(f"{prefix}{pretty}")
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        """Close the optional wandb run."""

        if self._wandb_run is not None:
            self._wandb_run.finish()
