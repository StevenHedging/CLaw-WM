"""Evaluate the toy continual world model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_toy_dataloader
from src.training.checkpoints import load_checkpoint
from src.training.eval import evaluate_model
from src.training.train import build_world_model, load_config, resolve_device
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    model = build_world_model(cfg)
    checkpoint_path = Path(cfg.checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path} at step {checkpoint.get('step')}")
    else:
        print(f"No checkpoint found at {checkpoint_path}; evaluating a fresh model.")
    dataloader = build_toy_dataloader(cfg.data, train=False)
    metrics = evaluate_model(model, dataloader, device, max_batches=int(cfg.eval.max_batches))
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
