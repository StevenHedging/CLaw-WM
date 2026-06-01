"""Save a rollout comparison figure for toy physics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_toy_dataset
from src.training.checkpoints import load_checkpoint
from src.training.train import build_world_model, load_config, resolve_device
from src.utils.seed import set_seed
from src.utils.visualization import save_rollout_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/rollout_comparison.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    model = build_world_model(cfg).to(device)
    checkpoint_path = Path(cfg.checkpoint_path)
    if checkpoint_path.exists():
        load_checkpoint(model, checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}; visualizing a fresh model.")

    dataset = build_toy_dataset(cfg.data, train=False)
    episode = dataset.episodes[0]
    context_length = int(cfg.data.context_length)
    horizon = min(int(cfg.eval.rollout_horizon), episode.frames.shape[0] - context_length)
    context = torch.from_numpy(episode.frames[:context_length]).unsqueeze(0).to(device)
    targets = torch.from_numpy(episode.frames[context_length : context_length + horizon])

    model.eval()
    model.selector.reset()
    with torch.no_grad():
        rollout = model.rollout(context, horizon=horizon)
    predictions = rollout["predictions"][0].detach().cpu()
    save_rollout_comparison(predictions, targets, args.output, max_frames=horizon)
    print(f"Saved rollout comparison to {args.output}")


if __name__ == "__main__":
    main()
