"""Run the minimal single-law PhyWorld validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.single_law import train_single_law


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/single_law_phyworld.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    metrics = train_single_law(args.config, max_steps=args.max_steps)
    print(f"Single-law validation metrics: {metrics}")


if __name__ == "__main__":
    main()
