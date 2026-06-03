"""Diagnose whether single-law failures come from dynamics or rendering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.single_law import build_phyworld_loader, build_single_law_model
from src.training.train import resolve_device


def _to_hwc(frame: torch.Tensor) -> torch.Tensor:
    return frame.detach().cpu().clamp(0, 1).permute(1, 2, 0)


def _foreground_center(frame: torch.Tensor) -> torch.Tensor:
    """Estimate center of non-white foreground pixels in normalized image coords."""

    gray = frame.mean(dim=1)
    mask = (gray < 0.95).float()
    batch, height, width = mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=frame.device),
        torch.arange(width, device=frame.device),
        indexing="ij",
    )
    weights = mask.flatten(start_dim=1).sum(dim=1).clamp_min(1e-6)
    cx = (mask * xx).flatten(start_dim=1).sum(dim=1) / weights / max(width - 1, 1)
    cy = (mask * yy).flatten(start_dim=1).sum(dim=1) / weights / max(height - 1, 1)
    return torch.stack([cx, cy], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/single_law_phyworld.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/single_law_phyworld/diagnosis.png")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    checkpoint_path = Path(args.checkpoint or cfg.checkpoint_path)
    device = resolve_device(str(cfg.device))
    model = build_single_law_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = build_phyworld_loader(cfg, train=False)
    batch = next(iter(loader))
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    with torch.no_grad():
        output = model(batch["context"], target_frame=batch["target"], force_head_id=0)
        pred_from_model = output["prediction"]
        pred_from_true_state = model.renderer(batch["next_state"])
        pred_from_encoded_state = model.renderer(output["z_next"])

    z_t_error = torch.linalg.vector_norm(output["z_t"] - batch["state"], dim=-1)
    z_next_error = torch.linalg.vector_norm(output["z_next"] - batch["next_state"], dim=-1)
    position_error = torch.linalg.vector_norm(output["z_next"][:, :2] - batch["next_state"][:, :2], dim=-1)
    velocity_error = torch.linalg.vector_norm(output["z_next"][:, 2:] - batch["next_state"][:, 2:], dim=-1)

    target_center = _foreground_center(batch["target"])
    model_center = _foreground_center(pred_from_model)
    oracle_render_center = _foreground_center(pred_from_true_state)
    model_center_error = torch.linalg.vector_norm(model_center - target_center, dim=-1)
    oracle_render_center_error = torch.linalg.vector_norm(oracle_render_center - target_center, dim=-1)

    print("state_error_mean", float(z_t_error.mean().cpu()))
    print("next_state_error_mean", float(z_next_error.mean().cpu()))
    print("position_error_mean", float(position_error.mean().cpu()))
    print("velocity_error_mean", float(velocity_error.mean().cpu()))
    print("model_image_mse", float(F.mse_loss(pred_from_model, batch["target"]).cpu()))
    print("oracle_state_render_mse", float(F.mse_loss(pred_from_true_state, batch["target"]).cpu()))
    print("model_center_error_mean", float(model_center_error.mean().cpu()))
    print("oracle_render_center_error_mean", float(oracle_render_center_error.mean().cpu()))
    print("example_true_next_state", batch["next_state"][0].detach().cpu().tolist())
    print("example_pred_next_state", output["z_next"][0].detach().cpu().tolist())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(6, batch["target"].shape[0])
    fig, axes = plt.subplots(3, columns, figsize=(1.8 * columns, 5.4), squeeze=False)
    for idx in range(columns):
        axes[0, idx].imshow(_to_hwc(batch["target"][idx]))
        axes[0, idx].set_title(f"GT {idx}", fontsize=9)
        axes[1, idx].imshow(_to_hwc(pred_from_model[idx]))
        axes[1, idx].set_title("E+F+R", fontsize=9)
        axes[2, idx].imshow(_to_hwc(pred_from_true_state[idx]))
        axes[2, idx].set_title("true z+R", fontsize=9)
        for row in range(3):
            axes[row, idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
