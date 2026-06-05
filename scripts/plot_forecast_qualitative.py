#!/usr/bin/env python3
"""Generate qualitative LSTM forecast overlays (predicted vs ground truth)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import lstm_out_dir, runs_dir, trajectory_tensor_path
from utils.forecast_visualize import (
    create_forecast_summary_figure,
    save_forecast_frame,
)


def _default_predicted_path(dataset: str, seed_id: str, variant_dir: str) -> Path:
    lstm_dir = lstm_out_dir(dataset) / variant_dir
    candidates = [
        lstm_dir / f"predicted_{seed_id}_A1_residual.json",
        lstm_dir / f"predicted_{seed_id}.json",
        lstm_dir / "predicted_tracks.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No predicted tracks for {dataset}/{seed_id} under {lstm_dir}. "
        "Run eval_lstm_ablations.py or predict_lstm.py first."
    )


def _load_obs_pred_len(dataset: str, variant_dir: str) -> tuple[int, int]:
    cfg_path = lstm_out_dir(dataset) / variant_dir / "train_config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("obs_len", 8)), int(cfg.get("pred_len", 4))
    return 8, 4


def main():
    parser = argparse.ArgumentParser(
        description="Overlay LSTM predicted trajectories with ground truth on frames."
    )
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--seed-id", default="offset_0s")
    parser.add_argument("--tensor", default=None, help="trajectory_tensors.json path")
    parser.add_argument("--predicted", default=None, help="predicted tracks JSON")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="LSTM checkpoint (.pt); default rollout from variant-dir/checkpoint.pt",
    )
    parser.add_argument(
        "--variant-dir",
        default="lstm_rule_features_residual",
        help="LSTM output subdir under lstm/",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--frames",
        default=None,
        help="JPEG frames directory (default: data/runs/{dataset}/frames)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: figures/forecast_qualitative.png)",
    )
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument(
        "--window-starts",
        default=None,
        help="Comma-separated sliding-window start indices (overrides --n-windows)",
    )
    parser.add_argument("--window-start", type=int, default=None, help="Single-window mode")
    parser.add_argument("--no-linear", action="store_true")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    tensor_path = Path(args.tensor) if args.tensor else trajectory_tensor_path(args.dataset, args.seed_id)
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Missing tensor file: {tensor_path}")

    lstm_dir = lstm_out_dir(args.dataset) / args.variant_dir
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else lstm_dir / "checkpoint.pt"
    if not checkpoint_path.is_file() and not args.predicted:
        predicted_path = _default_predicted_path(args.dataset, args.seed_id, args.variant_dir)
    else:
        predicted_path = Path(args.predicted) if args.predicted else None
        if predicted_path and not predicted_path.is_file():
            raise FileNotFoundError(f"Missing predicted tracks: {predicted_path}")
        if not checkpoint_path.is_file() and predicted_path is None:
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    frames_dir = Path(args.frames) if args.frames else runs_dir(args.dataset) / "frames"
    if not frames_dir.is_dir():
        frames_dir = None

    obs_len, pred_len = _load_obs_pred_len(args.dataset, args.variant_dir)
    show_linear = not args.no_linear

    figures_dir = runs_dir(args.dataset) / "figures"
    if args.window_start is not None:
        out = Path(args.output or figures_dir / f"forecast_window_{args.window_start}.jpg")
        save_forecast_frame(
            args.dataset,
            tensor_path,
            out,
            window_start=args.window_start,
            predicted_path=predicted_path,
            checkpoint_path=checkpoint_path if checkpoint_path.is_file() else None,
            device=device,
            obs_len=obs_len,
            pred_len=pred_len,
            frames_dir=frames_dir,
            seed_id=args.seed_id,
            show_linear=show_linear,
        )
        return

    window_starts = None
    if args.window_starts:
        window_starts = [int(x.strip()) for x in args.window_starts.split(",") if x.strip()]

    out = Path(args.output or figures_dir / "forecast_qualitative.png")
    create_forecast_summary_figure(
        args.dataset,
        tensor_path,
        out,
        predicted_path=predicted_path,
        checkpoint_path=checkpoint_path if checkpoint_path.is_file() else None,
        device=device,
        obs_len=obs_len,
        pred_len=pred_len,
        window_starts=window_starts,
        n_windows=args.n_windows,
        frames_dir=frames_dir,
        seed_id=args.seed_id,
        show_linear=show_linear,
        title=args.title,
    )


if __name__ == "__main__":
    main()
