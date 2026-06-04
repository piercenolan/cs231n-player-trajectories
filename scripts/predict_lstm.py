#!/usr/bin/env python3
"""Run LSTM rollout on validation tensors and write predicted_tracks.json."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import lstm_out_dir, trajectory_tensor_path
from utils.lstm_dataset import VAL_SEED as DATASET_VAL_SEED
from utils.lstm_dataset import load_tensor_file, resolve_tensor_paths
from utils.lstm_predict import (
    linear_extrapolation_positions,
    load_checkpoint,
    positions_to_tracks,
    rollout_positions,
    save_tracks,
)


def main():
    parser = argparse.ArgumentParser(description="LSTM trajectory prediction rollout")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint.pt (default: data/runs/.../lstm/checkpoint.pt)",
    )
    parser.add_argument(
        "--tensor",
        default=None,
        help="Tensor JSON to predict on (default: val seed offset_0s)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--stitch", choices=("last", "average"), default="last")
    parser.add_argument("--linear-baseline", action="store_true", help="Also write linear_baseline_tracks.json")
    args = parser.parse_args()

    ckpt_path = Path(
        args.checkpoint or lstm_out_dir(args.dataset) / "checkpoint.pt"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Run train_lstm.py first.")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, cfg = load_checkpoint(ckpt_path, device=device)

    if args.tensor:
        tensor_path = Path(args.tensor)
    else:
        val_path = trajectory_tensor_path(args.dataset, seed_id=DATASET_VAL_SEED)
        if not val_path.exists():
            paths = resolve_tensor_paths(args.dataset, "single")
            val_path = paths[0] if paths else val_path
        tensor_path = val_path

    seq = load_tensor_file(tensor_path)
    cfg = {**cfg, "scale": cfg.get("scale") or seq["scale"].tolist()}

    pred_pos, updated = rollout_positions(model, seq, cfg, device=device, stitch=args.stitch)
    tracks = positions_to_tracks(
        seq,
        pred_pos,
        meta_extra={
            "lstm_checkpoint": str(ckpt_path),
            "tensor_source": str(tensor_path),
            "obs_len": cfg["obs_len"],
            "pred_len": cfg["pred_len"],
            "stitch": args.stitch,
        },
    )

    out_dir = lstm_out_dir(args.dataset)
    pred_path = out_dir / "predicted_tracks.json"
    save_tracks(pred_path, tracks)

    manifest = {
        "predicted_tracks": str(pred_path),
        "lstm_checkpoint": str(ckpt_path),
        "tensor": str(tensor_path),
        "seed_id": seq["seed_id"],
        "forecast_frames": int(updated.sum(axis=1).astype(bool).sum()),
    }

    if args.linear_baseline:
        lin_pos, _ = linear_extrapolation_positions(seq, cfg)
        lin_tracks = positions_to_tracks(
            seq,
            lin_pos,
            meta_extra={"baseline": "linear_velocity", "tensor_source": str(tensor_path)},
        )
        lin_path = out_dir / "linear_baseline_tracks.json"
        save_tracks(lin_path, lin_tracks)
        manifest["linear_baseline_tracks"] = str(lin_path)

    manifest_path = out_dir / "predict_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Predicted tracks: {pred_path}")
    print(f"Seed: {seq['seed_id']} | tensor: {tensor_path}")


if __name__ == "__main__":
    main()
