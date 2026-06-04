#!/usr/bin/env python3
"""Run LSTM rollout and optional post-refine (A2)."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import lstm_out_dir, seed_augmented_tracks_path, trajectory_tensor_path
from utils.lstm_dataset import VAL_SEED as DATASET_VAL_SEED
from utils.lstm_dataset import load_tensor_file, resolve_tensor_paths
from utils.lstm_predict import (
    linear_extrapolation_positions,
    load_checkpoint,
    post_refine_tracks,
    positions_to_tracks,
    rollout_positions,
    save_tracks,
)


def main():
    parser = argparse.ArgumentParser(description="LSTM trajectory prediction rollout")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--seed-id", default=DATASET_VAL_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stitch", choices=("last", "average"), default="last")
    parser.add_argument("--linear-baseline", action="store_true")
    parser.add_argument(
        "--post-refine",
        default=None,
        choices=("physical", "game", "full"),
        help="Apply augmentation rules to predictions (A2)",
    )
    parser.add_argument(
        "--out-name",
        default="predicted_tracks.json",
        help="Output filename under lstm/ or lstm_<variant>/",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None:
        for cand in [
            lstm_out_dir(args.dataset) / "lstm_plain" / "checkpoint.pt",
            lstm_out_dir(args.dataset) / "checkpoint.pt",
        ]:
            if cand.is_file():
                ckpt_path = cand
                break
    if not ckpt_path or not ckpt_path.is_file():
        raise FileNotFoundError("Missing checkpoint. Run train_lstm.py first.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_checkpoint(ckpt_path, device=device)

    if args.tensor:
        tensor_path = Path(args.tensor)
    else:
        tensor_path = trajectory_tensor_path(args.dataset, seed_id=args.seed_id)
        if not tensor_path.exists():
            paths = resolve_tensor_paths(args.dataset, "single")
            tensor_path = paths[0]

    seq = load_tensor_file(tensor_path)
    cfg = {**cfg, "scale": cfg.get("scale") or seq["scale"].tolist()}

    pred_pos, updated = rollout_positions(model, seq, cfg, device=device, stitch=args.stitch)
    tracks = positions_to_tracks(
        seq,
        pred_pos,
        meta_extra={
            "lstm_checkpoint": str(ckpt_path),
            "model": cfg.get("model", "plain"),
            "tensor_source": str(tensor_path),
        },
    )

    if args.post_refine:
        tracks = post_refine_tracks(tracks, rules_preset=args.post_refine)

    out_dir = ckpt_path.parent
    pred_path = out_dir / args.out_name
    save_tracks(pred_path, tracks)

    manifest = {
        "predicted_tracks": str(pred_path),
        "lstm_checkpoint": str(ckpt_path),
        "tensor": str(tensor_path),
        "seed_id": seq["seed_id"],
        "post_refine": args.post_refine,
    }

    if args.linear_baseline:
        lin_pos, _ = linear_extrapolation_positions(seq, cfg)
        lin_tracks = positions_to_tracks(
            seq, lin_pos, meta_extra={"baseline": "linear_velocity"}
        )
        lin_path = out_dir / "linear_baseline_tracks.json"
        save_tracks(lin_path, lin_tracks)
        manifest["linear_baseline_tracks"] = str(lin_path)

    with open(out_dir / "predict_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Predicted tracks: {pred_path}")


if __name__ == "__main__":
    main()
