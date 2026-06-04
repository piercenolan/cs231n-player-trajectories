#!/usr/bin/env python3
"""Evaluate LSTM predicted tracks vs GT (ADE/FDE) with augmented baseline comparison."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import (
    lstm_out_dir,
    resolve_seed_gt_path,
    runs_dir,
    seed_augmented_tracks_path,
    trajectory_tensor_path,
)
from utils.lstm_dataset import VAL_SEED as DATASET_VAL_SEED
from utils.lstm_predict import load_checkpoint
from utils.lstm_dataset import load_tensor_file, norm_stats_from_meta, normalize_positions, denormalize_positions
from utils.trajectory_metrics import compute_ade_fde, forecast_min_frame_from_tracks


def teacher_forced_px_ade(model, seq, cfg, device):
    """Mean L2 px error predicting SAM positions (visible slots, teacher-forced)."""
    obs_len, pred_len = cfg["obs_len"], cfg["pred_len"]
    scale = seq.get("scale")
    if scale is None:
        scale = norm_stats_from_meta(seq["meta"])
    scale = np.asarray(scale, dtype=np.float32)
    pos_n = normalize_positions(seq["positions"], scale)
    pos = seq["positions"]
    vis = seq["visibility"]
    T = pos.shape[0]
    win = obs_len + pred_len
    errs, pers = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, T - win + 1):
            x = torch.from_numpy(pos_n[start : start + obs_len]).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).cpu().numpy()
            pred_px = denormalize_positions(pred, scale)
            gt = pos[start + obs_len : start + win]
            m = vis[start + obs_len : start + win]
            for k in range(pred_len):
                for p in range(pred.shape[1]):
                    if not m[k, p]:
                        continue
                    errs.append(float(np.hypot(pred_px[k, p, 0] - gt[k, p, 0], pred_px[k, p, 1] - gt[k, p, 1])))
                    pers.append(float(np.hypot(pos[start + obs_len - 1, p, 0] - gt[k, p, 0], pos[start + obs_len - 1, p, 1] - gt[k, p, 1])))
    return {
        "teacher_forced_ade_px": float(np.mean(errs)) if errs else float("nan"),
        "persistence_ade_px": float(np.mean(pers)) if pers else float("nan"),
        "num_pairs": len(errs),
    }


def main():
    parser = argparse.ArgumentParser(description="LSTM ADE/FDE evaluation vs SportsMOT GT")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--seed-id", default=DATASET_VAL_SEED)
    parser.add_argument(
        "--predicted",
        default=None,
        help="predicted_tracks.json (default: lstm/predicted_tracks.json)",
    )
    parser.add_argument(
        "--baseline-tracks",
        default=None,
        help="Augmented SAM tracks for baseline ADE",
    )
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--linear-baseline", action="store_true")
    args = parser.parse_args()

    out_dir = lstm_out_dir(args.dataset)
    pred_path = Path(args.predicted or out_dir / "predicted_tracks.json")
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"Missing {pred_path}. Run: py scripts/predict_lstm.py --dataset {args.dataset}"
        )

    baseline_path = Path(
        args.baseline_tracks
        or seed_augmented_tracks_path(args.dataset, args.seed_id)
    )
    if not baseline_path.is_file():
        baseline_path = runs_dir(args.dataset) / "ablations" / "sanitize_plus_velocity_cap" / "augmented_tracks.json"

    baseline_tracks = runs_dir(args.dataset, args.seed_id) / "baseline_tracks.json"
    gt_path = resolve_seed_gt_path(
        args.dataset,
        args.seed_id,
        baseline_tracks,
        align_if_missing=True,
    )

    cfg_path = out_dir / "train_config.json"
    obs_len = 8
    pred_len = 4
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        obs_len = int(cfg.get("obs_len", obs_len))
        pred_len = int(cfg.get("pred_len", pred_len))

    min_forecast_frame = forecast_min_frame_from_tracks(pred_path, obs_len)

    tf_metrics = {}
    manifest_path = out_dir / "predict_manifest.json"
    ckpt_path = out_dir / "checkpoint.pt"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        pred_ckpt = Path(man.get("lstm_checkpoint", ""))
        if pred_ckpt.is_file():
            ckpt_path = pred_ckpt
    if ckpt_path.is_file():
        device = torch.device("cpu")
        model, cfg = load_checkpoint(ckpt_path, device=device)
        val_tensor = trajectory_tensor_path(args.dataset, args.seed_id)
        if val_tensor.exists():
            seq = load_tensor_file(val_tensor)
            tf_metrics = teacher_forced_px_ade(model, seq, cfg, device)

    lstm_metrics = compute_ade_fde(
        pred_path, gt_path, max_distance=args.max_distance
    )
    aug_metrics = compute_ade_fde(
        baseline_path, gt_path, max_distance=args.max_distance
    )

    result = {
        "dataset": args.dataset,
        "seed_id": args.seed_id,
        "gt_path": str(gt_path),
        "obs_len": obs_len,
        "pred_len": pred_len,
        "lstm": lstm_metrics,
        "augmented_baseline": aug_metrics,
        "delta_ade": lstm_metrics["ade"] - aug_metrics["ade"],
        "delta_fde": lstm_metrics["fde"] - aug_metrics["fde"],
        "teacher_forced_on_sam": tf_metrics,
    }

    if min_forecast_frame is not None:
        result["forecast_horizon"] = {
            "min_frame": min_forecast_frame,
            "note": "ADE/FDE on frames where LSTM replaces SAM (index >= obs_len)",
        }
        result["lstm_forecast_only"] = compute_ade_fde(
            pred_path,
            gt_path,
            max_distance=args.max_distance,
            min_frame=min_forecast_frame,
        )
        result["augmented_forecast_only"] = compute_ade_fde(
            baseline_path,
            gt_path,
            max_distance=args.max_distance,
            min_frame=min_forecast_frame,
        )
        result["delta_ade_forecast_only"] = (
            result["lstm_forecast_only"]["ade"]
            - result["augmented_forecast_only"]["ade"]
        )

    if args.linear_baseline:
        lin_path = out_dir / "linear_baseline_tracks.json"
        if lin_path.is_file():
            lin_metrics = compute_ade_fde(
                lin_path, gt_path, max_distance=args.max_distance
            )
            result["linear_baseline"] = lin_metrics
            result["delta_ade_vs_linear"] = lstm_metrics["ade"] - lin_metrics["ade"]
            if min_forecast_frame is not None:
                result["linear_forecast_only"] = compute_ade_fde(
                    lin_path,
                    gt_path,
                    max_distance=args.max_distance,
                    min_frame=min_forecast_frame,
                )

    eval_path = out_dir / "lstm_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nLSTM EVALUATION (ADE / FDE vs GT)")
    print("=" * 50)
    print(f"Seed: {args.seed_id}")
    print(f"GT:   {gt_path}")
    print(f"\nAugmented SAM (baseline):")
    print(f"  ADE: {aug_metrics['ade']:.3f} px  FDE: {aug_metrics['fde']:.3f} px  matches: {aug_metrics['num_matches']}")
    print(f"\nLSTM predicted:")
    print(f"  ADE: {lstm_metrics['ade']:.3f} px  FDE: {lstm_metrics['fde']:.3f} px  matches: {lstm_metrics['num_matches']}")
    print(f"\nDelta ADE (LSTM - aug): {result['delta_ade']:+.3f} px")
    if tf_metrics:
        print(f"\nTeacher-forced on SAM tracks (val seed, visible slots):")
        print(f"  LSTM mean L2:        {tf_metrics['teacher_forced_ade_px']:.3f} px")
        print(f"  Persistence mean L2: {tf_metrics['persistence_ade_px']:.3f} px")
    if "lstm_forecast_only" in result:
        lf = result["lstm_forecast_only"]
        af = result["augmented_forecast_only"]
        print(f"\nForecast horizon only (frame >= {min_forecast_frame}):")
        print(f"  Augmented SAM: ADE {af['ade']:.3f} px")
        print(f"  LSTM:          ADE {lf['ade']:.3f} px")
        if args.linear_baseline and "linear_forecast_only" in result:
            linf = result["linear_forecast_only"]
            print(f"  Linear:        ADE {linf['ade']:.3f} px")
    print(f"Report: {eval_path}")


if __name__ == "__main__":
    main()
