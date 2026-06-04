#!/usr/bin/env python3
"""Compare A1 rollout with fixed vs autoregressive rule features."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import lstm_out_dir, resolve_seed_gt_path, runs_dir
from utils.lstm_dataset import load_tensor_file
from utils.lstm_predict import load_checkpoint, positions_to_tracks, rollout_positions, save_tracks
from utils.trajectory_metrics import compute_ade_fde, forecast_min_frame_from_tracks


def eval_one(ckpt, tensor_path, gt_path, device, autoregressive):
    model, cfg = load_checkpoint(ckpt, device=device)
    seq = load_tensor_file(tensor_path)
    scale = cfg.get("scale")
    if scale is None:
        import numpy as np

        scale = np.asarray(seq["scale"]).tolist()
    cfg = {**cfg, "scale": scale}
    pred_pos, _ = rollout_positions(model, seq, cfg, device=device, autoregressive=autoregressive)
    tracks = positions_to_tracks(
        seq, pred_pos, meta_extra={"autoregressive": autoregressive}
    )
    tag = "autoregressive" if autoregressive else "fixed_rules"
    out = ckpt.parent / f"_ar_compare_{tensor_path.parent.name}_{tag}.json"
    save_tracks(out, tracks)
    min_f = forecast_min_frame_from_tracks(out, cfg["obs_len"])
    m = compute_ade_fde(out, gt_path, min_frame=min_f) if min_f else compute_ade_fde(out, gt_path)
    return {"ade_forecast": m["ade"], "fde_forecast": m["fde"], "tracks": str(out)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = lstm_out_dir(args.dataset) / "lstm_rule_features" / "checkpoint.pt"
    seeds_root = runs_dir(args.dataset) / "seeds"
    rows = []
    for d in sorted(seeds_root.iterdir()):
        if not (d / "trajectory_tensors.json").is_file():
            continue
        sid = d.name
        tp = d / "trajectory_tensors.json"
        gt = resolve_seed_gt_path(
            args.dataset, sid, d / "baseline_tracks.json", align_if_missing=True
        )
        fixed = eval_one(ckpt, tp, gt, device, False)
        ar = eval_one(ckpt, tp, gt, device, True)
        rows.append(
            {
                "seed_id": sid,
                "fixed_rules_ade": fixed["ade_forecast"],
                "autoregressive_ade": ar["ade_forecast"],
                "delta_ar_minus_fixed": ar["ade_forecast"] - fixed["ade_forecast"],
            }
        )
        print(
            f"{sid}: fixed={fixed['ade_forecast']:.3f}  "
            f"autoregressive={ar['ade_forecast']:.3f}"
        )

    out = lstm_out_dir(args.dataset) / "lstm_autoregressive_compare.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)
    csv_path = lstm_out_dir(args.dataset) / "lstm_autoregressive_compare.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
