#!/usr/bin/env python3
"""Diagnose per-seed LSTM inputs: visibility, validation gate, GT alignment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import runs_dir, seed_augmented_tracks_path
from utils.lstm_dataset import load_tensor_file


def seed_diagnostics(dataset, seed_id):
    seed_dir = runs_dir(dataset) / "seeds" / seed_id
    row = {"seed_id": seed_id}
    tp = seed_dir / "trajectory_tensors.json"
    if not tp.is_file():
        row["error"] = "missing trajectory_tensors.json"
        return row

    seq = load_tensor_file(tp)
    vis = seq["visibility"]
    pos = seq["positions"]
    T, P = vis.shape
    row["T"] = T
    row["P"] = P
    row["global_visibility"] = float(vis.mean())
    row["frames_all_invisible"] = int((~vis.any(axis=1)).sum())
    row["slots_mostly_empty"] = float((vis.mean(axis=0) < 0.3).mean())
    row["position_nan"] = bool(np.isnan(pos).any())
    row["position_max"] = float(np.nanmax(pos)) if pos.size else 0.0

    val_path = seed_dir / "trajectory_validation.json"
    if val_path.is_file():
        with open(val_path, encoding="utf-8") as f:
            v = json.load(f)
        row["export_passed"] = v.get("passed")
        row["export_global_vis"] = v.get("global_visibility")

    gt_path = seed_dir / "gt_aligned.json"
    row["has_gt_aligned"] = gt_path.is_file()
    if gt_path.is_file():
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        row["gt_frames"] = len(gt.get("frames", []))

    aug = seed_augmented_tracks_path(dataset, seed_id)
    row["has_augmented"] = aug.is_file()
    bl = seed_dir / "baseline_tracks.json"
    row["has_baseline"] = bl.is_file()
    return row


def main():
    parser = argparse.ArgumentParser(description="Diagnose LSTM seed tensors")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds_root = runs_dir(args.dataset) / "seeds"
    rows = []
    for d in sorted(seeds_root.iterdir()):
        if d.is_dir() and (d / "trajectory_tensors.json").is_file():
            rows.append(seed_diagnostics(args.dataset, d.name))

    out_path = Path(args.out or (runs_dir(args.dataset) / "lstm" / "seed_diagnosis.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": rows}, f, indent=2)

    print(f"Wrote {out_path} ({len(rows)} seeds)")
    for r in rows:
        flag = ""
        if r.get("global_visibility", 1) < 0.7:
            flag = " LOW_VIS"
        if r.get("frames_all_invisible", 0) > 0:
            flag += " EMPTY_FRAMES"
        print(
            f"  {r['seed_id']}: vis={r.get('global_visibility', 0):.2f} "
            f"gt={r.get('has_gt_aligned')} export={r.get('export_passed', '?')}{flag}"
        )


if __name__ == "__main__":
    main()
