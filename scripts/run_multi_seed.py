#!/usr/bin/env python3
"""
Multi-seed validation: augmentation + metrics per seed, then aggregate ADE.

Expects baseline tracks at data/outputs/seeds/{seed_id}/baseline_tracks.json
or bootstraps frame windows from a single baseline when --bootstrap-from is set.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablations import run_one_ablation
from utils.trajectory_metrics import find_sportsmot_gt


DEFAULT_SEEDS = [
    ("offset_0s", 0.0),
    ("offset_10s", 10.0),
    ("offset_20s", 20.0),
]


def bootstrap_seed_baselines(source_baseline, seeds_root, num_frames=45):
    """
    Create pseudo-seeds by slicing one baseline into contiguous windows.
    Used when SAM3 multi-offset tracks are not yet available.
    """
    with open(source_baseline, encoding="utf-8") as f:
        data = json.load(f)
    frames = sorted(data.get("frames", []), key=lambda fr: int(fr["frame_number"]))
    meta = dict(data.get("meta") or {})
    chunk = max(1, len(frames) // 3)

    created = []
    for i, (seed_id, _offset) in enumerate(DEFAULT_SEEDS):
        start = i * chunk
        end = min(len(frames), start + chunk)
        if start >= len(frames):
            break
        subset = frames[start:end]
        for j, fr in enumerate(subset, start=1):
            fr["frame_number"] = j
        seed_dir = Path(seeds_root) / seed_id
        seed_dir.mkdir(parents=True, exist_ok=True)
        out = seed_dir / "baseline_tracks.json"
        payload = {
            "meta": {
                **meta,
                "seed_id": seed_id,
                "bootstrap": True,
                "source_window": [start, end],
            },
            "frames": subset,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        created.append((seed_id, out))
    return created


def main():
    parser = argparse.ArgumentParser(description="Multi-seed augmentation validation")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument(
        "--seeds-root",
        default=None,
        help="Root directory for per-seed outputs",
    )
    parser.add_argument(
        "--bootstrap-from",
        default=None,
        help="Bootstrap pseudo-seeds from one baseline JSON",
    )
    parser.add_argument(
        "--recommended-config",
        default=None,
        help="JSON with recommended_ablation name",
    )
    parser.add_argument("--gt", default=None)
    parser.add_argument("--sequence", default=None)
    args = parser.parse_args()

    from utils.datasets import ablations_dir, baseline_tracks_path, find_gt_path, runs_dir

    sequence = args.sequence or args.dataset
    seeds_root = Path(args.seeds_root or runs_dir(args.dataset) / "seeds")
    gt = args.gt or find_gt_path(args.dataset) or find_sportsmot_gt(sequence)
    rec_path = Path(
        args.recommended_config or ablations_dir(args.dataset) / "recommended_config.json"
    )

    if args.bootstrap_from:
        seed_list = bootstrap_seed_baselines(args.bootstrap_from, seeds_root)
        print(f"Bootstrapped {len(seed_list)} pseudo-seeds from {args.bootstrap_from}")
    else:
        seed_list = []
        for seed_id, _ in DEFAULT_SEEDS:
            p = seeds_root / seed_id / "baseline_tracks.json"
            if p.exists():
                seed_list.append((seed_id, p))
        if not seed_list:
            raise FileNotFoundError(
                f"No seed baselines under {seeds_root}. "
                "Run SAM3 with --seed-id or pass --bootstrap-from."
            )

    rec_name = "sanitize_plus_velocity_cap"
    if rec_path.exists():
        with open(rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        rec_name = rec.get("recommended_ablation_lstm_v1") or rec.get(
            "recommended_ablation", rec_name
        )

    # Map recommended ablation to rules/gap_fill
    from scripts.run_ablations import ABLATION_CONFIGS

    cfg = {c[0]: c for c in ABLATION_CONFIGS}.get(rec_name)
    if not cfg:
        cfg = ("sanitize_plus_velocity_cap", "velocity_cap", True, False, {})

    all_metrics = []
    for seed_id, baseline_path in seed_list:
        with open(baseline_path, encoding="utf-8") as f:
            meta = json.load(f).get("meta", {})
        fw = meta.get("frame_width")
        fh = meta.get("frame_height")
        out_dir = seeds_root / seed_id / rec_name
        print(f"\nSeed {seed_id} -> {rec_name}")
        m = run_one_ablation(
            name=rec_name,
            baseline_path=str(baseline_path),
            output_dir=out_dir,
            frame_width=int(fw),
            frame_height=int(fh),
            rules=cfg[1],
            sanitize=cfg[2],
            gap_fill=cfg[3],
            gt_path=gt,
            gt_sequence=sequence,
            extra_kwargs=cfg[4],
        )
        m["seed_id"] = seed_id
        all_metrics.append(m)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    # Aggregate ADE
    ades = [m["ade_fde"]["ade"] for m in all_metrics if "ade_fde" in m]
    summary = {
        "recommended_ablation": rec_name,
        "num_seeds": len(all_metrics),
        "ade_mean": sum(ades) / len(ades) if ades else None,
        "ade_values": ades,
        "seeds": [m["seed_id"] for m in all_metrics],
    }
    if len(ades) > 1:
        import numpy as np

        summary["ade_std"] = float(np.std(ades))

    out_summary = seeds_root / "multi_seed_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMulti-seed summary: {out_summary}")
    if summary.get("ade_mean") is not None:
        std = summary.get("ade_std", 0)
        print(f"ADE: {summary['ade_mean']:.3f} ± {std:.3f} over {len(ades)} seeds")


if __name__ == "__main__":
    main()
