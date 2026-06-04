#!/usr/bin/env python3
"""Grid search sanitize thresholds; rank by ADE and mean observed coverage."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.augmentation import run_augmentation
from utils.metrics import collect_metrics_dict, load_tracks
from utils.trajectory_metrics import compute_ade_fde, find_sportsmot_gt


def main():
    parser = argparse.ArgumentParser(description="Sanitize parameter grid search")
    parser.add_argument("--baseline", default="data/outputs/baseline_tracks.json")
    parser.add_argument("--output-root", default="data/outputs/sanitize_grid")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--sequence", default="video_1")
    parser.add_argument("--rules", default="velocity_cap")
    args = parser.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        meta = json.load(f).get("meta", {})
    fw = meta.get("frame_width")
    fh = meta.get("frame_height")
    gt = args.gt or find_sportsmot_gt(args.sequence)

    grid = []
    for max_w in (0.40, 0.35, 0.30):
        for y_min in (0.10, 0.15, 0.20):
            for max_p in (10, 12):
                grid.append(
                    {
                        "max_width_frac": max_w,
                        "y_min_court_frac": y_min,
                        "max_players": max_p,
                    }
                )

    rows = []
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(grid):
        tag = f"w{params['max_width_frac']}_y{params['y_min_court_frac']}_p{params['max_players']}"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        aug_path = out_dir / "augmented_tracks.json"

        run_augmentation(
            input_path=args.baseline,
            output_path=aug_path,
            frame_width=int(fw),
            frame_height=int(fh),
            rules=args.rules,
            sanitize=True,
            gap_fill=False,
            sanitize_max_width_frac=params["max_width_frac"],
            sanitize_y_min_court_frac=params["y_min_court_frac"],
            max_players=params["max_players"],
        )

        frames = load_tracks(aug_path)
        tm = collect_metrics_dict(frames, include_predicted=False)
        row = {**params, **tm, "tag": tag}
        if gt:
            row["ade"] = compute_ade_fde(str(aug_path), gt)["ade"]
            row["baseline_ade"] = compute_ade_fde(args.baseline, gt)["ade"]
            row["delta_ade"] = row["ade"] - row["baseline_ade"]
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r.get("delta_ade", 0),
            -r.get("mean_observed_per_frame", 0),
        )
    )

    csv_path = out_root / "sanitize_grid.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    best = rows[0] if rows else None
    if best:
        with open(out_root / "best_sanitize.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"Best sanitize params: {best['tag']} -> {out_root / 'best_sanitize.json'}")
    print(f"Wrote grid: {csv_path}")


if __name__ == "__main__":
    main()
