#!/usr/bin/env python3
"""
Aggregate metrics.json from multi-seed / multi-ablation runs (mean ± std).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics_files(root):
    """Find all metrics.json under root."""
    root = Path(root)
    files = list(root.rglob("metrics.json"))
    return files


def aggregate_metrics(files, group_key=None):
    """
    group_key: callable(Path) -> str grouping label (e.g. ablation name).
    Returns {group: {metric: {mean, std, n}}}.
    """
    groups = {}
    for path in files:
        label = group_key(path) if group_key else path.parent.name
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        flat = {}
        if "track_metrics" in data:
            flat.update(data["track_metrics"])
        elif "mean_players_per_frame" in data:
            flat.update(data)
        if "comparison" in data and "delta" in data["comparison"]:
            flat.update({f"delta_{k}": v for k, v in data["comparison"]["delta"].items()})
        if "ade_fde" in data:
            flat["ade"] = data["ade_fde"].get("ade")
            flat["fde"] = data["ade_fde"].get("fde")

        groups.setdefault(label, []).append(flat)

    summary = {}
    for label, rows in groups.items():
        keys = set()
        for row in rows:
            keys.update(row.keys())
        summary[label] = {}
        for key in sorted(keys):
            vals = [r[key] for r in rows if key in r and r[key] is not None]
            vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            summary[label][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": int(len(arr)),
            }
    return summary


def _ablation_from_path(path):
    parts = path.parts
    if "ablations" in parts:
        idx = parts.index("ablations")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "seeds" in parts:
        idx = parts.index("seeds")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}/{parts[idx + 2]}"
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics")
    parser.add_argument(
        "--root",
        default="data/outputs",
        help="Search root for metrics.json files",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/aggregated_metrics.json",
        help="Write aggregated summary JSON",
    )
    parser.add_argument(
        "--pattern",
        default="ablations",
        help="Only include paths containing this substring (empty = all)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = load_metrics_files(root)
    if args.pattern:
        files = [f for f in files if args.pattern in str(f)]

    if not files:
        print(f"No metrics.json found under {root}")
        return

    summary = aggregate_metrics(files, group_key=_ablation_from_path)

    print()
    print("AGGREGATED METRICS (mean ± std)")
    print("=" * 50)
    for label, metrics in sorted(summary.items()):
        print(f"\n[{label}]")
        for key, stats in sorted(metrics.items()):
            print(f"  {key:<28} {stats['mean']:8.3f} ± {stats['std']:.3f}  (n={stats['n']})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
