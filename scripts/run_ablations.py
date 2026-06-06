#!/usr/bin/env python3
"""
Run per-rule and combo augmentation ablations with ADE/FDE when GT is available.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.augmentation import ALL_RULES, GAME_RULES, PHYSICAL_RULES, run_augmentation
from utils.metrics import collect_metrics_dict, compare_reports, load_tracks
from utils.trajectory_metrics import compute_ade_fde, find_sportsmot_gt


# (name, rules, sanitize, gap_fill, extra_kwargs)
ABLATION_CONFIGS = [
    ("baseline", None, False, False, {}),
    ("sanitize_only", "none", True, False, {}),
    ("sanitize_plus_velocity_cap", "velocity_cap", True, False, {}),
    ("sanitize_plus_velocity_cap_hull", "velocity_cap,hull_containment", True, False, {}),
    ("sanitize_plus_velocity_cap_spacing", "velocity_cap,spacing_push", True, False, {}),
    ("full", None, True, True, {}),
]
for rule in sorted(ALL_RULES):
    if rule == "reid_gap_fill":
        ABLATION_CONFIGS.append((rule, rule, True, True, {}))
    else:
        ABLATION_CONFIGS.append((rule, rule, True, False, {}))


def run_one_ablation(
    name,
    baseline_path,
    output_dir,
    frame_width,
    frame_height,
    rules=None,
    sanitize=True,
    gap_fill=True,
    expected_players=10,
    gt_path=None,
    gt_sequence="sportsmot_example",
    extra_kwargs=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_kwargs = extra_kwargs or {}

    metrics = {
        "ablation": name,
        "rules": rules,
        "sanitize": sanitize,
        "gap_fill": gap_fill,
        **extra_kwargs,
    }

    if name == "baseline":
        frames = load_tracks(baseline_path)
        aug_path = baseline_path
        metrics["track_metrics"] = collect_metrics_dict(
            frames, expected_players=expected_players, include_predicted=True
        )
    else:
        aug_path = output_dir / "augmented_tracks.json"
        level = "full"
        if rules and "," not in rules and rules != "none":
            if rules in PHYSICAL_RULES:
                level = "physical"
            elif rules in GAME_RULES:
                level = "game"
        run_augmentation(
            input_path=baseline_path,
            output_path=aug_path,
            frame_width=frame_width,
            frame_height=frame_height,
            level=level,
            rules=rules,
            sanitize=sanitize,
            gap_fill=gap_fill,
            expected_players=expected_players,
            gap_fill_debug=name == "reid_gap_fill",
            **{k: v for k, v in extra_kwargs.items() if k.startswith("sanitize_") or k in ("max_gap_frames", "max_players")},
        )
        frames = load_tracks(aug_path)
        metrics["track_metrics"] = collect_metrics_dict(
            frames, expected_players=expected_players, include_predicted=False
        )
        pred = sum(
            1
            for fr in frames
            for p in fr.get("players", [])
            if p.get("predicted")
        )
        metrics["predicted_points"] = pred

    compare = compare_reports(
        baseline_path,
        str(aug_path),
        expected_players=expected_players,
        include_predicted_in_aug=False,
    )
    metrics["comparison"] = compare

    gt = gt_path or find_sportsmot_gt(gt_sequence)
    if gt:
        try:
            metrics["ade_fde"] = compute_ade_fde(str(aug_path), gt)
            metrics["ade_fde_baseline"] = compute_ade_fde(str(baseline_path), gt)
        except Exception as exc:
            metrics["ade_fde_error"] = str(exc)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def pick_best_by_ade(summary_rows):
    """Return ablation name with lowest ADE (excluding baseline)."""
    ranked = []
    for row in summary_rows:
        ade = row.get("ade")
        if ade == "" or ade is None:
            continue
        try:
            ranked.append((float(ade), row["ablation"]))
        except (TypeError, ValueError):
            continue
    if not ranked:
        return None
    ranked.sort(key=lambda t: t[0])
    return ranked[0][1]


def main():
    parser = argparse.ArgumentParser(description="Run augmentation ablations")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument("--expected-players", type=int, default=10)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--sequence", default=None, help="GT lookup key (default: --dataset)")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    from utils.datasets import ablations_dir, baseline_tracks_path, find_gt_path

    sequence = args.sequence or args.dataset
    baseline_path = Path(args.baseline or baseline_tracks_path(args.dataset))
    output_root = Path(args.output_root or ablations_dir(args.dataset))
    if args.gt is None:
        gt_default = find_gt_path(args.dataset)
        args.gt = str(gt_default) if gt_default else None
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    with open(baseline_path, encoding="utf-8") as f:
        meta = json.load(f).get("meta", {})
    frame_width = args.frame_width or meta.get("frame_width")
    frame_height = args.frame_height or meta.get("frame_height")
    if not frame_width or not frame_height:
        raise ValueError("frame_width/frame_height required")

    only = {x.strip() for x in args.only.split(",")} if args.only else None
    summary_rows = []

    for name, rules, sanitize, gap_fill, extra in ABLATION_CONFIGS:
        if only and name not in only:
            continue
        print(f"\n{'=' * 60}\nAblation: {name}\n{'=' * 60}")
        out_dir = output_root / name
        m = run_one_ablation(
            name=name,
            baseline_path=str(baseline_path),
            output_dir=out_dir,
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            rules=rules,
            sanitize=sanitize,
            gap_fill=gap_fill,
            expected_players=args.expected_players,
            gt_path=args.gt,
            gt_sequence=sequence,
            extra_kwargs=extra,
        )
        tm = m.get("track_metrics", {})
        delta = m.get("comparison", {}).get("delta", {})
        ade = m.get("ade_fde", {}).get("ade", "")
        base_ade = m.get("ade_fde_baseline", {}).get("ade", "")
        row = {
            "ablation": name,
            "mean_observed": tm.get("mean_observed_per_frame", ""),
            "delta_id_switches": delta.get("total_id_switches", ""),
            "delta_mean_streak": delta.get("mean_track_streak", ""),
            "delta_mean_displacement": delta.get("mean_displacement", ""),
            "ade": ade,
            "baseline_ade": base_ade,
            "delta_ade": (ade - base_ade) if ade != "" and base_ade != "" else "",
            "predicted_points": m.get("predicted_points", ""),
        }
        summary_rows.append(row)

    summary_path = output_root / "ablation_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nWrote summary: {summary_path}")

        best = pick_best_by_ade(summary_rows)
        rec = {
            "recommended_ablation_ade_proxy": best,
            "recommended_ablation_lstm_v1": "sanitize_plus_velocity_cap",
            "selection_metric_ade": "lowest_ade",
            "selection_metric_lstm": "best_smoothness_without_gap_fill",
            "lstm_v1_gap_fill": False,
        }
        rec_path = output_root / "recommended_config.json"
        if rec_path.exists():
            with open(rec_path, encoding="utf-8") as f:
                prev = json.load(f)
            rec.update({k: v for k, v in prev.items() if k not in rec})
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        if best:
            print(f"Lowest ADE (proxy): {best} -> {rec_path}")


if __name__ == "__main__":
    main()
