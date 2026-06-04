#!/usr/bin/env python3
"""Export and validate LSTM trajectory tensors for run root and/or all seeds."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import (
    LSTM_ABLATION,
    resolve_augmented_tracks_path,
    runs_dir,
    seed_augmented_tracks_path,
    trajectory_tensor_path,
)
from utils.seed_schedule import list_seed_entries
from utils.trajectory_export import export_trajectories, validate_export


def export_one(
    tracks_path,
    output_path,
    max_players,
    validate,
    min_vis,
    max_empty,
    seed_id=None,
):
    tracks_path = Path(tracks_path)
    output_path = Path(output_path)
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)

    export = export_trajectories(str(tracks_path), output_path, max_players=max_players)
    if seed_id:
        export["meta"]["seed_id"] = seed_id
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
    report = None
    if validate:
        passed, report = validate_export(
            export,
            min_global_visibility=min_vis,
            max_slot_empty_frac=max_empty,
        )
        report_path = output_path.with_name("trajectory_validation.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if not passed:
            raise RuntimeError(f"Validation failed for {output_path}: {report}")
    return output_path, report


def main():
    parser = argparse.ArgumentParser(description="Export LSTM tensors (root + seeds)")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Export per-seed tensors under seeds/{seed_id}/",
    )
    parser.add_argument(
        "--root",
        action="store_true",
        help="Export run-root trajectory_tensors.json (offset_0s canonical)",
    )
    parser.add_argument("--max-players", type=int, default=10)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", action="store_false", dest="validate")
    parser.add_argument("--min-global-visibility", type=float, default=0.7)
    parser.add_argument(
        "--max-slot-empty-frac",
        type=float,
        default=0.60,
        help="Per-slot empty fraction cap (0.30 strict; 0.60 for sparse 10th slots on seeds)",
    )
    args = parser.parse_args()

    do_root = args.root or not args.all_seeds
    do_seeds = args.all_seeds or not args.root
    if not do_root and not do_seeds:
        do_root = do_seeds = True

    exported = []
    failed = []

    if do_root:
        try:
            tracks = resolve_augmented_tracks_path(args.dataset)
            out = trajectory_tensor_path(args.dataset, seed_id=None)
            p, _ = export_one(
                tracks,
                out,
                args.max_players,
                args.validate,
                args.min_global_visibility,
                args.max_slot_empty_frac,
            )
            exported.append(("root", p))
        except Exception as e:
            failed.append(("root", str(e)))

    if do_seeds:
        seed_ids = [s[0] for s in list_seed_entries(args.dataset)]
        if not seed_ids:
            from utils.datasets import SEED_OFFSETS
            seed_ids = list(SEED_OFFSETS.keys())
        for seed_id in seed_ids:
            tracks = seed_augmented_tracks_path(args.dataset, seed_id)
            out = trajectory_tensor_path(args.dataset, seed_id=seed_id)
            try:
                p, _ = export_one(
                    tracks,
                    out,
                    args.max_players,
                    args.validate,
                    args.min_global_visibility,
                    args.max_slot_empty_frac,
                    seed_id=seed_id,
                )
                meta_out = out.parent / "tensor_export_meta.json"
                with open(meta_out, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "seed_id": seed_id,
                            "ablation": LSTM_ABLATION,
                            "tracks": str(tracks),
                            "tensor": str(out),
                        },
                        f,
                        indent=2,
                    )
                exported.append((seed_id, p))
            except Exception as e:
                failed.append((seed_id, str(e)))

    print("\nLSTM tensor export summary")
    print("=" * 40)
    for label, path in exported:
        print(f"  OK {label}: {path}")
    for label, err in failed:
        print(f"  FAIL {label}: {err}")

    summary_path = runs_dir(args.dataset) / "seeds" / "lstm_tensor_export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "ablation": LSTM_ABLATION,
                "exported": {k: str(v) for k, v in exported},
                "failed": {k: v for k, v in failed},
            },
            f,
            indent=2,
        )
    print(f"Summary: {summary_path}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
