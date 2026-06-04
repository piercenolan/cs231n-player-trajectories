"""
Export cleaned tracks to LSTM-ready position tensors.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def export_trajectories(tracks_path, output_path, max_players=10):
    """
    Build fixed-size arrays for sequence models.

    Returns dict with:
      positions: (T, P, 2) float32
      visibility: (T, P) bool — True when player observed (not predicted)
      frame_numbers: (T,) int
      player_ids: (P,) int — slot -> track id (-1 if empty)
    """
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)

    frames = sorted(data.get("frames", []), key=lambda fr: int(fr["frame_number"]))
    meta = dict(data.get("meta") or {})

    if not frames:
        raise ValueError("No frames in tracks file")

    id_counts = {}
    for fr in frames:
        for p in fr.get("players", []):
            if p.get("predicted"):
                continue
            pid = p["id"]
            id_counts[pid] = id_counts.get(pid, 0) + 1

    top_ids = sorted(id_counts.keys(), key=lambda i: id_counts[i], reverse=True)[:max_players]
    while len(top_ids) < max_players:
        top_ids.append(-1)

    T = len(frames)
    P = max_players
    positions = np.zeros((T, P, 2), dtype=np.float32)
    visibility = np.zeros((T, P), dtype=bool)
    frame_numbers = np.zeros(T, dtype=np.int32)

    id_to_slot = {pid: i for i, pid in enumerate(top_ids) if pid >= 0}

    for t, fr in enumerate(frames):
        fnum = int(fr["frame_number"])
        frame_numbers[t] = fnum
        for p in fr.get("players", []):
            pid = p["id"]
            if pid not in id_to_slot:
                continue
            slot = id_to_slot[pid]
            c = p.get("mask_center", {})
            if "x" not in c or "y" not in c:
                continue
            positions[t, slot, 0] = float(c["x"])
            positions[t, slot, 1] = float(c["y"])
            visibility[t, slot] = not bool(p.get("predicted", False))

    export = {
        "meta": {
            **meta,
            "max_players": max_players,
            "num_frames": T,
            "source_tracks": str(tracks_path),
        },
        "frame_numbers": frame_numbers.tolist(),
        "player_ids": top_ids,
        "positions": positions.tolist(),
        "visibility": visibility.tolist(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print(f"Exported LSTM trajectories: T={T}, P={P} -> {output_path}")
    return export


def validate_export(export, min_global_visibility=0.7, max_slot_empty_frac=0.30):
    """
    Validate LSTM tensor quality.

    Returns (passed: bool, report: dict).
    """
    visibility = np.array(export["visibility"], dtype=bool)
    T, P = visibility.shape
    if T == 0:
        return False, {"error": "no_frames"}

    global_vis = float(visibility.sum()) / float(T * P)
    per_slot = visibility.sum(axis=0) / float(T)
    frames_zero = int((visibility.sum(axis=1) == 0).sum())
    slot_empty_frac = 1.0 - per_slot

    report = {
        "global_visibility_fraction": global_vis,
        "frames_with_zero_visible": frames_zero,
        "per_slot_visibility": per_slot.tolist(),
        "per_slot_empty_fraction": slot_empty_frac.tolist(),
        "min_global_visibility_threshold": min_global_visibility,
        "max_slot_empty_fraction_threshold": max_slot_empty_frac,
    }

    passed = global_vis >= min_global_visibility and all(
        ef <= max_slot_empty_frac for ef in slot_empty_frac if ef < 1.0
    )
    report["passed"] = passed
    return passed, report


def main():
    parser = argparse.ArgumentParser(description="Export tracks to LSTM tensor JSON")
    parser.add_argument("--tracks", default="data/outputs/augmented_tracks.json")
    parser.add_argument("--output", default="data/outputs/trajectory_tensors.json")
    parser.add_argument("--max-players", type=int, default=10)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run LSTM-readiness checks (exit 1 if failed)",
    )
    parser.add_argument("--min-global-visibility", type=float, default=0.7)
    parser.add_argument("--max-slot-empty-frac", type=float, default=0.30)
    args = parser.parse_args()

    export = export_trajectories(args.tracks, args.output, max_players=args.max_players)

    if args.validate:
        passed, report = validate_export(
            export,
            min_global_visibility=args.min_global_visibility,
            max_slot_empty_frac=args.max_slot_empty_frac,
        )
        report_path = Path(args.output).with_name("trajectory_validation.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\nLSTM EXPORT VALIDATION")
        print("=" * 40)
        print(f"Global visibility: {report['global_visibility_fraction']:.3f}")
        print(f"Frames with zero visible: {report['frames_with_zero_visible']}")
        print(f"Passed: {passed}")
        print(f"Report: {report_path}")
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
