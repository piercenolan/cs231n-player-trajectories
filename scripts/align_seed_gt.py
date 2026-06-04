#!/usr/bin/env python3
"""Align SportsMOT gt.txt to one or all multi-seed baseline track windows."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import SEED_OFFSETS, align_seed_gt, baseline_tracks_path, runs_dir


def main():
    parser = argparse.ArgumentParser(description="Align GT per multi-seed window")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument(
        "--seed-id",
        default=None,
        help="One seed (offset_0s, offset_10s, offset_15s). Default: all with baselines.",
    )
    parser.add_argument("--tracks", default=None, help="Override baseline tracks path")
    parser.add_argument("--start-time-sec", type=float, default=None)
    parser.add_argument("--extract-fps", type=float, default=None)
    args = parser.parse_args()

    seeds_root = runs_dir(args.dataset) / "seeds"
    if args.seed_id:
        seeds = [(args.seed_id, args.start_time_sec)]
    else:
        seeds = list(SEED_OFFSETS.items())

    for seed_id, default_start in seeds:
        tracks = Path(args.tracks) if args.tracks and args.seed_id else baseline_tracks_path(
            args.dataset, seed_id
        )
        if not tracks.is_file():
            print(f"Skip {seed_id}: no baseline at {tracks}")
            continue
        start = args.start_time_sec if args.start_time_sec is not None else default_start
        out = align_seed_gt(
            args.dataset,
            seed_id,
            tracks_path=tracks,
            start_time_sec=start,
            extract_fps=args.extract_fps,
        )
        print(f"Aligned {seed_id} (start={start}s) -> {out}")


if __name__ == "__main__":
    main()
