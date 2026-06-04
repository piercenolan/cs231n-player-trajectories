#!/usr/bin/env python3
"""
Prepare aligned ground truth for ADE/FDE evaluation.

Modes:
  --raw-gt PATH     Align SportsMOT MOT gt.txt using tracks meta (preferred)
  --proxy-smooth    Build smoothed-baseline proxy GT (dev only when raw GT missing)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import get_dataset
from utils.gt_align import (
    align_mot_gt_to_tracks,
    build_smoothed_proxy_gt,
    save_aligned_gt_json,
)


def main():
    parser = argparse.ArgumentParser(description="Setup aligned GT for a dataset")
    parser.add_argument(
        "--dataset",
        default="sportsmot_example",
        help="Dataset key from utils.datasets",
    )
    parser.add_argument(
        "--tracks",
        default=None,
        help="Baseline tracks JSON (default: data/runs/{dataset}/baseline_tracks.json)",
    )
    parser.add_argument(
        "--raw-gt",
        default=None,
        help="Path to SportsMOT gt/gt.txt (default: dataset gt_mot path)",
    )
    parser.add_argument(
        "--seqinfo",
        default=None,
        help="Path to seqinfo.ini (default: dataset seqinfo path)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Aligned GT JSON output (default: dataset gt_json path)",
    )
    parser.add_argument(
        "--proxy-smooth",
        action="store_true",
        help="Build smoothed baseline proxy GT when raw SportsMOT GT unavailable",
    )
    parser.add_argument(
        "--extract-fps",
        type=float,
        default=None,
        help="FPS used when building track frames (25 for consecutive SportsMOT jpgs)",
    )
    parser.add_argument("--start-time-sec", type=float, default=0.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    tracks_path = Path(args.tracks or ROOT / "data" / "runs" / args.dataset / "baseline_tracks.json")
    output_path = args.output or str(ds["gt_json"])
    extract_fps = args.extract_fps if args.extract_fps is not None else float(ds["extract_fps"])

    with open(tracks_path, encoding="utf-8") as f:
        tracks_data = json.load(f)
    meta = dict(tracks_data.get("meta") or {})

    if args.raw_gt or (not args.proxy_smooth and ds["gt_mot"] and Path(ds["gt_mot"]).exists()):
        raw = Path(args.raw_gt or ds["gt_mot"])
        seqinfo = args.seqinfo or (str(ds["seqinfo"]) if ds["seqinfo"] else None)
        if seqinfo is None:
            seqinfo = str(raw.parent.parent / "seqinfo.ini")
        aligned = align_mot_gt_to_tracks(
            raw,
            meta,
            seqinfo_path=seqinfo,
            extract_fps=extract_fps,
            start_time_sec=args.start_time_sec,
        )
        out_meta = {
            **meta,
            "gt_source": "sportsmot_mot",
            "dataset": args.dataset,
            "raw_gt": str(raw),
            "extract_fps": extract_fps,
            "start_time_sec": args.start_time_sec,
        }
    elif args.proxy_smooth:
        aligned, proxy_meta = build_smoothed_proxy_gt(tracks_path, window=args.smooth_window)
        out_meta = {**proxy_meta, "dataset": args.dataset}
        print(
            "WARNING: Using smoothed-baseline proxy GT. Replace with SportsMOT gt.txt "
            "via --raw-gt for paper-quality ADE."
        )
    else:
        raise SystemExit(
            f"No raw GT at {ds['gt_mot']}. Upload gt.txt or use --proxy-smooth."
        )

    out = save_aligned_gt_json(aligned, output_path, meta=out_meta)
    num_frames = len(aligned)
    num_pts = sum(len(v) for v in aligned.values())
    print(f"Wrote aligned GT: {out} ({num_frames} frames, {num_pts} points)")


if __name__ == "__main__":
    main()
