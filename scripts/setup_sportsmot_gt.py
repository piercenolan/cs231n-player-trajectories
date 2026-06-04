#!/usr/bin/env python3
"""
Prepare aligned ground truth for video_1 ADE/FDE evaluation.

Modes:
  --raw-gt PATH     Align SportsMOT MOT gt.txt using tracks meta (preferred)
  --proxy-smooth    Build smoothed-baseline proxy GT (local dev when raw GT missing)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.gt_align import (
    align_mot_gt_to_tracks,
    build_smoothed_proxy_gt,
    save_aligned_gt_json,
)


def main():
    parser = argparse.ArgumentParser(description="Setup aligned GT for video_1")
    parser.add_argument(
        "--tracks",
        default="data/outputs/baseline_tracks.json",
        help="Baseline tracks JSON (provides meta for alignment)",
    )
    parser.add_argument(
        "--raw-gt",
        default=None,
        help="Path to SportsMOT gt/gt.txt (full resolution MOT format)",
    )
    parser.add_argument(
        "--seqinfo",
        default=None,
        help="Path to seqinfo.ini (defaults to raw-gt/../seqinfo.ini)",
    )
    parser.add_argument(
        "--output",
        default="data/gt/sportsmot/video_1/gt/gt.json",
        help="Aligned GT JSON output path",
    )
    parser.add_argument(
        "--proxy-smooth",
        action="store_true",
        help="Build smoothed baseline proxy GT when raw SportsMOT GT unavailable",
    )
    parser.add_argument("--extract-fps", type=float, default=1.0)
    parser.add_argument("--start-time-sec", type=float, default=0.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()

    tracks_path = Path(args.tracks)
    with open(tracks_path, encoding="utf-8") as f:
        tracks_data = json.load(f)
    meta = dict(tracks_data.get("meta") or {})

    if args.raw_gt:
        raw = Path(args.raw_gt)
        seqinfo = args.seqinfo or str(raw.parent.parent / "seqinfo.ini")
        aligned = align_mot_gt_to_tracks(
            raw,
            meta,
            seqinfo_path=seqinfo,
            extract_fps=args.extract_fps,
            start_time_sec=args.start_time_sec,
        )
        out_meta = {
            **meta,
            "gt_source": "sportsmot_mot",
            "raw_gt": str(raw),
            "extract_fps": args.extract_fps,
            "start_time_sec": args.start_time_sec,
        }
    elif args.proxy_smooth:
        aligned, proxy_meta = build_smoothed_proxy_gt(tracks_path, window=args.smooth_window)
        out_meta = proxy_meta
        print(
            "WARNING: Using smoothed-baseline proxy GT. Replace with SportsMOT gt.txt "
            "via --raw-gt for paper-quality ADE."
        )
    else:
        raise SystemExit("Provide --raw-gt or --proxy-smooth")

    out = save_aligned_gt_json(aligned, args.output, meta=out_meta)
    num_frames = len(aligned)
    num_pts = sum(len(v) for v in aligned.values())
    print(f"Wrote aligned GT: {out} ({num_frames} frames, {num_pts} points)")


if __name__ == "__main__":
    main()
