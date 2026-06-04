#!/usr/bin/env python3
"""
Generate temporal seeds every N seconds across the SportsMOT clip, run SAM3 on
Modal (one job at a time), download baselines from the Modal volume, align GT,
and optionally run augmentation + LSTM tensor export.

Default: step_sec=5 over 500 frames @ 25 FPS with 45-frame windows → offsets
0s, 5s, 10s, 15s (18.2s max start). Use --step-sec 2 for denser coverage.

Examples:
  # Full pipeline (Modal + download + post-process):
  py scripts/run_all_seeds_modal.py --dataset sportsmot_example

  # Plan only:
  py scripts/run_all_seeds_modal.py --dry-run

  # Re-download + post-process after Modal jobs finished elsewhere:
  py scripts/run_all_seeds_modal.py --skip-modal

  # Denser seeds (~10 windows):
  py scripts/run_all_seeds_modal.py --step-sec 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import baseline_tracks_path, get_dataset, runs_dir
from utils.seed_schedule import (
    build_seed_schedule,
    count_dataset_frames,
    max_start_time_sec,
    seed_manifest_path,
    write_seed_manifest,
)

MODAL_VOLUME = "sports-data"


def run_cmd(cmd: list[str], dry_run: bool = False) -> int:
    line = " ".join(cmd)
    print(f"\n>> {line}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def modal_available() -> bool:
    return shutil.which("modal") is not None


def run_modal_seed(
    dataset: str,
    seed_id: str,
    start_sec: float,
    args: argparse.Namespace,
    skip_upload: bool,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "run",
        "scripts/run_modal.py",
        "--dataset",
        dataset,
        "--skip-extract",
        f"--max-frames={args.max_frames}",
        f"--resize-scale={args.resize_scale}",
        f"--max-num-objects={args.max_num_objects}",
        f"--start-time-sec={start_sec}",
        f"--seed-id={seed_id}",
    ]
    if skip_upload:
        cmd.append("--skip-upload")
    return run_cmd(cmd, dry_run=dry_run)


def download_seed_baselines(dataset: str, schedule: list[dict], dry_run: bool) -> int:
    """Download each seed baseline; fall back to bulk seeds/ download."""
    seeds_root = runs_dir(dataset) / "seeds"
    seeds_root.mkdir(parents=True, exist_ok=True)
    rc = 0

    for entry in schedule:
        seed_id = entry["seed_id"]
        local = baseline_tracks_path(dataset, seed_id)
        local.parent.mkdir(parents=True, exist_ok=True)
        remote = f"runs/{dataset}/seeds/{seed_id}/baseline_tracks.json"
        code = run_cmd(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                MODAL_VOLUME,
                remote,
                str(local),
            ],
            dry_run=dry_run,
        )
        if code != 0 and not dry_run:
            rc = code
            print(f"WARNING: download failed for {seed_id} (exit {code})")

    bulk_rc = run_cmd(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            MODAL_VOLUME,
            f"runs/{dataset}/seeds",
            str(seeds_root),
            "--force",
        ],
        dry_run=dry_run,
    )
    return rc or bulk_rc


def post_process(dataset: str, schedule: list[dict], dry_run: bool) -> int:
    rc = 0
    for entry in schedule:
        seed_id = entry["seed_id"]
        code = run_cmd(
            [
                sys.executable,
                "scripts/align_seed_gt.py",
                "--dataset",
                dataset,
                "--seed-id",
                seed_id,
                f"--start-time-sec={entry['start_time_sec']}",
            ],
            dry_run=dry_run,
        )
        rc = rc or code

    for extra in (
        ["scripts/run_multi_seed.py", "--dataset", dataset, "--align-gt"],
        ["scripts/export_lstm_tensors.py", "--dataset", dataset, "--all-seeds"],
    ):
        code = run_cmd([sys.executable] + extra, dry_run=dry_run)
        rc = rc or code
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 Modal jobs for all temporal seeds and download results"
    )
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument(
        "--step-sec",
        type=float,
        default=5.0,
        help="Seconds between window starts (use 2–5 for more LSTM data)",
    )
    parser.add_argument("--max-frames", type=int, default=45)
    parser.add_argument("--resize-scale", type=float, default=0.67)
    parser.add_argument("--max-num-objects", type=int, default=12)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Override frame count (0 = count local dataset frames)",
    )
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument(
        "--end-sec",
        type=float,
        default=-1.0,
        help="Last start offset in seconds (-1 = auto from num_frames)",
    )
    parser.add_argument(
        "--modal-wait-sec",
        type=int,
        default=120,
        help="Pause between Modal jobs to reduce CUDA OOM on warm GPU",
    )
    parser.add_argument("--skip-modal", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--skip-post",
        action="store_true",
        help="Skip align_seed_gt, run_multi_seed, export_lstm_tensors after download",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip Modal job if local baseline_tracks.json already exists",
    )
    parser.add_argument(
        "--upload-frames",
        action="store_true",
        help="Upload local frames to volume on the first Modal job only",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    source_fps = float(ds["source_fps"])
    extract_fps = float(ds["extract_fps"])

    num_frames = args.num_frames or count_dataset_frames(args.dataset)
    if num_frames <= 0:
        print(
            f"ERROR: No frames in {ds['frames_dir']}. "
            "Extract SportsMOT frames into data/datasets/.../frames first."
        )
        sys.exit(1)

    end_sec = args.end_sec if args.end_sec >= 0 else None
    max_start = max_start_time_sec(
        num_frames, args.max_frames, source_fps, extract_fps
    )
    schedule = build_seed_schedule(
        num_frames=num_frames,
        step_sec=args.step_sec,
        max_frames=args.max_frames,
        source_fps=source_fps,
        extract_fps=extract_fps,
        start_sec=args.start_sec,
        end_sec=end_sec,
    )

    manifest_path = write_seed_manifest(
        args.dataset,
        schedule,
        num_frames=num_frames,
        step_sec=args.step_sec,
        max_frames=args.max_frames,
        resize_scale=args.resize_scale,
        extra={
            "source_fps": source_fps,
            "extract_fps": extract_fps,
            "max_start_time_sec": max_start,
            "modal_volume": MODAL_VOLUME,
        },
    )

    print("=" * 60)
    print("SEED SCHEDULE")
    print("=" * 60)
    print(f"Dataset:      {args.dataset}")
    print(f"Frames:       {num_frames} @ {source_fps} FPS")
    print(f"Window:       {args.max_frames} frames, resize {args.resize_scale}")
    print(f"Step:         {args.step_sec}s -> {len(schedule)} seeds")
    print(f"Max start:    {max_start:.2f}s (last offset)")
    print(f"Manifest:     {manifest_path}")
    for entry in schedule:
        print(f"  {entry['seed_id']:16s}  start={entry['start_time_sec']:.1f}s")
    print("=" * 60)

    if args.dry_run:
        print("Dry run — no Modal/download/post commands executed.")
        return

    if not args.skip_modal:
        if not modal_available():
            print("ERROR: `modal` CLI not found. Install: pip install modal")
            sys.exit(1)

        uploaded_frames = False
        for i, entry in enumerate(schedule):
            seed_id = entry["seed_id"]
            start_sec = entry["start_time_sec"]
            local_baseline = baseline_tracks_path(args.dataset, seed_id)

            if args.skip_existing and local_baseline.is_file():
                print(f"\n[{i+1}/{len(schedule)}] Skip {seed_id} (baseline exists)")
                continue

            skip_upload = True
            if args.upload_frames and not uploaded_frames:
                skip_upload = False
                uploaded_frames = True

            print(f"\n[{i+1}/{len(schedule)}] Modal: {seed_id} @ {start_sec}s")
            code = run_modal_seed(
                args.dataset,
                seed_id,
                start_sec,
                args,
                skip_upload=skip_upload,
            )
            if code != 0:
                print(f"ERROR: Modal failed for {seed_id} (exit {code})")
                sys.exit(code)

            if i < len(schedule) - 1 and args.modal_wait_sec > 0:
                print(f"Waiting {args.modal_wait_sec}s before next job (GPU cooldown)...")
                time.sleep(args.modal_wait_sec)

    if not args.skip_download:
        print("\n" + "=" * 60)
        print("DOWNLOAD FROM MODAL VOLUME")
        print("=" * 60)
        code = download_seed_baselines(args.dataset, schedule, dry_run=False)
        if code != 0:
            print(f"WARNING: some downloads failed (exit {code})")

        missing = [
            e["seed_id"]
            for e in schedule
            if not baseline_tracks_path(args.dataset, e["seed_id"]).is_file()
        ]
        if missing:
            print(f"WARNING: missing baselines after download: {missing}")
        else:
            print("All seed baselines present locally.")

    do_post = not args.skip_post
    if do_post:
        print("\n" + "=" * 60)
        print("POST-PROCESS (GT align, augmentation, LSTM export)")
        print("=" * 60)
        code = post_process(args.dataset, schedule, dry_run=False)
        if code != 0:
            print(f"WARNING: post-process returned exit {code}")

    print("\nDone.")
    print(f"  Manifest: {seed_manifest_path(args.dataset)}")
    print(f"  Seeds:    {runs_dir(args.dataset) / 'seeds'}")
    if not do_post:
        print(
            "Next (optional):\n"
            f"  py scripts/align_seed_gt.py --dataset {args.dataset}\n"
            f"  py scripts/run_multi_seed.py --dataset {args.dataset} --align-gt\n"
            f"  py scripts/export_lstm_tensors.py --dataset {args.dataset} --all-seeds"
        )


if __name__ == "__main__":
    main()
