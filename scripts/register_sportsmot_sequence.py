#!/usr/bin/env python3
"""
Register one SportsMOT sequence for the pipeline (frames + gt + seqinfo).

Copies from data/datasets/sportsmot_basketball/{train|val|test}/<seq_id>/
into data/datasets/<dataset_name>/ and records the entry in data/datasets/extra_datasets.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASKETBALL_ROOT = ROOT / "data" / "datasets" / "sportsmot_basketball"
EXTRA_JSON = ROOT / "data" / "datasets" / "extra_datasets.json"


def sanitize_dataset_name(seq_id: str, prefix: str = "sportsmot") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", seq_id).strip("_").lower()
    if len(slug) > 40:
        slug = slug[:40]
    return f"{prefix}_{slug}"


def find_sequence_dir(seq_id: str) -> tuple[Path, str]:
    for split in ("train", "val", "test"):
        cand = BASKETBALL_ROOT / split / seq_id
        if cand.is_dir():
            return cand, split
    raise FileNotFoundError(
        f"Sequence '{seq_id}' not under {BASKETBALL_ROOT}/{{train,val,test}}/. "
        "Run scripts/extract_sportsmot_basketball.py first."
    )


def register(seq_id: str, dataset_name: str | None, holdout: bool) -> Path:
    src, split = find_sequence_dir(seq_id)
    name = dataset_name or sanitize_dataset_name(seq_id)
    dest = ROOT / "data" / "datasets" / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    img1 = src / "img1"
    if not img1.is_dir():
        raise FileNotFoundError(f"Missing img1/ in {src}")
    shutil.copytree(img1, dest / "frames")

    gt_dir = dest / "gt"
    gt_dir.mkdir(parents=True)
    gt_txt = src / "gt" / "gt.txt"
    if gt_txt.is_file():
        shutil.copy2(gt_txt, gt_dir / "gt.txt")
    else:
        raise FileNotFoundError(f"Missing gt/gt.txt in {src}")

    seqinfo = src / "seqinfo.ini"
    if seqinfo.is_file():
        shutil.copy2(seqinfo, dest / "seqinfo.ini")

    entry = {
        "description": f"SportsMOT basketball {seq_id} ({split})",
        "frames_dir": str((dest / "frames").relative_to(ROOT)).replace("\\", "/"),
        "gt_mot": str((gt_dir / "gt.txt").relative_to(ROOT)).replace("\\", "/"),
        "gt_json": None,
        "seqinfo": str((dest / "seqinfo.ini").relative_to(ROOT)).replace("\\", "/")
        if (dest / "seqinfo.ini").is_file()
        else None,
        "video": None,
        "source_fps": 25.0,
        "extract_fps": 25.0,
        "sportsmot_seq_id": seq_id,
        "sportsmot_split": split,
        "eval_only_holdout": holdout,
    }

    extra = {}
    if EXTRA_JSON.is_file():
        extra = json.loads(EXTRA_JSON.read_text(encoding="utf-8"))
    extra[name] = entry
    EXTRA_JSON.write_text(json.dumps(extra, indent=2), encoding="utf-8")
    print(f"Registered dataset '{name}' from {split}/{seq_id}")
    print(f"  frames: {dest / 'frames'}")
    print(f"  wrote {EXTRA_JSON}")
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sequence_id", help="SportsMOT sequence folder name, e.g. v_-6Os86HzwCs_c001")
    p.add_argument("--dataset-name", default=None, help="Pipeline dataset key (default: sportsmot_<slug>)")
    p.add_argument(
        "--holdout",
        action="store_true",
        help="Mark as sequence holdout (eval/transfer only; documented in sprint manifest)",
    )
    args = p.parse_args()
    register(args.sequence_id, args.dataset_name, args.holdout)


if __name__ == "__main__":
    main()
