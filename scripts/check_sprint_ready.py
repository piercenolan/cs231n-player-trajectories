#!/usr/bin/env python3
"""Verify datasets are registered and extracted before Modal sprint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import all_datasets, frames_dir, get_dataset

SPRINT_JSON = ROOT / "data" / "datasets" / "sprint_sequences.json"


def main():
    if not SPRINT_JSON.is_file():
        print(f"Missing {SPRINT_JSON}")
        return 1
    spec = json.loads(SPRINT_JSON.read_text(encoding="utf-8"))
    extract_ids = spec.get("extract", [])
    ok = True
    print("Sprint sequence IDs:", extract_ids)
    for name, cfg in sorted(all_datasets().items()):
        if name in ("video_1_legacy",):
            continue
        fd = Path(cfg["frames_dir"])
        gt = Path(cfg["gt_mot"]) if cfg.get("gt_mot") else None
        n_frames = len(list(fd.glob("*.jpg"))) if fd.is_dir() else 0
        has_gt = gt.is_file() if gt else False
        status = "OK" if n_frames > 0 and has_gt else "MISSING"
        if status != "OK":
            ok = False
        print(f"  [{status}] {name}: {n_frames} frames, gt={has_gt}")
    if not ok:
        print("\nFix: see data/datasets/EXTRACTION_STATUS.md and docs/MODAL_SPRINT_RUNBOOK.md")
        return 1
    print("\nAll registered datasets ready for Modal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
