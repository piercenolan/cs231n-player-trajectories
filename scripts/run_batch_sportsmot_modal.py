#!/usr/bin/env python3
"""Run run_all_seeds_modal.py for multiple datasets (external terminal / Modal)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--step-sec", type=float, default=2.0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-modal", action="store_true", help="Post-process only")
    args = p.parse_args()

    for ds in args.datasets:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_all_seeds_modal.py"),
            "--dataset",
            ds,
            "--step-sec",
            str(args.step_sec),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.skip_modal:
            cmd.append("--skip-modal")
        print("\n" + "=" * 60)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"WARNING: {ds} exited {rc}")
    print("\nBatch complete.")


if __name__ == "__main__":
    main()
