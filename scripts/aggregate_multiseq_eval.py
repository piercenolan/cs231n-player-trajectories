#!/usr/bin/env python3
"""Aggregate per-dataset LSTM eval into one transfer-summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import all_datasets, lstm_out_dir

CLEAN_EXCLUDE = frozenset({"offset_0s", "offset_5s", "offset_15s"})


def median_from_robust(lstm_root: Path, variant: str, exclude=None) -> float:
    path = lstm_root / "lstm_ablation_robust.json"
    if not path.is_file():
        return float("nan")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("per_seed_delta") or []
    vals = []
    for r in rows:
        if exclude and r.get("seed_id") in exclude:
            continue
        if variant == "linear":
            v = r.get("linear_forecast_ade")
        elif variant == "residual":
            v = r.get("A1_residual_forecast_ade")
        else:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if vf == vf:
            vals.append(vf)
    if vals:
        return float(np.median(vals))
    agg = (data.get("robust_aggregate") or {}).get(
        "A1_residual" if variant == "residual" else "linear_baseline", {}
    )
    return float(agg.get("ade_forecast_median", float("nan")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset keys (default: all extra_datasets + sportsmot_example)",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "data" / "runs" / "multiseq_transfer_summary.csv"),
    )
    args = p.parse_args()

    names = args.datasets
    if not names:
        names = ["sportsmot_example"]
        names.extend(k for k in sorted(all_datasets()) if k != "sportsmot_example" and k != "video_1_legacy")

    rows = []
    for ds in names:
        lstm_root = lstm_out_dir(ds)
        if not (lstm_root / "lstm_ablation_robust.json").is_file():
            print(f"Skip {ds}: no lstm_ablation_robust.json")
            continue
        med_res = median_from_robust(lstm_root, "residual")
        med_lin = median_from_robust(lstm_root, "linear")
        med_res_clean = median_from_robust(lstm_root, "residual", exclude=CLEAN_EXCLUDE)
        med_lin_clean = median_from_robust(lstm_root, "linear", exclude=CLEAN_EXCLUDE)
        robust = json.loads((lstm_root / "lstm_ablation_robust.json").read_text(encoding="utf-8"))
        beats = robust.get("A1_residual_beats_linear_seeds", 0)
        n = robust.get("n_seeds", 0)
        rows.append(
            {
                "dataset": ds,
                "median_residual_ade": f"{med_res:.2f}",
                "median_linear_ade": f"{med_lin:.2f}",
                "median_residual_ade_clean": f"{med_res_clean:.2f}",
                "median_linear_ade_clean": f"{med_lin_clean:.2f}",
                "residual_beats_linear": "Y" if med_res < med_lin - 0.01 else ("tie" if abs(med_res - med_lin) <= 0.01 else "N"),
                "residual_beats_linear_seeds": str(beats),
                "n_seeds": str(n),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} datasets)")
    for r in rows:
        print(f"  {r['dataset']}: residual={r['median_residual_ade']} linear={r['median_linear_ade']} ({r['residual_beats_linear']})")


if __name__ == "__main__":
    main()
