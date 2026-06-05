#!/usr/bin/env python3
"""Bar chart: cross-sequence transfer (residual vs linear median ADE)."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=str(ROOT / "data" / "runs" / "multiseq_transfer_summary.csv"),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "data" / "runs" / "figures" / "multiseq_transfer_bar.png"),
    )
    args = p.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels = [r["dataset"].replace("sportsmot_", "") for r in rows]
    res = [float(r["median_residual_ade"]) for r in rows]
    lin = [float(r["median_linear_ade"]) for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, res, w, label="A1 Residual (transfer)", color="#17becf")
    ax.bar(x + w / 2, lin, w, label="Linear baseline", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median forecast ADE (px)")
    ax.set_title("Cross-sequence transfer (checkpoint trained on sportsmot_example only)")
    ax.legend()
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
