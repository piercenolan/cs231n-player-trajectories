#!/usr/bin/env python3
"""Bar charts: per-clip trained vs transfer residual ADE."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short_label(dataset):
    return dataset.replace("sportsmot_", "")


def plot_single(rows, title, output, residual_label):
    labels = [short_label(r["dataset"]) for r in rows]
    res = [float(r["median_residual_ade"]) for r in rows]
    lin = [float(r["median_linear_ade"]) for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, res, w, label=residual_label, color="#17becf")
    ax.bar(x + w / 2, lin, w, label="Linear baseline", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median forecast ADE (px)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_compare(perclip_rows, transfer_rows, output):
    by_ds = {r["dataset"]: r for r in transfer_rows}
    labels = [short_label(r["dataset"]) for r in perclip_rows]
    x = np.arange(len(labels))
    w = 0.22

    pc_res = [float(r["median_residual_ade"]) for r in perclip_rows]
    tr_res = [float(by_ds[r["dataset"]]["median_residual_ade"]) for r in perclip_rows]
    lin = [float(r["median_linear_ade"]) for r in perclip_rows]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w, pc_res, w, label="A1 Residual (per-clip train)", color="#17becf")
    ax.bar(x, tr_res, w, label="A1 Residual (transfer)", color="#aec7e8")
    ax.bar(x + w, lin, w, label="Linear baseline", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median forecast ADE (px)")
    ax.set_title("Per-clip training vs transfer (example checkpoint)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=str(ROOT / "data" / "runs" / "multiseq_perclip_summary.csv"),
    )
    p.add_argument(
        "--transfer-csv",
        default=str(ROOT / "data" / "runs" / "multiseq_transfer_baseline.csv"),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "data" / "runs" / "figures" / "multiseq_perclip_bar.png"),
    )
    p.add_argument(
        "--compare-output",
        default=str(ROOT / "data" / "runs" / "figures" / "multiseq_train_vs_transfer.png"),
    )
    args = p.parse_args()

    rows = load_csv(args.csv)
    plot_single(
        rows,
        "Per-clip trained A1 Residual vs linear (held_out_seed, 80 epochs)",
        args.output,
        "A1 Residual (per-clip train)",
    )

    transfer_path = Path(args.transfer_csv)
    if transfer_path.is_file():
        plot_compare(rows, load_csv(transfer_path), args.compare_output)


if __name__ == "__main__":
    main()
