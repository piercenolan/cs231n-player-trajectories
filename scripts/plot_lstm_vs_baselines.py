#!/usr/bin/env python3
"""Two-panel median forecast ADE: all seeds vs clean seeds."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import runs_dir

CLEAN_SEEDS_EXCLUDE = frozenset({"offset_0s", "offset_5s", "offset_15s"})
BAR_ORDER = [
    ("linear_forecast_ade", "Linear", "#ff7f0e"),
    ("A0_forecast_ade", "A0 Plain LSTM", "#1f77b4"),
    ("A1_forecast_ade", "A1 Rule-Conditioned", "#2ca02c"),
    ("A1_residual_forecast_ade", "A1 Residual", "#17becf"),
    ("A3_forecast_ade", "A3 Graph", "#9467bd"),
]


def load_rows_with_residual(csv_path, multi_seed_path):
    rows = load_rows(csv_path)
    if not multi_seed_path.is_file():
        return rows
    import json

    with open(multi_seed_path, encoding="utf-8") as f:
        ms = json.load(f)
    by_seed = {}
    for r in ms.get("per_seed", []):
        if r.get("variant") == "A1_residual":
            by_seed[r["seed_id"]] = r.get("ade_forecast")
    for row in rows:
        sid = row.get("seed_id")
        if sid in by_seed:
            row["A1_residual_forecast_ade"] = by_seed[sid]
    return rows


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_clean(rows):
    return [r for r in rows if r["seed_id"] not in CLEAN_SEEDS_EXCLUDE]


def collect_stats(rows, col):
    vals = []
    for r in rows:
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            continue
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(vals)), float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


def plot_panel(ax, rows, title, footnote=None):
    medians, yerr_lo, yerr_hi, labels, colors = [], [], [], [], []
    for col, label, color in BAR_ORDER:
        med, q25, q75 = collect_stats(rows, col)
        medians.append(med)
        yerr_lo.append(max(0.0, med - q25) if med == med else 0)
        yerr_hi.append(max(0.0, q75 - med) if med == med else 0)
        labels.append(label)
        colors.append(color)
    x = np.arange(len(labels))
    ax.bar(
        x,
        medians,
        color=colors,
        edgecolor="white",
        yerr=[yerr_lo, yerr_hi],
        capsize=4,
        error_kw={"elinewidth": 1.2},
    )
    lin_med = medians[0]
    if lin_med == lin_med:
        ax.axhline(lin_med, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(0.02, 0.98, "Linear baseline", transform=ax.transAxes, va="top", fontsize=8, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Forecast ADE (pixels)")
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(medians):
        if v == v:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    if footnote:
        ax.text(0.5, -0.22, footnote, transform=ax.transAxes, ha="center", fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv or runs_dir(args.dataset) / "lstm" / "lstm_per_seed_delta.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}. Run eval_lstm_ablations.py --all-seeds first.")

    multi_path = csv_path.parent / "lstm_ablation_multi_seed.json"
    rows = load_rows_with_residual(csv_path, multi_path)
    out = Path(args.output or runs_dir(args.dataset) / "figures" / "lstm_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_panel(axes[0], rows, "Forecast ADE — All 12 Seeds (Median ± IQR)")
    plot_panel(
        axes[1],
        filter_clean(rows),
        "Forecast ADE — Clean Seeds (SAM3.1 tracking quality > threshold)",
        footnote="3 seeds excluded: SAM3.1 tracking ADE > 35px on all models",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
