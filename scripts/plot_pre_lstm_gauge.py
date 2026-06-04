#!/usr/bin/env python3
"""Generate pre-LSTM gauge figures: ADE ablations, multi-seed robustness, summary text."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_ablation_ade(csv_path, output_path, highlight=None):
    import csv

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ade = float(row["ade"])
            except (TypeError, ValueError):
                continue
            rows.append((row["ablation"], ade))

    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    ades = [r[1] for r in rows]

    colors = []
    for n in names:
        if n == "baseline":
            colors.append("#1f77b4")
        elif n == highlight:
            colors.append("#2ca02c")
        elif n in ("full", "convergence_pull", "cluster_cohesion"):
            colors.append("#d62728")
        elif n == "dead_ball_freeze":
            colors.append("#ff7f0e")
        else:
            colors.append("#aec7e8")

    fig_h = max(5, 0.35 * len(names))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(names))
    ax.barh(y, ades, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("ADE (px) vs SportsMOT GT")
    ax.set_title("Augmentation ablations (45-frame window, lower is better)")
    ax.axvline(ades[names.index("baseline")] if "baseline" in names else 0, color="#1f77b4", linestyle="--", alpha=0.5, label="baseline ADE")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_multi_seed(summary_path, output_path):
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)

    seeds = s.get("seeds", [])
    ades = s.get("ade_values", [])
    if not seeds or not ades:
        raise ValueError(f"No ADE values in {summary_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(seeds))
    bars = ax.bar(x, ades, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(seeds)], edgecolor="white")
    mean_ade = s.get("ade_mean")
    std_ade = s.get("ade_std", 0)
    if mean_ade is not None:
        ax.axhline(mean_ade, color="#444444", linestyle="--", linewidth=1.5, label=f"mean = {mean_ade:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=15, ha="right")
    ax.set_ylabel("ADE (px)")
    title = f"Multi-seed robustness (sanitize_plus_velocity_cap)"
    if mean_ade is not None and std_ade is not None:
        title += f"\nADE = {mean_ade:.2f} ± {std_ade:.2f} px"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, ades):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_lstm_ade(eval_path, output_path):
    """Bar chart: augmented SAM vs LSTM vs linear (forecast horizon if available)."""
    if not Path(eval_path).is_file():
        print(f"Skip LSTM plot (missing {eval_path})")
        return

    with open(eval_path, encoding="utf-8") as f:
        ev = json.load(f)

    use_forecast = "lstm_forecast_only" in ev
    if use_forecast:
        names = ["Aug SAM (forecast)", "LSTM (forecast)", "Linear (forecast)"]
        ades = [
            ev["augmented_forecast_only"]["ade"],
            ev["lstm_forecast_only"]["ade"],
            ev.get("linear_forecast_only", {}).get("ade", float("nan")),
        ]
        title = "ADE on forecast horizon (frames >= obs_len)"
    else:
        names = ["Augmented SAM", "LSTM", "Linear"]
        ades = [
            ev["augmented_baseline"]["ade"],
            ev["lstm"]["ade"],
            ev.get("linear_baseline", {}).get("ade", float("nan")),
        ]
        title = "ADE vs GT (full clip)"

    colors = ["#2ca02c", "#9467bd", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x, ades, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ADE (px)")
    ax.set_title(title)
    for i, v in enumerate(ades):
        if v == v:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def write_gauge_report(fig_dir, validation_path, compare_log_path):
    with open(validation_path, encoding="utf-8") as f:
        val = json.load(f)

    lines = [
        "# Pre-LSTM gauge summary",
        "",
        "## LSTM export gate",
        f"- **Passed:** {val.get('passed')}",
        f"- **Global visibility:** {val.get('global_visibility_fraction', 0):.2%}",
        f"- **Frames with zero visible players:** {val.get('frames_with_zero_visible')}",
        f"- Thresholds: visibility ≥ {val.get('min_global_visibility_threshold')}, slot empty ≤ {val.get('max_slot_empty_fraction_threshold')}",
        "",
        "## Figures",
        f"- `baseline_metrics.png` — SAM3 coverage + ID continuity",
        f"- `summary_figure.png` — baseline vs augmented (qualitative)",
        f"- `ablation_ade_bar.png` — ADE across ablations (real GT)",
        f"- `multi_seed_ade_bar.png` — ADE across temporal offsets",
        "",
    ]
    if compare_log_path and Path(compare_log_path).exists():
        lines.append("## Baseline vs augmented (console compare)")
        lines.append("```")
        lines.append(Path(compare_log_path).read_text(encoding="utf-8")[-4000:])
        lines.append("```")

    out = Path(fig_dir) / "PRE_LSTM_GAUGE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Pre-LSTM gauge plots")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()

    from utils.datasets import runs_dir

    fig_dir = Path(args.figures_dir or runs_dir(args.dataset) / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation_ade(
        runs_dir(args.dataset) / "ablations" / "ablation_summary.csv",
        fig_dir / "ablation_ade_bar.png",
        highlight="sanitize_plus_velocity_cap",
    )
    plot_multi_seed(
        runs_dir(args.dataset) / "seeds" / "multi_seed_summary.json",
        fig_dir / "multi_seed_ade_bar.png",
    )
    write_gauge_report(
        fig_dir,
        runs_dir(args.dataset) / "trajectory_validation.json",
        fig_dir / "compare_table.txt",
    )
    plot_lstm_ade(
        runs_dir(args.dataset) / "lstm" / "lstm_eval.json",
        fig_dir / "lstm_ade_bar.png",
    )


if __name__ == "__main__":
    main()
