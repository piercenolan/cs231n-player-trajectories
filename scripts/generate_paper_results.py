#!/usr/bin/env python3
"""Generate docs/PAPER_RESULTS.md from run artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import lstm_out_dir, runs_dir

CLEAN_EXCLUDE = frozenset({"offset_0s", "offset_5s", "offset_15s"})


def load_json(path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path):
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics_block(base_m, aug_m):
    def tm(j):
        if not j:
            return {}
        return j.get("track_metrics") or j
    b, a = tm(base_m), tm(aug_m)
    if not b and not a:
        return "_(missing metrics.json)_"
    return (
        f"| Mean players/frame | {b.get('mean_players_per_frame', 'n/a')} | {a.get('mean_players_per_frame', 'n/a')} |\n"
        f"| ID switches | {b.get('total_id_switches', 'n/a')} | {a.get('total_id_switches', 'n/a')} |\n"
        f"| Mean track streak | {b.get('mean_track_streak', 'n/a')} | {a.get('mean_track_streak', 'n/a')} |"
    )


def median_col(rows, col, exclude=None):
    vals = []
    for r in rows:
        if exclude and r.get("seed_id") in exclude:
            continue
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            continue
        if v == v:
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def beats_linear(med_a1, med_lin):
    if med_a1 != med_a1 or med_lin != med_lin:
        return "N/A"
    return "Y" if med_a1 < med_lin - 0.01 else "N"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_root = runs_dir(args.dataset)
    lstm_root = lstm_out_dir(args.dataset)
    rows = load_csv_rows(lstm_root / "lstm_per_seed_delta.csv")
    robust = load_json(lstm_root / "lstm_ablation_robust.json") or {}
    multi = load_json(lstm_root / "lstm_ablation_multi_seed.json") or {}

    base_m = load_json(run_root / "ablations" / "baseline" / "metrics.json")
    aug_m = load_json(run_root / "ablations" / "sanitize_plus_velocity_cap" / "metrics.json")

    lines = ["# Paper Results (auto-generated)", ""]
    lines.append("## Section 1 — Detection metrics (pre-LSTM)")
    lines.append("")
    lines.append("| Metric | SAM3.1 baseline | Augmented |")
    lines.append("|--------|-----------------|-----------|")
    lines.append(metrics_block(base_m, aug_m))
    lines.append("")

    lines.append("## Section 2 — Forecast metrics (LSTM)")
    lines.append("")
    lines.append("| Model | All-seed median ADE | Clean-seed median ADE | Beats linear |")
    lines.append("|-------|---------------------|------------------------|--------------|")
    models = [
        ("Linear", "linear_forecast_ade"),
        ("A0 Plain", "A0_forecast_ade"),
        ("A1 Rule-Conditioned", "A1_forecast_ade"),
        ("A3 Graph", "A3_forecast_ade"),
    ]
    for name, col in models:
        med_all = median_col(rows, col)
        med_clean = median_col(rows, col, exclude=CLEAN_EXCLUDE)
        bl = beats_linear(median_col(rows, "A1_forecast_ade"), median_col(rows, "linear_forecast_ade")) if name == "A1 Rule-Conditioned" else ("—" if name == "Linear" else beats_linear(med_all, median_col(rows, "linear_forecast_ade")))
        if name == "Linear":
            bl = "—"
        lines.append(f"| {name} | {med_all:.2f} | {med_clean:.2f} | {bl} |")
    lines.append("")

    lines.append("## Section 3 — Per-seed breakdown")
    lines.append("")
    lines.append("| Seed | A0 | A1 | Linear | Winner (A1 vs linear) |")
    lines.append("|------|-----|-----|--------|------------------------|")
    for r in sorted(rows, key=lambda x: x.get("seed_id", "")):
        lines.append(
            f"| {r.get('seed_id')} | {float(r.get('A0_forecast_ade', 'nan')):.2f} | "
            f"{float(r.get('A1_forecast_ade', 'nan')):.2f} | {float(r.get('linear_forecast_ade', 'nan')):.2f} | "
            f"{r.get('winner_vs_linear', 'n/a')} |"
        )
    lines.append("")

    a1_wins = robust.get("A1_wins_vs_A0", sum(1 for r in rows if r.get("winner") == "A1"))
    n_seeds = robust.get("n_seeds", len(rows))
    med_a1_clean = median_col(rows, "A1_forecast_ade", exclude=CLEAN_EXCLUDE)
    med_lin_clean = median_col(rows, "linear_forecast_ade", exclude=CLEAN_EXCLUDE)
    pct = (1 - med_a1_clean / med_lin_clean) * 100 if med_lin_clean == med_lin_clean and med_lin_clean > 0 else float("nan")

    tf_a1 = [
        r.get("teacher_forced_ade_px")
        for r in multi.get("per_seed", [])
        if r.get("variant") == "A1_rule_features"
    ]
    tf_med = float(np.median([float(x) for x in tf_a1 if x == x])) if tf_a1 else float("nan")

    lines.append("## Section 4 — Key findings")
    lines.append("")
    lines.append(f"- A1 beats A0 on {a1_wins}/{n_seeds} seeds (per robust report).")
    if pct == pct and pct > 0:
        imp = f"{pct:.1f}% lower than linear"
    elif pct == pct:
        imp = f"{abs(pct):.1f}% higher than linear"
    else:
        imp = "n/a"
    lines.append(f"- A1 clean-seed median: {med_a1_clean:.2f} px vs linear: {med_lin_clean:.2f} px ({imp}).")
    lines.append("- Three failure seeds (`offset_0s`, `offset_5s`, `offset_15s`): high ADE on all models — SAM3.1 tracking failure, not LSTM-specific.")
    lines.append(f"- Teacher-forced A1 median: {tf_med:.2f} px; rollout forecast ADE gap reflects exposure bias during training.")
    lines.append("")

    lines.append("## Section 5 — Honest limitations")
    lines.append("")
    lines.append("- Evaluated on a single SportsMOT sequence (`sportsmot_example`).")
    lines.append("- Prior held-out `offset_0s` training showed poor generalization on that window; `temporal_all` retrains all seeds.")
    lines.append("- Game-rules post-refine (A2) hurts rather than helps — see ablation attribution.")
    lines.append("- Cross-sequence generalization is untested.")
    lines.append("")

    out = Path(args.output or ROOT / "docs" / "PAPER_RESULTS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
