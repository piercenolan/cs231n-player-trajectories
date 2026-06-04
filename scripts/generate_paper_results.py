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


def beats_linear(med_model, med_lin):
    if med_model != med_model or med_lin != med_lin:
        return "N/A"
    return "Y" if med_model < med_lin - 0.01 else "N"


def residual_by_seed(multi):
    out = {}
    for r in multi.get("per_seed", []):
        if r.get("variant") == "A1_residual":
            out[r.get("seed_id")] = r.get("ade_forecast")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--multiseq-csv",
        default=None,
        help="Optional aggregate CSV from scripts/aggregate_multiseq_eval.py",
    )
    args = parser.parse_args()

    run_root = runs_dir(args.dataset)
    lstm_root = lstm_out_dir(args.dataset)
    rows = load_csv_rows(lstm_root / "lstm_per_seed_delta.csv")
    robust = load_json(lstm_root / "lstm_ablation_robust.json") or {}
    multi = load_json(lstm_root / "lstm_ablation_multi_seed.json") or {}
    res_map = residual_by_seed(multi)

    if rows and res_map and "A1_residual_forecast_ade" not in rows[0]:
        for r in rows:
            r["A1_residual_forecast_ade"] = res_map.get(r.get("seed_id"))

    base_m = load_json(run_root / "ablations" / "baseline" / "metrics.json")
    aug_m = load_json(run_root / "ablations" / "sanitize_plus_velocity_cap" / "metrics.json")

    lines = ["# Paper Results (auto-generated)", ""]
    lines.append("## Section 1 - Detection metrics (pre-LSTM)")
    lines.append("")
    lines.append("| Metric | SAM3.1 baseline | Augmented |")
    lines.append("|--------|-----------------|-----------|")
    lines.append(metrics_block(base_m, aug_m))
    lines.append("")

    lines.append("## Section 2 - Forecast metrics (LSTM)")
    lines.append("")
    lines.append("| Model | All-seed median ADE | Clean-seed median ADE | Beats linear |")
    lines.append("|-------|---------------------|------------------------|--------------|")
    med_lin_all = median_col(rows, "linear_forecast_ade")
    med_lin_clean = median_col(rows, "linear_forecast_ade", exclude=CLEAN_EXCLUDE)
    models = [
        ("Linear", "linear_forecast_ade"),
        ("A0 Plain", "A0_forecast_ade"),
        ("A1 Rule-Conditioned", "A1_forecast_ade"),
        ("A1 Residual (headline)", "A1_residual_forecast_ade"),
        ("A3 Graph", "A3_forecast_ade"),
    ]
    agg = robust.get("robust_aggregate") or {}
    for name, col in models:
        med_all = median_col(rows, col)
        if med_all != med_all and name == "A1 Residual (headline)":
            med_all = agg.get("A1_residual", {}).get("ade_forecast_median", float("nan"))
        med_clean = median_col(rows, col, exclude=CLEAN_EXCLUDE)
        if med_clean != med_clean and name == "A1 Residual (headline)":
            vals = [
                float(r[col])
                for r in rows
                if r.get("seed_id") not in CLEAN_EXCLUDE and col in r
            ]
            vals = [v for v in vals if v == v]
            if not vals and res_map:
                vals = [
                    res_map[s]
                    for s in res_map
                    if s not in CLEAN_EXCLUDE and res_map[s] == res_map[s]
                ]
            med_clean = float(np.median(vals)) if vals else float("nan")
        bl = "—" if name == "Linear" else beats_linear(med_all, med_lin_all)
        lines.append(f"| {name} | {med_all:.2f} | {med_clean:.2f} | {bl} |")
    lines.append("")

    lines.append("## Section 3 - Per-seed breakdown")
    lines.append("")
    lines.append("| Seed | A0 | A1 | A1 Residual | Linear | Residual vs linear |")
    lines.append("|------|-----|-----|-------------|--------|---------------------|")
    for r in sorted(rows, key=lambda x: x.get("seed_id", "")):
        res = r.get("A1_residual_forecast_ade") or res_map.get(r.get("seed_id"), float("nan"))
        try:
            res_f = float(res)
        except (TypeError, ValueError):
            res_f = float("nan")
        wres = r.get("winner_residual_vs_linear", "n/a")
        lines.append(
            f"| {r.get('seed_id')} | {float(r.get('A0_forecast_ade', 'nan')):.2f} | "
            f"{float(r.get('A1_forecast_ade', 'nan')):.2f} | {res_f:.2f} | "
            f"{float(r.get('linear_forecast_ade', 'nan')):.2f} | {wres} |"
        )
    lines.append("")

    a1_wins = robust.get("A1_wins_vs_A0", sum(1 for r in rows if r.get("winner") == "A1"))
    n_seeds = robust.get("n_seeds", len(rows))
    med_res_clean = median_col(rows, "A1_residual_forecast_ade", exclude=CLEAN_EXCLUDE)
    if med_res_clean != med_res_clean:
        med_res_clean = agg.get("A1_residual", {}).get("ade_forecast_median", float("nan"))
    med_res_all = agg.get("A1_residual", {}).get("ade_forecast_median", median_col(rows, "A1_residual_forecast_ade"))
    res_beats = robust.get("A1_residual_beats_linear_seeds", 0)

    tf_a1 = [
        r.get("teacher_forced_ade_px")
        for r in multi.get("per_seed", [])
        if r.get("variant") == "A1_rule_features"
    ]
    tf_med = float(np.median([float(x) for x in tf_a1 if x == x])) if tf_a1 else float("nan")

    lines.append("## Section 4 - Key findings")
    lines.append("")
    lines.append(f"- A1 beats A0 on {a1_wins}/{n_seeds} seeds (held-out-seed training eval).")
    lines.append(
        f"- **A1 Residual** median forecast ADE: {med_res_all:.2f} px (all seeds) / "
        f"{med_res_clean:.2f} px (clean); linear: {med_lin_all:.2f} / {med_lin_clean:.2f} px."
    )
    lines.append(f"- Residual beats linear on {res_beats}/{n_seeds} seeds individually.")
    lines.append("- Three failure seeds (`offset_0s`, `offset_5s`, `offset_15s`): high ADE on all models — SAM3.1 tracking failure, not LSTM-specific.")
    lines.append(f"- Teacher-forced A1 median: {tf_med:.2f} px; rollout gap reflects exposure bias during training.")
    lines.append("")

    multiseq_path = Path(args.multiseq_csv or ROOT / "data" / "runs" / "multiseq_transfer_summary.csv")
    if multiseq_path.is_file():
        ms_rows = load_csv_rows(multiseq_path)
        lines.append("## Section 5 - Cross-sequence transfer (eval-only)")
        lines.append("")
        lines.append("| Dataset | Median residual ADE | Median linear ADE | Residual beats linear |")
        lines.append("|---------|---------------------|-------------------|------------------------|")
        for r in ms_rows:
            lines.append(
                f"| {r.get('dataset')} | {r.get('median_residual_ade', 'n/a')} | "
                f"{r.get('median_linear_ade', 'n/a')} | {r.get('residual_beats_linear', 'n/a')} |"
            )
        lines.append("")
        sec_lim = 6
    else:
        sec_lim = 5

    lines.append(f"## Section {sec_lim} - Honest limitations")
    lines.append("")
    if not multiseq_path.is_file():
        lines.append("- Cross-sequence transfer table pending (`scripts/aggregate_multiseq_eval.py`).")
    lines.append("- LSTM trained only on `sportsmot_example`; other clips use the same checkpoint (transfer).")
    lines.append("- Game-rules post-refine (A2) hurts rather than helps — see ablation attribution.")
    lines.append("- SAM augmented tracks on future frames are a detection ceiling, not a fair forecast baseline.")
    lines.append("")

    out = Path(args.output or ROOT / "docs" / "PAPER_RESULTS.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
