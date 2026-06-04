#!/usr/bin/env python3
"""Evaluate LSTM variants A0-A3 + per-rule post-refine attribution."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import (
    lstm_out_dir,
    resolve_seed_gt_path,
    runs_dir,
    seed_augmented_tracks_path,
)
from utils.lstm_dataset import load_tensor_file, norm_stats_from_meta, normalize_positions
from utils.lstm_predict import (
    linear_extrapolation_positions,
    load_checkpoint,
    post_refine_tracks,
    positions_to_tracks,
    rollout_positions,
    save_tracks,
)
from utils.trajectory_metrics import compute_ade_fde, forecast_min_frame_from_tracks


def teacher_forced_px(model, cfg, seq, device):
    obs_len, pred_len = cfg["obs_len"], cfg["pred_len"]
    scale = seq.get("scale")
    if scale is None:
        scale = norm_stats_from_meta(seq["meta"])
    scale = np.asarray(scale, dtype=np.float32)
    pos_n = normalize_positions(seq["positions"], scale)
    pos = seq["positions"]
    vis = seq["visibility"]
    rf = seq.get("rule_features")
    model_name = cfg.get("model", "plain")
    T = pos.shape[0]
    win = obs_len + pred_len
    errs, pers = [], []
    with torch.no_grad():
        for start in range(0, T - win + 1):
            x = torch.from_numpy(pos_n[start : start + obs_len]).unsqueeze(0).to(device)
            if model_name == "rule_features":
                rules = torch.from_numpy(rf[start : start + obs_len]).unsqueeze(0).to(device)
                pred = model(x, rules).squeeze(0).cpu().numpy()
            elif model_name == "graph":
                mx = torch.from_numpy(vis[start : start + obs_len]).unsqueeze(0).to(device)
                pred = model(x, mx).squeeze(0).cpu().numpy()
            else:
                pred = model(x).squeeze(0).cpu().numpy()
            pred_px = pred * scale.reshape(1, 1, 2)
            gt = pos[start + obs_len : start + win]
            m = vis[start + obs_len : start + win]
            for k in range(pred_len):
                for p in range(pred.shape[1]):
                    if not m[k, p]:
                        continue
                    errs.append(float(np.hypot(pred_px[k, p, 0] - gt[k, p, 0], pred_px[k, p, 1] - gt[k, p, 1])))
                    pers.append(
                        float(
                            np.hypot(
                                pos[start + obs_len - 1, p, 0] - gt[k, p, 0],
                                pos[start + obs_len - 1, p, 1] - gt[k, p, 1],
                            )
                        )
                    )
    return {
        "teacher_forced_ade_px": float(np.mean(errs)) if errs else float("nan"),
        "persistence_ade_px": float(np.mean(pers)) if pers else float("nan"),
    }


def eval_tracks_ade(tracks_path, gt_path, min_frame=None, max_distance=80.0):
    kw = {"max_distance": max_distance}
    if min_frame is not None:
        kw["min_frame"] = min_frame
    return compute_ade_fde(tracks_path, gt_path, **kw)




def linear_forecast_ade_for_seed(tensor_path, gt_path, lstm_root, seed_id, obs_len=8, pred_len=4):
    seq = load_tensor_file(tensor_path)
    cfg = {"obs_len": obs_len, "pred_len": pred_len}
    lin_pos, _ = linear_extrapolation_positions(seq, cfg)
    lin_tracks = positions_to_tracks(seq, lin_pos)
    lin_p = lstm_root / f"_lin_eval_{seed_id}.json"
    save_tracks(lin_p, lin_tracks)
    min_f = forecast_min_frame_from_tracks(lin_p, obs_len)
    if min_f is not None:
        return eval_tracks_ade(lin_p, gt_path, min_frame=min_f)
    return eval_tracks_ade(lin_p, gt_path)


def sam_forecast_ade_for_seed(dataset, seed_id, gt_path, obs_len=8):
    aug_path = seed_augmented_tracks_path(dataset, seed_id)
    if not aug_path.is_file():
        return {"ade": float("nan"), "fde": float("nan")}
    min_f = forecast_min_frame_from_tracks(aug_path, obs_len)
    if min_f is not None:
        return eval_tracks_ade(aug_path, gt_path, min_frame=min_f)
    return eval_tracks_ade(aug_path, gt_path)


def run_variant(
    variant_id,
    ckpt_path,
    tensor_path,
    gt_path,
    device,
    post_refine=None,
    out_name="predicted_tracks.json",
    autoregressive=False,
):
    model, cfg = load_checkpoint(ckpt_path, device=device)
    seq = load_tensor_file(tensor_path)
    scale = cfg.get("scale")
    if scale is None:
        scale = np.asarray(seq["scale"]).tolist()
    cfg = {**cfg, "scale": scale}
    pred_pos, _ = rollout_positions(
        model, seq, cfg, device=device, autoregressive=autoregressive
    )
    tracks = positions_to_tracks(
        seq, pred_pos, meta_extra={"variant": variant_id, "checkpoint": str(ckpt_path)}
    )
    if post_refine:
        tracks = post_refine_tracks(tracks, rules_preset=post_refine)
    out_path = ckpt_path.parent / out_name
    save_tracks(out_path, tracks)
    min_f = forecast_min_frame_from_tracks(out_path, cfg["obs_len"])
    full = eval_tracks_ade(out_path, gt_path)
    horizon = eval_tracks_ade(out_path, gt_path, min_frame=min_f) if min_f else full
    tf = teacher_forced_px(model, cfg, seq, device)
    return {
        "variant": variant_id,
        "checkpoint": str(ckpt_path),
        "tracks_path": str(out_path),
        "ade_full": full["ade"],
        "fde_full": full["fde"],
        "ade_forecast": horizon["ade"],
        "fde_forecast": horizon["fde"],
        "num_matches": full["num_matches"],
        **tf,
        "post_refine": post_refine,
    }


def per_rule_attribution(plain_ckpt, tensor_path, gt_path, device, dataset):
    """A2 per-rule: refine plain predictions with one rule at a time."""
    from utils.augmentation import GAME_RULES

    model, cfg = load_checkpoint(plain_ckpt, device=device)
    seq = load_tensor_file(tensor_path)
    scale = cfg.get("scale")
    if scale is None:
        scale = np.asarray(seq["scale"]).tolist()
    cfg = {**cfg, "scale": scale}
    pred_pos, _ = rollout_positions(model, seq, cfg, device=device)
    base_tracks = positions_to_tracks(seq, pred_pos, meta_extra={"variant": "A0_plain"})
    tmp_base = plain_ckpt.parent / "_tmp_base_for_attribution.json"
    save_tracks(tmp_base, base_tracks)
    min_f = forecast_min_frame_from_tracks(tmp_base, cfg["obs_len"])
    base_h = eval_tracks_ade(tmp_base, gt_path, min_frame=min_f)

    rows = []
    for rule in ("velocity_cap", "hull_containment", "spacing_push") + tuple(sorted(GAME_RULES)):
        refined = post_refine_tracks(base_tracks, rules_preset=rule)
        p = plain_ckpt.parent / f"attrib_{rule}.json"
        save_tracks(p, refined)
        m = eval_tracks_ade(p, gt_path, min_frame=min_f)
        rows.append(
            {
                "rule": rule,
                "ade_forecast": m["ade"],
                "delta_ade_vs_plain": m["ade"] - base_h["ade"],
            }
        )
    out_csv = lstm_out_dir(dataset) / "lstm_rule_attribution.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "ade_forecast", "delta_ade_vs_plain"])
        w.writeheader()
        w.writerows(rows)

    fig_path = runs_dir(dataset) / "figures" / "lstm_per_rule_delta_ade.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    rules = [r["rule"] for r in rows]
    deltas = [r["delta_ade_vs_plain"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in deltas]
    ax.bar(range(len(rules)), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(rules, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ forecast ADE vs plain LSTM (px)")
    ax.set_title("Per-rule post-refine attribution (negative = improved)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return rows, str(out_csv)


def write_robust_report(seed_rows, lstm_root, dataset, linear_by_seed=None):
    """Per-seed A1 vs A0 delta, median ADE, win rate."""
    linear_by_seed = linear_by_seed or {}
    by_seed = {}
    for row in seed_rows:
        sid = row["seed_id"]
        by_seed.setdefault(sid, {})[row["variant"]] = row

    delta_rows = []
    wins = losses = 0
    a1_beats_linear = 0
    for sid in sorted(by_seed.keys()):
        a0 = by_seed[sid].get("A0_plain", {})
        a1 = by_seed[sid].get("A1_rule_features", {})
        a3 = by_seed[sid].get("A3_graph", {})
        lin_row = by_seed[sid].get("linear_baseline", {})
        fc0 = a0.get("ade_forecast", float("nan"))
        fc1 = a1.get("ade_forecast", float("nan"))
        fc_lin = lin_row.get("ade_forecast", linear_by_seed.get(sid, float("nan")))
        delta = fc1 - fc0 if fc0 == fc0 and fc1 == fc1 else float("nan")
        if delta == delta:
            if delta < -0.01:
                wins += 1
                winner = "A1"
            elif delta > 0.01:
                losses += 1
                winner = "A0"
            else:
                winner = "tie"
        else:
            winner = "na"
        if fc1 == fc1 and fc_lin == fc_lin:
            if fc1 < fc_lin - 0.01:
                winner_vs_linear = "A1"
                a1_beats_linear += 1
            elif fc1 > fc_lin + 0.01:
                winner_vs_linear = "linear"
            else:
                winner_vs_linear = "tie"
        else:
            winner_vs_linear = "na"
        delta_rows.append(
            {
                "seed_id": sid,
                "A0_forecast_ade": fc0,
                "A1_forecast_ade": fc1,
                "A3_forecast_ade": a3.get("ade_forecast"),
                "linear_forecast_ade": fc_lin,
                "delta_A1_minus_A0": delta,
                "winner": winner,
                "winner_vs_linear": winner_vs_linear,
                "A0_teacher_forced": a0.get("teacher_forced_ade_px"),
                "A1_teacher_forced": a1.get("teacher_forced_ade_px"),
            }
        )

    def collect(variant):
        return [
            r["ade_forecast"]
            for r in seed_rows
            if r["variant"] == variant and r.get("ade_forecast") == r.get("ade_forecast")
        ]

    summary = {
        "n_seeds": len(delta_rows),
        "A1_wins_vs_A0": wins,
        "A0_wins_vs_A1": losses,
        "per_seed_delta": delta_rows,
        "robust_aggregate": {},
    }
    for variant in (
        "A0_plain",
        "A1_rule_features",
        "A1_residual",
        "A3_graph",
        "linear_baseline",
        "SAM_augmented",
    ):
        ades = collect(variant)
        if ades:
            summary["robust_aggregate"][variant] = {
                "ade_forecast_mean": float(np.mean(ades)),
                "ade_forecast_median": float(np.median(ades)),
                "ade_forecast_std": float(np.std(ades)),
                "n_seeds": len(ades),
            }

    summary["A1_beats_linear_seeds"] = a1_beats_linear

    report_path = lstm_root / "lstm_ablation_robust.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = lstm_root / "lstm_per_seed_delta.csv"
    if delta_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            w.writeheader()
            w.writerows(delta_rows)

    print(
        f"Robust report: A1 wins {wins}/{len(delta_rows)} seeds | "
        f"median A1={summary['robust_aggregate'].get('A1_rule_features', {}).get('ade_forecast_median', float('nan')):.2f} px"
    )
    return report_path


def plot_bars(rows, out_path):
    names = [r["variant"] for r in rows]
    ades = [r["ade_forecast"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"][: len(names)]
    ax.bar(range(len(names)), ades, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Forecast ADE (px)")
    ax.set_title("LSTM variants (forecast horizon)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LSTM ablation evaluation")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--seed-id", default="offset_0s")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Evaluate all seeds with exported tensors and write multi-seed aggregate",
    )
    parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="Recompute rule features from rolled-out positions (A1)",
    )
    parser.add_argument(
        "--a1-checkpoint",
        default=None,
        help="Override A1 checkpoint path",
    )
    parser.add_argument(
        "--a1-residual-checkpoint",
        default=None,
        help="Override A1 residual checkpoint path",
    )
    parser.add_argument(
        "--diagnose-seeds",
        action="store_true",
        help="Run seed diagnosis and write lstm/seed_diagnosis.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    lstm_root = lstm_out_dir(args.dataset)
    tensor_path = runs_dir(args.dataset) / "seeds" / args.seed_id / "trajectory_tensors.json"
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Missing {tensor_path}")

    gt_path = resolve_seed_gt_path(
        args.dataset,
        args.seed_id,
        runs_dir(args.dataset, args.seed_id) / "baseline_tracks.json",
        align_if_missing=True,
    )

    a1_ckpt = Path(args.a1_checkpoint) if args.a1_checkpoint else (lstm_root / "lstm_rule_features" / "checkpoint.pt")
    a1_res_ckpt = (
        Path(args.a1_residual_checkpoint)
        if args.a1_residual_checkpoint
        else (lstm_root / "lstm_rule_features_residual" / "checkpoint.pt")
    )
    variants = [
        ("A0_plain", lstm_root / "lstm_plain" / "checkpoint.pt", None),
        ("A1_rule_features", a1_ckpt, None),
        ("A3_graph", lstm_root / "lstm_graph" / "checkpoint.pt", None),
        ("A2_post_game", lstm_root / "lstm_plain" / "checkpoint.pt", "game"),
        ("A2_post_physical", lstm_root / "lstm_plain" / "checkpoint.pt", "physical"),
    ]
    if a1_res_ckpt.is_file():
        variants.insert(2, ("A1_residual", a1_res_ckpt, None))

    aug_path = seed_augmented_tracks_path(args.dataset, args.seed_id)
    if aug_path.is_file():
        sam = eval_tracks_ade(aug_path, gt_path)
        lin_pos, _ = linear_extrapolation_positions(
            load_tensor_file(tensor_path),
            {"obs_len": 8, "pred_len": 4},
        )
        lin_tracks = positions_to_tracks(load_tensor_file(tensor_path), lin_pos)
        lin_p = lstm_root / "_lin_eval.json"
        save_tracks(lin_p, lin_tracks)
        lin_m = eval_tracks_ade(lin_p, gt_path)
    else:
        sam = lin_m = {"ade": float("nan"), "fde": float("nan")}

    results = []
    for vid, ckpt, pr in variants:
        if not ckpt.is_file():
            print(f"Skip {vid}: missing {ckpt}")
            continue
        out_name = f"predicted_{vid}.json"
        r = run_variant(
            vid,
            ckpt,
            tensor_path,
            gt_path,
            device,
            post_refine=pr,
            out_name=out_name,
            autoregressive=args.autoregressive,
        )
        results.append(r)
        print(f"{vid}: forecast ADE={r['ade_forecast']:.3f}  teacher-forced={r['teacher_forced_ade_px']:.3f}")

    results.append(
        {
            "variant": "SAM_augmented",
            "ade_forecast": sam["ade"],
            "fde_forecast": sam.get("fde"),
            "teacher_forced_ade_px": float("nan"),
            "persistence_ade_px": float("nan"),
        }
    )
    results.append(
        {
            "variant": "linear_baseline",
            "ade_forecast": lin_m["ade"],
            "fde_forecast": lin_m.get("fde"),
            "teacher_forced_ade_px": float("nan"),
            "persistence_ade_px": float("nan"),
        }
    )

    summary_path = lstm_root / "lstm_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"seed_id": args.seed_id, "gt_path": str(gt_path), "results": results}, f, indent=2)

    csv_path = lstm_root / "lstm_ablation_summary.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

    fig_path = runs_dir(args.dataset) / "figures" / "lstm_rule_ablation_bar.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_bars([r for r in results if r["variant"].startswith("A")], fig_path)

    if not args.skip_attribution and (lstm_root / "lstm_plain" / "checkpoint.pt").is_file():
        per_rule_attribution(
            lstm_root / "lstm_plain" / "checkpoint.pt",
            tensor_path,
            gt_path,
            device,
            args.dataset,
        )

    print(f"Wrote {summary_path}")

    if args.diagnose_seeds:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "diagnose_lstm_seeds.py"),
                "--dataset",
                args.dataset,
            ],
            check=True,
        )

    if args.all_seeds:
        seeds_root = runs_dir(args.dataset) / "seeds"
        seed_ids = sorted(
            d.name
            for d in seeds_root.iterdir()
            if d.is_dir() and (d / "trajectory_tensors.json").is_file()
        )
        seed_rows = []
        linear_by_seed = {}
        for sid in seed_ids:
            tp = runs_dir(args.dataset) / "seeds" / sid / "trajectory_tensors.json"
            if not tp.is_file():
                continue
            gt = resolve_seed_gt_path(
                args.dataset,
                sid,
                runs_dir(args.dataset, sid) / "baseline_tracks.json",
                align_if_missing=True,
            )
            lin_m = linear_forecast_ade_for_seed(tp, gt, lstm_root, sid)
            linear_by_seed[sid] = lin_m["ade"]
            seed_rows.append(
                {
                    "variant": "linear_baseline",
                    "seed_id": sid,
                    "ade_forecast": lin_m["ade"],
                    "fde_forecast": lin_m.get("fde"),
                    "teacher_forced_ade_px": float("nan"),
                    "persistence_ade_px": float("nan"),
                }
            )
            sam_m = sam_forecast_ade_for_seed(args.dataset, sid, gt)
            seed_rows.append(
                {
                    "variant": "SAM_augmented",
                    "seed_id": sid,
                    "ade_forecast": sam_m["ade"],
                    "fde_forecast": sam_m.get("fde"),
                    "teacher_forced_ade_px": float("nan"),
                    "persistence_ade_px": float("nan"),
                }
            )
            for vid, ckpt, pr in variants:
                if pr is not None:
                    continue
                if not ckpt.is_file():
                    continue
                r = run_variant(
                    vid,
                    ckpt,
                    tp,
                    gt,
                    device,
                    post_refine=pr,
                    out_name=f"predicted_{sid}_{vid}.json",
                    autoregressive=args.autoregressive,
                )
                seed_rows.append({**r, "seed_id": sid})
        agg_path = lstm_root / "lstm_ablation_multi_seed.json"
        by_var = {}
        for row in seed_rows:
            by_var.setdefault(row["variant"], []).append(row["ade_forecast"])
        aggregate = {}
        for v, ades in by_var.items():
            aggregate[v] = {
                "ade_forecast_mean": float(np.mean(ades)),
                "ade_forecast_median": float(np.median(ades)),
                "ade_forecast_std": float(np.std(ades)),
                "n_seeds": len(ades),
            }
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump({"per_seed": seed_rows, "aggregate": aggregate}, f, indent=2)
        write_robust_report(seed_rows, lstm_root, args.dataset, linear_by_seed=linear_by_seed)
        print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
