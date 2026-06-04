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


def run_variant(
    variant_id,
    ckpt_path,
    tensor_path,
    gt_path,
    device,
    post_refine=None,
    out_name="predicted_tracks.json",
):
    model, cfg = load_checkpoint(ckpt_path, device=device)
    seq = load_tensor_file(tensor_path)
    scale = cfg.get("scale")
    if scale is None:
        scale = np.asarray(seq["scale"]).tolist()
    cfg = {**cfg, "scale": scale}
    pred_pos, _ = rollout_positions(model, seq, cfg, device=device)
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

    variants = [
        ("A0_plain", lstm_root / "lstm_plain" / "checkpoint.pt", None),
        ("A1_rule_features", lstm_root / "lstm_rule_features" / "checkpoint.pt", None),
        ("A3_graph", lstm_root / "lstm_graph" / "checkpoint.pt", None),
        ("A2_post_game", lstm_root / "lstm_plain" / "checkpoint.pt", "game"),
        ("A2_post_physical", lstm_root / "lstm_plain" / "checkpoint.pt", "physical"),
    ]

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
        r = run_variant(vid, ckpt, tensor_path, gt_path, device, post_refine=pr, out_name=out_name)
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

    if args.all_seeds:
        seeds_root = runs_dir(args.dataset) / "seeds"
        seed_ids = sorted(
            d.name
            for d in seeds_root.iterdir()
            if d.is_dir() and (d / "trajectory_tensors.json").is_file()
        )
        seed_rows = []
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
            for vid, ckpt, pr in variants:
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
                )
                seed_rows.append({**r, "seed_id": sid})
        agg_path = lstm_root / "lstm_ablation_multi_seed.json"
        by_var = {}
        for row in seed_rows:
            by_var.setdefault(row["variant"], []).append(row["ade_forecast"])
        aggregate = {
            v: {
                "ade_forecast_mean": float(np.mean(ades)),
                "ade_forecast_std": float(np.std(ades)),
                "n_seeds": len(ades),
            }
            for v, ades in by_var.items()
        }
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump({"per_seed": seed_rows, "aggregate": aggregate}, f, indent=2)
        print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
