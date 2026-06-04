#!/usr/bin/env python3
"""A1 feature-group ablation: mask rule feature columns at inference."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.datasets import lstm_out_dir, resolve_seed_gt_path, runs_dir
from utils.lstm_dataset import load_tensor_file, norm_stats_from_meta, normalize_positions
from utils.lstm_predict import load_checkpoint, positions_to_tracks, rollout_positions, save_tracks
from utils.rule_features import FEATURE_GROUPS, RULE_FEATURE_DIM, mask_rule_features
from utils.trajectory_metrics import compute_ade_fde, forecast_min_frame_from_tracks

# Ablation configs: name -> groups kept (empty = all features = full A1)
ABLATION_CONFIGS = {
    "full_all_features": list(FEATURE_GROUPS.keys()),
    "kinematic_only": ["kinematic"],
    "social_geometry_only": ["social_geometry"],
    "game_state_only": ["game_state"],
    "kinematic_plus_social": ["kinematic", "social_geometry"],
    "kinematic_plus_game": ["kinematic", "game_state"],
    "no_game_state": ["kinematic", "social_geometry"],
}


@torch.no_grad()
def eval_masked_rollout(model, cfg, seq, device, groups_to_keep):
    """Rollout with masked rule features."""
    obs_len, pred_len = cfg["obs_len"], cfg["pred_len"]
    scale = seq.get("scale")
    if scale is None:
        scale = norm_stats_from_meta(seq["meta"])
    scale = np.asarray(scale, dtype=np.float32)

    seq = dict(seq)
    rf = seq.get("rule_features")
    if rf is None:
        raise ValueError("rule_features required")
    seq["rule_features"] = mask_rule_features(rf, groups_to_keep)

    pred_pos, _ = rollout_positions(model, seq, cfg, device=device)
    return pred_pos


def teacher_forced_masked(model, cfg, seq, device, groups_to_keep):
    obs_len, pred_len = cfg["obs_len"], cfg["pred_len"]
    scale = seq.get("scale")
    if scale is None:
        scale = norm_stats_from_meta(seq["meta"])
    scale = np.asarray(scale, dtype=np.float32)
    pos_n = normalize_positions(seq["positions"], scale)
    pos = seq["positions"]
    vis = seq["visibility"]
    rf = mask_rule_features(seq["rule_features"], groups_to_keep)
    T = pos.shape[0]
    win = obs_len + pred_len
    errs = []
    with torch.no_grad():
        for start in range(0, T - win + 1):
            x = torch.from_numpy(pos_n[start : start + obs_len]).unsqueeze(0).to(device)
            rules = torch.from_numpy(rf[start : start + obs_len]).unsqueeze(0).to(device)
            pred = model(x, rules).squeeze(0).cpu().numpy()
            pred_px = pred * scale.reshape(1, 1, 2)
            gt = pos[start + obs_len : start + win]
            m = vis[start + obs_len : start + win]
            for k in range(pred_len):
                for p in range(pred.shape[1]):
                    if not m[k, p]:
                        continue
                    errs.append(
                        float(np.hypot(pred_px[k, p, 0] - gt[k, p, 0], pred_px[k, p, 1] - gt[k, p, 1]))
                    )
    return float(np.mean(errs)) if errs else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Rule feature group ablation for A1")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--seed-id", default="offset_0s")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    lstm_root = lstm_out_dir(args.dataset)
    ckpt = lstm_root / "lstm_rule_features" / "checkpoint.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {ckpt}")

    model, cfg = load_checkpoint(ckpt, device=device)
    if cfg.get("model") != "rule_features":
        raise ValueError("Checkpoint must be rule_features model")

    seeds_root = runs_dir(args.dataset) / "seeds"
    seed_ids = sorted(
        d.name
        for d in seeds_root.iterdir()
        if d.is_dir() and (d / "trajectory_tensors.json").is_file()
    )
    if not args.all_seeds:
        seed_ids = [args.seed_id]

    rows = []
    for sid in seed_ids:
        tp = seeds_root / sid / "trajectory_tensors.json"
        gt = resolve_seed_gt_path(
            args.dataset,
            sid,
            runs_dir(args.dataset, sid) / "baseline_tracks.json",
            align_if_missing=True,
        )
        seq = load_tensor_file(tp)
        scale = cfg.get("scale")
        if scale is None:
            scale = np.asarray(seq["scale"]).tolist()
        run_cfg = {**cfg, "scale": scale}

        for ab_name, groups in ABLATION_CONFIGS.items():
            pred_pos = eval_masked_rollout(model, run_cfg, seq, device, groups)
            tracks = positions_to_tracks(
                seq, pred_pos, meta_extra={"ablation": ab_name, "groups": groups}
            )
            out = lstm_root / f"predicted_{sid}_{ab_name}.json"
            save_tracks(out, tracks)
            min_f = forecast_min_frame_from_tracks(out, cfg["obs_len"])
            m = compute_ade_fde(out, gt, min_frame=min_f) if min_f else compute_ade_fde(out, gt)
            tf = teacher_forced_masked(model, run_cfg, seq, device, groups)
            rows.append(
                {
                    "seed_id": sid,
                    "ablation": ab_name,
                    "groups_kept": ",".join(groups),
                    "n_features_active": sum(len(FEATURE_GROUPS[g]) for g in groups),
                    "ade_forecast": m["ade"],
                    "fde_forecast": m["fde"],
                    "teacher_forced_ade_px": tf,
                }
            )
            print(f"{sid} {ab_name}: forecast ADE={m['ade']:.3f}  tf={tf:.3f}")

    out_json = lstm_root / "lstm_rule_feature_group_ablation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"configs": ABLATION_CONFIGS, "results": rows}, f, indent=2)

    csv_path = lstm_root / "lstm_rule_feature_group_ablation.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
