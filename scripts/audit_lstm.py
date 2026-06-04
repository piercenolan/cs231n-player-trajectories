#!/usr/bin/env python3
"""Diagnose LSTM ADE: teacher-forced error vs rollout vs data size."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trajectory_lstm import TrajectoryLSTM, masked_mse
from utils.datasets import lstm_out_dir, trajectory_tensor_path
from utils.lstm_dataset import (
    TrajectoryWindowDataset,
    build_dataloaders,
    denormalize_positions,
    load_tensor_file,
    normalize_positions,
)
from utils.lstm_predict import load_checkpoint, rollout_positions
from utils.trajectory_metrics import compute_ade_fde, match_frame, load_ground_truth, load_tracks_centers


def teacher_forced_px_error(model, seq, cfg, device):
    """Per-window L2 in pixels on visible future slots only (matches training target)."""
    obs_len, pred_len = cfg["obs_len"], cfg["pred_len"]
    scale = np.array(cfg.get("scale") or seq["scale"], dtype=np.float32)
    pos_norm = normalize_positions(seq["positions"], scale)
    vis = seq["visibility"]
    T = pos_norm.shape[0]
    win = obs_len + pred_len
    errs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, T - win + 1):
            x = torch.from_numpy(pos_norm[start : start + obs_len]).unsqueeze(0).to(device)
            y_true = pos_norm[start + obs_len : start + win]
            mask = vis[start + obs_len : start + win]
            pred = model(x).squeeze(0).cpu().numpy()
            pred_px = denormalize_positions(pred, scale)
            gt_px = denormalize_positions(y_true, scale)
            for k in range(pred_len):
                for p in range(pred.shape[1]):
                    if not mask[k, p]:
                        continue
                    d = np.hypot(
                        pred_px[k, p, 0] - gt_px[k, p, 0],
                        pred_px[k, p, 1] - gt_px[k, p, 1],
                    )
                    errs.append(d)
    return np.array(errs) if errs else np.array([np.nan])


def main():
    dataset = "sportsmot_example"
    device = torch.device("cpu")
    ckpt_path = lstm_out_dir(dataset) / "checkpoint.pt"
    model, cfg = load_checkpoint(ckpt_path, device=device)

    val_path = trajectory_tensor_path(dataset, "offset_0s")
    seq = load_tensor_file(val_path)
    cfg = {**cfg, "scale": cfg.get("scale") or seq["scale"].tolist()}

    tf_errs = teacher_forced_px_error(model, seq, cfg, device)
    print("=== Teacher-forced (SAM history, visible slots only) ===")
    print(f"  Windows: {45 - cfg['obs_len'] - cfg['pred_len'] + 1}")
    print(f"  Matches: {len(tf_errs)}")
    print(f"  Mean L2 (px): {np.nanmean(tf_errs):.3f}")
    print(f"  Median L2:    {np.nanmedian(tf_errs):.3f}")
    print(f"  P95 L2:       {np.nanpercentile(tf_errs, 95):.3f}")

    pred_pos, updated = rollout_positions(model, seq, cfg, device=device)
    orig = seq["positions"]
    vis = seq["visibility"]
    stitch_errs = []
    for t in range(orig.shape[0]):
        for p in range(orig.shape[1]):
            if not updated[t, p] or not vis[t, p]:
                continue
            d = np.hypot(
                pred_pos[t, p, 0] - orig[t, p, 0],
                pred_pos[t, p, 1] - orig[t, p, 1],
            )
            stitch_errs.append(d)
    print("\n=== Stitched LSTM vs SAM input (visible + updated frames) ===")
    print(f"  Mean displacement vs SAM: {np.mean(stitch_errs):.3f} px")

    # Invisible slots given non-zero LSTM output on updated frames
    ghost = 0
    for t in range(orig.shape[0]):
        if not updated.any(axis=1)[t] if updated.ndim == 2 else False:
            pass
        for p in range(orig.shape[1]):
            if updated[t, p] and not vis[t, p]:
                if np.abs(pred_pos[t, p]).sum() > 1e-3:
                    ghost += 1
    print(f"  Ghost predictions (updated but slot invisible): {ghost}")

    train_loader, val_loader, _, split = build_dataloaders(dataset, "multi", cfg["obs_len"], cfg["pred_len"])
    print(f"\n=== Data scale ===")
    print(f"  Train windows: {len(train_loader.dataset)}")
    print(f"  Val windows:   {len(val_loader.dataset)}")

    # Normalized val loss check
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["x"].to(device))
            loss = masked_mse(pred, batch["y"].to(device), batch["mask_y"].to(device))
            total += loss.item()
            n += 1
    print(f"  Val masked MSE (norm): {total / max(n, 1):.6f}")

    gt_path = val_path.parent / "gt_aligned.json"
    pred_tracks = lstm_out_dir(dataset) / "predicted_tracks.json"
    if pred_tracks.exists() and gt_path.exists():
        m = compute_ade_fde(pred_tracks, gt_path)
        print(f"\n=== Full-track ADE vs GT (current eval) ===")
        print(f"  ADE: {m['ade']:.3f} px  matches: {m['num_matches']}")


if __name__ == "__main__":
    main()
