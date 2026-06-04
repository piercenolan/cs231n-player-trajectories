"""
Roll LSTM forecasts into full-sequence position arrays and tracks JSON.
"""

import json
from pathlib import Path

import numpy as np
import torch

from models.trajectory_lstm import TrajectoryLSTM
from utils.lstm_dataset import (
    denormalize_positions,
    load_tensor_file,
    normalize_positions,
    norm_stats_from_meta,
)


def load_checkpoint(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TrajectoryLSTM(
        num_players=cfg["num_players"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        pred_len=cfg["pred_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def rollout_positions(model, seq, cfg, device="cpu", stitch="last"):
    """
    Sliding-window LSTM rollout on one sequence dict from load_tensor_file.

    Returns (T, P, 2) pixel positions and (T, P) bool forecast_updated mask.
    """
    obs_len = cfg["obs_len"]
    pred_len = cfg["pred_len"]
    # Always use this sequence's scale (seeds may differ from checkpoint / root export).
    scale = seq.get("scale")
    if scale is None:
        scale = norm_stats_from_meta(seq["meta"])
    scale = np.asarray(scale, dtype=np.float32)

    pos_px = seq["positions"].copy()
    vis = seq["visibility"]
    pos_norm = normalize_positions(pos_px, scale)
    T, P, _ = pos_norm.shape
    win = obs_len + pred_len

    accum = np.zeros((T, P, 2), dtype=np.float32)
    counts = np.zeros((T, P), dtype=np.float32)
    updated = np.zeros((T, P), dtype=bool)

    for start in range(0, T - win + 1):
        x = pos_norm[start : start + obs_len]
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        pred_norm = model(xt).squeeze(0).cpu().numpy()
        pred_px = denormalize_positions(pred_norm, scale)
        t0 = start + obs_len
        for k in range(pred_len):
            t = t0 + k
            for p in range(pred_px.shape[1]):
                if not vis[t, p]:
                    continue
                if stitch == "last":
                    accum[t, p] = pred_px[k, p]
                    counts[t, p] = 1.0
                    updated[t, p] = True
                else:
                    accum[t, p] += pred_px[k, p]
                    counts[t, p] += 1.0
                    updated[t, p] = True

    out = pos_px.copy()
    if stitch == "average":
        mask = counts > 0
        out[mask] = accum[mask] / counts[mask, None]
    else:
        mask = counts > 0
        out[mask] = accum[mask]

    return out, updated


def positions_to_tracks(seq, positions, meta_extra=None):
    """Build tracks JSON compatible with trajectory_metrics.load_tracks_centers."""
    frame_numbers = seq["frame_numbers"]
    player_ids = seq["player_ids"]
    visibility = seq["visibility"]
    T = positions.shape[0]

    frames = []
    for t in range(T):
        players = []
        for slot, pid in enumerate(player_ids):
            if pid < 0:
                continue
            if not visibility[t, slot]:
                continue
            x, y = float(positions[t, slot, 0]), float(positions[t, slot, 1])
            players.append(
                {
                    "id": int(pid),
                    "mask_center": {"x": x, "y": y},
                }
            )
        frames.append({"frame_number": int(frame_numbers[t]), "players": players})

    meta = dict(seq["meta"])
    if meta_extra:
        meta.update(meta_extra)
    return {"meta": meta, "frames": frames}


def linear_extrapolation_positions(seq, cfg):
    """Last-velocity linear baseline for pred_len steps per window, stitched."""
    obs_len = cfg["obs_len"]
    pred_len = cfg["pred_len"]
    pos = seq["positions"].copy()
    vis = seq["visibility"]
    T, P, _ = pos.shape
    win = obs_len + pred_len
    out = pos.copy()
    updated = np.zeros((T, P), dtype=bool)

    for start in range(0, T - win + 1):
        hist = pos[start : start + obs_len]
        if obs_len >= 2:
            vel = hist[-1] - hist[-2]
        else:
            vel = np.zeros((P, 2), dtype=np.float32)
        for k in range(pred_len):
            t = start + obs_len + k
            for p in range(pos.shape[1]):
                if not vis[t, p]:
                    continue
                out[t, p] = hist[-1, p] + vel[p] * (k + 1)
                updated[t, p] = True
    return out, updated


def save_tracks(path, tracks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)
