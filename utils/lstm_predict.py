"""
Roll LSTM forecasts into full-sequence position arrays and tracks JSON.
"""

import json
from pathlib import Path

import numpy as np
import torch

from models.trajectory_graph_lstm import TrajectoryGraphLSTM
from models.trajectory_lstm import RuleConditionedLSTM, TrajectoryLSTM
from utils.linear_baseline import linear_prediction_norm
from utils.lstm_dataset import (
    denormalize_positions,
    load_tensor_file,
    normalize_positions,
    norm_stats_from_meta,
)


def build_model_from_config(cfg):
    model_name = cfg.get("model", "plain")
    if model_name == "rule_features":
        return RuleConditionedLSTM(
            num_players=cfg["num_players"],
            rule_feature_dim=cfg.get("rule_feature_dim", 15),
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            pred_len=cfg["pred_len"],
        )
    if model_name == "graph":
        return TrajectoryGraphLSTM(
            num_players=cfg["num_players"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            pred_len=cfg["pred_len"],
        )
    return TrajectoryLSTM(
        num_players=cfg["num_players"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        pred_len=cfg["pred_len"],
    )


def load_checkpoint(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def _rule_features_for_window(seq, pos_px, scale, start, obs_len, autoregressive):
    """Rule feature slice for one obs window."""
    if not autoregressive:
        rf = seq.get("rule_features")
        if rf is None:
            raise ValueError("rule_features required for rule_features model")
        return rf[start : start + obs_len]

    from utils.rule_features import compute_rule_features_from_positions

    meta = seq.get("meta") or {}
    fw = int(meta.get("frame_width", 640))
    fh = int(meta.get("frame_height", 360))
    full_rf = compute_rule_features_from_positions(
        pos_px,
        seq["visibility"],
        seq["frame_numbers"],
        seq["player_ids"],
        frame_width=fw,
        frame_height=fh,
    )
    return full_rf[start : start + obs_len]


@torch.no_grad()
def rollout_positions(model, seq, cfg, device="cpu", stitch="last", autoregressive=False):
    """
    Sliding-window LSTM rollout on one sequence dict from load_tensor_file.

    autoregressive: feed predicted positions back into history and recompute rule
    features each window (rule_features model only).

    Returns (T, P, 2) pixel positions and (T, P) bool forecast_updated mask.
    """
    obs_len = cfg["obs_len"]
    pred_len = cfg["pred_len"]
    scale = seq.get("scale")
    if scale is None:
        scale = norm_stats_from_meta(seq["meta"])
    scale = np.asarray(scale, dtype=np.float32)

    pos_px = seq["positions"].copy().astype(np.float32)
    vis = seq["visibility"]
    model_name = cfg.get("model", "plain")
    T, P, _ = pos_px.shape
    win = obs_len + pred_len

    accum = np.zeros((T, P, 2), dtype=np.float32)
    counts = np.zeros((T, P), dtype=np.float32)
    updated = np.zeros((T, P), dtype=bool)

    for start in range(0, T - win + 1):
        pos_norm = normalize_positions(pos_px, scale)
        x = pos_norm[start : start + obs_len]
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        if model_name == "rule_features":
            rf = _rule_features_for_window(
                seq, pos_px, scale, start, obs_len, autoregressive
            )
            rft = torch.from_numpy(rf).unsqueeze(0).to(device)
            delta_norm = model(xt, rft).squeeze(0).cpu().numpy()
        elif model_name == "graph":
            mx = torch.from_numpy(vis[start : start + obs_len]).unsqueeze(0).to(device)
            delta_norm = model(xt, mx).squeeze(0).cpu().numpy()
        else:
            delta_norm = model(xt).squeeze(0).cpu().numpy()
        if cfg.get("residual"):
            lin_norm = linear_prediction_norm(x, pred_len)
            pred_norm = lin_norm + delta_norm
        else:
            pred_norm = delta_norm
        pred_px = denormalize_positions(pred_norm, scale)
        t0 = start + obs_len
        for k in range(pred_len):
            t = t0 + k
            for p in range(pred_px.shape[1]):
                if not vis[t, p]:
                    continue
                if autoregressive:
                    pos_px[t, p] = pred_px[k, p]
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


RULE_PRESETS = {
    "physical": "velocity_cap,hull_containment,spacing_push",
    "game": "collective_momentum,stationary_persistence,cut_continuation,"
    "mirror_prediction,cluster_cohesion,convergence_pull,divergence_spread,"
    "isolated_player_hold,dead_ball_freeze",
    "full": None,
}


def post_refine_tracks(tracks, rules_preset="game", gap_fill=False):
    """Apply augmentation rules to predicted tracks JSON (A2)."""
    from utils.augmentation import apply_augmentation, resolve_enabled_rules

    meta = dict(tracks.get("meta") or {})
    fw = int(meta.get("frame_width", 640))
    fh = int(meta.get("frame_height", 360))
    frames = tracks.get("frames", [])
    if rules_preset == "full":
        enabled = resolve_enabled_rules(level="full", gap_fill=gap_fill)
    elif rules_preset in RULE_PRESETS:
        enabled = resolve_enabled_rules(rules=RULE_PRESETS[rules_preset], gap_fill=gap_fill)
    else:
        enabled = resolve_enabled_rules(rules=rules_preset, gap_fill=gap_fill)
    aug_frames, _log = apply_augmentation(
        frames,
        fw,
        fh,
        enabled_rules=enabled,
        sanitize=False,
        gap_fill=gap_fill,
    )
    return {"meta": {**meta, "post_refine": rules_preset}, "frames": aug_frames}
