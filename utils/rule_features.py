"""
Per-frame rule / social features for LSTM conditioning (geometry-free).

Features reuse detectors from utils.augmentation without mutating tracks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.augmentation import (
    build_position_history,
    convex_hull_margin,
    detect_collective_direction,
    detect_convergence,
    detect_dead_ball,
    detect_divergence,
    detect_direction_change,
    detect_fast_break,
    detect_half_court_set,
    detect_player_clustering,
    detect_spacing_violation,
    detect_stationary_players,
    detect_transition,
    distance,
    estimate_speed,
    estimate_velocity,
)

# Per-slot feature names (order fixed for export / model)
RULE_FEATURE_NAMES = [
    "speed_norm",
    "vel_x_norm",
    "vel_y_norm",
    "hull_margin_norm",
    "nearest_neighbor_norm",
    "min_pair_spacing_norm",
    "is_stationary",
    "direction_change_norm",
    "frame_fast_break",
    "frame_transition",
    "frame_half_court",
    "frame_convergence",
    "frame_divergence",
    "frame_dead_ball",
    "frame_cluster",
]

RULE_FEATURE_DIM = len(RULE_FEATURE_NAMES)


def _slot_players_from_frame(frame_players, player_ids, id_to_slot):
    """Build list of player dicts aligned to fixed slots."""
    by_id = {p["id"]: p for p in frame_players}
    out = []
    for pid in player_ids:
        if pid < 0 or pid not in by_id:
            out.append(None)
        else:
            p = by_id[pid]
            out.append(
                {
                    "id": pid,
                    "mask_center": dict(p.get("mask_center", {})),
                }
            )
    return out


def compute_rule_features_tensor(
    tracks_path=None,
    frames=None,
    player_ids=None,
    frame_width=640,
    frame_height=360,
    max_players=10,
):
    """
    Compute (T, P, F) rule features aligned with trajectory_export slots.

    Provide tracks_path or pre-loaded frames list (sorted).
    """
    if frames is None:
        if tracks_path is None:
            raise ValueError("tracks_path or frames required")
        with open(tracks_path, encoding="utf-8") as f:
            data = json.load(f)
        frames = sorted(data.get("frames", []), key=lambda fr: int(fr["frame_number"]))
        meta = data.get("meta") or {}
        frame_width = int(meta.get("frame_width", frame_width))
        frame_height = int(meta.get("frame_height", frame_height))

    if not frames:
        raise ValueError("no frames")

    if player_ids is None:
        id_counts = {}
        for fr in frames:
            for p in fr.get("players", []):
                if not p.get("predicted"):
                    pid = p["id"]
                    id_counts[pid] = id_counts.get(pid, 0) + 1
        player_ids = sorted(id_counts.keys(), key=lambda i: id_counts[i], reverse=True)[
            :max_players
        ]
        while len(player_ids) < max_players:
            player_ids.append(-1)
    else:
        player_ids = list(player_ids)

    P = len(player_ids)
    T = len(frames)
    feats = np.zeros((T, P, RULE_FEATURE_DIM), dtype=np.float32)
    id_to_slot = {pid: i for i, pid in enumerate(player_ids) if pid >= 0}

    history = build_position_history(frames)
    w, h = float(frame_width), float(frame_height)
    diag = max(np.hypot(w, h), 1.0)

    for t, frame in enumerate(frames):
        fnum = int(frame["frame_number"])
        frame_players = [
            {
                "id": p["id"],
                "mask_center": dict(p.get("mask_center", {})),
            }
            for p in frame.get("players", [])
            if not p.get("predicted")
        ]

        player_positions = {}
        for pid in id_to_slot:
            if pid in history:
                player_positions[pid] = [
                    e for e in history[pid] if e[0] <= fnum
                ]

        is_fb, _ = detect_fast_break(player_positions, fnum)
        is_tr, _, _, _ = detect_transition(player_positions, fnum)
        is_hc, _ = detect_half_court_set(player_positions, fnum)
        is_conv, _, _ = detect_convergence(player_positions, fnum)
        is_div, _ = detect_divergence(player_positions, fnum)
        is_dead, _ = detect_dead_ball(player_positions, fnum)
        cluster_labels = detect_player_clustering(frame_players)
        stationary = set(detect_stationary_players(player_positions, fnum))

        frame_flags = np.array(
            [
                float(is_fb),
                float(is_tr),
                float(is_hc),
                float(is_conv),
                float(is_div),
                float(is_dead),
                float(len(cluster_labels) >= 4),
            ],
            dtype=np.float32,
        )

        slot_players = _slot_players_from_frame(frame_players, player_ids, id_to_slot)
        visible_centers = []
        for sp in slot_players:
            if sp is not None:
                c = sp["mask_center"]
                visible_centers.append((float(c["x"]), float(c["y"])))

        for slot, pid in enumerate(player_ids):
            if pid < 0:
                feats[t, slot, 8:15] = frame_flags
                continue
            sp = slot_players[slot]
            if sp is None:
                feats[t, slot, 8:15] = frame_flags
                continue

            x = float(sp["mask_center"]["x"])
            y = float(sp["mask_center"]["y"])
            hist = player_positions.get(pid, [])
            dx, dy = estimate_velocity(hist, fnum)
            spd = estimate_speed(dx, dy)
            others = [p for p in frame_players if p["id"] != pid]
            hull_m = convex_hull_margin(x, y, others, margin=80.0)
            hull_norm = 0.0 if hull_m else min(1.0, 80.0 / diag)

            nn_dist = diag
            min_pair = diag
            if visible_centers:
                dists = [
                    np.hypot(x - ox, y - oy)
                    for ox, oy in visible_centers
                    if not (abs(ox - x) < 1e-3 and abs(oy - y) < 1e-3)
                ]
                if dists:
                    nn_dist = min(dists)
            spacing_pairs = detect_spacing_violation(frame_players, min_spacing=35)
            for a, b in spacing_pairs:
                if pid in (a, b):
                    other = b if a == pid else a
                    for p in frame_players:
                        if p["id"] == other:
                            c = p["mask_center"]
                            d = np.hypot(x - c["x"], y - c["y"])
                            min_pair = min(min_pair, d)

            _, dir_chg = detect_direction_change(player_positions, fnum, pid)
            dir_norm = min(1.0, dir_chg / 180.0)

            feats[t, slot, 0] = min(1.0, spd / 50.0)
            feats[t, slot, 1] = dx / w
            feats[t, slot, 2] = dy / h
            feats[t, slot, 3] = hull_norm
            feats[t, slot, 4] = min(1.0, nn_dist / diag)
            feats[t, slot, 5] = min(1.0, min_pair / diag)
            feats[t, slot, 6] = 1.0 if pid in stationary else 0.0
            feats[t, slot, 7] = dir_norm
            feats[t, slot, 8:15] = frame_flags

    return feats, player_ids, RULE_FEATURE_NAMES


def attach_rule_features_to_export(export, tracks_path, max_players=10):
    """Add rule_features array to an export dict (mutates and returns export)."""
    feats, pids, names = compute_rule_features_tensor(
        tracks_path=tracks_path,
        player_ids=export.get("player_ids"),
        frame_width=export["meta"].get("frame_width", 640),
        frame_height=export["meta"].get("frame_height", 360),
        max_players=max_players,
    )
    export["rule_features"] = feats.tolist()
    export["rule_feature_names"] = names
    export["meta"]["rule_feature_dim"] = RULE_FEATURE_DIM
    return export
