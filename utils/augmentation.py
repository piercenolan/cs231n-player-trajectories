"""
Basketball-domain, geometry-free augmentation for SAM3.1 tracks.

This module post-processes raw tracking output using only relative player
relationships and velocity patterns. It intentionally avoids court coordinates,
basket locations, and camera calibration assumptions.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_tracks(tracks_path):
    """
    Load tracks.json and return frames sorted by frame number.

    Sorting stabilizes temporal logic for velocity- and continuity-based rules,
    and remains geometry-free because it uses only time ordering.
    """
    tracks_path = Path(tracks_path)
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    return sorted(frames, key=lambda fr: int(fr.get("frame_number", 0)))


def build_position_history(frames):
    """
    Build per-player temporal position history from frame data.

    Returns {player_id: [(frame_number, x, y), ...]} for all observed players.
    """
    history = defaultdict(list)
    for frame in sorted(frames, key=lambda fr: int(fr["frame_number"])):
        fnum = int(frame["frame_number"])
        for p in frame.get("players", []):
            history[p["id"]].append((fnum, float(p["mask_center"]["x"]), float(p["mask_center"]["y"])))
    return dict(history)


def _history_until_frame(history, frame_number):
    return [entry for entry in history if entry[0] <= frame_number]


def _position_at_or_before(history, frame_number):
    subset = _history_until_frame(history, frame_number)
    if not subset:
        return None
    _, x, y = subset[-1]
    return x, y


def estimate_velocity(history, frame_number, window=3):
    """
    Estimate mean velocity (dx, dy) over recent steps up to frame_number.

    Velocity is computed from temporal displacements in history only, independent
    of any absolute court coordinate assumptions.
    """
    subset = _history_until_frame(history, frame_number)
    if len(subset) < 2:
        return 0.0, 0.0
    displacements = []
    for (f0, x0, y0), (f1, x1, y1) in zip(subset, subset[1:]):
        dt = max(1, f1 - f0)
        displacements.append(((x1 - x0) / dt, (y1 - y0) / dt))
    displacements = displacements[-window:]
    if not displacements:
        return 0.0, 0.0
    arr = np.array(displacements, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def estimate_speed(dx, dy):
    """Return scalar speed from velocity components."""
    return float(math.hypot(dx, dy))


def direction_angle(dx, dy):
    """
    Return motion direction angle in degrees using atan2.

    Returns None when stationary to avoid forcing meaningless directions.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return float((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0)


def angle_difference(a1, a2):
    """Return smallest circular angular difference in degrees (0..180)."""
    if a1 is None or a2 is None:
        return 180.0
    diff = abs((a1 - a2 + 180.0) % 360.0 - 180.0)
    return float(diff)


def predict_next_position(x, y, dx, dy):
    """One-step kinematic prediction at 1 FPS."""
    return float(x + dx), float(y + dy)


def distance(x1, y1, x2, y2):
    """Euclidean distance between two points."""
    return float(math.hypot(x2 - x1, y2 - y1))


def player_centroid(frame_players):
    """
    Mean player center for a frame.

    Uses relative group position only (no court map), returning None for empty.
    """
    if not frame_players:
        return None
    pts = np.array([[p["mask_center"]["x"], p["mask_center"]["y"]] for p in frame_players], dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def player_spread(frame_players):
    """
    Scalar spread of player positions.

    High spread suggests transition/spacing; low spread suggests clustering.
    """
    if len(frame_players) < 2:
        return 0.0
    pts = np.array([[p["mask_center"]["x"], p["mask_center"]["y"]] for p in frame_players], dtype=float)
    centroid = pts.mean(axis=0)
    dists = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
    return float(dists.std())


def _convex_hull(points):
    """Monotonic chain convex hull with numpy-friendly tuples."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _point_in_poly(x, y, poly):
    """Ray-casting point in polygon."""
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        intersect = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        )
        if intersect:
            inside = not inside
    return inside


def _point_segment_distance(px, py, x1, y1, x2, y2):
    """Distance from point to line segment."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-9:
        return distance(px, py, x1, y1)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    projx, projy = x1 + t * vx, y1 + t * vy
    return distance(px, py, projx, projy)


def convex_hull_margin(x, y, frame_players, margin=60):
    """
    Test whether a point is inside or near the convex hull of other players.

    This replaces court-boundary heuristics with a purely relational geometry:
    players far outside the group's hull are anomalous for broadcast-agnostic
    augmentation.
    """
    pts = [(p["mask_center"]["x"], p["mask_center"]["y"]) for p in frame_players]
    if len(pts) < 3:
        return True
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return True
    if _point_in_poly(x, y, hull):
        return True
    min_dist = min(
        _point_segment_distance(x, y, hx1, hy1, hx2, hy2)
        for (hx1, hy1), (hx2, hy2) in zip(hull, hull[1:] + hull[:1])
    )
    return min_dist <= margin


def kmeans2(points, max_iter=20):
    """
    K=2 clustering using only numpy.

    Returns integer labels for each point. Handles degenerate inputs by
    returning all zeros.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=int)

    c1 = points[0].copy()
    far_idx = int(np.argmax(np.sum((points - c1) ** 2, axis=1)))
    c2 = points[far_idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d1 = np.sum((points - c1) ** 2, axis=1)
        d2 = np.sum((points - c2) ** 2, axis=1)
        new_labels = (d2 < d1).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if np.any(labels == 0):
            c1 = points[labels == 0].mean(axis=0)
        if np.any(labels == 1):
            c2 = points[labels == 1].mean(axis=0)
    return labels


def detect_collective_direction(player_positions, frame_number, agreement_threshold=0.6):
    """
    Detect dominant team movement direction via 8-bin velocity quantization.

    Returns (dominant_angle_or_None, confidence). This captures transition flow
    without any notion of "left basket/right basket" coordinates.
    """
    angles = []
    for history in player_positions.values():
        dx, dy = estimate_velocity(history, frame_number)
        a = direction_angle(dx, dy)
        if a is not None and estimate_speed(dx, dy) > 1e-3:
            angles.append(a)
    if not angles:
        return None, 0.0

    bins = np.zeros(8, dtype=int)
    for a in angles:
        bins[int(a // 45) % 8] += 1
    dominant_bin = int(np.argmax(bins))
    agree = bins[dominant_bin]
    confidence = float(agree / max(1, len(angles)))
    if confidence < agreement_threshold:
        return None, confidence
    return float(dominant_bin * 45.0), confidence


def detect_fast_break(
    player_positions,
    frame_number,
    speed_threshold=25,
    agreement_threshold=0.5,
    min_players=3,
):
    """
    Detect fast-break style motion: many players moving fast in one direction.

    Uses only speed and directional agreement, robust to camera pan/zoom.
    """
    movers = []
    for pid, history in player_positions.items():
        dx, dy = estimate_velocity(history, frame_number)
        spd = estimate_speed(dx, dy)
        ang = direction_angle(dx, dy)
        if spd >= speed_threshold and ang is not None:
            movers.append((pid, spd, ang))
    if len(movers) < min_players:
        return False, 0.0

    angles = np.array([m[2] for m in movers], dtype=float)
    best_frac = 0.0
    for a in angles:
        aligned = sum(1 for b in angles if angle_difference(a, b) <= 60.0)
        best_frac = max(best_frac, aligned / len(angles))
    ok = best_frac >= agreement_threshold
    return ok, float(best_frac)


def detect_transition(player_positions, frame_number, flip_threshold=120, min_players=3):
    """
    Detect possession-like transitions via multi-player direction reversal.

    Compares each player's current heading to heading two frames earlier.
    """
    flipped = 0
    used = 0
    old_angles = []
    new_angles = []
    for history in player_positions.values():
        now_dx, now_dy = estimate_velocity(history, frame_number)
        old_dx, old_dy = estimate_velocity(history, frame_number - 2)
        a_now = direction_angle(now_dx, now_dy)
        a_old = direction_angle(old_dx, old_dy)
        if a_now is None or a_old is None:
            continue
        used += 1
        old_angles.append(a_old)
        new_angles.append(a_now)
        if angle_difference(a_old, a_now) >= flip_threshold:
            flipped += 1
    if used == 0:
        return False, 0.0, None, None
    confidence = float(flipped / used)
    if flipped < min_players:
        return False, confidence, (float(np.mean(old_angles)) if old_angles else None), (
            float(np.mean(new_angles)) if new_angles else None
        )
    return True, confidence, float(np.mean(old_angles)), float(np.mean(new_angles))


def detect_half_court_set(player_positions, frame_number, speed_threshold=12):
    """
    Detect slower, deliberate half-court behavior through mean speed.

    Purely motion-based, independent of absolute court zones.
    """
    speeds = []
    for history in player_positions.values():
        dx, dy = estimate_velocity(history, frame_number)
        speeds.append(estimate_speed(dx, dy))
    if not speeds:
        return False, 0.0
    mean_speed = float(np.mean(speeds))
    confidence = max(0.0, min(1.0, 1.0 - mean_speed / max(1e-6, speed_threshold)))
    return mean_speed < speed_threshold, confidence


def detect_stationary_players(player_positions, frame_number, threshold_pixels=6, min_frames=3):
    """
    Detect players with minimal movement over recent frames.

    Useful for screeners/post defenders/dead-ball moments using only temporal
    displacement.
    """
    stationary = []
    for pid, history in player_positions.items():
        subset = _history_until_frame(history, frame_number)
        if len(subset) < min_frames:
            continue
        recent = subset[-min_frames:]
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        move = distance(xs[0], ys[0], xs[-1], ys[-1])
        if move <= threshold_pixels:
            stationary.append(pid)
    return stationary


def detect_spacing_violation(frame_players, min_spacing=35):
    """
    Detect suspiciously close player pairs (possible mask merge errors).

    Basketball players can contact briefly, but repeated severe overlaps
    suggest tracker artifacts.
    """
    pairs = []
    for i in range(len(frame_players)):
        p1 = frame_players[i]
        x1, y1 = p1["mask_center"]["x"], p1["mask_center"]["y"]
        for j in range(i + 1, len(frame_players)):
            p2 = frame_players[j]
            x2, y2 = p2["mask_center"]["x"], p2["mask_center"]["y"]
            if distance(x1, y1, x2, y2) < min_spacing:
                pairs.append((p1["id"], p2["id"]))
    return pairs


def detect_direction_change(history, frame_number, player_id, threshold=65):
    """
    Detect abrupt personal direction changes (cuts/rotations/hesitations).

    Compares two recent velocity vectors from the same player's history.
    """
    h = history.get(player_id, [])
    subset = _history_until_frame(h, frame_number)
    if len(subset) < 4:
        return False, 0.0
    v1 = (
        subset[-2][1] - subset[-3][1],
        subset[-2][2] - subset[-3][2],
    )
    v2 = (
        subset[-1][1] - subset[-2][1],
        subset[-1][2] - subset[-2][2],
    )
    a1 = direction_angle(v1[0], v1[1])
    a2 = direction_angle(v2[0], v2[1])
    change = angle_difference(a1, a2)
    return change >= threshold, float(change)


def detect_convergence(player_positions, frame_number, convergence_threshold=0.6):
    """
    Detect whether majority of players move toward shared centroid.

    Captures rebounding/scramble behavior using relative vectors only.
    """
    current_points = []
    for history in player_positions.values():
        pos = _position_at_or_before(history, frame_number)
        if pos is not None:
            current_points.append(pos)
    if len(current_points) < 2:
        return False, None, 0.0
    centroid = np.mean(np.array(current_points, dtype=float), axis=0)

    toward = 0
    used = 0
    for history in player_positions.values():
        pos = _position_at_or_before(history, frame_number)
        if pos is None:
            continue
        dx, dy = estimate_velocity(history, frame_number)
        if estimate_speed(dx, dy) < 1e-3:
            continue
        to_c = np.array([centroid[0] - pos[0], centroid[1] - pos[1]], dtype=float)
        vel = np.array([dx, dy], dtype=float)
        if np.linalg.norm(to_c) < 1e-6:
            continue
        used += 1
        cos_val = float(np.dot(vel, to_c) / (np.linalg.norm(vel) * np.linalg.norm(to_c) + 1e-9))
        if cos_val > 0:
            toward += 1
    if used == 0:
        return False, (float(centroid[0]), float(centroid[1])), 0.0
    conf = float(toward / used)
    return conf >= convergence_threshold, (float(centroid[0]), float(centroid[1])), conf


def detect_divergence(player_positions, frame_number, divergence_threshold=0.6):
    """
    Detect whether majority of players move away from group centroid.

    Geometry-free indicator of spacing-up or reset behavior.
    """
    is_conv, _, conv_conf = detect_convergence(player_positions, frame_number, 0.0)
    _ = is_conv
    div_conf = float(max(0.0, 1.0 - conv_conf))
    return div_conf >= divergence_threshold, div_conf


def detect_player_clustering(frame_players):
    """
    Split players into two clusters and label by side relative to frame centroid.

    This provides team-like grouping from relative positions only.
    """
    if not frame_players:
        return {}
    points = np.array(
        [[p["mask_center"]["x"], p["mask_center"]["y"]] for p in frame_players], dtype=float
    )
    labels = kmeans2(points)
    c = points.mean(axis=0)
    out = {}
    for idx, p in enumerate(frame_players):
        out[p["id"]] = int(labels[idx])

    # Relabel clusters by horizontal side of their centroids relative to frame centroid.
    unique_labels = sorted(set(out.values()))
    if len(unique_labels) == 2:
        centroids = {}
        for lb in unique_labels:
            pts = points[[i for i, p in enumerate(frame_players) if out[p["id"]] == lb]]
            centroids[lb] = pts.mean(axis=0)
        left_label = unique_labels[0] if centroids[unique_labels[0]][0] <= centroids[unique_labels[1]][0] else unique_labels[1]
        right_label = unique_labels[1] if left_label == unique_labels[0] else unique_labels[0]
        for pid in list(out.keys()):
            out[pid] = 0 if out[pid] == left_label else 1
        _ = right_label, c
    return out


def detect_isolation(frame_players, player_id, isolation_threshold=120):
    """
    Detect if a player is far from all others (nearest-neighbor criterion).

    Useful for identifying corner/outlet-style isolated roles or outlier tracks.
    """
    me = None
    others = []
    for p in frame_players:
        if p["id"] == player_id:
            me = p
        else:
            others.append(p)
    if me is None or not others:
        return False, 0.0
    x, y = me["mask_center"]["x"], me["mask_center"]["y"]
    nearest = min(
        distance(x, y, q["mask_center"]["x"], q["mask_center"]["y"]) for q in others
    )
    return nearest > isolation_threshold, float(nearest)


def detect_mirroring(player_positions, frame_number, player_id_a, player_id_b, mirror_threshold=30):
    """
    Detect whether player B mirrors player A (similar direction and speed).

    This is relation-based and independent of camera geometry.
    """
    ha = player_positions.get(player_id_a, [])
    hb = player_positions.get(player_id_b, [])
    dx_a, dy_a = estimate_velocity(ha, frame_number)
    dx_b, dy_b = estimate_velocity(hb, frame_number)
    sa, sb = estimate_speed(dx_a, dy_a), estimate_speed(dx_b, dy_b)
    aa, ab = direction_angle(dx_a, dy_a), direction_angle(dx_b, dy_b)
    if aa is None or ab is None:
        return False, 0.0
    ang_diff = angle_difference(aa, ab)
    speed_ratio = min(sa, sb) / max(sa, sb, 1e-6)
    conf = max(0.0, min(1.0, (1.0 - ang_diff / max(1e-6, mirror_threshold)) * speed_ratio))
    return ang_diff <= mirror_threshold and speed_ratio >= 0.5, float(conf)


def detect_dead_ball(player_positions, frame_number, speed_threshold=6, min_slow_fraction=0.7):
    """
    Detect dead-ball-like states from widespread low motion.

    No absolute location assumptions; purely based on velocity magnitudes.
    """
    speeds = []
    for history in player_positions.values():
        dx, dy = estimate_velocity(history, frame_number)
        speeds.append(estimate_speed(dx, dy))
    if not speeds:
        return False, 0.0
    slow_frac = float(sum(s <= speed_threshold for s in speeds) / len(speeds))
    return slow_frac >= min_slow_fraction, slow_frac


def rule_velocity_cap(x, y, prev_x, prev_y, dx, dy, max_pixels=75):
    """
    Enforce per-frame motion cap at 1 FPS.

    Limits implausible jumps that often come from tracker glitches.
    """
    raw_step = distance(prev_x, prev_y, x, y)
    if raw_step <= max_pixels:
        return x, y, False
    speed = max(1e-6, estimate_speed(dx, dy))
    scale = min(1.0, max_pixels / speed)
    capped_dx, capped_dy = dx * scale, dy * scale
    px, py = predict_next_position(prev_x, prev_y, capped_dx, capped_dy)
    return px, py, True


def rule_hull_containment(x, y, dx, dy, frame_players):
    """
    Pull outlier points back toward history prediction when outside group hull.

    Geometry-free replacement for court-boundary rules.
    """
    if convex_hull_margin(x, y, frame_players, margin=80):
        return x, y, False
    px, py = predict_next_position(x, y, dx, dy)
    return px, py, True


def rule_spacing_push(player_id, x, y, other_players, min_spacing=35):
    """
    Push crowded players apart by half overlap to reduce merge artifacts.
    """
    if not other_players:
        return x, y, False
    nearest = None
    nearest_d = float("inf")
    for p in other_players:
        if p["id"] == player_id:
            continue
        d = distance(x, y, p["mask_center"]["x"], p["mask_center"]["y"])
        if d < nearest_d:
            nearest_d = d
            nearest = p
    if nearest is None or nearest_d >= min_spacing:
        return x, y, False
    ox = x - nearest["mask_center"]["x"]
    oy = y - nearest["mask_center"]["y"]
    norm = math.hypot(ox, oy)
    if norm < 1e-6:
        ox, oy, norm = 1.0, 0.0, 1.0
    overlap = min_spacing - nearest_d
    push = 0.5 * overlap
    return x + push * (ox / norm), y + push * (oy / norm), True


def rule_collective_momentum(x, y, dx, dy, collective_angle, confidence, blend_weight=0.4):
    """
    Blend individual velocity with collective movement direction.

    Used in transition/fast-break states where team flow carries individuals.
    """
    if collective_angle is None or confidence <= 0:
        px, py = predict_next_position(x, y, dx, dy)
        return px, py, False
    spd = estimate_speed(dx, dy)
    ca = math.radians(collective_angle)
    cdx, cdy = spd * math.cos(ca), spd * math.sin(ca)
    w = max(0.0, min(1.0, blend_weight * confidence))
    ndx = (1.0 - w) * dx + w * cdx
    ndy = (1.0 - w) * dy + w * cdy
    px, py = predict_next_position(x, y, ndx, ndy)
    return px, py, True


def rule_stationary_persistence(x, y, history, frame_number):
    """
    Keep stationary players near current location.

    Captures screeners/post defenders without requiring court zones.
    """
    recent = _history_until_frame(history, frame_number)
    if len(recent) < 3:
        return x, y, False
    move = distance(recent[-3][1], recent[-3][2], recent[-1][1], recent[-1][2])
    if move <= 6:
        return x, y, True
    return x, y, False


def rule_cut_continuation(x, y, dx, dy, pre_cut_dx, pre_cut_dy, frames_since_cut):
    """Persist new cut direction for short horizon after sharp change."""
    if frames_since_cut is None or frames_since_cut > 2:
        px, py = predict_next_position(x, y, dx, dy)
        return px, py, False
    px, py = predict_next_position(x, y, dx, dy)
    return px, py, True


def rule_mirror_prediction(
    player_id,
    x,
    y,
    mirrored_player_x,
    mirrored_player_y,
    mirrored_player_dx,
    mirrored_player_dy,
):
    """
    Predict defender-like mirroring by inheriting counterpart movement.
    """
    _ = player_id, mirrored_player_x, mirrored_player_y
    px, py = predict_next_position(x, y, mirrored_player_dx, mirrored_player_dy)
    return px, py, True


def rule_cluster_cohesion(
    x,
    y,
    dx,
    dy,
    cluster_centroid_x,
    cluster_centroid_y,
    cohesion_weight=0.2,
):
    """
    Bias predictions slightly toward own-cluster centroid to preserve grouping.
    """
    px, py = predict_next_position(x, y, dx, dy)
    w = max(0.0, min(1.0, cohesion_weight))
    cx = (1.0 - w) * px + w * cluster_centroid_x
    cy = (1.0 - w) * py + w * cluster_centroid_y
    return cx, cy, True


def rule_convergence_pull(
    x, y, dx, dy, convergence_point_x, convergence_point_y, confidence, pull_weight=0.25
):
    """
    Pull predictions toward convergence point in scramble/rebound-like states.
    """
    px, py = predict_next_position(x, y, dx, dy)
    w = max(0.0, min(1.0, pull_weight * confidence))
    return (1.0 - w) * px + w * convergence_point_x, (1.0 - w) * py + w * convergence_point_y, True


def rule_divergence_spread(x, y, dx, dy, centroid_x, centroid_y, confidence):
    """
    Push predictions away from centroid during divergence/spacing-up behavior.
    """
    px, py = predict_next_position(x, y, dx, dy)
    vx, vy = px - centroid_x, py - centroid_y
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        vx, vy, norm = 1.0, 0.0, 1.0
    push = confidence * 0.3 * estimate_speed(dx, dy)
    return px + push * vx / norm, py + push * vy / norm, True


def rule_isolated_player_hold(x, y, dx, dy, is_isolated):
    """
    Dampen velocity for isolated players likely waiting in space.
    """
    if not is_isolated:
        return predict_next_position(x, y, dx, dy) + (False,)
    return predict_next_position(x, y, dx * 0.4, dy * 0.4) + (True,)


def rule_dead_ball_freeze(x, y, dx, dy, confidence):
    """
    Freeze-like damping during dead-ball periods.
    """
    factor = max(0.0, 1.0 - confidence * 0.85)
    return predict_next_position(x, y, dx * factor, dy * factor) + (True,)


def _clip_xy(x, y, frame_width, frame_height):
    return float(min(max(x, 0.0), frame_width - 1.0)), float(min(max(y, 0.0), frame_height - 1.0))


def apply_augmentation(frames, frame_width, frame_height, level="full"):
    """
    Apply geometry-free basketball augmentation rules to frame tracks.

    Levels:
      - physical: velocity cap, hull containment, spacing push
      - game: context-aware game rules only
      - full: physical + game
    """
    if level not in {"physical", "game", "full"}:
        raise ValueError("level must be one of: physical, game, full")

    frames = sorted(frames, key=lambda fr: int(fr["frame_number"]))
    augmented = []
    correction_log = []

    history = defaultdict(list)
    cut_state = {}  # pid -> {"frames_since_cut": int}
    seen_ids = set()

    for frame in frames:
        fnum = int(frame["frame_number"])
        players = []
        for p in frame.get("players", []):
            players.append(
                {
                    "id": p["id"],
                    "bbox": dict(p.get("bbox", {})),
                    "mask_center": dict(p["mask_center"]),
                }
            )
            seen_ids.add(p["id"])

        by_id = {p["id"]: p for p in players}
        frame_players = list(by_id.values())

        # Build state detectors from history up to current frame.
        player_positions = dict(history)
        for p in frame_players:
            player_positions.setdefault(p["id"], [])
            player_positions[p["id"]] = list(player_positions[p["id"]]) + [
                (fnum, float(p["mask_center"]["x"]), float(p["mask_center"]["y"]))
            ]

        collective_angle, collective_conf = detect_collective_direction(player_positions, fnum)
        is_fast_break, fast_break_conf = detect_fast_break(player_positions, fnum)
        is_transition, transition_conf, old_angle, new_angle = detect_transition(player_positions, fnum)
        _ = old_angle, new_angle
        is_half_court, half_court_conf = detect_half_court_set(player_positions, fnum)
        stationary_ids = set(detect_stationary_players(player_positions, fnum))
        is_conv, conv_point, conv_conf = detect_convergence(player_positions, fnum)
        is_div, div_conf = detect_divergence(player_positions, fnum)
        is_dead, dead_conf = detect_dead_ball(player_positions, fnum)
        clusters = detect_player_clustering(frame_players)

        # Cluster centroids for cohesion rule.
        cluster_centroids = {}
        for label in {clusters.get(pid) for pid in clusters}:
            ids = [pid for pid, lb in clusters.items() if lb == label]
            pts = [by_id[pid]["mask_center"] for pid in ids if pid in by_id]
            if pts:
                cluster_centroids[label] = (
                    float(np.mean([p["x"] for p in pts])),
                    float(np.mean([p["y"] for p in pts])),
                )

        corrected_by_id = {}
        for p in frame_players:
            pid = p["id"]
            x = float(p["mask_center"]["x"])
            y = float(p["mask_center"]["y"])
            original = (x, y)

            prev = _position_at_or_before(history.get(pid, []), fnum - 1)
            dx, dy = estimate_velocity(player_positions.get(pid, []), fnum)
            if prev is None:
                prev = (x - dx, y - dy)

            fired_rule = None
            fired_conf = 0.0
            game_state_active = None

            # Physical rules
            if level in {"physical", "full"}:
                nx, ny, fired = rule_velocity_cap(x, y, prev[0], prev[1], dx, dy)
                if fired:
                    x, y = nx, ny
                    fired_rule = "velocity_cap"
                    fired_conf = 1.0
                    game_state_active = "physical"

                nx, ny, fired = rule_hull_containment(x, y, dx, dy, [q for q in frame_players if q["id"] != pid])
                if fired:
                    x, y = nx, ny
                    fired_rule = "hull_containment"
                    fired_conf = 0.9
                    game_state_active = "physical"

                nx, ny, fired = rule_spacing_push(pid, x, y, [q for q in frame_players if q["id"] != pid])
                if fired:
                    x, y = nx, ny
                    fired_rule = "spacing_push"
                    fired_conf = 0.85
                    game_state_active = "physical"

            # Game rule selection: one per player/frame.
            if level in {"game", "full"}:
                candidates = []
                if is_dead:
                    candidates.append(("dead_ball", dead_conf))
                if is_conv:
                    candidates.append(("convergence", conv_conf))
                if is_div:
                    candidates.append(("divergence", div_conf))
                if is_fast_break:
                    candidates.append(("fast_break", fast_break_conf))
                if is_transition:
                    candidates.append(("transition", transition_conf))
                if is_half_court:
                    candidates.append(("half_court", half_court_conf))
                if pid in stationary_ids:
                    candidates.append(("stationary", 0.8))

                changed, change_amount = detect_direction_change(history, fnum, pid)
                if changed:
                    candidates.append(("cut", min(1.0, change_amount / 180.0)))
                    cut_state[pid] = {"frames_since_cut": 0}
                elif pid in cut_state:
                    cut_state[pid]["frames_since_cut"] += 1

                iso, nearest = detect_isolation(frame_players, pid)
                if iso:
                    candidates.append(("isolation", min(1.0, nearest / 300.0)))

                # Mirroring candidate: nearest neighbor relation.
                nearest_id = None
                nearest_d = float("inf")
                for q in frame_players:
                    if q["id"] == pid:
                        continue
                    d = distance(x, y, q["mask_center"]["x"], q["mask_center"]["y"])
                    if d < nearest_d:
                        nearest_d = d
                        nearest_id = q["id"]
                if nearest_id is not None:
                    mirr, mirr_conf = detect_mirroring(player_positions, fnum, pid, nearest_id)
                    if mirr:
                        candidates.append(("mirror", mirr_conf))

                if candidates:
                    game_state_active, best_conf = max(candidates, key=lambda t: t[1])
                    fired_conf = float(best_conf)
                    if game_state_active in {"fast_break", "transition"}:
                        nx, ny, fired = rule_collective_momentum(
                            x, y, dx, dy, collective_angle, collective_conf
                        )
                        if fired:
                            x, y = nx, ny
                            fired_rule = "collective_momentum"
                    elif game_state_active == "stationary":
                        nx, ny, fired = rule_stationary_persistence(
                            x, y, player_positions.get(pid, []), fnum
                        )
                        if fired:
                            x, y = nx, ny
                            fired_rule = "stationary_persistence"
                    elif game_state_active == "cut":
                        cstate = cut_state.get(pid, {"frames_since_cut": None})
                        nx, ny, fired = rule_cut_continuation(
                            x, y, dx, dy, dx, dy, cstate["frames_since_cut"]
                        )
                        if fired:
                            x, y = nx, ny
                            fired_rule = "cut_continuation"
                    elif game_state_active == "mirror" and nearest_id in by_id:
                        ndx, ndy = estimate_velocity(player_positions.get(nearest_id, []), fnum)
                        nx, ny, fired = rule_mirror_prediction(
                            pid,
                            x,
                            y,
                            by_id[nearest_id]["mask_center"]["x"],
                            by_id[nearest_id]["mask_center"]["y"],
                            ndx,
                            ndy,
                        )
                        if fired:
                            x, y = nx, ny
                            fired_rule = "mirror_prediction"
                    elif game_state_active == "convergence" and conv_point is not None:
                        nx, ny, fired = rule_convergence_pull(
                            x, y, dx, dy, conv_point[0], conv_point[1], conv_conf
                        )
                        if fired:
                            x, y = nx, ny
                            fired_rule = "convergence_pull"
                    elif game_state_active == "divergence":
                        c = player_centroid(frame_players) or (x, y)
                        nx, ny, fired = rule_divergence_spread(
                            x, y, dx, dy, c[0], c[1], div_conf
                        )
                        if fired:
                            x, y = nx, ny
                            fired_rule = "divergence_spread"
                    elif game_state_active == "isolation":
                        nx, ny, fired = rule_isolated_player_hold(x, y, dx, dy, True)
                        if fired:
                            x, y = nx, ny
                            fired_rule = "isolated_player_hold"
                    elif game_state_active == "dead_ball":
                        nx, ny, fired = rule_dead_ball_freeze(x, y, dx, dy, dead_conf)
                        if fired:
                            x, y = nx, ny
                            fired_rule = "dead_ball_freeze"
                    else:
                        # Default game-only cohesion if no specific higher rule fired.
                        lb = clusters.get(pid)
                        if lb in cluster_centroids:
                            nx, ny, fired = rule_cluster_cohesion(
                                x, y, dx, dy, cluster_centroids[lb][0], cluster_centroids[lb][1]
                            )
                            if fired:
                                x, y = nx, ny
                                fired_rule = "cluster_cohesion"
                                game_state_active = "cluster"
                                fired_conf = 0.5

            x, y = _clip_xy(x, y, frame_width, frame_height)
            corrected = {
                "id": pid,
                "bbox": dict(p["bbox"]),
                "mask_center": {"x": round(x, 1), "y": round(y, 1)},
            }
            corrected_by_id[pid] = corrected

            if fired_rule is not None:
                correction_log.append(
                    {
                        "frame_number": fnum,
                        "player_id": pid,
                        "rule_fired": fired_rule,
                        "confidence": float(max(0.0, min(1.0, fired_conf))),
                        "game_state_active": game_state_active,
                        "original_pos": {"x": round(original[0], 1), "y": round(original[1], 1)},
                        "corrected_pos": {"x": round(x, 1), "y": round(y, 1)},
                        "predicted": False,
                    }
                )

        # Re-identification for missing players seen previously.
        present_ids = set(corrected_by_id.keys())
        recent_threshold = 10
        missing_ids = sorted(
            pid for pid in (seen_ids - present_ids)
            if any(fn >= fnum - recent_threshold
                for fn, _, _ in history.get(pid, []))
        )
        for pid in missing_ids:
            h = history.get(pid, [])
            if not h:
                continue
            last = _position_at_or_before(h, fnum - 1)
            if last is None:
                continue
            dx, dy = estimate_velocity(h, fnum - 1)

            # pick best active game context for missing player
            contexts = [
                ("dead_ball", dead_conf if is_dead else 0.0),
                ("convergence", conv_conf if is_conv else 0.0),
                ("divergence", div_conf if is_div else 0.0),
                ("transition", transition_conf if is_transition else 0.0),
                ("fast_break", fast_break_conf if is_fast_break else 0.0),
                ("half_court", half_court_conf if is_half_court else 0.0),
            ]
            game_state_active, conf = max(contexts, key=lambda t: t[1])
            x, y = last[0], last[1]
            fired_rule = "collective_momentum"
            if level in {"game", "full"}:
                if game_state_active == "dead_ball":
                    x, y, _ = rule_dead_ball_freeze(x, y, dx, dy, conf)
                    fired_rule = "dead_ball_freeze"
                elif game_state_active == "convergence" and conv_point is not None:
                    x, y, _ = rule_convergence_pull(x, y, dx, dy, conv_point[0], conv_point[1], conf)
                    fired_rule = "convergence_pull"
                elif game_state_active == "divergence":
                    c = player_centroid(list(corrected_by_id.values())) or (x, y)
                    x, y, _ = rule_divergence_spread(x, y, dx, dy, c[0], c[1], conf)
                    fired_rule = "divergence_spread"
                elif game_state_active in {"fast_break", "transition"}:
                    x, y, _ = rule_collective_momentum(x, y, dx, dy, collective_angle, collective_conf)
                    fired_rule = "collective_momentum"
                else:
                    x, y = predict_next_position(x, y, dx, dy)
                    fired_rule = "cluster_cohesion"
            else:
                x, y = predict_next_position(x, y, dx, dy)
                fired_rule = "velocity_cap"

            x, y = _clip_xy(x, y, frame_width, frame_height)
            last_entry = next(
                (p for frame in reversed(augmented) 
                for p in frame.get("players", []) 
                if p["id"] == pid), None
            )
            last_bbox = last_entry["bbox"] if last_entry else {"x": int(round(x)), "y": int(round(y)), "w": 30, "h": 60}

            corrected_by_id[pid] = {
                "id": pid,
                "bbox": {"x": int(round(x)), "y": int(round(y)), 
                        "w": last_bbox["w"], "h": last_bbox["h"]},
                "mask_center": {"x": round(x, 1), "y": round(y, 1)},
                "predicted": True,
            }
            correction_log.append(
                {
                    "frame_number": fnum,
                    "player_id": pid,
                    "rule_fired": fired_rule,
                    "confidence": float(max(0.0, min(1.0, conf))),
                    "game_state_active": game_state_active,
                    "original_pos": {"x": round(last[0], 1), "y": round(last[1], 1)},
                    "corrected_pos": {"x": round(x, 1), "y": round(y, 1)},
                    "predicted": True,
                }
            )

        out_players = [corrected_by_id[pid] for pid in sorted(corrected_by_id)]
        augmented.append({"frame_number": fnum, "players": out_players})

        # Update history with augmented frame for recursive temporal consistency.
        for p in out_players:
            history[p["id"]].append((fnum, float(p["mask_center"]["x"]), float(p["mask_center"]["y"])))
            seen_ids.add(p["id"])

    return augmented, correction_log


def save_augmented_tracks(frames, correction_log, output_path):
    """
    Save augmented tracks and correction log.

    Tracks are saved with the same frame/player structure as input so metrics.py
    remains compatible.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, indent=2)

    corrections_path = output_path.parent / "corrections.json"
    with open(corrections_path, "w", encoding="utf-8") as f:
        json.dump({"corrections": correction_log}, f, indent=2)

    return corrections_path


def run_augmentation(input_path, output_path, frame_width, frame_height, level="full"):
    """
    Run full augmentation pipeline: load, augment, save, and print summary.
    """
    frames = load_tracks(input_path)
    augmented_frames, correction_log = apply_augmentation(
        frames, frame_width=frame_width, frame_height=frame_height, level=level
    )
    corrections_path = save_augmented_tracks(augmented_frames, correction_log, output_path)

    counts = defaultdict(int)
    predicted_count = 0
    for entry in correction_log:
        counts[entry["rule_fired"]] += 1
        if entry.get("predicted"):
            predicted_count += 1

    physical_rules = ["velocity_cap", "hull_containment", "spacing_push"]
    game_rules = [
        "collective_momentum",
        "stationary_persistence",
        "cut_continuation",
        "mirror_prediction",
        "cluster_cohesion",
        "convergence_pull",
        "divergence_spread",
        "isolated_player_hold",
        "dead_ball_freeze",
    ]
    players_processed = sum(len(fr.get("players", [])) for fr in augmented_frames)

    print("AUGMENTATION SUMMARY")
    print("=====================")
    print(f"Level: {level} | Frames: {len(augmented_frames)} | Players processed: {players_processed}")
    print()
    print("Physical Rules Fired:")
    for name in physical_rules:
        print(f"  {name:<20} {counts[name]}")
    print()
    print("Game Rules Fired:")
    for name in game_rules:
        print(f"  {name:<24} {counts[name]}")
    print()
    print("Re-identification:")
    print(f"  Players re-added via prediction: {predicted_count}")
    print()
    print(f"Total corrections: {len(correction_log)}")
    print(f"Saved augmented tracks to: {Path(output_path)}")
    print(f"Saved correction log to:   {corrections_path}")

    return augmented_frames, correction_log


def main():
    parser = argparse.ArgumentParser(description="Run geometry-free basketball augmentation on SAM3.1 tracks.")
    parser.add_argument("--tracks", default="data/outputs/tracks.json", help="Input tracks.json path")
    parser.add_argument(
        "--output",
        default="data/outputs/augmented_tracks.json",
        help="Output augmented tracks.json path",
    )
    parser.add_argument("--frame-width", type=int, required=True, help="Frame width in pixels")
    parser.add_argument("--frame-height", type=int, required=True, help="Frame height in pixels")
    parser.add_argument(
        "--level",
        choices=["physical", "game", "full"],
        default="full",
        help="Ablation level: physical, game, or full",
    )
    args = parser.parse_args()

    run_augmentation(
        input_path=args.tracks,
        output_path=args.output,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        level=args.level,
    )


if __name__ == "__main__":
    main()

