"""
Overlay LSTM forecast trajectories on video frames (observed / predicted / GT).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.lstm_dataset import load_tensor_file
from utils.lstm_predict import linear_extrapolation_positions, load_checkpoint, rollout_positions
from utils.visualize import (
    _frame_path_for_number,
    _get_sorted_frame_paths,
    _track_display_scales,
    load_tracks_by_frame,
)

# BGR colors aligned with matplotlib legend in create_forecast_summary_figure
COLOR_OBS_BGR = (208, 208, 208)      # #d0d0d0
COLOR_GT_BGR = (0, 220, 0)           # #00dc00
COLOR_PRED_BGR = (255, 229, 0)       # #00e5ff (cyan in RGB)
COLOR_LINEAR_BGR = (0, 165, 255)     # #ffa500 (orange in RGB)
COLOR_ANCHOR_BGR = (255, 255, 255)


def resolve_run_frames_dir(dataset: str, seed_id: str | None = None) -> Path | None:
    """Prefer per-run extracted frames (45-frame seed window)."""
    from utils.datasets import runs_dir

    for candidate in (
        runs_dir(dataset, seed_id) / "frames" if seed_id else None,
        runs_dir(dataset) / "frames",
    ):
        if candidate and candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    return None


def resolve_source_frame_path(
    dataset: str,
    frame_number: int,
    meta: dict,
) -> Path | None:
    """Map 1-based seed-local frame_number to a JPEG in the full dataset."""
    from utils.datasets import frames_dir as dataset_frames_dir
    from utils.datasets import get_dataset

    ds = get_dataset(dataset)
    source_dir = Path(dataset_frames_dir(dataset))
    if not source_dir.is_dir():
        return None

    start_time_sec = float(meta.get("start_time_sec", 0.0))
    source_fps = float(meta.get("source_fps", ds.get("source_fps", 25.0)))
    extract_fps = float(meta.get("extract_fps", ds.get("extract_fps", 25.0)))
    start_mot_frame = int(round(start_time_sec * source_fps)) + 1
    interval = max(int(round(source_fps / extract_fps)), 1)
    mot_frame = start_mot_frame + (int(frame_number) - 1) * interval

    path = source_dir / f"{mot_frame:06d}.jpg"
    if path.is_file():
        return path

    matches = sorted(source_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    for p in matches:
        if int(p.stem) == mot_frame:
            return p
    return None


def load_frame_bgr(
    dataset: str,
    frame_number: int,
    meta: dict,
    frames_dir: Path | str | None = None,
    seed_id: str | None = None,
):
    """Load display frame for a seed-local frame_number."""
    if frames_dir:
        frame_paths = _get_sorted_frame_paths(frames_dir)
        frame_path = _frame_path_for_number(frame_paths, frame_number)
        if frame_path is not None:
            img = cv2.imread(str(frame_path))
            if img is not None:
                return img, frame_path

    run_frames = resolve_run_frames_dir(dataset, seed_id)
    if run_frames is not None:
        frame_paths = _get_sorted_frame_paths(run_frames)
        frame_path = _frame_path_for_number(frame_paths, frame_number)
        if frame_path is not None:
            img = cv2.imread(str(frame_path))
            if img is not None:
                return img, frame_path

    src_path = resolve_source_frame_path(dataset, frame_number, meta)
    if src_path is not None:
        img = cv2.imread(str(src_path))
        if img is not None:
            resize_scale = float(meta.get("resize_scale", 1.0))
            if resize_scale != 1.0:
                img = cv2.resize(
                    img,
                    (0, 0),
                    fx=resize_scale,
                    fy=resize_scale,
                    interpolation=cv2.INTER_AREA,
                )
            return img, src_path
    return None, None


def rollout_predicted_positions(
    seq: dict,
    checkpoint_path: Path | str,
    device: str = "cpu",
    stitch: str = "last",
) -> np.ndarray:
    """Run LSTM rollout on a tensor sequence and return pixel positions."""
    import torch

    model, cfg = load_checkpoint(checkpoint_path, device=device)
    cfg = {**cfg, "scale": cfg.get("scale") or seq["scale"].tolist()}
    pred_pos, _ = rollout_positions(model, seq, cfg, device=device, stitch=stitch)
    return pred_pos


def tracks_to_position_array(tracks_path, seq: dict) -> np.ndarray:
    """Align tracks JSON centers to tensor slot order."""
    by_frame = load_tracks_by_frame(tracks_path)
    frame_numbers = seq["frame_numbers"]
    player_ids = seq["player_ids"]
    visibility = seq["visibility"]
    T = len(frame_numbers)
    P = len(player_ids)
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pid_to_slot = {int(pid): i for i, pid in enumerate(player_ids) if int(pid) >= 0}

    for t, fn in enumerate(frame_numbers):
        for player in by_frame.get(int(fn), []):
            slot = pid_to_slot.get(int(player["id"]))
            if slot is None:
                continue
            center = player.get("mask_center") or {}
            pos[t, slot, 0] = float(center.get("x", 0.0))
            pos[t, slot, 1] = float(center.get("y", 0.0))
    pos[~visibility] = 0.0
    return pos


def _scale_point(x: float, y: float, scale_x: float, scale_y: float) -> tuple[int, int]:
    return int(round(x * scale_x)), int(round(y * scale_y))


def _draw_polyline(
    image_bgr,
    points,
    color,
    thickness=2,
    dashed=False,
    dash_len=8,
    gap_len=6,
):
    """Draw connected polyline; optional dashed style."""
    pts = [tuple(p) for p in points if p is not None]
    if len(pts) < 2:
        if len(pts) == 1:
            cv2.circle(image_bgr, pts[0], 4, color, -1, cv2.LINE_AA)
        return

    if not dashed:
        cv2.polylines(
            np.asarray(image_bgr),
            [np.array(pts, dtype=np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        if dist < 1e-3:
            continue
        ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
        pos = 0.0
        draw = True
        while pos < dist:
            seg = dash_len if draw else gap_len
            nxt = min(pos + seg, dist)
            if draw:
                p_a = (int(round(x0 + ux * pos)), int(round(y0 + uy * pos)))
                p_b = (int(round(x0 + ux * nxt)), int(round(y0 + uy * nxt)))
                cv2.line(image_bgr, p_a, p_b, color, thickness, cv2.LINE_AA)
            pos = nxt
            draw = not draw


def _window_forecast_ade(gt_pos, pred_pos, vis, window_start, obs_len, pred_len):
    """Mean per-step ADE over visible player-steps in one forecast window."""
    t0 = window_start + obs_len
    errs = []
    for k in range(pred_len):
        t = t0 + k
        for p in range(gt_pos.shape[1]):
            if not vis[t, p]:
                continue
            dx = float(pred_pos[t, p, 0] - gt_pos[t, p, 0])
            dy = float(pred_pos[t, p, 1] - gt_pos[t, p, 1])
            errs.append(float(np.hypot(dx, dy)))
    return float(np.mean(errs)) if errs else float("nan")


def pick_forecast_windows(
    gt_pos,
    pred_pos,
    vis,
    obs_len,
    pred_len,
    n_windows=3,
    window_starts=None,
):
    """Choose representative sliding-window starts (spread + best ADE)."""
    win = obs_len + pred_len
    max_start = gt_pos.shape[0] - win
    if max_start < 0:
        return []

    if window_starts:
        return [int(s) for s in window_starts if 0 <= int(s) <= max_start]

    candidates = list(range(0, max_start + 1))
    scored = [
        (s, _window_forecast_ade(gt_pos, pred_pos, vis, s, obs_len, pred_len))
        for s in candidates
    ]
    valid = [(s, ade) for s, ade in scored if ade == ade]
    if not valid:
        return candidates[:n_windows]

    valid.sort(key=lambda x: x[1])
    best = valid[0][0]
    if n_windows == 1:
        return [best]

    thirds = [max_start // 3, (2 * max_start) // 3, max_start]
    picks = []
    for s in thirds:
        if s not in picks:
            picks.append(s)
    if best not in picks:
        picks[-1] = best
    return picks[:n_windows]


def _select_player_slots(
    gt_pos,
    vis,
    player_ids,
    window_start,
    obs_len,
    pred_len,
    max_players=4,
):
    """Pick visible players with largest motion in the window (reduces overlay clutter)."""
    anchor_t = window_start + obs_len - 1
    t_end = window_start + obs_len + pred_len
    scores = []
    for slot, pid in enumerate(player_ids):
        if int(pid) < 0 or not vis[anchor_t, slot]:
            continue
        pts = []
        for t in range(window_start, min(t_end, gt_pos.shape[0])):
            if vis[t, slot]:
                pts.append(gt_pos[t, slot])
        if len(pts) < 2:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        scores.append((length, slot))
    scores.sort(reverse=True)
    if max_players is None or max_players <= 0:
        return [slot for _, slot in scores]
    return [slot for _, slot in scores[:max_players]]


def draw_forecast_window(
    frame_bgr,
    gt_pos,
    pred_pos,
    linear_pos,
    vis,
    player_ids,
    window_start,
    obs_len,
    pred_len,
    scale_x=1.0,
    scale_y=1.0,
    show_linear=True,
    draw_panel_legend=True,
    max_players=4,
):
    """
    Draw one forecast window on the last-observed frame.

    Observed history: solid gray (matches legend).
    GT future: green solid.
    LSTM prediction: cyan dashed.
    Linear baseline: orange dashed (optional).
    """
    out = frame_bgr.copy()
    anchor_t = window_start + obs_len - 1
    t_pred_start = window_start + obs_len
    t_pred_end = t_pred_start + pred_len

    slots = _select_player_slots(
        gt_pos, vis, player_ids, window_start, obs_len, pred_len, max_players=max_players
    )

    for slot in slots:
        pid = player_ids[slot]
        if int(pid) < 0:
            continue
        if not vis[anchor_t, slot]:
            continue
        if not np.any(vis[window_start:t_pred_end, slot]):
            continue

        obs_pts = []
        for t in range(window_start, window_start + obs_len):
            if not vis[t, slot]:
                continue
            obs_pts.append(_scale_point(gt_pos[t, slot, 0], gt_pos[t, slot, 1], scale_x, scale_y))

        gt_future = [
            _scale_point(gt_pos[t, slot, 0], gt_pos[t, slot, 1], scale_x, scale_y)
            for t in range(t_pred_start, t_pred_end)
            if vis[t, slot]
        ]
        pred_future = [
            _scale_point(pred_pos[t, slot, 0], pred_pos[t, slot, 1], scale_x, scale_y)
            for t in range(t_pred_start, t_pred_end)
            if vis[t, slot]
        ]
        lin_future = []
        if show_linear and linear_pos is not None:
            lin_future = [
                _scale_point(linear_pos[t, slot, 0], linear_pos[t, slot, 1], scale_x, scale_y)
                for t in range(t_pred_start, t_pred_end)
                if vis[t, slot]
            ]

        # Draw back-to-front: observed, GT, LSTM, then linear on top (often ~1px from GT).
        if obs_pts:
            _draw_polyline(out, obs_pts, COLOR_OBS_BGR, thickness=2, dashed=False)
        if obs_pts and gt_future:
            _draw_polyline(out, [obs_pts[-1]] + gt_future, COLOR_GT_BGR, thickness=3, dashed=False)
            for pt in gt_future:
                cv2.circle(out, pt, 5, COLOR_GT_BGR, -1, cv2.LINE_AA)
        if obs_pts and pred_future:
            _draw_polyline(out, [obs_pts[-1]] + pred_future, COLOR_PRED_BGR, thickness=3, dashed=True)
            for pt in pred_future:
                cv2.circle(out, pt, 4, COLOR_PRED_BGR, -1, cv2.LINE_AA)
        if obs_pts and lin_future:
            _draw_polyline(out, [obs_pts[-1]] + lin_future, COLOR_LINEAR_BGR, thickness=3, dashed=True)
            for pt in lin_future:
                cv2.circle(out, pt, 7, COLOR_LINEAR_BGR, 2, cv2.LINE_AA)

        ax, ay = _scale_point(
            gt_pos[anchor_t, slot, 0], gt_pos[anchor_t, slot, 1], scale_x, scale_y
        )
        cv2.circle(out, (ax, ay), 5, COLOR_ANCHOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(out, (ax, ay), 6, (0, 0, 0), 1, cv2.LINE_AA)

    if draw_panel_legend:
        _draw_forecast_legend(out, obs_len, pred_len, show_linear=show_linear)
    return out


def _forecast_legend_handles(obs_len: int, pred_len: int, show_linear: bool = True):
    """Matplotlib legend handles matching overlay line styles."""
    handles = [
        Line2D([0], [0], color="#d0d0d0", linewidth=2.5, label=f"Observed ({obs_len}f)"),
        Line2D([0], [0], color="#00dc00", linewidth=2.5, label=f"Ground truth ({pred_len}f)"),
        Line2D(
            [0],
            [0],
            color="#00e5ff",
            linewidth=2.5,
            linestyle="--",
            label=f"LSTM forecast ({pred_len}f)",
        ),
    ]
    if show_linear:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#ffa500",
                linewidth=2.5,
                linestyle="--",
                label=f"Linear baseline ({pred_len}f)",
            )
        )
    return handles


def _draw_forecast_legend(image_bgr, obs_len, pred_len, show_linear=True):
    """Compact legend for trajectory styles."""
    h, w = image_bgr.shape[:2]
    pad = 8
    box_w, box_h = 220, 92 if show_linear else 74
    x0, y0 = pad, h - box_h - pad
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image_bgr, 0.45, 0, image_bgr)

    entries = [
        (f"Observed ({obs_len}f)", COLOR_OBS_BGR, False),
        (f"Ground truth ({pred_len}f)", COLOR_GT_BGR, False),
        (f"LSTM forecast ({pred_len}f)", COLOR_PRED_BGR, True),
    ]
    if show_linear:
        entries.append((f"Linear baseline ({pred_len}f)", COLOR_LINEAR_BGR, True))
    y = y0 + 18
    for label, color, dashed in entries:
        x1, x2 = x0 + 12, x0 + 52
        if dashed:
            _draw_polyline(image_bgr, [(x1, y), (x2, y)], color, thickness=2, dashed=True)
        else:
            cv2.line(image_bgr, (x1, y), (x2, y), color, 2, cv2.LINE_AA)
        cv2.putText(
            image_bgr,
            label,
            (x0 + 60, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18


def render_forecast_panel(
    dataset: str,
    seq: dict,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    linear_pos: np.ndarray | None,
    window_start: int,
    obs_len: int,
    pred_len: int,
    frames_dir: Path | str | None = None,
    seed_id: str | None = None,
    show_linear: bool = True,
    draw_panel_legend: bool = True,
    max_players: int = 4,
):
    """Render one forecast window panel."""
    anchor_t = window_start + obs_len - 1
    frame_number = int(seq["frame_numbers"][anchor_t])
    meta = seq["meta"]
    frame_bgr, frame_path = load_frame_bgr(
        dataset,
        frame_number,
        meta,
        frames_dir=frames_dir,
        seed_id=seed_id,
    )
    if frame_bgr is None:
        raise FileNotFoundError(
            f"No JPEG for frame_number={frame_number} (dataset={dataset}, seed={seed_id})"
        )

    sx, sy = _track_display_scales(meta, frame_bgr)
    annotated = draw_forecast_window(
        frame_bgr,
        gt_pos,
        pred_pos,
        linear_pos,
        seq["visibility"],
        seq["player_ids"],
        window_start,
        obs_len,
        pred_len,
        scale_x=sx,
        scale_y=sy,
        show_linear=show_linear,
        draw_panel_legend=draw_panel_legend,
        max_players=max_players,
    )
    ade = _window_forecast_ade(
        gt_pos, pred_pos, seq["visibility"], window_start, obs_len, pred_len
    )
    return annotated, frame_number, ade, frame_path


def resolve_predicted_positions(
    seq: dict,
    predicted_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
):
    """Load predictions from checkpoint rollout (preferred) or tracks JSON."""
    if checkpoint_path and Path(checkpoint_path).is_file():
        return rollout_predicted_positions(seq, checkpoint_path, device=device)
    if predicted_path and Path(predicted_path).is_file():
        return tracks_to_position_array(predicted_path, seq)
    raise FileNotFoundError("Need --checkpoint or --predicted tracks JSON.")


def create_forecast_summary_figure(
    dataset: str,
    tensor_path: Path | str,
    output_path: Path | str,
    predicted_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
    obs_len: int = 8,
    pred_len: int = 4,
    window_starts=None,
    n_windows: int = 3,
    frames_dir: Path | str | None = None,
    seed_id: str | None = None,
    show_linear: bool = True,
    title: str | None = None,
    max_players: int = 4,
):
    """Build multi-row publication figure with forecast overlays."""
    seq = load_tensor_file(tensor_path)
    gt_pos = seq["positions"]
    pred_pos = resolve_predicted_positions(
        seq,
        predicted_path=predicted_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    cfg = {"obs_len": obs_len, "pred_len": pred_len}
    linear_pos, _ = linear_extrapolation_positions(seq, cfg)

    seed_id = seed_id or seq.get("seed_id")
    windows = pick_forecast_windows(
        gt_pos,
        pred_pos,
        seq["visibility"],
        obs_len,
        pred_len,
        n_windows=n_windows,
        window_starts=window_starts,
    )
    if not windows:
        raise ValueError("No valid forecast windows for this sequence.")

    panels = []
    for ws in windows:
        panel, frame_number, ade, _ = render_forecast_panel(
            dataset,
            seq,
            gt_pos,
            pred_pos,
            linear_pos,
            ws,
            obs_len,
            pred_len,
            frames_dir=frames_dir,
            seed_id=seed_id,
            show_linear=show_linear,
            draw_panel_legend=False,
            max_players=max_players,
        )
        panels.append((panel, ws, frame_number, ade))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n + 0.35), squeeze=False)
    default_title = title or (
        f"LSTM Forecast vs Ground Truth ({seq.get('seed_id', 'seed')}, "
        f"obs={obs_len}, pred={pred_len})"
    )
    fig.suptitle(default_title, fontsize=13, y=0.995)

    for row, (panel, ws, frame_number, ade) in enumerate(panels):
        ax = axes[row, 0]
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        ade_txt = f"{ade:.1f}px" if ade == ade else "n/a"
        ax.set_title(
            f"Window start t={ws} · anchor frame {frame_number} · window ADE {ade_txt}",
            fontsize=10,
        )

    legend_handles = _forecast_legend_handles(obs_len, pred_len, show_linear=show_linear)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        fontsize=10,
        handlelength=2.8,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved forecast summary figure to {output_path}")
    return str(output_path), windows


def save_forecast_frame(
    dataset: str,
    tensor_path: Path | str,
    output_path: Path | str,
    window_start: int,
    predicted_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
    obs_len: int = 8,
    pred_len: int = 4,
    frames_dir: Path | str | None = None,
    seed_id: str | None = None,
    show_linear: bool = True,
):
    """Save a single annotated forecast frame."""
    seq = load_tensor_file(tensor_path)
    gt_pos = seq["positions"]
    pred_pos = resolve_predicted_positions(
        seq,
        predicted_path=predicted_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    cfg = {"obs_len": obs_len, "pred_len": pred_len}
    linear_pos, _ = linear_extrapolation_positions(seq, cfg)
    seed_id = seed_id or seq.get("seed_id")

    panel, frame_number, ade, _ = render_forecast_panel(
        dataset,
        seq,
        gt_pos,
        pred_pos,
        linear_pos,
        window_start,
        obs_len,
        pred_len,
        frames_dir=frames_dir,
        seed_id=seed_id,
        show_linear=show_linear,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved forecast frame {frame_number} (window={window_start}, ADE={ade:.2f}px) to {output_path}")
    return str(output_path)
