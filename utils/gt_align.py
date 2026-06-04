"""
Align SportsMOT MOT ground truth to SAM3 track coordinate space.

Maps full-resolution MOT frames to extracted track frame_numbers using
fps subsampling, start_time_sec, and resize_scale from tracks meta.
"""

import json
from pathlib import Path

from utils.trajectory_metrics import load_gt_mot, load_gt_json


def _parse_seqinfo(seqinfo_path):
    info = {"frameRate": 25.0, "imWidth": 1280, "imHeight": 720}
    if not seqinfo_path.exists():
        return info
    with open(seqinfo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("["):
                continue
            key, val = line.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "framerate":
                info["frameRate"] = float(val)
            elif key == "imwidth":
                info["imWidth"] = int(float(val))
            elif key == "imheight":
                info["imHeight"] = int(float(val))
    return info


def mot_frame_to_track_frame(mot_frame, extract_fps, source_fps, start_time_sec):
    """Map 1-based MOT frame index to 1-based track frame_number."""
    source_frame_0 = mot_frame - 1
    start_source_frame = int(round(start_time_sec * source_fps))
    rel = source_frame_0 - start_source_frame
    if rel < 0:
        return None
    interval = max(int(round(source_fps / extract_fps)), 1)
    if rel % interval != 0:
        return None
    return rel // interval + 1


def align_mot_gt_to_tracks(
    raw_gt_path,
    tracks_meta,
    seqinfo_path=None,
    extract_fps=1.0,
    start_time_sec=0.0,
):
    """
    Return {frame_number: {gt_id: (cx, cy)}} in track pixel space.
    """
    raw_gt_path = Path(raw_gt_path)
    gt_by_mot = load_gt_mot(raw_gt_path)

    meta = tracks_meta or {}
    resize_scale = float(meta.get("resize_scale", 1.0))
    fw = float(meta.get("frame_width", 1280 * resize_scale))
    fh = float(meta.get("frame_height", 720 * resize_scale))

    seqinfo = _parse_seqinfo(Path(seqinfo_path) if seqinfo_path else raw_gt_path.parent / "seqinfo.ini")
    source_fps = float(seqinfo["frameRate"])
    src_w = float(seqinfo["imWidth"])
    src_h = float(seqinfo["imHeight"])

    sx = (fw / src_w) if src_w > 0 else resize_scale
    sy = (fh / src_h) if src_h > 0 else resize_scale

    aligned = {}
    for mot_frame, players in gt_by_mot.items():
        tfn = mot_frame_to_track_frame(mot_frame, extract_fps, source_fps, start_time_sec)
        if tfn is None:
            continue
        aligned.setdefault(tfn, {})
        for gid, (cx, cy) in players.items():
            aligned[tfn][gid] = (cx * sx, cy * sy)

    return aligned


def save_aligned_gt_json(aligned_by_frame, output_path, meta=None):
    """Save aligned GT as tracks-compatible JSON."""
    frames = []
    for fnum in sorted(aligned_by_frame):
        players = []
        for gid, (cx, cy) in sorted(aligned_by_frame[fnum].items()):
            players.append(
                {
                    "id": gid,
                    "mask_center": {"x": round(cx, 1), "y": round(cy, 1)},
                }
            )
        frames.append({"frame_number": fnum, "players": players})

    payload = {"frames": frames}
    if meta:
        payload["meta"] = meta

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def build_smoothed_proxy_gt(baseline_tracks_path, window=3):
    """
    Build temporally smoothed GT from baseline (proxy when SportsMOT raw GT unavailable).

    Uses moving average on each track ID's mask_center. Not a substitute for real
    SportsMOT labels in the paper, but enables ADE-based ablation ranking locally.
    """
    with open(baseline_tracks_path, encoding="utf-8") as f:
        data = json.load(f)

    frames = sorted(data.get("frames", []), key=lambda fr: int(fr["frame_number"]))
    meta = dict(data.get("meta") or {})
    meta["gt_source"] = "smoothed_baseline_proxy"
    meta["smooth_window"] = window

    id_series = {}
    for fr in frames:
        fnum = int(fr["frame_number"])
        for p in fr.get("players", []):
            c = p.get("mask_center", {})
            if "x" not in c or "y" not in c:
                continue
            id_series.setdefault(p["id"], []).append((fnum, float(c["x"]), float(c["y"])))

    smoothed = {pid: [] for pid in id_series}
    half = max(1, window // 2)
    for pid, series in id_series.items():
        series = sorted(series, key=lambda t: t[0])
        for i, (fnum, x, y) in enumerate(series):
            lo = max(0, i - half)
            hi = min(len(series), i + half + 1)
            chunk = series[lo:hi]
            sx = sum(t[1] for t in chunk) / len(chunk)
            sy = sum(t[2] for t in chunk) / len(chunk)
            smoothed[pid].append((fnum, sx, sy))

    by_frame = {}
    for pid, points in smoothed.items():
        for fnum, x, y in points:
            by_frame.setdefault(fnum, {})[pid] = (x, y)

    return by_frame, meta
