"""
Trajectory accuracy metrics (ADE / FDE) against ground truth.

Supports SportsMOT-style MOT gt.txt and a simple JSON GT format.
Primary use: evaluate augmented tracks on video_1 (SportsMOT source).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_gt_mot(gt_path):
    """
    Load MOT challenge gt.txt: frame,id,x,y,w,h,...

    Returns {frame_number: {track_id: (cx, cy)}} with 1-based frame numbers.
    """
    gt_path = Path(gt_path)
    by_frame = defaultdict(dict)
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cx, cy = x + w / 2.0, y + h / 2.0
            by_frame[frame][tid] = (cx, cy)
    return dict(by_frame)


def load_gt_json(gt_path):
    """
    Load JSON GT: {"frames": [{"frame_number": 1, "players": [{"id": 1, "mask_center": {...}}]}]}
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    by_frame = {}
    for frame in data.get("frames", []):
        fnum = int(frame["frame_number"])
        by_frame[fnum] = {}
        for p in frame.get("players", []):
            c = p.get("mask_center", p.get("center", {}))
            by_frame[fnum][p["id"]] = (float(c["x"]), float(c["y"]))
    return by_frame


def load_ground_truth(gt_path, format="auto"):
    gt_path = Path(gt_path)
    if format == "auto":
        format = "mot" if gt_path.suffix == ".txt" else "json"
    if format == "mot":
        return load_gt_mot(gt_path)
    if format == "json":
        return load_gt_json(gt_path)
    raise ValueError(f"Unknown GT format: {format}")


def load_tracks_centers(tracks_path):
    """Return {frame_number: {player_id: (x, y)}} from tracks JSON."""
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)
    by_frame = {}
    for frame in data.get("frames", []):
        fnum = int(frame["frame_number"])
        by_frame[fnum] = {}
        for p in frame.get("players", []):
            if p.get("predicted"):
                continue
            c = p.get("mask_center", {})
            if "x" in c and "y" in c:
                by_frame[fnum][p["id"]] = (float(c["x"]), float(c["y"]))
    return by_frame


def _hungarian_match(cost_matrix):
    """Greedy min-cost matching (small matrices); returns list of (i, j)."""
    if cost_matrix.size == 0:
        return []
    n, m = cost_matrix.shape
    used_j = set()
    pairs = []
    for i in range(n):
        best_j = None
        best_c = float("inf")
        for j in range(m):
            if j in used_j:
                continue
            if cost_matrix[i, j] < best_c:
                best_c = cost_matrix[i, j]
                best_j = j
        if best_j is not None and best_c < 1e6:
            pairs.append((i, best_j))
            used_j.add(best_j)
    return pairs


def match_frame(pred_pts, gt_pts, max_distance=80.0):
    """
    Match predicted track IDs to GT IDs on one frame.

    pred_pts: {pred_id: (x,y)}
    gt_pts: {gt_id: (x,y)}
    Returns list of (pred_id, gt_id, distance).
    """
    pred_ids = list(pred_pts.keys())
    gt_ids = list(gt_pts.keys())
    if not pred_ids or not gt_ids:
        return []

    cost = np.zeros((len(pred_ids), len(gt_ids)), dtype=float)
    for i, pid in enumerate(pred_ids):
        px, py = pred_pts[pid]
        for j, gid in enumerate(gt_ids):
            gx, gy = gt_pts[gid]
            d = np.hypot(px - gx, py - gy)
            cost[i, j] = d if d <= max_distance else 1e9

    pairs = _hungarian_match(cost)
    matches = []
    for i, j in pairs:
        pid, gid = pred_ids[i], gt_ids[j]
        px, py = pred_pts[pid]
        gx, gy = gt_pts[gid]
        matches.append((pid, gid, float(np.hypot(px - gx, py - gy))))
    return matches


def compute_ade_fde(
    tracks_path,
    gt_path,
    max_distance=80.0,
    gt_format="auto",
    min_frame=None,
    max_frame=None,
):
    """
    Compute ADE (mean L2 over matched frame-player pairs) and FDE (last frame).

    Optional min_frame / max_frame restrict which frame numbers are scored.

    Returns dict with ade, fde, num_matches, num_frames.
    """
    pred_by_frame = load_tracks_centers(tracks_path)
    gt_by_frame = load_ground_truth(gt_path, format=gt_format)

    common_frames = sorted(set(pred_by_frame) & set(gt_by_frame))
    if min_frame is not None:
        common_frames = [f for f in common_frames if f >= min_frame]
    if max_frame is not None:
        common_frames = [f for f in common_frames if f <= max_frame]
    errors = []
    last_frame_errors = []

    for fnum in common_frames:
        matches = match_frame(
            pred_by_frame[fnum], gt_by_frame[fnum], max_distance=max_distance
        )
        for _pid, _gid, dist in matches:
            errors.append(dist)
        if matches:
            last_frame_errors = [m[2] for m in matches]

    ade = float(np.mean(errors)) if errors else float("nan")
    fde = float(np.mean(last_frame_errors)) if last_frame_errors else float("nan")

    return {
        "ade": ade,
        "fde": fde,
        "num_matches": len(errors),
        "num_frames": len(common_frames),
        "min_frame": min_frame,
        "max_frame": max_frame,
        "tracks_path": str(tracks_path),
        "gt_path": str(gt_path),
    }


def forecast_min_frame_from_tracks(tracks_path, obs_len):
    """First frame number at which LSTM rollout replaces SAM (index obs_len)."""
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = sorted(data.get("frames", []), key=lambda fr: int(fr["frame_number"]))
    if len(frames) <= obs_len:
        return None
    return int(frames[obs_len]["frame_number"])


def find_sportsmot_gt(sequence_name="sportsmot_example", gt_root=None):
    """
    Resolve aligned GT for a dataset or legacy SportsMOT sequence folder.

    Prefer aligned gt.json, then raw gt.txt.
    """
    from utils.datasets import DATASETS, find_gt_path

    if sequence_name in DATASETS:
        path = find_gt_path(sequence_name)
        if path is not None:
            return path

    root = Path(gt_root or "data/gt/sportsmot")
    candidates = [
        root / sequence_name / "gt" / "gt.json",
        root / sequence_name / "gt" / "gt.txt",
        root / sequence_name / "gt.txt",
        root / f"{sequence_name}.txt",
        root / "gt.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute ADE/FDE vs ground truth")
    parser.add_argument("--tracks", required=True, help="Predicted/augmented tracks JSON")
    parser.add_argument("--gt", default=None, help="Ground truth path (MOT txt or JSON)")
    parser.add_argument(
        "--sequence",
        default="sportsmot_example",
        help="Dataset key or legacy SportsMOT sequence when --gt omitted",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Alias for --sequence (dataset key from utils.datasets)",
    )
    parser.add_argument("--gt-root", default="data/gt/sportsmot")
    parser.add_argument("--gt-format", default="auto", choices=["auto", "mot", "json"])
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--output", default=None, help="Save metrics JSON")
    args = parser.parse_args()
    sequence = args.dataset or args.sequence

    gt_path = args.gt
    if gt_path is None:
        gt_path = find_sportsmot_gt(sequence, args.gt_root)
        if gt_path is None:
            raise FileNotFoundError(
                f"No GT found for '{sequence}'. Place gt.txt under "
                f"data/datasets/{sequence}/gt/ and run setup_sportsmot_gt.py, "
                "or pass --gt explicitly."
            )

    result = compute_ade_fde(
        args.tracks,
        gt_path,
        max_distance=args.max_distance,
        gt_format=args.gt_format,
    )

    print()
    print("TRAJECTORY METRICS (ADE / FDE)")
    print("=" * 40)
    print(f"Tracks: {result['tracks_path']}")
    print(f"GT:     {result['gt_path']}")
    print(f"Frames matched: {result['num_frames']}")
    print(f"Point matches:  {result['num_matches']}")
    print(f"ADE: {result['ade']:.3f} px")
    print(f"FDE: {result['fde']:.3f} px")
    print("=" * 40)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out}")

    return result


if __name__ == "__main__":
    main()
