"""
Visualization utilities for SAM3.1 tracking outputs.

Supports:
1) Annotating one tracks file on frames
2) Side-by-side baseline vs augmentation comparisons
3) Publication summary figure generation
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_player_color(player_id, predicted=False):
    """
    Return BGR color for a player track.

    Predicted (re-identified) players are highlighted in bright green.
    """
    if predicted:
        return (0, 255, 0)
    return (
        int((player_id * 67) % 255),
        int((player_id * 113) % 255),
        int((player_id * 181) % 255),
    )


def draw_tracks_on_frame(frame_bgr, frame_tracks, scale_x=1.0, scale_y=1.0):
    """
    Draw tracking annotations for one frame.

    Works on a copy and returns the annotated frame.
    """
    out = frame_bgr.copy()
    players = scale_players_to_image(frame_tracks or [], scale_x, scale_y)

    if not players:
        cv2.putText(
            out,
            "No tracks",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    for player in players:
        player_id = int(player["id"])
        predicted = bool(player.get("predicted", False))
        color = get_player_color(player_id, predicted=predicted)

        bbox = player.get("bbox", {})
        x = int(round(bbox.get("x", 0)))
        y = int(round(bbox.get("y", 0)))
        w = int(round(bbox.get("w", 0)))
        h = int(round(bbox.get("h", 0)))

        center = player.get("mask_center", {})
        cx = int(round(center.get("x", x + w / 2.0)))
        cy = int(round(center.get("y", y + h / 2.0)))

        # Draw bbox border
        if w > 0 and h > 0:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        else:
            # For predicted players with no bbox, draw a small diamond at center
            pts = np.array(
                [[cx, cy - 8], [cx + 8, cy], [cx, cy + 8], [cx - 8, cy]],
                dtype=np.int32,
            )
            cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)

        # Draw label with black outline then white text
        label = f"ID:{player_id}" + ("*" if predicted else "")
        tx = x
        ty = max(14, y - 8)
        cv2.putText(
            out,
            label,
            (tx + 1, ty + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def load_tracks_by_frame(tracks_path):
    """
    Load tracks.json into frame-number keyed dictionary.

    Returns:
        {frame_number: [players...]}
    """
    tracks_path = Path(tracks_path)
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for frame in data.get("frames", []):
        frame_number = int(frame.get("frame_number", 0))
        mapping[frame_number] = frame.get("players", [])

    print(f"Loaded tracks for {len(mapping)} frames from {tracks_path}")
    return mapping


def load_tracks_meta(tracks_path):
    """Load optional metadata written by run_sam3.py."""
    tracks_path = Path(tracks_path)
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("meta", {})


def scale_players_to_image(players, scale_x, scale_y):
    """Scale bbox and mask_center coordinates to match display image size."""
    if scale_x == 1.0 and scale_y == 1.0:
        return players

    scaled = []
    for player in players:
        p = copy.deepcopy(player)
        bbox = p.get("bbox", {})
        bbox["x"] = int(round(float(bbox.get("x", 0)) * scale_x))
        bbox["y"] = int(round(float(bbox.get("y", 0)) * scale_y))
        bbox["w"] = int(round(float(bbox.get("w", 0)) * scale_x))
        bbox["h"] = int(round(float(bbox.get("h", 0)) * scale_y))
        p["bbox"] = bbox

        center = p.get("mask_center", {})
        if center:
            p["mask_center"] = {
                "x": round(float(center.get("x", 0)) * scale_x, 1),
                "y": round(float(center.get("y", 0)) * scale_y, 1),
            }
        if "mask_area" in p:
            p["mask_area"] = int(float(p["mask_area"]) * scale_x * scale_y)
        scaled.append(p)
    return scaled


def _track_display_scales(meta, image_bgr):
    """Return scale factors from track coordinate space to image pixels."""
    if image_bgr is None:
        return 1.0, 1.0

    img_h, img_w = image_bgr.shape[:2]
    track_w = int(meta.get("frame_width", img_w))
    track_h = int(meta.get("frame_height", img_h))
    if track_w <= 0 or track_h <= 0:
        return 1.0, 1.0
    if track_w == img_w and track_h == img_h:
        return 1.0, 1.0
    return img_w / track_w, img_h / track_h


def _frame_path_for_number(frame_paths, frame_number):
    """Map 1-based frame_number to sorted JPEG path."""
    idx = int(frame_number) - 1
    if 0 <= idx < len(frame_paths):
        return frame_paths[idx]
    return None


def _tracked_frame_numbers(tracks_by_frame):
    """Sorted frame numbers that have track entries."""
    return sorted(int(n) for n in tracks_by_frame.keys() if int(n) > 0)


def _get_sorted_frame_paths(frames_dir):
    """Return JPEG frame paths sorted numerically by stem."""
    frames_dir = Path(frames_dir)
    return sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def _write_jpeg(path, image_bgr, quality=95):
    """Write JPEG with fixed quality."""
    cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def save_annotated_frames(frames_dir, tracks_path, output_dir, every_n=1):
    """
    Save annotated frames for one tracks file.

    Uses frame_number = file_index + 1 mapping.
    """
    frame_paths = _get_sorted_frame_paths(frames_dir)
    tracks_by_frame = load_tracks_by_frame(tracks_path)
    meta = load_tracks_meta(tracks_path)
    tracked_numbers = _tracked_frame_numbers(tracks_by_frame)

    if len(frame_paths) != len(tracked_numbers):
        print(
            f"[WARN] Frame count mismatch: {len(frame_paths)} JPEGs vs "
            f"{len(tracked_numbers)} tracked frames. Visualizing tracked frames only."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    print(f"Annotating {len(tracked_numbers)} tracked frames (every_n={every_n})...")

    for frame_number in tracked_numbers:
        if every_n > 1 and ((frame_number - 1) % every_n) != 0:
            continue

        frame_path = _frame_path_for_number(frame_paths, frame_number)
        if frame_path is None:
            print(f"Skipping frame {frame_number}: no matching JPEG")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        sx, sy = _track_display_scales(meta, frame)
        frame_tracks = tracks_by_frame.get(frame_number, [])
        annotated = draw_tracks_on_frame(frame, frame_tracks, scale_x=sx, scale_y=sy)

        out_name = f"annotated_{frame_number:04d}.jpg"
        out_path = output_dir / out_name
        _write_jpeg(out_path, annotated, quality=95)
        saved.append(str(out_path))

        if len(saved) % 50 == 0:
            print(f"Saved {len(saved)} annotated frames...")

    print(f"Done. Saved {len(saved)} annotated frames to {output_dir}")
    return saved


def _add_comparison_label_bar(image_bgr, left_label, right_label):
    """Add dark top bar with left/right labels."""
    h, w = image_bgr.shape[:2]
    bar_h = 34
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:bar_h, :, :] = (35, 35, 35)
    out[bar_h:, :, :] = image_bgr

    half = w // 2
    cv2.putText(
        out,
        left_label,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        right_label,
        (half + 12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def save_comparison_frames(frames_dir, baseline_path, augmented_path, output_dir, every_n=1):
    """
    Save side-by-side baseline vs augmented comparisons per frame.
    """
    frame_paths = _get_sorted_frame_paths(frames_dir)
    baseline = load_tracks_by_frame(baseline_path)
    augmented = load_tracks_by_frame(augmented_path)
    meta = load_tracks_meta(baseline_path)
    tracked_numbers = _tracked_frame_numbers(baseline)

    if len(frame_paths) != len(tracked_numbers):
        print(
            f"[WARN] Frame count mismatch: {len(frame_paths)} JPEGs vs "
            f"{len(tracked_numbers)} tracked frames. Visualizing tracked frames only."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    print(f"Creating comparison frames for {len(tracked_numbers)} tracked frames...")

    for frame_number in tracked_numbers:
        if every_n > 1 and ((frame_number - 1) % every_n) != 0:
            continue

        frame_path = _frame_path_for_number(frame_paths, frame_number)
        if frame_path is None:
            print(f"Skipping frame {frame_number}: no matching JPEG")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        sx, sy = _track_display_scales(meta, frame)
        left = draw_tracks_on_frame(
            frame, baseline.get(frame_number, []), scale_x=sx, scale_y=sy
        )
        right = draw_tracks_on_frame(
            frame, augmented.get(frame_number, []), scale_x=sx, scale_y=sy
        )
        comparison = np.hstack([left, right])
        comparison = _add_comparison_label_bar(
            comparison, "SAM3.1 Raw", "SAM3.1 + Augmentation"
        )

        out_name = f"comparison_{frame_number:04d}.jpg"
        out_path = output_dir / out_name
        _write_jpeg(out_path, comparison, quality=95)
        saved.append(str(out_path))

        if len(saved) % 50 == 0:
            print(f"Saved {len(saved)} comparison frames...")

    print(f"Done. Saved {len(saved)} comparison frames to {output_dir}")
    return saved


def create_summary_figure(frames_dir, baseline_path, augmented_path, output_path, n_frames=4):
    """
    Create publication-quality figure with baseline vs augmentation pairs.

    Frame rows are evenly spaced across tracked frame numbers (not orphan JPEGs).
    """
    frame_paths = _get_sorted_frame_paths(frames_dir)
    baseline = load_tracks_by_frame(baseline_path)
    augmented = load_tracks_by_frame(augmented_path)
    meta = load_tracks_meta(baseline_path)
    tracked_numbers = _tracked_frame_numbers(baseline)

    if not frame_paths:
        raise ValueError("No JPEG frames found for summary figure.")
    if not tracked_numbers:
        raise ValueError("No tracked frames found in baseline tracks.json.")

    if len(frame_paths) != len(tracked_numbers):
        print(
            f"[WARN] Frame count mismatch: {len(frame_paths)} JPEGs vs "
            f"{len(tracked_numbers)} tracked frames."
        )

    n_rows = max(1, min(int(n_frames), len(tracked_numbers)))
    pick_idx = np.linspace(0, len(tracked_numbers) - 1, n_rows, dtype=int)
    summary_frame_numbers = [tracked_numbers[int(i)] for i in pick_idx]

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 2.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, frame_number in enumerate(summary_frame_numbers):
        frame_path = _frame_path_for_number(frame_paths, frame_number)
        frame = cv2.imread(str(frame_path)) if frame_path else None
        if frame is None:
            frame = np.zeros((270, 480, 3), dtype=np.uint8)

        sx, sy = _track_display_scales(meta, frame)
        left = draw_tracks_on_frame(
            frame, baseline.get(frame_number, []), scale_x=sx, scale_y=sy
        )
        right = draw_tracks_on_frame(
            frame, augmented.get(frame_number, []), scale_x=sx, scale_y=sy
        )

        left = cv2.resize(left, (480, 270), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (480, 270), interpolation=cv2.INTER_AREA)

        # Convert BGR -> RGB for matplotlib display
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        ax_l = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]
        ax_l.imshow(left_rgb)
        ax_r.imshow(right_rgb)

        ax_l.set_ylabel(f"Frame {frame_number}", fontsize=10)
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_r.set_xticks([])
        ax_r.set_yticks([])

    axes[0, 0].set_title("SAM3.1 Raw", fontsize=12, pad=10)
    axes[0, 1].set_title("SAM3.1 + Augmentation", fontsize=12, pad=10)
    fig.suptitle("SAM3.1 Baseline vs Augmentation", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary figure to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3.1 tracking outputs.")
    parser.add_argument("--frames", required=True, help="Directory containing JPEG frames")
    parser.add_argument("--tracks", help="Single tracks.json path for annotation mode")
    parser.add_argument("--baseline", help="Baseline tracks.json path for compare/summary mode")
    parser.add_argument("--augmented", help="Augmented tracks.json path for compare/summary mode")
    parser.add_argument("--output", required=True, help="Output directory or figure path")
    parser.add_argument("--every-n", type=int, default=1, help="Process every n-th frame")
    parser.add_argument("--compare", action="store_true", help="Run side-by-side comparison mode")
    parser.add_argument("--summary", action="store_true", help="Run summary figure mode")
    parser.add_argument(
        "--n-frames",
        type=int,
        default=4,
        help="Number of frame rows in summary figure",
    )
    args = parser.parse_args()

    if args.summary:
        if not args.baseline or not args.augmented:
            raise ValueError("--summary mode requires --baseline and --augmented")
        create_summary_figure(
            frames_dir=args.frames,
            baseline_path=args.baseline,
            augmented_path=args.augmented,
            output_path=args.output,
            n_frames=args.n_frames,
        )
    elif args.compare:
        if not args.baseline or not args.augmented:
            raise ValueError("--compare mode requires --baseline and --augmented")
        save_comparison_frames(
            frames_dir=args.frames,
            baseline_path=args.baseline,
            augmented_path=args.augmented,
            output_dir=args.output,
            every_n=max(1, int(args.every_n)),
        )
    else:
        if not args.tracks:
            raise ValueError("Annotation mode requires --tracks")
        save_annotated_frames(
            frames_dir=args.frames,
            tracks_path=args.tracks,
            output_dir=args.output,
            every_n=max(1, int(args.every_n)),
        )


if __name__ == "__main__":
    main()

