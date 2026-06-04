"""
Baseline tracking quality metrics for SAM3.1 on basketball video.

Loads tracks.json from run_sam3.py and measures ID stability, coverage,
and continuity before any basketball-domain augmentation is applied.

Usage:
    python utils/metrics.py --tracks data/outputs/tracks.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_tracks(tracks_path):
    """
    Load SAM3.1 tracking output from disk.

    Returns the frames list used by all other metric functions.
    """
    tracks_path = Path(tracks_path)
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    print(f"Loaded {len(frames)} frames from {tracks_path}")
    return frames


def _sorted_frames(frames):
    """Return frames sorted by frame_number; empty list if no frames."""
    if not frames:
        return []
    return sorted(frames, key=lambda f: int(f["frame_number"]))


def _is_predicted(player):
    return bool(player.get("predicted", False))


def _players_per_frame(frames, include_predicted=True):
    """Return parallel lists of frame numbers and player counts."""
    sorted_frames = _sorted_frames(frames)
    if not sorted_frames:
        return [], []
    numbers = [f["frame_number"] for f in sorted_frames]
    counts = []
    for f in sorted_frames:
        players = f.get("players", [])
        if include_predicted:
            counts.append(len(players))
        else:
            counts.append(sum(1 for p in players if not _is_predicted(p)))
    return numbers, counts


def observed_players_per_frame(frames):
    """Count non-predicted players per frame."""
    return _players_per_frame(frames, include_predicted=False)


def predicted_players_per_frame(frames):
    """Count predicted (gap-filled) players per frame."""
    sorted_frames = _sorted_frames(frames)
    if not sorted_frames:
        return [], []
    numbers = [f["frame_number"] for f in sorted_frames]
    counts = [
        sum(1 for p in f.get("players", []) if _is_predicted(p)) for f in sorted_frames
    ]
    return numbers, counts


def compute_trajectory_smoothness(frames, include_predicted=False):
    """
    Mean per-frame center displacement and max jump (proxy for jerk).

    Lower is smoother. Uses mask_center when present.
    """
    sorted_frames = _sorted_frames(frames)
    if len(sorted_frames) < 2:
        return {"mean_displacement": 0.0, "max_jump": 0.0, "total_jumps": 0}

    id_to_series = defaultdict(list)
    for frame in sorted_frames:
        fnum = int(frame["frame_number"])
        for player in frame.get("players", []):
            if not include_predicted and _is_predicted(player):
                continue
            center = player.get("mask_center", {})
            if "x" not in center or "y" not in center:
                continue
            id_to_series[player["id"]].append(
                (fnum, float(center["x"]), float(center["y"]))
            )

    jumps = []
    for _pid, series in id_to_series.items():
        series = sorted(series, key=lambda t: t[0])
        for (_, x0, y0), (_, x1, y1) in zip(series, series[1:]):
            jumps.append(float(np.hypot(x1 - x0, y1 - y0)))

    if not jumps:
        return {"mean_displacement": 0.0, "max_jump": 0.0, "total_jumps": 0}

    arr = np.array(jumps, dtype=float)
    return {
        "mean_displacement": float(np.mean(arr)),
        "max_jump": float(np.max(arr)),
        "total_jumps": int(len(arr)),
    }


def count_id_switches(frames, gap_threshold=5, include_predicted=True):
    """
    Count identity re-acquisitions after extended dropout.

    For each player ID we record every frame it appears in. If an ID vanishes
    for more than `gap_threshold` consecutive frames and then returns, SAM3.1
    likely lost the track and assigned a new identity — a core failure mode
    for basketball research (ID switches during occlusion or fast motion).

    Returns:
        (total_switches, reappearances): total count and
        {frame_number: [ids that reappeared on that frame]}.
    """
    if not frames:
        return 0, {}

    id_to_frames = defaultdict(list)
    for frame in _sorted_frames(frames):
        frame_num = frame["frame_number"]
        for player in frame.get("players", []):
            if not include_predicted and _is_predicted(player):
                continue
            id_to_frames[player["id"]].append(frame_num)

    reappearances = defaultdict(list)
    total_switches = 0
    min_gap = gap_threshold + 1  # more than 5 missing frames => gap >= 6

    for player_id, appearances in id_to_frames.items():
        appearances = sorted(set(appearances))
        if len(appearances) < 2:
            continue
        for prev, curr in zip(appearances, appearances[1:]):
            if curr - prev > min_gap:
                total_switches += 1
                reappearances[curr].append(player_id)

    return total_switches, dict(reappearances)


def count_tracking_loss(frames, window=10, drop_threshold=0.5):
    """
    Flag frames where coverage drops sharply vs a rolling recent peak.

    Uses the maximum player count over the previous `window` frames as a
    dynamic baseline (handles camera pans and fast breaks without assuming
    a fixed roster size). A loss event occurs when the current count falls
    more than `drop_threshold` below that rolling maximum.

    Returns:
        (loss_frames, loss_count): list of frame numbers flagged and count.
    """
    frame_numbers, counts = _players_per_frame(frames)
    if not frame_numbers:
        return [], 0

    loss_frames = []
    for i, (frame_num, count) in enumerate(zip(frame_numbers, counts)):
        start = max(0, i - window)
        if i == 0:
            continue
        rolling_max = max(counts[start:i])
        if rolling_max == 0:
            continue
        threshold = (1.0 - drop_threshold) * rolling_max
        if count < threshold:
            loss_frames.append(frame_num)

    return loss_frames, len(loss_frames)


def count_sudden_drops(frames, drop_size=3):
    """
    Flag abrupt single-step drops in tracked player count.

    Simpler complement to rolling-window loss detection: marks any frame
    where the count falls by at least `drop_size` versus the immediately
    prior frame. Sensitive to sudden SAM3.1 failures on one timestep.

    Returns:
        (drop_frames, drop_count): list of frame numbers and count.
    """
    frame_numbers, counts = _players_per_frame(frames)
    if len(frame_numbers) < 2:
        return [], 0

    drop_frames = []
    for i in range(1, len(counts)):
        if counts[i - 1] - counts[i] >= drop_size:
            drop_frames.append(frame_numbers[i])

    return drop_frames, len(drop_frames)


def compute_track_continuity(frames, include_predicted=True):
    """
    Measure how long each player ID stays continuously visible.

    For every unique ID, finds the longest run of consecutive frame numbers
    in which that ID is tracked. Longer streaks indicate stable SAM3.1
    association; short streaks indicate flickering or fragmentary tracks.

    Returns:
        (longest_streaks, mean_streak, min_streak):
        dict {player_id: longest_streak}, mean across IDs, minimum streak.
    """
    id_to_frames = defaultdict(list)
    for frame in _sorted_frames(frames):
        frame_num = frame["frame_number"]
        for player in frame.get("players", []):
            if not include_predicted and _is_predicted(player):
                continue
            id_to_frames[player["id"]].append(frame_num)

    if not id_to_frames:
        return {}, 0.0, 0

    longest_streaks = {}
    for player_id, appearances in id_to_frames.items():
        appearances = sorted(set(appearances))
        best = 1
        current = 1
        for prev, curr in zip(appearances, appearances[1:]):
            if curr == prev + 1:
                current += 1
            else:
                best = max(best, current)
                current = 1
        best = max(best, current)
        longest_streaks[player_id] = best

    streak_values = list(longest_streaks.values())
    mean_streak = float(np.mean(streak_values))
    min_streak = int(min(streak_values))
    return longest_streaks, mean_streak, min_streak


def compute_id_consistency(frames, include_predicted=True):
    """
    Summarize how stable the number of active tracks is over time.

    Counts players per frame (unique IDs present). High variance means
    SAM3.1 is inconsistently maintaining the set of tracked players —
    a problem for downstream trajectory and team analytics.

    Returns:
        (mean_count, std_count, min_count, max_count).
    """
    _, counts = _players_per_frame(frames, include_predicted=include_predicted)
    if not counts:
        return 0.0, 0.0, 0, 0

    arr = np.array(counts, dtype=float)
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        int(np.min(arr)),
        int(np.max(arr)),
    )


def collect_metrics_dict(frames, expected_players=10, include_predicted=False):
    """Return a flat metrics dict suitable for JSON export."""
    mean_players, std_players, min_players, max_players = compute_id_consistency(
        frames, include_predicted=include_predicted
    )
    total_id_switches, _ = count_id_switches(frames, include_predicted=include_predicted)
    _, mean_streak, min_streak = compute_track_continuity(
        frames, include_predicted=include_predicted
    )
    _, obs_counts = observed_players_per_frame(frames)
    _, pred_counts = predicted_players_per_frame(frames)
    smooth = compute_trajectory_smoothness(frames, include_predicted=include_predicted)
    loss_count = count_tracking_loss(frames)[1]
    sudden_drop_count = count_sudden_drops(frames)[1]

    total_frames = len(_sorted_frames(frames))
    full_cov = sum(1 for c in obs_counts if c >= expected_players) if obs_counts else 0

    return {
        "total_frames": total_frames,
        "include_predicted_in_counts": include_predicted,
        "mean_players_per_frame": mean_players,
        "std_players_per_frame": std_players,
        "min_players_per_frame": min_players,
        "max_players_per_frame": max_players,
        "mean_observed_per_frame": float(np.mean(obs_counts)) if obs_counts else 0.0,
        "mean_predicted_per_frame": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "frames_with_full_coverage": full_cov,
        "total_id_switches": total_id_switches,
        "mean_track_streak": mean_streak,
        "min_track_streak": min_streak,
        "mean_displacement": smooth["mean_displacement"],
        "max_jump": smooth["max_jump"],
        "rolling_loss_frames": loss_count,
        "sudden_drop_frames": sudden_drop_count,
    }


def compare_reports(
    baseline_path,
    augmented_path,
    expected_players=10,
    include_predicted_in_aug=False,
):
    """
    Print baseline vs augmented deltas for research ablations.
    """
    baseline_frames = load_tracks(baseline_path)
    augmented_frames = load_tracks(augmented_path)

    base = collect_metrics_dict(
        baseline_frames, expected_players=expected_players, include_predicted=True
    )
    aug = collect_metrics_dict(
        augmented_frames,
        expected_players=expected_players,
        include_predicted=include_predicted_in_aug,
    )

    print()
    print("=" * 50)
    print("BASELINE vs AUGMENTED COMPARISON")
    print("=" * 50)
    print(f"Baseline:  {baseline_path}")
    print(f"Augmented: {augmented_path}")
    print(f"(Augmented counts exclude predicted: {not include_predicted_in_aug})")
    print()

    keys = [
        ("mean_observed_per_frame", "Mean observed players/frame", False),
        ("mean_predicted_per_frame", "Mean predicted players/frame", False),
        ("total_id_switches", "ID switches", True),
        ("mean_track_streak", "Mean track streak", False),
        ("mean_displacement", "Mean displacement (smoothness)", True),
        ("max_jump", "Max jump", True),
        ("rolling_loss_frames", "Rolling loss frames", True),
    ]

    for key, label, lower_is_better in keys:
        b = base.get(key, 0)
        a = aug.get(key, 0)
        delta = a - b
        direction = "better" if (delta < 0) == lower_is_better else "worse"
        if delta == 0:
            direction = "same"
        print(f"  {label:<32} base={b:8.2f}  aug={a:8.2f}  delta={delta:+8.2f}  ({direction})")

    print("=" * 50)
    print()

    return {"baseline": base, "augmented": aug, "delta": {k: aug[k] - base[k] for k in base}}


def summary_report(
    tracks_path,
    expected_players=10,
    frames=None,
    label="SAM3.1 BASELINE",
    include_predicted=True,
):
    """
    Run all baseline metrics and print a formatted research report.

    Produces the SAM3.1 pre-augmentation baseline numbers used to judge
    whether basketball rule layers reduce ID switches and improve coverage.

    Returns:
        Dict with fixed keys for Weights & Biases logging.
    """
    if frames is None:
        if tracks_path is None:
            raise ValueError("tracks_path is required when frames is None")
        frames = load_tracks(tracks_path)

    mean_players, std_players, min_players, max_players = compute_id_consistency(
        frames, include_predicted=include_predicted
    )
    total_id_switches, _ = count_id_switches(frames, include_predicted=include_predicted)
    _, mean_streak, min_streak = compute_track_continuity(
        frames, include_predicted=include_predicted
    )
    smooth = compute_trajectory_smoothness(frames, include_predicted=include_predicted)
    _, obs_counts = observed_players_per_frame(frames)
    _, pred_counts = predicted_players_per_frame(frames)

    total_frames = len(frames)
    _, counts = _players_per_frame(frames, include_predicted=include_predicted)
    frames_with_full_coverage = sum(1 for c in counts if c >= expected_players)
    frames_with_loss = sum(1 for c in counts if c < expected_players)

    full_pct = (
        100.0 * frames_with_full_coverage / total_frames if total_frames else 0.0
    )
    loss_pct = 100.0 * frames_with_loss / total_frames if total_frames else 0.0

    print()
    print("=" * 42)
    print(f"{label} METRICS REPORT")
    print("=" * 42)
    print(f"Total frames analyzed: {total_frames}")
    print(f"Include predicted in counts: {include_predicted}")
    print()
    print("--- Tracking Coverage ---")
    print(
        f"Mean players tracked per frame: {mean_players:.1f} / {expected_players}"
    )
    print(f"Std deviation: {std_players:.1f}")
    print(f"Min players in a frame: {min_players}")
    print(f"Max players in a frame: {max_players}")
    print(
        f"Frames with full coverage ({expected_players} players): "
        f"{frames_with_full_coverage} ({full_pct:.1f}%)"
    )
    print(
        f"Frames with tracking loss (<{expected_players} players): "
        f"{frames_with_loss} ({loss_pct:.1f}%)"
    )
    if pred_counts:
        print(f"Mean observed per frame: {np.mean(obs_counts):.1f}")
        print(f"Mean predicted per frame: {np.mean(pred_counts):.1f}")
    print()
    print("--- Trajectory Smoothness ---")
    print(f"Mean displacement: {smooth['mean_displacement']:.2f} px")
    print(f"Max jump: {smooth['max_jump']:.2f} px")
    print()
    print("--- ID Stability ---")
    print(f"Total ID switches detected: {total_id_switches}")
    print(f"Mean track streak before loss: {mean_streak:.1f} frames")
    print(f"Min track streak: {min_streak} frames")
    print()
    print("=" * 42)
    print(f"These are {label} numbers.")
    print("Target: augmentation layer should reduce")
    print("ID switches and increase mean streak.")
    print("=" * 42)
    print()

    # Smart loss detection — handles camera pans and fast breaks
    loss_frames, loss_count = count_tracking_loss(frames)
    sudden_drop_frames, sudden_drop_count = count_sudden_drops(frames)
    print()
    print("--- Smart Loss Detection ---")
    print(f"Rolling-window loss events: {loss_count} frames")
    print(f"Sudden drop events (3+ players): {sudden_drop_count} frames")
    print("(These exclude camera pans and fast breaks)")

    metrics = collect_metrics_dict(
        frames, expected_players=expected_players, include_predicted=include_predicted
    )
    metrics["frames_with_loss"] = frames_with_loss
    return metrics


def plot_metrics(
    frames,
    output_path="data/outputs/baseline_metrics.png",
    expected_players=10,
    label="SAM3.1 BASELINE",
    include_predicted=True,
):
    """
    Save a two-panel baseline figure for the research paper.

    Panel 1: players tracked per frame (coverage over time).
    Panel 2: longest continuous streak per player ID (continuity).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_numbers, counts = _players_per_frame(frames, include_predicted=include_predicted)
    longest_streaks, _, _ = compute_track_continuity(
        frames, include_predicted=include_predicted
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: coverage over time
    ax1 = axes[0]
    if frame_numbers:
        colors = [
            "#2ca02c" if c >= expected_players else "#d62728" for c in counts
        ]
        ax1.scatter(frame_numbers, counts, c=colors, s=12, alpha=0.85, edgecolors="none")
        ax1.plot(frame_numbers, counts, color="#1f77b4", linewidth=0.8, alpha=0.5)
        ax1.axhline(
            expected_players,
            color="#444444",
            linestyle="--",
            linewidth=1,
            label=f"Full coverage ({expected_players})",
        )
    ax1.set_xlim(left=0)
    if counts:
        ymax = max(12, int(max(counts) + 2))
        ax1.set_ylim(0, ymax)
    else:
        ax1.set_ylim(0, 12)
    ax1.set_xlabel("Frame number")
    ax1.set_ylabel("Player count")
    ax1.set_title(f"{label}: Players Tracked Per Frame")
    ax1.grid(True, alpha=0.3)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ca02c",
            markersize=8,
            label=f"= {expected_players} players",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markersize=8,
            label=f"< {expected_players} players",
        ),
        Line2D([0], [0], color="#444444", linestyle="--", label="Expected roster"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Panel 2: continuity per ID
    ax2 = axes[1]
    if longest_streaks:
        ids = sorted(longest_streaks.keys())
        streaks = [longest_streaks[i] for i in ids]
        ax2.bar(ids, streaks, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Player ID")
    ax2.set_ylabel("Longest streak (frames)")
    ax2.set_title(f"{label}: Track Continuity Per Player")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline metrics figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute SAM3.1 baseline tracking metrics from tracks.json"
    )
    parser.add_argument(
        "--tracks",
        default="data/outputs/tracks.json",
        help="Path to tracks.json from run_sam3.py",
    )
    parser.add_argument(
        "--output-figure",
        default="data/outputs/baseline_metrics.png",
        help="Path to save the baseline metrics figure",
    )
    parser.add_argument(
        "--expected-players",
        type=int,
        default=10,
        help="Roster size for full-coverage reporting and plot thresholds",
    )
    parser.add_argument(
        "--label",
        default="SAM3.1 BASELINE",
        help="Label for the metrics report and plot",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to augmented tracks; print baseline vs augmented comparison",
    )
    parser.add_argument(
        "--exclude-predicted",
        action="store_true",
        help="Exclude predicted (gap-filled) players from augmented metrics",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Save metrics dict to JSON path",
    )
    args = parser.parse_args()

    include_predicted = not args.exclude_predicted

    if args.compare:
        compare_reports(
            args.tracks,
            args.compare,
            expected_players=args.expected_players,
            include_predicted_in_aug=include_predicted,
        )
        return

    frames = load_tracks(args.tracks)
    metrics = summary_report(
        args.tracks,
        expected_players=args.expected_players,
        frames=frames,
        label=args.label,
        include_predicted=include_predicted,
    )
    plot_metrics(
        frames,
        output_path=args.output_figure,
        expected_players=args.expected_players,
        label=args.label,
        include_predicted=include_predicted,
    )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to {out}")


if __name__ == "__main__":
    main()
