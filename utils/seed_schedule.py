"""
Build temporal seed schedules for multi-window SAM3 / LSTM training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from utils.datasets import frames_dir, runs_dir

SEED_ID_RE = re.compile(r"^offset_(\d+(?:\.\d+)?)s$")


def seed_id_from_offset_sec(sec: float) -> str:
    """Canonical seed folder name, e.g. offset_5s or offset_12.5s."""
    if abs(sec - round(sec)) < 1e-6:
        return f"offset_{int(round(sec))}s"
    return f"offset_{sec:g}s"


def parse_seed_offset_sec(seed_id: str) -> float | None:
    m = SEED_ID_RE.match(seed_id)
    if m:
        return float(m.group(1))
    return None


def count_dataset_frames(dataset: str = "sportsmot_example") -> int:
    fd = Path(frames_dir(dataset))
    if not fd.is_dir():
        return 0
    return len(list(fd.glob("*.jpg")))


def max_start_time_sec(
    num_frames: int,
    max_frames: int = 45,
    source_fps: float = 25.0,
    extract_fps: float = 25.0,
) -> float:
    """
    Latest start_time_sec so a window of max_frames still fits in num_frames.

    Matches prepare_frames_from_dir: start_mot = round(sec*fps)+1, interval=1
    when extract_fps equals source_fps.
    """
    interval = max(int(round(source_fps / extract_fps)), 1)
    # Last MOT index consumed: start_mot + (max_frames - 1) * interval
    # start_mot = round(sec * source_fps) + 1  (1-based frame names)
    last_start_mot = num_frames - (max_frames - 1) * interval
    if last_start_mot < 1:
        return 0.0
    return max(0.0, (last_start_mot - 1) / float(source_fps))


def build_seed_schedule(
    num_frames: int = 500,
    step_sec: float = 5.0,
    max_frames: int = 45,
    source_fps: float = 25.0,
    extract_fps: float = 25.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[dict]:
    """
    Return list of {seed_id, start_time_sec} covering the clip every step_sec.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if step_sec <= 0:
        raise ValueError("step_sec must be positive")

    end = end_sec
    if end is None:
        end = max_start_time_sec(num_frames, max_frames, source_fps, extract_fps)

    schedule = []
    t = float(start_sec)
    while t <= end + 1e-9:
        schedule.append(
            {
                "seed_id": seed_id_from_offset_sec(t),
                "start_time_sec": round(t, 6),
            }
        )
        t += step_sec

    if not schedule:
        schedule.append({"seed_id": seed_id_from_offset_sec(0.0), "start_time_sec": 0.0})
    return schedule


def seed_manifest_path(dataset: str = "sportsmot_example") -> Path:
    return runs_dir(dataset) / "seeds" / "seed_manifest.json"


def write_seed_manifest(
    dataset: str,
    schedule: list[dict],
    *,
    num_frames: int,
    step_sec: float,
    max_frames: int,
    resize_scale: float,
    extra: dict | None = None,
) -> Path:
    path = seed_manifest_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "num_frames": num_frames,
        "step_sec": step_sec,
        "max_frames": max_frames,
        "resize_scale": resize_scale,
        "seeds": schedule,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_seed_manifest(dataset: str = "sportsmot_example") -> dict | None:
    path = seed_manifest_path(dataset)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_seed_entries(dataset: str = "sportsmot_example") -> list[tuple[str, float]]:
    """
    (seed_id, start_time_sec) from manifest, else scan seeds/*/baseline_tracks.json.
    """
    manifest = load_seed_manifest(dataset)
    if manifest and manifest.get("seeds"):
        return [
            (s["seed_id"], float(s["start_time_sec"]))
            for s in manifest["seeds"]
        ]

    entries = []
    seeds_root = runs_dir(dataset) / "seeds"
    if not seeds_root.is_dir():
        return entries

    for seed_dir in sorted(seeds_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        if not (seed_dir / "baseline_tracks.json").is_file():
            continue
        sid = seed_dir.name
        off = parse_seed_offset_sec(sid)
        entries.append((sid, off if off is not None else 0.0))
    return entries
