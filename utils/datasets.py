"""
Canonical dataset paths for the CS231N tracking pipeline.

Use --dataset sportsmot_example (default) so scripts resolve frames, GT, and
run outputs from one place instead of scattered data/frames, data/videos, etc.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

DATASETS = {
    "sportsmot_example": {
        "description": "Official SportsMOT example zip (500 frames + gt.txt)",
        "frames_dir": DATA_ROOT / "datasets" / "sportsmot_example" / "frames",
        "gt_mot": DATA_ROOT / "datasets" / "sportsmot_example" / "gt" / "gt.txt",
        "gt_json": DATA_ROOT / "datasets" / "sportsmot_example" / "gt" / "gt.json",
        "seqinfo": DATA_ROOT / "datasets" / "sportsmot_example" / "seqinfo.ini",
        "video": None,
        "source_fps": 25.0,
        "extract_fps": 25.0,
    },
    "video_1_legacy": {
        "description": "Legacy unknown-source clip (deprecated; use sportsmot_example)",
        "frames_dir": DATA_ROOT / "archive" / "video_1_legacy" / "frames",
        "gt_mot": DATA_ROOT / "archive" / "video_1_legacy" / "gt" / "gt.txt",
        "gt_json": DATA_ROOT / "archive" / "video_1_legacy" / "gt" / "gt.json",
        "seqinfo": None,
        "video": DATA_ROOT / "archive" / "video_1_legacy" / "video_1.mp4",
        "source_fps": 25.0,
        "extract_fps": 1.0,
    },
}


def get_dataset(name="sportsmot_example"):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Valid: {sorted(DATASETS)}")
    return DATASETS[name]


def runs_dir(dataset="sportsmot_example", seed_id=None):
    """Pipeline outputs: data/runs/{dataset}/ or data/runs/{dataset}/seeds/{seed_id}/."""
    base = DATA_ROOT / "runs" / dataset
    if seed_id:
        return base / "seeds" / seed_id
    return base


def baseline_tracks_path(dataset="sportsmot_example", seed_id=None):
    return runs_dir(dataset, seed_id) / "baseline_tracks.json"


def augmented_tracks_path(dataset="sportsmot_example", seed_id=None):
    return runs_dir(dataset, seed_id) / "augmented_tracks.json"


def ablations_dir(dataset="sportsmot_example", seed_id=None):
    return runs_dir(dataset, seed_id) / "ablations"


def find_gt_path(dataset="sportsmot_example", prefer_json=True):
    """Return best available GT file for a dataset."""
    cfg = get_dataset(dataset)
    if prefer_json and cfg["gt_json"] and Path(cfg["gt_json"]).exists():
        return Path(cfg["gt_json"])
    if cfg["gt_mot"] and Path(cfg["gt_mot"]).exists():
        return Path(cfg["gt_mot"])
    return None


def frames_dir(dataset="sportsmot_example"):
    return get_dataset(dataset)["frames_dir"]
