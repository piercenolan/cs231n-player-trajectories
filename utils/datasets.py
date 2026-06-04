"""
Canonical dataset paths for the CS231N tracking pipeline.

Use --dataset sportsmot_example (default) so scripts resolve frames, GT, and
run outputs from one place instead of scattered data/frames, data/videos, etc.
"""

import json
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


# Temporal offsets for multi-seed SAM3 (seconds into 25 FPS SportsMOT clip).
SEED_OFFSETS = {
    "offset_0s": 0.0,
    "offset_10s": 10.0,
    "offset_15s": 15.0,
}


def seed_gt_aligned_path(dataset="sportsmot_example", seed_id="offset_0s"):
    return runs_dir(dataset, seed_id) / "gt_aligned.json"


def start_time_sec_for_seed(seed_id, tracks_meta=None):
    """Resolve start_time_sec from tracks meta or SEED_OFFSETS."""
    if tracks_meta and tracks_meta.get("start_time_sec") is not None:
        return float(tracks_meta["start_time_sec"])
    if seed_id in SEED_OFFSETS:
        return float(SEED_OFFSETS[seed_id])
    return 0.0


def align_seed_gt(
    dataset="sportsmot_example",
    seed_id="offset_0s",
    tracks_path=None,
    start_time_sec=None,
    extract_fps=None,
    output_path=None,
):
    """
    Build gt_aligned.json for one seed window from raw gt.txt + baseline tracks meta.
    """
    from utils.gt_align import align_mot_gt_to_tracks, save_aligned_gt_json

    ds = get_dataset(dataset)
    tracks_path = Path(tracks_path or baseline_tracks_path(dataset, seed_id))
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Missing baseline tracks: {tracks_path}")

    with open(tracks_path, encoding="utf-8") as f:
        meta = dict(json.load(f).get("meta") or {})

    start = (
        start_time_sec
        if start_time_sec is not None
        else start_time_sec_for_seed(seed_id, meta)
    )
    fps = extract_fps if extract_fps is not None else float(ds["extract_fps"])
    out = Path(output_path or seed_gt_aligned_path(dataset, seed_id))

    raw = Path(ds["gt_mot"])
    seqinfo = ds["seqinfo"]
    seqinfo_path = str(seqinfo) if seqinfo and Path(seqinfo).exists() else None
    if seqinfo_path is None:
        seqinfo_path = str(raw.parent.parent / "seqinfo.ini")

    aligned = align_mot_gt_to_tracks(
        raw,
        meta,
        seqinfo_path=seqinfo_path,
        extract_fps=fps,
        start_time_sec=start,
    )
    out_meta = {
        **meta,
        "gt_source": "sportsmot_mot",
        "dataset": dataset,
        "seed_id": seed_id,
        "raw_gt": str(raw),
        "extract_fps": fps,
        "start_time_sec": start,
    }
    save_aligned_gt_json(aligned, out, meta=out_meta)
    return out


def resolve_seed_gt_path(dataset, seed_id, baseline_path, align_if_missing=True):
    """
    Per-seed aligned GT; falls back to dataset gt.json only for offset_0s.
    """
    per_seed = seed_gt_aligned_path(dataset, seed_id)
    if per_seed.exists():
        return per_seed

    if align_if_missing:
        baseline_path = Path(baseline_path)
        if baseline_path.is_file():
            align_seed_gt(dataset, seed_id, tracks_path=baseline_path)
            if per_seed.exists():
                return per_seed

    if seed_id == "offset_0s":
        global_gt = find_gt_path(dataset)
        if global_gt:
            return global_gt

    raise FileNotFoundError(
        f"No aligned GT for seed '{seed_id}'. Run:\n"
        f"  py scripts/align_seed_gt.py --dataset {dataset} --seed-id {seed_id}"
    )


def resolve_augmented_tracks_path(dataset="sportsmot_example"):
    """
    Return path to augmented tracks for export / metrics.

    Prefers data/runs/{dataset}/augmented_tracks.json, then the LSTM v1 ablation
    from recommended_config.json, then any ablation output if present.
    """
    import json

    primary = augmented_tracks_path(dataset)
    if primary.exists():
        return primary

    rec_path = ablations_dir(dataset) / "recommended_config.json"
    if rec_path.exists():
        with open(rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        for key in ("recommended_ablation_lstm_v1", "recommended_ablation_ade_proxy"):
            name = rec.get(key)
            if not name:
                continue
            cand = ablations_dir(dataset) / name / "augmented_tracks.json"
            if cand.exists():
                return cand

    ab_root = ablations_dir(dataset)
    if ab_root.is_dir():
        for sub in sorted(ab_root.iterdir()):
            cand = sub / "augmented_tracks.json"
            if cand.is_file():
                return cand

    raise FileNotFoundError(
        f"No augmented tracks for dataset '{dataset}'. Run:\n"
        f"  py utils/augmentation.py --dataset {dataset} --rules velocity_cap --no-gap-fill\n"
        "or complete run_ablations.py first."
    )
