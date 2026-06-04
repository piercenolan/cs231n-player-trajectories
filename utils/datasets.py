"""
Canonical dataset paths for the CS231N tracking pipeline.

Use --dataset sportsmot_example (default) so scripts resolve frames, GT, and
run outputs from one place instead of scattered data/frames, data/videos, etc.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
EXTRA_DATASETS_JSON = DATA_ROOT / "datasets" / "extra_datasets.json"


def _resolve_dataset_cfg(cfg: dict) -> dict:
    """Turn repo-relative paths in extra_datasets.json into absolute Paths."""
    def _p(key):
        v = cfg.get(key)
        if not v:
            return None
        path = Path(v)
        return path if path.is_absolute() else REPO_ROOT / path

    return {
        **cfg,
        "frames_dir": _p("frames_dir"),
        "gt_mot": _p("gt_mot"),
        "gt_json": _p("gt_json"),
        "seqinfo": _p("seqinfo"),
    }


# Sprint basketball clips (Modal must know these even if extra_datasets.json is missing).
SPRINT_BUILTIN_DATASETS = {
    "sportsmot_v_6os86hzwcs_c001": {
        "description": "SportsMOT basketball v_-6Os86HzwCs_c001 (train)",
        "frames_dir": DATA_ROOT / "datasets" / "sportsmot_v_6os86hzwcs_c001" / "frames",
        "gt_mot": DATA_ROOT / "datasets" / "sportsmot_v_6os86hzwcs_c001" / "gt" / "gt.txt",
        "gt_json": None,
        "seqinfo": DATA_ROOT / "datasets" / "sportsmot_v_6os86hzwcs_c001" / "seqinfo.ini",
        "video": None,
        "source_fps": 25.0,
        "extract_fps": 25.0,
    },
    "sportsmot_v_6os86hzwcs_c003": {
        "description": "SportsMOT basketball v_-6Os86HzwCs_c003 (train)",
        "frames_dir": DATA_ROOT / "datasets" / "sportsmot_v_6os86hzwcs_c003" / "frames",
        "gt_mot": DATA_ROOT / "datasets" / "sportsmot_v_6os86hzwcs_c003" / "gt" / "gt.txt",
        "gt_json": None,
        "seqinfo": DATA_ROOT / "datasets" / "sportsmot_v_6os86hzwcs_c003" / "seqinfo.ini",
        "video": None,
        "source_fps": 25.0,
        "extract_fps": 25.0,
    },
    "sportsmot_v_00hrwkvvjtq_c001": {
        "description": "SportsMOT basketball v_00HRwkvvjtQ_c001 (val holdout)",
        "frames_dir": DATA_ROOT / "datasets" / "sportsmot_v_00hrwkvvjtq_c001" / "frames",
        "gt_mot": DATA_ROOT / "datasets" / "sportsmot_v_00hrwkvvjtq_c001" / "gt" / "gt.txt",
        "gt_json": None,
        "seqinfo": DATA_ROOT / "datasets" / "sportsmot_v_00hrwkvvjtq_c001" / "seqinfo.ini",
        "video": None,
        "source_fps": 25.0,
        "extract_fps": 25.0,
    },
}


def _load_extra_datasets():
    out = dict(SPRINT_BUILTIN_DATASETS)
    if not EXTRA_DATASETS_JSON.is_file():
        return out
    with open(EXTRA_DATASETS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    for name, cfg in raw.items():
        out[name] = _resolve_dataset_cfg(cfg)
    return out


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


def all_datasets():
    merged = dict(DATASETS)
    merged.update(_load_extra_datasets())
    return merged


def get_dataset(name="sportsmot_example"):
    merged = all_datasets()
    if name not in merged:
        raise ValueError(f"Unknown dataset '{name}'. Valid: {sorted(merged)}")
    return merged[name]


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


def start_time_sec_for_seed(seed_id, tracks_meta=None, dataset="sportsmot_example"):
    """Resolve start_time_sec from tracks meta, manifest, SEED_OFFSETS, or seed_id name."""
    if tracks_meta and tracks_meta.get("start_time_sec") is not None:
        return float(tracks_meta["start_time_sec"])
    if seed_id in SEED_OFFSETS:
        return float(SEED_OFFSETS[seed_id])
    from utils.seed_schedule import load_seed_manifest, parse_seed_offset_sec

    manifest = load_seed_manifest(dataset)
    if manifest:
        for s in manifest.get("seeds", []):
            if s.get("seed_id") == seed_id:
                return float(s["start_time_sec"])
    parsed = parse_seed_offset_sec(seed_id)
    if parsed is not None:
        return parsed
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
        else start_time_sec_for_seed(seed_id, meta, dataset=dataset)
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


LSTM_ABLATION = "sanitize_plus_velocity_cap"


def seed_augmented_tracks_path(dataset="sportsmot_example", seed_id="offset_0s"):
    """Per-seed augmented tracks from multi-seed ablation run."""
    return (
        runs_dir(dataset, seed_id)
        / LSTM_ABLATION
        / "augmented_tracks.json"
    )


def trajectory_tensor_path(dataset="sportsmot_example", seed_id=None):
    """LSTM tensor JSON: run root (offset_0s canonical) or per-seed under seeds/."""
    if seed_id:
        return runs_dir(dataset, seed_id) / "trajectory_tensors.json"
    return runs_dir(dataset) / "trajectory_tensors.json"


def lstm_tensor_paths(dataset="sportsmot_example", mode="multi"):
    """
    Resolve trajectory_tensors.json paths for LSTM training.

    mode: 'single' -> run-root tensor (offset_0s canonical export)
          'multi'  -> all SEED_OFFSETS with existing per-seed tensors
    """
    if mode == "single":
        p = trajectory_tensor_path(dataset, seed_id=None)
        if not p.exists():
            p = trajectory_tensor_path(dataset, seed_id="offset_0s")
        return [p] if p.exists() else []

    from utils.seed_schedule import list_seed_entries

    paths = []
    for seed_id, _ in list_seed_entries(dataset):
        p = trajectory_tensor_path(dataset, seed_id=seed_id)
        if p.exists():
            paths.append(p)
    if not paths:
        for seed_id in SEED_OFFSETS:
            p = trajectory_tensor_path(dataset, seed_id=seed_id)
            if p.exists():
                paths.append(p)
    if not paths:
        root = trajectory_tensor_path(dataset, seed_id=None)
        if root.exists():
            paths.append(root)
    return paths


def lstm_out_dir(dataset="sportsmot_example"):
    return runs_dir(dataset) / "lstm"
