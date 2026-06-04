"""
Run the SAM3.1 tracking pipeline on Modal GPU.

Tuned for A10G (24 GB): ~67% resolution and 45 frames balances detection
quality against VRAM. Uses CPU video offload and bf16 multiplex inference.

Usage:
    # SportsMOT example (upload frames once, then skip video):
    py -m modal run scripts/run_modal.py --dataset sportsmot_example --skip-extract

    # Legacy video clip:
    py -m modal run scripts/run_modal.py --video-path data/archive/video_1_legacy/video_1.mp4
"""

import modal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Sync code + small dataset registry JSON; exclude frames/GT blobs (use Modal volume).
_MODAL_DATA_ALLOW = frozenset(
    {
        "extra_datasets.json",
        "sprint_sequences.json",
        "EXTRACTION_STATUS.md",
    }
)


def _modal_sync_ignore(path) -> bool:
    p = Path(path)
    parts = p.parts
    if ".git" in parts or "__pycache__" in parts:
        return True
    if "data" not in parts:
        return False
    if "datasets" in parts and p.name in _MODAL_DATA_ALLOW:
        return False
    if "datasets" in parts and p.suffix == ".md":
        return False
    return True


app = modal.App("sports-trajectory")

volume = modal.Volume.from_name("sports-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install("git", "libgl1", "libglib2.0-0", "libglib2.0-dev")
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "opencv-python-headless",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
        "einops",
        "timm",
        "hydra-core",
        "omegaconf",
        "pycocotools",
        "psutil",
        "iopath",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /sam3",
        "cd /sam3 && pip install -e .",
    )
    .add_local_dir(
        str(ROOT),
        remote_path="/project",
        ignore=lambda p: _modal_sync_ignore(p),
    )
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_pipeline(
    dataset: str = "sportsmot_example",
    video_path: str = "",
    skip_extract: bool = True,
    frames_source: str = "",
    max_frames: int = 45,
    resize_scale: float = 0.67,
    output_path: str = "",
    max_num_objects: int = 14,
    start_time_sec: float = 0.0,
    extract_fps: float = 25.0,
    source_fps: float = 25.0,
    seed_id: str = "",
):
    import gc
    import glob
    import json
    import os
    import sys

    import huggingface_hub
    import torch

    volume.reload()

    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    sys.path.insert(0, "/project")
    sys.path.insert(0, "/sam3")

    from utils.datasets import baseline_tracks_path, get_dataset, runs_dir

    ds = get_dataset(dataset)
    run_root = runs_dir(dataset)
    frames_dir = str(run_root / "frames")
    if not output_path:
        output_path = str(baseline_tracks_path(dataset))

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for f in glob.glob(f"{frames_dir}/*.jpg"):
        os.remove(f)

    from scripts.run_sam3 import (
        extract_frames,
        free_cuda_memory,
        prepare_frames_from_dir,
        run_tracking,
    )

    print(
        f"Memory settings: max_frames={max_frames}, resize_scale={resize_scale}, "
        f"max_num_objects={max_num_objects}, offload_video_to_cpu=True"
    )

    if skip_extract:
        source = frames_source or f"/data/datasets/{dataset}/frames"
        if not os.path.isdir(source) or not glob.glob(f"{source}/*.jpg"):
            raise FileNotFoundError(
                f"No frames on volume at {source}. Upload with:\n"
                f"  py -m modal volume put sports-data "
                f"data/datasets/{dataset}/frames datasets/{dataset}/frames"
            )
        print(f"Preparing frames from {source} (max={max_frames}, start={start_time_sec}s)...")
        prepare_frames_from_dir(
            source,
            frames_dir,
            max_frames=max_frames,
            resize_scale=resize_scale,
            start_time_sec=start_time_sec,
            source_fps=source_fps,
            extract_fps=extract_fps,
        )
    else:
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Video not found on Modal volume: {video_path}. "
                "Re-run with --video-path or use --skip-extract."
            )
        print(
            f"Extracting frames from {video_path} (max={max_frames}, start={start_time_sec}s)..."
        )
        extract_frames(
            video_path,
            frames_dir,
            fps=extract_fps,
            max_frames=max_frames,
            resize_scale=resize_scale,
            start_time_sec=start_time_sec,
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_cuda_memory("After frame extraction")

    print("Running SAM3.1 tracking...")
    run_tracking(
        frames_dir=frames_dir,
        output_path=output_path,
        use_fp16_weights=False,
        use_fa3=False,
        offload_video_to_cpu=True,
        max_num_objects=max_num_objects,
        async_loading_frames=True,
        resize_scale=resize_scale,
    )

    with open(output_path, encoding="utf-8") as f:
        track_data = json.load(f)
    track_data.setdefault("meta", {})
    track_data["meta"]["start_time_sec"] = start_time_sec
    track_data["meta"]["extract_fps"] = extract_fps
    track_data["meta"]["source_fps"] = source_fps
    track_data["meta"]["max_frames"] = max_frames
    track_data["meta"]["resize_scale"] = resize_scale
    if seed_id:
        track_data["meta"]["seed_id"] = seed_id
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=2)
    print(f"Track metadata: {track_data.get('meta', {})}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    free_cuda_memory("End of pipeline")

    volume.commit()
    print(f"Done. Results saved to {output_path}")


@app.local_entrypoint()
def main(
    dataset: str = "sportsmot_example",
    video_path: str = "",
    skip_extract: bool = True,
    max_frames: int = 45,
    resize_scale: float = 0.67,
    output_path: str = "",
    max_num_objects: int = 14,
    skip_upload: bool = False,
    start_time_sec: float = 0.0,
    seed_id: str = "",
    extract_fps: float = 0.0,
):
    import sys

    sys.path.insert(0, str(ROOT))
    from utils.datasets import baseline_tracks_path, get_dataset

    ds = get_dataset(dataset)
    fps = extract_fps if extract_fps > 0 else float(ds["extract_fps"])
    source_fps = float(ds["source_fps"])

    out_local = baseline_tracks_path(dataset, seed_id or None)
    remote_out = output_path or (
        "/data/" + out_local.relative_to(ROOT / "data").as_posix()
    )
    rel_baseline = str(out_local.relative_to(ROOT)).replace("\\", "/")
    rel_frames = f"data/runs/{dataset}/frames"

    if skip_extract:
        local_frames = ds["frames_dir"]
        remote_frames = f"datasets/{dataset}/frames"
        if not skip_upload and local_frames.is_dir() and any(local_frames.glob("*.jpg")):
            print(f"Uploading {local_frames} -> volume:{remote_frames} ...")
            with volume.batch_upload(force=True) as batch:
                batch.put_directory(str(local_frames), remote_frames)
        elif skip_upload:
            print(f"Skipping frame upload; using volume:{remote_frames}")
        else:
            print(
                f"WARNING: No local frames in {local_frames}. "
                f"Ensure volume has datasets/{dataset}/frames/*.jpg"
            )
        run_pipeline.remote(
            dataset=dataset,
            skip_extract=True,
            frames_source=f"/data/{remote_frames}",
            max_frames=max_frames,
            resize_scale=resize_scale,
            output_path=remote_out,
            max_num_objects=max_num_objects,
            start_time_sec=start_time_sec,
            extract_fps=fps,
            source_fps=source_fps,
            seed_id=seed_id,
        )
    else:
        if not video_path:
            video_path = str(ds.get("video") or "")
        local_video = Path(video_path)
        if not local_video.is_file():
            raise FileNotFoundError(
                f"Local video not found: {local_video.resolve()}. "
                "Pass --video-path or use --skip-extract with SportsMOT frames."
            )

        remote_video = f"videos/{local_video.name}"
        remote_video_abs = f"/data/{remote_video}"

        if not skip_upload:
            print(f"Uploading {local_video} -> volume:{remote_video} ...")
            with volume.batch_upload(force=True) as batch:
                batch.put_file(str(local_video), remote_video)
        else:
            print(f"Skipping upload; using existing volume file {remote_video}")

        run_pipeline.remote(
            dataset=dataset,
            video_path=remote_video_abs,
            skip_extract=False,
            max_frames=max_frames,
            resize_scale=resize_scale,
            output_path=remote_out,
            max_num_objects=max_num_objects,
            start_time_sec=start_time_sec,
            extract_fps=fps,
            source_fps=source_fps,
            seed_id=seed_id,
        )

    vol_baseline = rel_baseline.replace("data/", "", 1)
    vol_frames = rel_frames.replace("data/", "", 1)
    print(
        "Pipeline finished. Download results:\n"
        f"  py -m modal volume get sports-data {vol_baseline} {rel_baseline}\n"
        f"  py -m modal volume get sports-data {vol_frames} {rel_frames} --force\n"
        "Then run setup_sportsmot_gt.py, ablations, and visualize."
    )
