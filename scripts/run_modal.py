"""
Run the SAM3.1 tracking pipeline on Modal GPU.

Tuned for A10G (24 GB): ~67% resolution and 45 frames balances detection
quality against VRAM. Uses CPU video offload and bf16 multiplex inference.

Usage:
    py -m modal run scripts/run_modal.py
    py -m modal run scripts/run_modal.py --video-path data/videos/video_1.mp4
    py -m modal run scripts/run_modal.py --max-frames 45 --resize-scale 0.67
"""

import modal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
        ignore=lambda p: (
            ".git" in p.parts
            or "__pycache__" in p.parts
            or "data" in p.parts
        ),
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
    video_path: str = "/data/videos/video_1.mp4",
    max_frames: int = 45,
    resize_scale: float = 0.67,
    output_path: str = "/data/outputs/baseline_tracks.json",
    max_num_objects: int = 14,
    start_time_sec: float = 0.0,
):
    import gc
    import glob
    import json
    import os
    import sys

    import huggingface_hub
    import torch

    volume.reload()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video not found on Modal volume: {video_path}. "
            "Re-run with --video-path pointing to a local file so it is uploaded."
        )

    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    sys.path.insert(0, "/project")
    sys.path.insert(0, "/sam3")

    os.makedirs("/data/frames", exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for f in glob.glob("/data/frames/*.jpg"):
        os.remove(f)

    from scripts.run_sam3 import extract_frames, free_cuda_memory, run_tracking

    print(
        f"Memory settings: max_frames={max_frames}, resize_scale={resize_scale}, "
        f"max_num_objects={max_num_objects}, offload_video_to_cpu=True"
    )

    print(
        f"Extracting frames from {video_path} (max={max_frames}, start={start_time_sec}s)..."
    )
    extract_frames(
        video_path,
        "/data/frames",
        fps=1,
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
        frames_dir="/data/frames",
        output_path=output_path,
        use_fp16_weights=False,
        use_fa3=False,
        offload_video_to_cpu=True,
        max_num_objects=max_num_objects,
        async_loading_frames=True,
        resize_scale=resize_scale,
    )

    with open(output_path, encoding="utf-8") as f:
        meta = json.load(f).get("meta", {})
    print(f"Track metadata: {meta}")

    volume.commit()
    print(f"Done. Results saved to {output_path}")


@app.local_entrypoint()
def main(
    video_path: str = "data/videos/video_1.mp4",
    max_frames: int = 45,
    resize_scale: float = 0.67,
    output_path: str = "/data/outputs/baseline_tracks.json",
    max_num_objects: int = 14,
    skip_upload: bool = False,
    start_time_sec: float = 0.0,
    seed_id: str = "",
):
    local_video = Path(video_path)
    if not local_video.is_file():
        raise FileNotFoundError(
            f"Local video not found: {local_video.resolve()}. "
            "Pass --video-path to an existing .mp4 file."
        )

    remote_video = f"videos/{local_video.name}"
    remote_video_abs = f"/data/{remote_video}"

    if not skip_upload:
        print(f"Uploading {local_video} -> volume:{remote_video} ...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(local_video), remote_video)
    else:
        print(f"Skipping upload; using existing volume file {remote_video}")

    remote_output = output_path
    if seed_id:
        remote_output = f"/data/outputs/seeds/{seed_id}/baseline_tracks.json"

    run_pipeline.remote(
        video_path=remote_video_abs,
        max_frames=max_frames,
        resize_scale=resize_scale,
        output_path=remote_output,
        max_num_objects=max_num_objects,
        start_time_sec=start_time_sec,
    )

    print(
        "Pipeline finished. Download results:\n"
        f"  py -m modal volume get sports-data outputs/baseline_tracks.json data/outputs/baseline_tracks.json\n"
        "Download matching frames for visualization (same run resolution):\n"
        "  py -m modal volume get sports-data frames data/frames --force\n"
        "Then run metrics / augmentation / visualize using data/frames and the tracks meta dimensions."
    )
