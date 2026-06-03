"""
Run the SAM3.1 tracking pipeline on Modal GPU.

Usage:
    py -m modal run scripts/run_modal.py
    py -m modal run scripts/run_modal.py --video-path data/videos/video_1.mp4 --max-frames 100
"""

import modal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

app = modal.App("sports-trajectory")

volume = modal.Volume.from_name("sports-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
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
    max_frames: int = 100,
    output_path: str = "/data/outputs/baseline_tracks.json",
):
    import glob
    import os
    import sys

    import huggingface_hub

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

    from scripts.run_sam3 import extract_frames, run_tracking

    print(f"Extracting frames from {video_path} (max={max_frames})...")
    extract_frames(
        video_path,
        "/data/frames",
        fps=1,
        max_frames=max_frames,
    )

    print("Running SAM3.1 tracking...")
    run_tracking(
        frames_dir="/data/frames",
        output_path=output_path,
        use_fp16_weights=False,
        use_fa3=False,
    )

    volume.commit()
    print(f"Done. Results saved to {output_path}")


@app.local_entrypoint()
def main(
    video_path: str = "data/videos/video_1.mp4",
    max_frames: int = 100,
    output_path: str = "/data/outputs/baseline_tracks.json",
):
    local_video = Path(video_path)
    if not local_video.is_file():
        raise FileNotFoundError(
            f"Local video not found: {local_video.resolve()}. "
            "Pass --video-path to an existing .mp4 file."
        )

    remote_video = f"videos/{local_video.name}"
    remote_video_abs = f"/data/{remote_video}"

    print(f"Uploading {local_video} -> volume:{remote_video} ...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_video), remote_video)

    run_pipeline.remote(
        video_path=remote_video_abs,
        max_frames=max_frames,
        output_path=output_path,
    )

    print(
        "Pipeline finished. Download results with:\n"
        f"  py -m modal volume get sports-data {output_path.lstrip('/data/')} "
        f"data/outputs/{Path(output_path).name}"
    )
