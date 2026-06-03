import modal
from pathlib import Path

app = modal.App("sports-trajectory")

volume = modal.Volume.from_name("sports-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "libglib2.0-dev")
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128"
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
)
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /sam3",
        "cd /sam3 && pip install -e .",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_pipeline(max_frames: int = 100):
    import sys
    import os
    import subprocess

    # Auth HF
    import huggingface_hub
    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    # Clone latest code from GitHub
    subprocess.run([
        "git", "clone",
        "https://github.com/piercenolan/cs231n-player-trajectories.git",
        "/project"
    ], check=True)

    sys.path.insert(0, "/project")
    sys.path.insert(0, "/sam3")

    os.makedirs("/data/frames", exist_ok=True)
    os.makedirs("/data/outputs", exist_ok=True)

    # Clear old frames
    import glob
    for f in glob.glob("/data/frames/*.jpg"):
        os.remove(f)

    # Run extraction and tracking
    from scripts.run_sam3 import extract_frames, run_tracking

    print(f"Extracting frames (max={max_frames})...")
    extract_frames(
        "/data/videos/video_1.mp4",
        "/data/frames",
        fps=1,
        max_frames=max_frames
    )

    print("Running SAM3.1 tracking...")
    run_tracking(
        frames_dir="/data/frames",
        output_path="/data/outputs/baseline_tracks.json",
    )

    volume.commit()
    print("Done. Results saved to /data/outputs/baseline_tracks.json")

@app.local_entrypoint()
def main():
    run_pipeline.remote(max_frames=100)