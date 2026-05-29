"""
Run SAM3 video tracking on extracted basketball frames.
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path
import torch

# stability flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3.model_builder import (
    build_sam3_video_predictor,
    download_ckpt_from_hf,
)


# -----------------------------
# dtype helper (IMPORTANT FIX)
# -----------------------------
def get_autocast_dtype():
    """Use bf16 if supported, else fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# -----------------------------
# Model loading
# -----------------------------
def load_sam3_predictor(checkpoint_path=None):
    if checkpoint_path is None:
        print("Downloading SAM3 checkpoint...")
        checkpoint_path = download_ckpt_from_hf()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SAM3")

    print(f"Loading checkpoint: {checkpoint_path}")

    predictor = build_sam3_video_predictor(
        checkpoint_path=checkpoint_path,
    )

    # IMPORTANT: keep FP32 weights, move to GPU only
    predictor.model = predictor.model.cuda()
    predictor.model.eval()

    return predictor


# -----------------------------
# Frame extraction
# -----------------------------
def extract_frames(video_path, output_dir, fps=1, max_frames=150, resize_scale=1.0):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = fps  # fallback safety

    frame_interval = max(int(video_fps / fps), 1)

    frame_paths = []
    i = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS downsampling control
        if i % frame_interval == 0:

            # Optional resize to reduce downstream memory
            if resize_scale != 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_scale,
                    fy=resize_scale,
                    interpolation=cv2.INTER_AREA,
                )

            path = f"{output_dir}/{saved}.jpg"
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1

            # HARD STOP: prevents SAM3 overload
            if saved >= max_frames:
                print(f"[INFO] Reached max_frames={max_frames}, stopping extraction.")
                break

        i += 1

    cap.release()
    return frame_paths


# -----------------------------
# Session
# -----------------------------
def start_video_session(predictor, frames_dir):
    resp = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(frames_dir),
    })

    if not resp or "session_id" not in resp:
        raise RuntimeError(f"Failed to start session: {resp}")

    return resp["session_id"]


def add_prompt(predictor, session_id):
    print("Adding prompt: basketball players")

    resp = predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": "basketball players",
    })

    if not resp:
        raise RuntimeError("Empty prompt response")

    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(resp["error"])


# -----------------------------
# Tracking
# -----------------------------
def propagate_tracking(predictor, session_id):
    for r in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):
        yield r["frame_index"], r["outputs"]


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def box_xywh(box, w, h):
    x, y, bw, bh = to_numpy(box).reshape(-1).tolist()

    if max(abs(x), abs(y), abs(bw), abs(bh)) <= 1.0:
        x, y, bw, bh = x * w, y * h, bw * w, bh * h

    return int(x), int(y), int(bw), int(bh)


def parse_outputs(outputs, w, h):
    obj_ids = to_numpy(outputs["out_obj_ids"]).astype(int).tolist()
    boxes = outputs["out_boxes_xywh"]

    players = []

    for i, oid in enumerate(obj_ids):
        try:
            x, y, bw, bh = box_xywh(boxes[i], w, h)

            players.append({
                "id": int(oid),
                "bbox": {"x": x, "y": y, "w": bw, "h": bh},
            })
        except Exception as e:
            print("parse warning:", e)

    return players


# -----------------------------
# Main pipeline
# -----------------------------
def run_tracking(frames_dir, output_path, checkpoint=None):
    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob("*.jpg"))

    if not frames:
        raise FileNotFoundError("No frames found")

    img = cv2.imread(str(frames[0]))
    h, w = img.shape[:2]

    predictor = load_sam3_predictor(checkpoint)

    session_id = None

    try:
        session_id = start_video_session(predictor, frames_dir)
        add_prompt(predictor, session_id)

        results = {"frames": []}

        amp_dtype = get_autocast_dtype()
        print(f"Using autocast dtype: {amp_dtype}")

        # CRITICAL FIX: only use autocast, NOT model half()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            for frame_idx, outputs in propagate_tracking(predictor, session_id):

                frame_num = frame_idx + 1
                players = parse_outputs(outputs, w, h)

                results["frames"].append({
                    "frame": frame_num,
                    "players": players
                })

                if frame_num % 10 == 0:
                    print(f"Frame {frame_num}: {len(players)} players")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print("Saved:", output_path)

    finally:
        if session_id:
            try:
                predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id,
                })
            except Exception:
                pass


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output", default="tracks.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    frames_dir = "data/frames"

    extract_frames(args.video_path, frames_dir)

    run_tracking(
        frames_dir=frames_dir,
        output_path=args.output,
        checkpoint=args.checkpoint
    )


if __name__ == "__main__":
    main()