"""
Run SAM3 video tracking on extracted basketball frames.

Usage:
    python scripts/run_sam3.py --frames_dir data/frames/clip1 --output data/outputs/tracks.json

Requires: pip install sam3  (CONTEXT.md: segment-anything-3)
Hugging Face access to facebook/sam3.1 for the SAM 3.1 multiplex checkpoint.
"""

import argparse
import json
import sys
import cv2
import numpy as np
import torch
import traceback

from pathlib import Path

# Project root on path when invoked as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    # SAM 3.1 Multiplex: recommended builder for multi-object video tracking
    from sam3.model_builder import build_sam3_multiplex_video_predictor, download_ckpt_from_hf
except ImportError as exc:
    raise ImportError(
        "SAM3 not found. Install with: pip install sam3"
    ) from exc


def load_sam3_predictor(checkpoint_path=None):
    """Load SAM 3.1 Multiplex video predictor (facebook/sam3.1 checkpoint)."""
    if checkpoint_path is None:
        print("Downloading SAM 3.1 multiplex checkpoint from Hugging Face...")
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")

    # Remove comments after GPU has been acquired
    if torch.cuda.is_available():
        gpus_to_use = [torch.cuda.current_device()]
    else:
        raise RuntimeError("SAM3 video tracking requires a CUDA GPU.")

    print(f"Loading SAM 3.1 multiplex video predictor (checkpoint: {checkpoint_path})...")
    try:
        return build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=gpus_to_use,
        )
    except TypeError:
        print("gpus_to_use not supported, loading without it...")
        return build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
        )
    


def extract_frames(video_path, output_dir, fps=5):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps else 1  # None = every frame

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            path = f"{output_dir}/{saved_count}.jpg"
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved_count += 1
        frame_count += 1

    cap.release()
    return frame_paths


def start_video_session(predictor, frames_dir):
    """Open a frame directory as a SAM3 video session."""
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(frames_dir),
        }
    )
    return response["session_id"]


def prompt_players_on_first_frame(predictor, session_id, frame_paths):
    """
    Prompt SAM 3.1 Multiplex on frame 0 with a text concept to detect basketball players.

    Uses PCS (Promptable Concept Segmentation): a single text prompt asks the
    multiplex detector to find all instances of "basketball players" and assign
    each a unique object ID before temporal propagation.
    """
    prompt_text = "basketball players"
    print(f'Adding text prompt "{prompt_text}" on frame 0...')

    try:
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt_text,
            }
        )
    except Exception as exc:
        print(f"  Warning: text prompt failed: {exc}")


def propagate_tracking(predictor, session_id):
    """Propagate prompts across all frames; yields (frame_index, outputs)."""
    for response in predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
        }
    ):
        yield response["frame_index"], response["outputs"]


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mask_center_xy(binary_mask):
    """Centroid of a binary mask in pixel coordinates."""
    mask = _to_numpy(binary_mask).astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def box_xywh_pixels(box_xywh, width, height):
    """Convert a relative or absolute xywh box to pixel x, y, w, h."""
    x, y, w, h = [float(v) for v in _to_numpy(box_xywh).reshape(-1).tolist()]
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.0:
        x, y, w, h = x * width, y * height, w * width, h * height
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def parse_frame_outputs(outputs, width, height):
    """Extract per-player tracks from one frame of SAM3 outputs."""
    obj_ids = _to_numpy(outputs["out_obj_ids"]).astype(int).tolist()
    boxes = outputs["out_boxes_xywh"]
    masks = outputs["out_binary_masks"]

    players = []
    for i, obj_id in enumerate(obj_ids):
        try:
            x, y, w, h = box_xywh_pixels(boxes[i], width, height)
            center = mask_center_xy(masks[i])
            if center is None:
                center = (x + w / 2.0, y + h / 2.0)

            players.append(
                {
                    "id": int(obj_id),
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "mask_center": {"x": round(center[0], 1), "y": round(center[1], 1)},
                }
            )
        except Exception as exc:
            print(f"  Warning: could not parse object {obj_id}: {exc}")

    return players


def run_tracking(frames_dir, output_path, checkpoint_path=None):
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPEG frames found in {frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]

    predictor = load_sam3_predictor(checkpoint_path=checkpoint_path)
    session_id = None

    try:
        session_id = start_video_session(predictor, frames_dir)
        prompt_players_on_first_frame(predictor, session_id, frame_paths)

        results = {"frames": []}
        for frame_index, outputs in propagate_tracking(predictor, session_id):
            frame_number = frame_index + 1  # match frame_0001.jpg naming

            try:
                players = parse_frame_outputs(outputs, width, height)
            except Exception as exc:
                print(f"Frame {frame_number}: parse failed ({exc}), saving empty tracks")
                players = []

            results["frames"].append(
                {"frame_number": frame_number, "players": players}
            )

            if frame_number % 10 == 0:
                print(f"Frame {frame_number}: tracking {len(players)} players")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Saved tracks for {len(results['frames'])} frames to {output_path}")
    finally:
        try:
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            print("Session closed.")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Track basketball players with SAM3 on extracted frames."
    )
    parser.add_argument(
        "--video_path",
        required=False,
        help="Path of video from where the frames are to be extracted",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/tracks.json",
        help="Path to write tracking JSON (default: data/outputs/tracks.json)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local path to SAM3-large checkpoint (sam3.pt)",
    )
    args = parser.parse_args()

    output_dir = "data/frames_dir/"

    extract_frames(args.video_path, output_dir)
    run_tracking(output_dir, args.output, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
