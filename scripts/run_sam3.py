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
    build_sam3_multiplex_video_predictor,
    download_ckpt_from_hf,
)


def patch_sam3_start_session():
    """
    Filter init_state kwargs for SAM3.1 multiplex predictors.

    Upstream sam3#544: start_session() forwards offload_state_to_cpu to
    init_state(), but Sam3MultiplexTrackingWithInteractivity does not
    accept that argument. add_prompt/propagate_in_video already filter
    kwargs; this applies the same pattern to start_session.
    """
    import inspect
    import time
    import uuid

    from sam3.model.sam3_base_predictor import Sam3BasePredictor

    if getattr(Sam3BasePredictor, "_start_session_patched", False):
        return

    def start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        init_kwargs = dict(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type

        sig = inspect.signature(self.model.init_state)
        valid_params = set(sig.parameters.keys())
        init_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_params}

        inference_state = self.model.init_state(**init_kwargs)

        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    Sam3BasePredictor.start_session = start_session
    Sam3BasePredictor._start_session_patched = True


# -----------------------------
# Model loading
# -----------------------------
def load_sam3_predictor(checkpoint_path=None, use_fp16_weights=True):
    patch_sam3_start_session()

    if checkpoint_path is None:
        print("Downloading SAM3 checkpoint...")
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SAM3")

    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=[torch.cuda.current_device()],
        )
    except TypeError:
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
        )

    # IMPORTANT: half weights + fp16 autocast must match for stable memory attention
    if use_fp16_weights:
        predictor.model = predictor.model.cuda().half()
    else:
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

            path = f"{output_dir}/{saved:05d}.jpg"
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
        "text": "basketball player",
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

    if max(abs(bw), abs(bh)) < 2.0:
        x, y, bw, bh = x * w, y * h, bw * w, bh * h

    return int(x), int(y), int(bw), int(bh)


def parse_outputs(outputs, w, h):
    obj_ids = to_numpy(outputs["out_obj_ids"]).astype(int).tolist()
    boxes = outputs["out_boxes_xywh"]
    masks = outputs.get("out_binary_masks") or []

    players = []

    for i, oid in enumerate(obj_ids):
        try:
            x, y, bw, bh = box_xywh(boxes[i], w, h)
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            mask_center = {"x": round(cx, 1), "y": round(cy, 1)}
            mask_area = int(bw * bh)

            if i < len(masks):
                try:
                    mask = to_numpy(masks[i])
                    while mask.ndim > 2:
                        mask = mask[0]
                    if mask.ndim == 2:
                        binary = mask > 0
                        area = int(np.sum(binary))
                        if area > 0:
                            ys, xs = np.where(binary)
                            mask_center = {
                                "x": round(float(xs.mean()), 1),
                                "y": round(float(ys.mean()), 1),
                            }
                            mask_area = area
                except Exception as mask_exc:
                    print(f"  mask parse warning id={oid}: {mask_exc}")

            players.append({
                "id": int(oid),
                "bbox": {"x": x, "y": y, "w": bw, "h": bh},
                "mask_center": mask_center,
                "mask_area": mask_area,
            })
        except Exception as e:
            print("parse warning:", e)

    return players


def filter_detections(
    players,
    frame_width,
    frame_height,
    top_y_fraction=0.14,
    min_area_fraction=0.0008,
    max_area_fraction=0.015,
    max_objects=13,
):
    """
    Filter SAM3.1 detections using mask geometry only — no court calibration.

    All thresholds are relative to frame dimensions so the filter works
    correctly across different broadcast resolutions (360p, 720p, 1080p)
    and camera zoom levels without any modification.

    Filters applied in sequence:
    1. Top-band removal: drops detections whose mask center y is in the
       upper top_y_fraction of the frame. Crowd, scoreboards, and
       upper-bowl fans dominate this region in broadcast views.

    2. Minimum area: removes tiny detections (distant fans, noise).
       Threshold is min_area_fraction * frame_area pixels — scales
       automatically with resolution. At 720p (921600px) this is ~737px.
       At 360p (230400px) this is ~184px.

    3. Maximum area: removes oversized blobs — merged detections,
       court graphics, and scoreboards. Threshold is max_area_fraction
       * frame_area pixels. At 720p this is ~13824px. At 360p ~3456px.

    4. Top-N cap: after size and position filtering, keep at most
       max_objects detections sorted by mask_area descending. On-court
       players tend to have larger visible mask area than partially
       visible crowd members that survive the earlier filters.
    """
    if not players:
        return []

    frame_area = frame_width * frame_height
    top_y_limit = top_y_fraction * frame_height
    min_mask_area = int(min_area_fraction * frame_area)
    max_mask_area = int(max_area_fraction * frame_area)

    kept = []
    for p in players:
        cy = float(p["mask_center"]["y"])
        area = int(p["mask_area"])

        if cy < top_y_limit:
            continue
        if area < min_mask_area:
            continue
        if area > max_mask_area:
            continue
        kept.append(p)

    kept.sort(key=lambda p: p["mask_area"], reverse=True)
    return kept[:max_objects]


# -----------------------------
# Main pipeline
# -----------------------------
def run_tracking(frames_dir, output_path, checkpoint=None, use_fp16_weights=True):
    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))

    if not frames:
        raise FileNotFoundError("No frames found")

    img = cv2.imread(str(frames[0]))
    if img is None:
        raise RuntimeError(
            f"Could not read first frame: {frames[0]}. "
            "Check that the file exists and is a valid JPEG."
        )
    h, w = img.shape[:2]

    predictor = load_sam3_predictor(checkpoint, use_fp16_weights=use_fp16_weights)

    session_id = None

    try:
        session_id = start_video_session(predictor, frames_dir)
        add_prompt(predictor, session_id)

        results = {"frames": []}

        autocast_dtype = torch.float16 if use_fp16_weights else torch.bfloat16
        print(f"Using autocast dtype: {autocast_dtype}")

        # CRITICAL FIX: fp16 weights and fp16 autocast must match
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            for frame_idx, outputs in propagate_tracking(predictor, session_id):

                frame_num = frame_idx + 1
                players = parse_outputs(outputs, w, h)
                raw_count = len(players)
                players = filter_detections(players, frame_width=w, frame_height=h)
                print(f"Frame {frame_num}: {raw_count} raw -> {len(players)} filtered detections")

                results["frames"].append({
                    "frame_number": frame_num,
                    "players": players
                })

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
    parser.add_argument(
        "--fps",
        type=float,
        default=1,
        help="Frames per second to extract from video (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum number of frames to extract (default: 150)",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=1.0,
        help="Optional downscale factor when saving frames (default: 1.0)",
    )
    args = parser.parse_args()

    frames_dir = "data/frames"

    # Clear frames directory before extraction so old frames from
    # previous runs do not mix with new ones. SAM3 loads all JPEGs
    # in the directory — stale frames from a prior run with different
    # max_frames would be included silently.
    frames_path = Path(frames_dir)
    if frames_path.exists():
        for old_frame in frames_path.glob("*.jpg"):
            old_frame.unlink()
        print(f"[INFO] Cleared {frames_dir} before extraction.")

    extract_frames(
        args.video_path,
        frames_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        resize_scale=args.resize_scale,
    )

    run_tracking(
        frames_dir=frames_dir,
        output_path=args.output,
        checkpoint=args.checkpoint
    )


if __name__ == "__main__":
    main()
