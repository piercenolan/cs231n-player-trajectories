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


def configure_cuda_memory():
    """Reduce fragmentation-related OOMs (see PyTorch CUDA memory notes)."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def free_cuda_memory(label=""):
    """Release cached GPU allocations before loading the model."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        prefix = f"[INFO] {label} " if label else "[INFO] "
        print(
            f"{prefix}GPU memory: "
            f"{free_bytes / 1e9:.2f} GB free / {total_bytes / 1e9:.2f} GB total"
        )


configure_cuda_memory()

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
def load_sam3_predictor(
    checkpoint_path=None,
    use_fp16_weights=True,
    use_fa3=True,
    max_num_objects=16,
    async_loading_frames=True,
):
    patch_sam3_start_session()

    if checkpoint_path is None:
        print("Downloading SAM3 checkpoint...")
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SAM3")

    free_cuda_memory("Before model load")

    print(f"Loading checkpoint: {checkpoint_path}")
    if not use_fa3:
        print("Flash Attention 3 disabled (use_fa3=False); using PyTorch SDPA fallback.")
    print(
        f"Model memory settings: max_num_objects={max_num_objects}, "
        f"async_loading_frames={async_loading_frames}, "
        f"use_fp16_weights={use_fp16_weights}"
    )

    predictor_kwargs = {
        "checkpoint_path": checkpoint_path,
        "use_fa3": use_fa3,
        "use_rope_real": use_fa3,
        "max_num_objects": max_num_objects,
        "async_loading_frames": async_loading_frames,
    }

    try:
        predictor = build_sam3_multiplex_video_predictor(
            gpus_to_use=[torch.cuda.current_device()],
            **predictor_kwargs,
        )
    except TypeError:
        predictor = build_sam3_multiplex_video_predictor(**predictor_kwargs)

    # Sam3MultiplexVideoPredictor runs with bf16 autocast internally. Converting
    # weights to half() causes Float/Half mismatches in the decoder (e.g. mat1
    # Float vs mat2 Half). Keep fp32 weights; memory is managed via frame count,
    # resolution, offload_video_to_cpu, and max_num_objects instead.
    if use_fp16_weights:
        print(
            "Note: model.half() skipped for multiplex predictor "
            "(incompatible with internal bf16 path)."
        )
    predictor.model = predictor.model.cuda()
    predictor.model.eval()
    free_cuda_memory("After model load")

    return predictor


# -----------------------------
# Frame extraction
# -----------------------------
def extract_frames(
    video_path,
    output_dir,
    fps=1,
    max_frames=150,
    resize_scale=1.0,
    start_time_sec=0.0,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = fps  # fallback safety

    if start_time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000.0)
        print(f"[INFO] Skipped to start_time_sec={start_time_sec}")

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


def prepare_frames_from_dir(
    source_dir,
    output_dir,
    max_frames=45,
    resize_scale=1.0,
    start_time_sec=0.0,
    source_fps=25.0,
    extract_fps=25.0,
):
    """
    Copy subsampled JPEGs from an existing frame folder (e.g. SportsMOT img1).

    Output files are 00000.jpg, 00001.jpg, ... for SAM3. Use extract_fps=25 when
    taking consecutive SportsMOT frames so GT frame N aligns with track frame N.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_frame in output_dir.glob("*.jpg"):
        old_frame.unlink()

    all_frames = sorted(source_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not all_frames:
        raise FileNotFoundError(f"No JPEG frames in {source_dir}")

    start_mot_frame = int(round(start_time_sec * source_fps)) + 1
    interval = max(int(round(source_fps / extract_fps)), 1)

    saved = 0
    frame_paths = []
    for frame_path in all_frames:
        mot_frame = int(frame_path.stem)
        if mot_frame < start_mot_frame:
            continue
        rel = mot_frame - start_mot_frame
        if rel % interval != 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        if resize_scale != 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_AREA,
            )

        path = output_dir / f"{saved:05d}.jpg"
        cv2.imwrite(str(path), frame)
        frame_paths.append(str(path))
        saved += 1
        if saved >= max_frames:
            print(f"[INFO] Reached max_frames={max_frames}, stopping frame prep.")
            break

    print(
        f"[INFO] Prepared {saved} frames from {source_dir} "
        f"(start_mot={start_mot_frame}, interval={interval}, scale={resize_scale})"
    )
    return frame_paths


# -----------------------------
# Session
# -----------------------------
def start_video_session(predictor, frames_dir, offload_video_to_cpu=False):
    resp = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(frames_dir),
        "offload_video_to_cpu": offload_video_to_cpu,
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
        "text": "basketball players on the court",
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
    masks = outputs.get("out_binary_masks")
    if masks is None:
        masks = []

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


def filter_defaults_for_resolution(frame_width, frame_height):
    """
    Resolution-aware filter defaults.

    Smaller frames (Modal downscale) need a lower min_area_fraction so
    on-court players are not removed as noise.
    """
    frame_area = frame_width * frame_height
    if frame_area <= 280_000:
        min_area_fraction = 0.00030
        top_y_fraction = 0.11
    elif frame_area <= 450_000:
        min_area_fraction = 0.00040
        top_y_fraction = 0.12
    else:
        min_area_fraction = 0.00055
        top_y_fraction = 0.13
    return {
        "top_y_fraction": top_y_fraction,
        "min_area_fraction": min_area_fraction,
        "max_area_fraction": 0.018,
        "max_objects": 14,
    }


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
def run_tracking(
    frames_dir,
    output_path,
    checkpoint=None,
    use_fp16_weights=True,
    use_fa3=True,
    offload_video_to_cpu=False,
    max_num_objects=16,
    async_loading_frames=True,
    resize_scale=1.0,
):
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
    print(f"Frame dimensions: {w}x{h} ({len(frames)} frames)")

    free_cuda_memory("Before tracking")

    predictor = load_sam3_predictor(
        checkpoint,
        use_fp16_weights=use_fp16_weights,
        use_fa3=use_fa3,
        max_num_objects=max_num_objects,
        async_loading_frames=async_loading_frames,
    )

    session_id = None

    try:
        session_id = start_video_session(
            predictor,
            frames_dir,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        add_prompt(predictor, session_id)

        filter_kwargs = filter_defaults_for_resolution(w, h)
        print(f"Filter settings for {w}x{h}: {filter_kwargs}")

        results = {
            "meta": {
                "frame_width": w,
                "frame_height": h,
                "num_source_frames": len(frames),
                "resize_scale": resize_scale,
            },
            "frames": [],
        }

        # Match Sam3MultiplexVideoPredictor's internal bf16 inference path.
        autocast_dtype = torch.bfloat16
        print(f"Using autocast dtype: {autocast_dtype}")

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            for frame_idx, outputs in propagate_tracking(predictor, session_id):

                frame_num = frame_idx + 1
                players = parse_outputs(outputs, w, h)
                raw_count = len(players)
                players = filter_detections(
                    players,
                    frame_width=w,
                    frame_height=h,
                    **filter_kwargs,
                )
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
    parser.add_argument(
        "--dataset",
        default="sportsmot_example",
        help="Dataset key from utils.datasets (default: sportsmot_example)",
    )
    parser.add_argument(
        "--video_path",
        default=None,
        help="Input video (required unless --skip-extract)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Use existing dataset frames instead of extracting from video",
    )
    parser.add_argument(
        "--frames-source",
        default=None,
        help="Override source frame directory when --skip-extract",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Native video FPS for frame prep (default: dataset source_fps)",
    )
    parser.add_argument("--output", default="tracks.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Extract/sample FPS (default: 25 for SportsMOT frames, 1 for video)",
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
    parser.add_argument(
        "--start-time-sec",
        type=float,
        default=0.0,
        help="Start extraction at this timestamp in the video (multi-seed offsets)",
    )
    parser.add_argument(
        "--seed-id",
        default=None,
        help="Seed label for outputs (e.g. offset_10s). Writes under data/runs/{dataset}/seeds/{seed_id}/",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Override prepared frames output directory",
    )
    args = parser.parse_args()

    from utils.datasets import baseline_tracks_path, get_dataset, runs_dir

    ds = get_dataset(args.dataset)
    source_fps = args.source_fps if args.source_fps is not None else float(ds["source_fps"])
    if args.fps is not None:
        extract_fps = args.fps
    elif args.skip_extract:
        extract_fps = float(ds["extract_fps"])
    else:
        extract_fps = 1.0

    if args.seed_id:
        seed_root = runs_dir(args.dataset, args.seed_id)
        frames_dir = args.frames_dir or str(seed_root / "frames")
        output_path = args.output if args.output != "tracks.json" else str(
            baseline_tracks_path(args.dataset, args.seed_id)
        )
    else:
        run_root = runs_dir(args.dataset)
        frames_dir = args.frames_dir or str(run_root / "frames")
        output_path = args.output if args.output != "tracks.json" else str(
            baseline_tracks_path(args.dataset)
        )

    if args.skip_extract:
        source = Path(args.frames_source or ds["frames_dir"])
        if not source.is_dir() or not any(source.glob("*.jpg")):
            raise FileNotFoundError(
                f"No frames in {source}. Upload SportsMOT img1/*.jpg to "
                f"{ds['frames_dir']} (see data/datasets/sportsmot_example/README.md)."
            )
        prepare_frames_from_dir(
            source,
            frames_dir,
            max_frames=args.max_frames,
            resize_scale=args.resize_scale,
            start_time_sec=args.start_time_sec,
            source_fps=source_fps,
            extract_fps=extract_fps,
        )
    else:
        if not args.video_path:
            video = ds.get("video")
            if video and Path(video).is_file():
                args.video_path = str(video)
            else:
                raise SystemExit(
                    "Provide --video_path or use --skip-extract with SportsMOT frames in "
                    f"{ds['frames_dir']}"
                )

        frames_path = Path(frames_dir)
        if frames_path.exists():
            for old_frame in frames_path.glob("*.jpg"):
                old_frame.unlink()
            print(f"[INFO] Cleared {frames_dir} before extraction.")

        extract_frames(
            args.video_path,
            frames_dir,
            fps=extract_fps,
            max_frames=args.max_frames,
            resize_scale=args.resize_scale,
            start_time_sec=args.start_time_sec,
        )

    run_tracking(
        frames_dir=frames_dir,
        output_path=output_path,
        checkpoint=args.checkpoint,
        resize_scale=args.resize_scale,
    )


if __name__ == "__main__":
    main()
