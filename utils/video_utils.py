"""Utilities for extracting and loading video frames."""

from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path, output_dir, max_frames=500):
    """
    Extract frames from a video at 1 frame per second of video time.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where JPEG frames will be saved.
        max_frames: Stop after extracting this many frames (default 500).

    Returns:
        (num_extracted, video_fps): Count of saved frames and the source video FPS.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback when metadata is missing

    # Sample one frame per second: skip (fps - 1) frames between saves
    frame_interval = max(1, int(round(video_fps)))

    source_index = 0
    num_extracted = 0

    while num_extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if source_index % frame_interval == 0:
            num_extracted += 1
            out_path = output_dir / f"frame_{num_extracted:04d}.jpg"
            cv2.imwrite(str(out_path), frame)

            if num_extracted % 50 == 0:
                print(f"Extracted {num_extracted} frames...")

        source_index += 1

    cap.release()
    print(f"Finished: {num_extracted} frames saved to {output_dir}")
    return num_extracted, video_fps


def load_frames(frames_dir):
    """
    Load all JPEG frames from a directory in filename order.

    Args:
        frames_dir: Directory containing frame JPEGs (e.g. frame_0001.jpg).

    Returns:
        List of numpy arrays in BGR format (OpenCV default).
    """
    frames_dir = Path(frames_dir)
    paths = sorted(frames_dir.glob("*.jpg"))

    frames = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        frames.append(frame)

    return frames
