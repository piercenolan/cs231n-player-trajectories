# CS231N Player Trajectories (SAM3.1)

Research project for evaluating **SAM3.1** tracking on basketball broadcast video, then improving it with a basketball-domain, geometry-free augmentation layer.

## Project Goal

Phase 1 objective:
- run SAM3.1 on basketball clips,
- extract per-frame player tracks,
- measure baseline failures (ID switches, tracking loss),
- apply augmentation rules,
- compare baseline vs augmented outputs quantitatively and qualitatively.

## Current Architecture

End-to-end pipeline:

1. **Tracking (`scripts/run_sam3.py`)**
   - Extracts JPEG frames from input video (`extract_frames`, default 5 FPS sampling).
   - Loads SAM3.1 multiplex predictor.
   - Starts session and prompts frame 0 with text `"basketball players"`.
   - Propagates tracking across video.
   - Saves raw tracks JSON to `data/outputs/tracks.json`.

2. **Baseline / Evaluation (`utils/metrics.py`)**
   - Loads tracks JSON.
   - Computes ID stability and coverage metrics.
   - Generates report + plots (baseline or augmented labels supported).

3. **Augmentation (`utils/augmentation.py`)**
   - Stage 0: `sanitize_detections` (drop merged boxes, cap roster).
   - Stage 1: per-rule corrections (`--rules velocity_cap` for single-rule ablations).
   - Stage 2: gated `reid_gap_fill` (only when roster is short; `--no-gap-fill` to disable).
   - Saves augmented tracks + `corrections.json`.

4. **Visualization (`utils/visualize.py`)**
   - Draws tracks on frames.
   - Creates baseline vs augmented side-by-side comparisons.
   - Creates summary figure for paper/slides.

## Repository Layout

```text
cs231n-player-trajectories/
├── scripts/
│   └── run_sam3.py
├── utils/
│   ├── augmentation.py
│   ├── metrics.py
│   ├── video_utils.py
│   └── visualize.py
├── data/
│   ├── clips/                 # input .mp4 clips (not committed)
│   ├── frames_dir/            # extracted JPEG frames (generated)
│   └── outputs/               # tracks, metrics, visualizations (generated)
└── CONTEXT.md
```

## Requirements

- Python 3.11+
- CUDA-capable GPU (for SAM3.1 tracking)
- Hugging Face access to gated repo: `facebook/sam3.1`

Python packages used in current code:
- `sam3`
- `torch`
- `opencv-python` (or `opencv-python-headless` in headless env)
- `numpy`
- `matplotlib`

Install example:

```bash
pip install sam3 torch opencv-python numpy matplotlib
```

If `ImportError: No module named cv2`:

```bash
pip install opencv-python
```

## Data Expectations

- Input videos in `data/clips/` (example names in `CONTEXT.md`).
- Extracted frames are saved by `run_sam3.py` to `data/frames_dir/` as:
  - `0.jpg`, `1.jpg`, `2.jpg`, ...
- `tracks.json` frame numbering is 1-indexed:
  - `frame_number=1` corresponds to `0.jpg`

## Alternate tracks.json format

If your tracks file uses `"frame"` instead of `"frame_number"` or lacks `mask_center`,
normalize it before running the pipeline:

```bash
python utils/convert_tracks.py --input tracks.json --output data/outputs/tracks.json
```

This sets `frame_number`, derives `mask_center` from each bbox center, and preserves `predicted` flags.

## Quick Start (End-to-End)

### 1) Run SAM3.1 tracking

```bash
python scripts/run_sam3.py \
  --video_path data/clips/60.0_clip.mp4 \
  --output data/outputs/tracks.json
```

Optional local checkpoint:

```bash
python scripts/run_sam3.py \
  --video_path data/clips/60.0_clip.mp4 \
  --output data/outputs/tracks.json \
  --checkpoint /path/to/sam3.1_multiplex.pt
```

### 2) Compute baseline metrics

```bash
python utils/metrics.py \
  --tracks data/outputs/tracks.json \
  --output-figure data/outputs/baseline_metrics.png \
  --label "SAM3.1 BASELINE"
```

### 3) Run augmentation

```bash
python utils/augmentation.py \
  --tracks data/outputs/tracks.json \
  --output data/outputs/augmented_tracks.json \
  --frame-width 1280 \
  --frame-height 720 \
  --level full
```

Ablation modes:
- `--level physical` / `--level game` / `--level full`
- `--rules velocity_cap` — single rule (comma-separated for small combos)
- `--no-sanitize` / `--no-gap-fill` — stage ablations

Per-rule ablation sweep:

```bash
python scripts/setup_sportsmot_gt.py --tracks data/outputs/baseline_tracks.json --proxy-smooth
python scripts/run_ablations.py --baseline data/outputs/baseline_tracks.json
python scripts/run_sanitize_grid.py --baseline data/outputs/baseline_tracks.json
python scripts/aggregate_experiments.py --root data/outputs/ablations
```

Multi-seed validation (bootstrap from one baseline, or run SAM3 with `--seed-id`):

```bash
python scripts/run_multi_seed.py --bootstrap-from data/outputs/baseline_tracks.json
```

LSTM export + validation gate:

```bash
python utils/trajectory_export.py --tracks data/outputs/augmented_tracks.json --validate
```

Compare baseline vs augmented metrics (observed players only):

```bash
python utils/metrics.py --tracks data/outputs/baseline_tracks.json \
  --compare data/outputs/augmented_tracks.json --exclude-predicted
```

SportsMOT ADE/FDE on `video_1` (place GT at `data/gt/sportsmot/<sequence>/gt/gt.txt`):

```bash
python utils/trajectory_metrics.py --tracks data/outputs/augmented_tracks.json --sequence video_1
```

Multi-seed tracking (local):

```bash
python scripts/run_sam3.py --video_path data/clips/60.0_clip.mp4 \
  --seed-id offset_10s --start-time-sec 10 --max-frames 45
```

Export LSTM-ready tensors:

```bash
python utils/trajectory_export.py --tracks data/outputs/augmented_tracks.json
```

### 4) Compute augmented metrics

```bash
python utils/metrics.py \
  --tracks data/outputs/augmented_tracks.json \
  --output-figure data/outputs/augmented_metrics.png \
  --label "SAM3.1 + AUGMENTATION"
```

### 5) Visualize qualitative results

Annotate one tracks file:

```bash
python utils/visualize.py \
  --frames data/frames/frames_dir \
  --tracks data/outputs/tracks.json \
  --output data/outputs/annotated_baseline \
  --every-n 5
```

Baseline vs augmented side-by-side:

```bash
python utils/visualize.py \
  --frames data/frames/frames_dir \
  --baseline data/outputs/tracks.json \
  --augmented data/outputs/augmented_tracks.json \
  --output data/outputs/comparison \
  --compare \
  --every-n 5
```

Paper summary figure:

```bash
python utils/visualize.py \
  --frames data/frames/frames_dir \
  --baseline data/outputs/tracks.json \
  --augmented data/outputs/augmented_tracks.json \
  --output data/outputs/summary_figure.png \
  --summary \
  --n-frames 4
```

## Output Files

Typical generated artifacts:

- `data/outputs/tracks.json` – raw SAM3.1 tracks
- `data/outputs/augmented_tracks.json` – augmented tracks
- `data/outputs/corrections.json` – augmentation correction log
- `data/outputs/baseline_metrics.png` – baseline metric figure
- `data/outputs/augmented_metrics.png` – augmented metric figure
- `data/outputs/comparison/*.jpg` – side-by-side qualitative frames
- `data/outputs/summary_figure.png` – paper-ready summary panel

## Notes on Methodology

- Augmentation intentionally avoids hardcoded court geometry, basket positions, and camera calibration.
- Rules rely on relative player relationships and velocity patterns so they transfer across broadcast camera pans/zooms/cuts.
- `predicted: true` marks players re-added by augmentation re-identification logic.

## Common Issues

1. **`No module named cv2`**
   - Install OpenCV in the same Python environment used to run scripts.

2. **No annotated frames saved**
   - Check `--frames` directory path.
   - Confirm JPEG frames exist and are numerically named (`0.jpg`, `1.jpg`, ... or zero-padded numeric stems).

3. **HF checkpoint access errors**
   - Ensure access to `facebook/sam3.1` and that your HF auth token is active in the runtime environment.

## Citation / Internal Use

This repository is currently structured for CS231N milestone and paper workflows:
- reproducible baseline,
- augmentation ablations,
- metrics and qualitative comparisons for reporting.