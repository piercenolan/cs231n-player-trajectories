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
│   ├── run_sam3.py              # SAM3 tracking (video or SportsMOT frames)
│   ├── run_modal.py             # Modal GPU runner
│   ├── setup_sportsmot_gt.py    # Align gt.txt → gt.json
│   └── run_ablations.py         # Per-rule ablation sweep
├── utils/
│   ├── datasets.py              # Canonical paths (use --dataset)
│   ├── augmentation.py
│   ├── metrics.py
│   └── visualize.py
├── data/
│   ├── datasets/
│   │   └── sportsmot_example/   # PRIMARY: upload zip frames + gt.txt here
│   ├── runs/
│   │   └── sportsmot_example/   # SAM3 + augmentation outputs (generated)
│   ├── archive/                 # Legacy video_1 era artifacts
│   └── outputs/                 # Old run outputs (deprecated)
└── CONTEXT.md
```

See `data/datasets/sportsmot_example/README.md` for upload instructions.

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

**Primary dataset:** SportsMOT example zip → `data/datasets/sportsmot_example/`

| Zip contents | Repo path |
|--------------|-----------|
| `img1/*.jpg` | `data/datasets/sportsmot_example/frames/` |
| `gt/gt.txt` | `data/datasets/sportsmot_example/gt/gt.txt` |
| `seqinfo.ini` | `data/datasets/sportsmot_example/seqinfo.ini` |

Legacy `data/videos/video_1.mp4` has unknown provenance — do not use for ADE/FDE.

Prepared SAM frames for a run live at `data/runs/sportsmot_example/frames/` (`00000.jpg`, …).
Track `frame_number` is 1-indexed: frame 1 = first prepared JPEG.

## Alternate tracks.json format

If your tracks file uses `"frame"` instead of `"frame_number"` or lacks `mask_center`,
normalize it before running the pipeline:

```bash
python utils/convert_tracks.py --input tracks.json --output data/outputs/tracks.json
```

This sets `frame_number`, derives `mask_center` from each bbox center, and preserves `predicted` flags.

## Quick Start (SportsMOT example)

### 0) Upload data from the SportsMOT example zip

Copy `img1/*.jpg` and `gt/gt.txt` into `data/datasets/sportsmot_example/` (see README there).

### 1) Run SAM3.1 tracking (Modal or local)

Modal (uploads frames, runs on GPU):

```bash
py -m modal run scripts/run_modal.py --dataset sportsmot_example --max-frames 45 --resize-scale 0.67
```

Local (if you have a GPU):

```bash
python scripts/run_sam3.py --dataset sportsmot_example --skip-extract --max-frames 45 --resize-scale 0.67
```

### 2) Align ground truth

```bash
python scripts/setup_sportsmot_gt.py --dataset sportsmot_example --start-time-sec 0
```

Uses real `gt.txt` (not proxy). For consecutive SportsMOT frames, alignment uses 25 FPS by default.

### 3) Augmentation + ablations

```bash
python utils/augmentation.py --dataset sportsmot_example
python scripts/run_ablations.py --dataset sportsmot_example
python scripts/run_sanitize_grid.py --dataset sportsmot_example
python scripts/run_multi_seed.py --dataset sportsmot_example
python utils/trajectory_export.py --dataset sportsmot_example --validate
```

### 4) Metrics and visualization

```bash
python utils/metrics.py --tracks data/runs/sportsmot_example/baseline_tracks.json \
  --compare data/runs/sportsmot_example/augmented_tracks.json --exclude-predicted

python utils/trajectory_metrics.py --tracks data/runs/sportsmot_example/augmented_tracks.json

python utils/visualize.py \
  --frames data/runs/sportsmot_example/frames \
  --baseline data/runs/sportsmot_example/baseline_tracks.json \
  --augmented data/runs/sportsmot_example/augmented_tracks.json \
  --output data/runs/sportsmot_example/figures/summary_figure.png \
  --summary --n-frames 4
```

## Quick Start (custom video clip)

```bash
python scripts/run_sam3.py \
  --dataset sportsmot_example \
  --video_path data/clips/60.0_clip.mp4 \
  --max-frames 45
```

## Output Files

Typical generated artifacts under `data/runs/sportsmot_example/`:

- `baseline_tracks.json` – raw SAM3.1 tracks
- `augmented_tracks.json` – augmented tracks
- `ablations/` – per-rule metrics + `recommended_config.json`
- `frames/` – prepared JPEGs used for that SAM run
- `figures/summary_figure.png` – paper-ready summary panel

Legacy paths under `data/outputs/` are from the old `video_1` layout.

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