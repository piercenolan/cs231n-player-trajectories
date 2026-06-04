# CS231N Project Context

## Project

Evaluate **SAM3.1** on basketball broadcast video, improve tracks with a **geometry-free augmentation** layer, measure against **SportsMOT** ground truth, then train an **LSTM** on exported trajectory tensors.

## Stack

- Python 3.12 (local) / 3.12 (Modal image)
- PyTorch, SAM3.1 (`facebook/sam3.1`, Hugging Face gated)
- OpenCV (frame I/O)
- Modal A10G for SAM3 tracking (`scripts/run_modal.py`)
- NumPy, Matplotlib (metrics / viz)

## Data (current)

| Role | Location |
|------|----------|
| **Primary eval set** | `data/datasets/sportsmot_example/` — 500 JPEGs + `gt/gt.txt` |
| **Run artifacts** | `data/runs/sportsmot_example/` — tracks, ablations, tensors |
| **Legacy (do not cite)** | `data/outputs/`, `data/videos/video_1.mp4`, proxy GT under `data/gt/sportsmot/video_1/` |
| **NBA clips (optional)** | `data/clips/` — 60.0, 690.0, 2700.0 clips at 360p |

## Repository structure

```text
cs231n-player-trajectories/
├── docs/                    # MILESTONE_CHECKLIST.md, PROJECT_PLAN.md
├── scripts/                 # run_sam3, run_modal, ablations, multi_seed, setup GT
├── utils/                   # augmentation, metrics, datasets, export, visualize
├── data/datasets/           # static inputs (SportsMOT example)
├── data/runs/               # generated per-dataset outputs
└── data/archive/            # legacy layout notes
```

Path helper: `utils/datasets.py` — always pass `--dataset sportsmot_example` unless using legacy paths.

## Phase goals

| Phase | Goal | Status |
|-------|------|--------|
| 1 | SAM3.1 baseline tracks | Done (45-frame window) |
| 2 | Augmentation + ablations + ADE | Done |
| 3 | Pre-LSTM validation (GT, export gate) | Done; multi-seed pending |
| 4 | LSTM forecaster | Not started |

See `docs/MILESTONE_CHECKLIST.md` for report-ready checklist and `docs/PROJECT_PLAN.md` for architecture and commands.

## Current results snapshot (SportsMOT 45f)

- Baseline: ~10.8 players/frame, 0 ID switches, ADE ~7.16 px vs aligned GT
- `velocity_cap` inactive on this clip; sanitize drives most changes
- LSTM export validation: **passed** (global visibility 0.94)
- Recommended for write-up: minimal aug (`sanitize_plus_velocity_cap` or grid winner `w0.4_y0.1_p10`); avoid `full` game-rule stack

## Next actions (ordered)

1. Multi-seed Modal runs (0s / 10s / 15s) + `run_multi_seed.py`
2. Regenerate `figures/summary_figure.png` under `data/runs/sportsmot_example/`
3. LSTM training on `trajectory_tensors.json`

## Conventions

- Track `frame_number` is 1-based; SAM prep frames are `00000.jpg` … under `runs/.../frames/`
- SportsMOT MOT frame *N* ↔ `frames/00000N.jpg` when using consecutive 25 FPS prep
- Augmentation: `--no-gap-fill` for LSTM v1; game rules hurt ADE on real GT
