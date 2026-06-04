# CS231N Project Context

## Project

Evaluate **SAM3.1** on basketball broadcast video, improve tracks with a **geometry-free augmentation** layer, measure against **SportsMOT** ground truth, and forecast trajectories with **rule-aware LSTM** variants (A0 plain, A1 rule features, A2 post-refine, A3 graph).

## Stack

- Python 3.12 (local) / 3.12 (Modal image)
- PyTorch, SAM3.1 (`facebook/sam3.1`, Hugging Face gated)
- OpenCV (frame I/O)
- Modal A10G for SAM3 tracking (`scripts/run_modal.py`, `scripts/run_all_seeds_modal.py`)
- NumPy, Matplotlib (metrics / viz)

## Data (current)

| Role | Location |
|------|----------|
| **Primary eval set** | `data/datasets/sportsmot_example/` — 500 JPEGs + `gt/gt.txt` |
| **Run artifacts** | `data/runs/sportsmot_example/` — seeds, ablations, LSTM |
| **Legacy (do not cite)** | `data/outputs/`, `data/videos/video_1.mp4`, proxy GT under `data/gt/sportsmot/video_1/` |
| **NBA clips (optional)** | `data/clips/` — 60.0, 690.0, 2700.0 clips at 360p |

## Repository structure

```text
cs231n-player-trajectories/
├── docs/                    # MILESTONE_CHECKLIST, PROJECT_PLAN, MULTI_SEED_COMMANDS
├── scripts/                 # Modal, ablations, export, train_lstm, eval_lstm_ablations
├── models/                  # trajectory_lstm, trajectory_graph_lstm
├── utils/                   # augmentation, rule_features, lstm_*, metrics
├── data/datasets/           # static inputs (SportsMOT example)
├── data/runs/               # generated per-dataset outputs
└── data/archive/            # legacy layout notes
```

Path helper: `utils/datasets.py` — always pass `--dataset sportsmot_example`.

## Phase goals

| Phase | Goal | Status |
|-------|------|--------|
| 1 | SAM3.1 baseline tracks | Done |
| 2 | Augmentation + ablations + ADE | Done |
| 3 | Pre-LSTM validation (GT, export, multi-seed) | Done — 12 seeds @ 2s |
| 4 | Rule-aware LSTM (A0–A3) | Done |

See `docs/MILESTONE_CHECKLIST.md` for report tables and `docs/PROJECT_PLAN.md` for architecture.

## Current results snapshot

**Detection (augmented vs GT):** baseline ADE ~7.16 px; `sanitize_plus_velocity_cap` ~7.34 px; game-rule `full` stack hurts ADE.

**Forecasting (12 seeds, `temporal_all` train):**

| Variant | Mean forecast ADE | Best on `offset_0s` |
|---------|-------------------|---------------------|
| A3 graph | 18.64 px | 9.09 px |
| A1 rule features | 18.88 px | **8.86 px** (beats A0) |
| A0 plain | 20.29 px | 9.19 px |
| Linear / SAM aug | — | 6.35 / 7.48 px |

**Write-up framing:** SAM3 provides perception; sanitize cleans inputs; rule **features** in A1 improve short-horizon forecasts vs plain LSTM; post-hoc game rules on predictions (A2) mirror detection findings and hurt ADE.

## Conventions

- Track `frame_number` is 1-based; SAM prep frames are `00000.jpg` … under `runs/.../frames/`
- Multi-seed tensors: **resize_scale 0.5** under `seeds/offset_*` — do not mix with run-root export at 0.67
- LSTM eval: **forecast-horizon** ADE via `scripts/eval_lstm_ablations.py`
- Augmentation: `--no-gap-fill` for LSTM export; game rules hurt detection ADE on real GT
