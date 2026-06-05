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
| **Primary train/eval clip** | `data/datasets/sportsmot_example/` — 500 frames |
| **Transfer clips (3)** | `sportsmot_v_6os86hzwcs_c001`, `sportsmot_v_6os86hzwcs_c003`, `sportsmot_v_00hrwkvvjtq_c001` (holdout) |
| **Run artifacts** | `data/runs/{dataset}/` — seeds, ablations, LSTM |
| **Transfer summary** | `data/runs/multiseq_transfer_summary.csv` |
| **Legacy (do not cite)** | `data/outputs/`, proxy GT under `data/gt/` |

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

**Forecasting (12 seeds @ 2s, `held_out_seed` training — val `offset_0s`):**

| Variant | Median forecast ADE (all) | Median forecast ADE (clean seeds) |
|---------|---------------------------|-----------------------------------|
| **A1 residual** (headline) | **5.81 px** | 5.95 px |
| Linear baseline | 5.81 px | 5.94 px |
| A1 rule features | 7.51 px | 7.22 px |
| A3 graph | 8.67 px | 8.47 px |
| A0 plain | 10.68 px | 10.43 px |

A1 beats A0 on **10/12** seeds; **residual LSTM ties linear** on median and beats linear on **5/12** seeds. Three failure seeds (`offset_0s`, `offset_5s`, `offset_15s`) reflect SAM tracking collapse. See `docs/PAPER_RESULTS.md`.

**Write-up framing:** SAM3 provides perception; sanitize cleans inputs; **residual rule-LSTM** learns corrections over constant-velocity linear; post-hoc game rules on predictions (A2) hurt ADE. **Cross-sequence transfer** complete — see `data/runs/multiseq_transfer_summary.csv` and `docs/PAPER_RESULTS.md` Section 5.

## Conventions

- Track `frame_number` is 1-based; SAM prep frames are `00000.jpg` … under `runs/.../frames/`
- Multi-seed tensors: **resize_scale 0.5** under `seeds/offset_*` — do not mix with run-root export at 0.67
- LSTM eval: **forecast-horizon** ADE via `scripts/eval_lstm_ablations.py`
- Augmentation: `--no-gap-fill` for LSTM export; game rules hurt detection ADE on real GT
