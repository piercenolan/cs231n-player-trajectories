# Player Trajectory Forecasting on Basketball Broadcast Video

**CS231N Final Project Report**

## Abstract

We build an end-to-end pipeline for multi-player tracking and short-horizon trajectory forecasting on SportsMOT basketball footage. SAM3.1 produces per-frame detections; a geometry-free augmentation layer (sanitize + velocity cap) stabilizes tracks before forecasting. We compare plain LSTM, rule-conditioned LSTM, graph LSTM, and a **residual LSTM** that predicts corrections over a constant-velocity linear baseline. On our primary clip (`sportsmot_example`, 12 temporal seeds at 2s spacing), the residual model **ties the linear baseline** on median rollout ADE (~5.8 px) while plain LSTM remains far worse (~10.7 px). Rule features improve A1 over A0 but only match linear once trained as a residual head. We evaluate **transfer** of the same checkpoint on three additional basketball clips (47 Modal seeds): holdout median ADE slightly favors residual (4.99 vs 5.01 px); two train-split clips slightly favor linear.

---

## 1. Introduction

Sports analytics increasingly rely on accurate player tracks and short-term motion forecasts for tactics, broadcast graphics, and automated officiating aids. Modern segment-anything models (SAM) provide strong open-vocabulary detection, but broadcast basketball introduces occlusion, fast cuts, and crowded scenes that degrade track continuity.

**Goals:**

1. Evaluate SAM3.1 tracking with SportsMOT ground truth on a standard example clip.
2. Improve track quality with lightweight, geometry-free augmentation (no court homography).
3. Forecast 4-step horizons (~0.16 s at 25 FPS export) with rule-aware LSTMs.
4. Compare fairly against a **constant-velocity linear** baseline (not SAM tracks on future frames).

---

## 2. Related Work

- **Multi-object tracking (MOT):** SportsMOT provides 240 clips across basketball, football, and volleyball with MOTChallenge-format annotations.
- **Segmentation-based tracking:** SAM and successors enable promptable instance masks; we use SAM3.1 via Modal GPU jobs.
- **Trajectory forecasting:** RNNs and graph networks model social dynamics; we use a compact LSTM with optional 15-dimensional rule features (game phase, velocity hints) derived without court geometry.

---

## 3. Method

### 3.1 Data and evaluation protocol

- **Primary dataset:** `sportsmot_example` — 500 frames, 1280x720 @ 25 FPS, official SportsMOT example.
- **Multi-seed protocol:** 12 windows starting every **2 seconds** (45-frame SAM context); per-seed aligned GT from `gt.txt`.
- **Metrics:** Forecast-horizon ADE (pixels) under rollout (model feeds its own predictions). Teacher-forced ADE reported for exposure-bias analysis.
- **Clean seeds:** Exclude `offset_0s`, `offset_5s`, `offset_15s` where SAM tracking fails (high ADE on all models).

### 3.2 Tracking (SAM3.1)

SAM3.1 runs on Modal (A10G), one seed job at a time. Outputs `baseline_tracks.json` per seed under `data/runs/{dataset}/seeds/{seed_id}/`.

### 3.3 Augmentation

Recommended stack: `sanitize_plus_velocity_cap` (prune spurious detections, cap per-frame velocity). Game-rule post-refinement on **tracks** hurts detection ADE on real GT (consistent with prior ablations); we do not use A2 post-refine in the headline forecast.

### 3.4 Forecasting models

| Variant | Description |
|---------|-------------|
| **A0** | Plain LSTM on positions |
| **A1** | LSTM + 15-d rule features |
| **A3** | Graph LSTM variant |
| **A1 residual (headline)** | Predicts residual over linear extrapolation; checkpoint selected by **validation rollout ADE** |
| **Linear** | Constant-velocity extrapolation from last two observed positions |
| **SAM aug** | Augmented tracks including future frames — **detection ceiling**, not a fair forecast peer |

**Training (headline model):** `held_out_seed` split — train 11 seeds, validate `offset_0s`; 80 epochs with scheduled sampling (p: 0 to 0.3) and `--optimize-forecast-ade`.

### 3.5 Cross-sequence transfer (planned)

Checkpoint trained only on `sportsmot_example` is evaluated on held-out basketball sequences without fine-tuning. Registration via `scripts/register_sportsmot_sequence.py` and `data/datasets/extra_datasets.json`. See `docs/MODAL_SPRINT_RUNBOOK.md`.

---

## 4. Experiments

### 4.1 Detection (pre-LSTM)

| Metric | SAM3.1 baseline | Augmented |
|--------|-----------------|-----------|
| Mean players/frame | 10.76 | 10.42 |
| ID switches | 0 | 0 |
| Mean track streak | 32.3 | 27.9 |

Baseline ADE vs aligned GT ~7.16 px; best sanitize-grid ADE ~5.87 px (reported in milestone checklist).

### 4.2 Forecasting on `sportsmot_example` (12 seeds)

| Model | Median ADE (all) | Median ADE (clean) | Beats linear (median) |
|-------|------------------|--------------------|------------------------|
| Linear | 5.81 | 5.94 | — |
| A0 plain | 10.68 | 10.43 | No |
| A1 rule-conditioned | 7.51 | 7.22 | No |
| **A1 residual** | **5.81** | **5.95** | Tie |
| A3 graph | 8.67 | 8.47 | No |

- A1 beats A0 on **10/12** seeds.
- **Residual beats linear on 5/12** seeds individually; medians tie within 0.01 px.
- Teacher-forced A1 median ~4.97 px vs rollout ~7.5 px on clean seeds — exposure bias gap.

**Failure seeds:** `offset_0s`, `offset_5s`, `offset_15s` show catastrophic ADE for A0, A1, residual, and linear alike → upstream tracking failure.

**Figures:** `data/runs/sportsmot_example/figures/lstm_comparison.png`, `lstm_ade_bar.png`, `PRE_LSTM_GAUGE.md`.

Auto-generated tables: `docs/PAPER_RESULTS.md`.

### 4.3 Cross-sequence transfer (eval-only)

Same **A1 residual** checkpoint trained on `sportsmot_example` only; no fine-tuning on new clips.

| Dataset | Seeds eval | Median residual ADE | Median linear ADE | Residual vs linear |
|---------|------------|---------------------|-------------------|--------------------|
| sportsmot_example (train) | 12 | 5.81 | 5.81 | tie |
| sportsmot_v_6os86hzwcs_c001 | 16 | 4.94 | 4.84 | linear wins |
| sportsmot_v_6os86hzwcs_c003 | 8 | 5.56 | 5.41 | linear wins |
| sportsmot_v_00hrwkvvjtq_c001 (holdout) | 23 | **4.99** | 5.01 | **residual wins** |

Holdout val clip (`v_00HRwkvvjtQ_c001`) shows slight transfer gain over linear on median ADE; train-split clips are within ~0.1–0.2 px of linear. See `data/runs/multiseq_transfer_summary.csv`.

Late-window seeds excluded from export where tensor validation failed (end-of-clip tracking dropout).

---

## 5. Discussion

**Why linear was hard to beat:** Basketball player motion is smooth over 4 frames; constant-velocity is a strong inductive bias. Teacher-forced training underestimates rollout error; residual formulation and forecast-ADE checkpointing align training with evaluation.

**Value of rule features:** A1 improves over A0 on most seeds but still loses to linear under rollout until residuals are explicit. Rule features encode phase hints without court geometry.

**Augmentation role:** Primarily reduces false positives; does not fix broken seeds where SAM loses players.

---

## 6. Limitations and Future Work

1. **Single training clip** for LSTM; other clips evaluated via transfer only.
2. **No pooled multi-sequence training** (deferred; see `docs/DEFERRED_MULTISEQ.md`).
3. SAM-on-future-frames is an oracle ceiling, not comparable to causal forecasters.
4. NBA clips in `data/clips/` lack SportsMOT GT for the same protocol.
5. End-of-clip seeds may fail tensor validation on shorter or noisy windows.

---

## 7. Reproducibility

```powershell
# Example clip (complete)
py scripts/eval_lstm_ablations.py --dataset sportsmot_example --all-seeds --skip-attribution
py scripts/plot_lstm_vs_baselines.py
py scripts/generate_paper_results.py --multiseq-csv data/runs/multiseq_transfer_summary.csv

# After re-download SportsMOT zip
py scripts/extract_sportsmot_basketball.py --zip data/sportsmot_publish.zip --sequences v_-6Os86HzwCs_c001 v_-6Os86HzwCs_c003 v_00HRwkvvjtQ_c001
py scripts/register_sportsmot_sequence.py v_00HRwkvvjtQ_c001 --holdout
# External terminal:
py scripts/run_batch_sportsmot_modal.py --step-sec 2 --skip-existing --datasets <registered_names>
```

Artifacts: `data/runs/sportsmot_example/lstm/lstm_rule_features_residual/checkpoint.pt`, `lstm_ablation_robust.json`, `lstm_per_seed_delta.csv`.

---

## References

- SportsMOT: Yang et al., ICCV 2023.
- SAM / SAM3 family: Meta AI segmentation models.
- MOTChallenge format and metrics conventions.
