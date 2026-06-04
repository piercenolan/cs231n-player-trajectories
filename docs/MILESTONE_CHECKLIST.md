# CS231N Milestone Checklist (Write-Up)

One-page status for the report. Primary dataset: **SportsMOT example**, 45-frame SAM window, `data/runs/sportsmot_example/`.

## Pipeline at a glance

```mermaid
flowchart LR
  subgraph data [Data]
    F[frames 000001-000500]
    GT[gt.txt / gt.json]
  end
  subgraph track [Tracking]
    SAM[SAM3.1 Modal]
    BT[baseline_tracks.json]
  end
  subgraph aug [Augmentation]
    SAN[sanitize]
    RULES[physical / game rules]
    AUG[augmented_tracks.json]
  end
  subgraph eval [Evaluation]
    ADE[ADE / FDE]
    MET[metrics + ablations]
    EXP[trajectory_tensors.json]
  end
  subgraph lstm [Rule-aware LSTM]
    A0[A0 plain]
    A1[A1 rule features]
    A3[A3 graph]
  end
  F --> SAM --> BT
  GT --> ADE
  BT --> SAN --> RULES --> AUG
  AUG --> ADE
  AUG --> MET
  AUG --> EXP
  EXP --> A0
  EXP --> A1
  EXP --> A3
```

---

## Milestone table

| # | Deliverable | Status | Evidence / notes |
|---|-------------|--------|----------------|
| 1 | **Dataset**: SportsMOT example frames + `gt.txt` | Done | `data/datasets/sportsmot_example/` |
| 2 | **SAM3 baseline** (45 frames, 0.67 scale) | Done | `data/runs/sportsmot_example/baseline_tracks.json` |
| 3 | **Aligned GT** (real MOT, not proxy) | Done | `data/datasets/sportsmot_example/gt/gt.json` |
| 4 | **Augmentation layer** (sanitize + rules + gap-fill gate) | Done | `utils/augmentation.py` |
| 5 | **Per-rule ablations** + ADE/FDE | Done | `ablations/ablation_summary.csv` |
| 6 | **Sanitize grid** (ADE-ranked) | Done | `sanitize_grid/best_sanitize.json` → `w0.4_y0.1_p10` |
| 7 | **LSTM export** + validation gate | Done | `trajectory_tensors.json`, `trajectory_validation.json` (`passed: true`) |
| 8 | **Multi-seed SAM3** (12 offsets @ 2s) | Done | `seeds/seed_manifest.json`, `multi_seed_summary.json` |
| 9 | **Paper figures** (SportsMOT run) | Done | `figures/PRE_LSTM_GAUGE.md` + gauge PNGs |
| 10 | **LSTM v1** train / eval | Done | `lstm/lstm_plain/`, `eval_lstm.py` |
| 11 | **Rule-aware LSTM** (A0–A3 ablations) | Done | `lstm/lstm_ablation_summary.csv`, `figures/lstm_rule_ablation_bar.png` |
| 12 | **Rule feature export** (15-dim) | Done | `utils/rule_features.py`; tensors include `rule_features` |
| 13 | **Step-sec 2 multi-seed (12 windows)** | Done | 12 seeds @ 2s step; tensors + rule features exported |

---

## Results to cite (SportsMOT 45-frame window)

| Metric | Baseline | Best augmentation (ADE) | LSTM v1 policy config |
|--------|----------|---------------------------|------------------------|
| Mean observed players/frame | 10.76 | 10.42 (post-sanitize) | `sanitize_plus_velocity_cap` |
| ID switches | 0 | 0 | same |
| Mean displacement (px) | 3.57 | ~3.67 (`sanitize_plus_velocity_cap`) | velocity_cap **inactive** on this clip |
| ADE vs aligned GT (px) | **7.16** | **7.08** (`dead_ball_freeze`) | sanitize+velocity_cap **7.34** |
| Sanitize grid best ADE | — | **5.87** (`w0.4_y0.1_p10`, velocity_cap rule) | consider for report |
| Export global visibility | — | **0.94** (gate passed) | from `sanitize_plus_velocity_cap` export |

**Report framing:** Baseline SAM3.1 is already strong on this clip; augmentation mainly **prunes crowd/over-detections** (sanitize). Game rules and `full` **hurt ADE** on real GT. Prefer **minimal physical + sanitize** for LSTM inputs.

---

## Blockers (original → current)

| Blocker | Was | Now |
|---------|-----|-----|
| Unknown `video_1` + proxy GT | Invalid ADE | **Cleared** — SportsMOT `gt.txt` + aligned `gt.json` |
| Export validation | Failed on old run | **Cleared** — `passed: true`, visibility 0.94 |
| Multi-seed stability | Bootstrap only | **Cleared** — 12 offsets @ 2s step |
| LSTM train (local) | **Cleared** | A0–A3 trained; `lstm_ablation_summary.csv` |

---

## Optional / out of scope for milestone

- Full 500-frame SAM3 session (VRAM); 45-frame window is sufficient for milestone.
- `seqinfo.ini` (defaults 1280×720 @ 25 FPS used if missing).
- NBA clips in `data/clips/` (secondary; not on real SportsMOT GT).

---

## Rule-aware LSTM (forecast horizon, 12 seeds @ 2s step)

| Variant | Mean forecast ADE (px) | Notes |
|---------|------------------------|--------|
| **A3 graph** | **18.64 ± 16.71** | Best mean across 12 seeds |
| **A1 rule features** | **18.88 ± 16.92** | Beats A0; best teacher-forced (~4.2 px on `offset_0s`) |
| A0 plain | 20.29 ± 15.20 | Positions-only baseline |
| A2 post-refine (game) | 21.83 ± 15.26 | Hurts forecasts |
| A2 post-refine (physical) | 21.56 ± 15.62 | Slightly worse than A0 |

**`offset_0s` slice:** A1 forecast ADE **8.86** vs A0 **9.19** vs linear **6.35** vs SAM aug **7.48** px.

Per-rule post-refine (ΔADE vs plain on `offset_0s`): worst `convergence_pull` (+10.4), `cluster_cohesion` (+6.2); near-neutral physical rules.

Per-rule post-refine attribution: `lstm/lstm_rule_attribution.csv`, `figures/lstm_per_rule_delta_ade.png`.

---

## Commands (copy-paste)

```bash
py scripts/export_lstm_tensors.py --dataset sportsmot_example --all-seeds --with-rule-features
py scripts/train_lstm.py --model plain --split temporal_all
py scripts/train_lstm.py --model rule_features --split temporal_all
py scripts/train_lstm.py --model graph --split temporal_all
py scripts/eval_lstm_ablations.py --all-seeds
py scripts/predict_lstm.py --post-refine game   # A2 on plain checkpoint
```

See **README.md** (full pipeline) and **docs/PROJECT_PLAN.md** (multi-seed + LSTM ordering).
