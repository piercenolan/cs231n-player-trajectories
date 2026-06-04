# Pre-LSTM gauge summary

## LSTM export gate
- **Passed:** True
- **Global visibility:** 94.00%
- **Frames with zero visible players:** 0
- Thresholds: visibility ≥ 0.7, slot empty ≤ 0.6

## Figures
- `baseline_metrics.png` — SAM3 coverage + ID continuity
- `summary_figure.png` — baseline vs augmented (qualitative)
- `ablation_ade_bar.png` — ADE across ablations (real GT)
- `multi_seed_ade_bar.png` — ADE across temporal offsets
- `lstm_rule_ablation_bar.png` — forecast ADE: A0–A3 LSTM variants
- `lstm_per_rule_delta_ade.png` — per-rule post-refine ΔADE vs plain LSTM

## Post-LSTM metrics
See `../lstm/lstm_ablation_summary.csv` and `docs/MILESTONE_CHECKLIST.md` (12 seeds @ 2s step).

## Baseline vs augmented (console compare)
```
﻿Loaded 45 frames from data\runs\sportsmot_example\baseline_tracks.json
Loaded 45 frames from data\runs\sportsmot_example\ablations\sanitize_plus_velocity_cap\augmented_tracks.json

==================================================
BASELINE vs AUGMENTED COMPARISON
==================================================
Baseline:  data/runs/sportsmot_example/baseline_tracks.json
Augmented: data/runs/sportsmot_example/ablations/sanitize_plus_velocity_cap/augmented_tracks.json
(Augmented counts exclude predicted: True)

  Mean observed players/frame      base=   10.76  aug=   10.42  delta=   -0.33  (worse)
  Mean predicted players/frame     base=    0.00  aug=    0.00  delta=   +0.00  (same)
  ID switches                      base=    0.00  aug=    0.00  delta=   +0.00  (same)
  Mean track streak                base=   32.27  aug=   27.93  delta=   -4.33  (worse)
  Mean displacement (smoothness)   base=    3.57  aug=    3.67  delta=   +0.10  (worse)
  Max jump                         base=   15.01  aug=   15.01  delta=   +0.00  (same)
  Rolling loss frames              base=    0.00  aug=    0.00  delta=   +0.00  (same)
==================================================


```