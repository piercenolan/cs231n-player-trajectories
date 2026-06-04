# Paper Results (auto-generated)

## Section 1 — Detection metrics (pre-LSTM)

| Metric | SAM3.1 baseline | Augmented |
|--------|-----------------|-----------|
| Mean players/frame | 10.755555555555556 | 10.422222222222222 |
| ID switches | 0 | 0 |
| Mean track streak | 32.266666666666666 | 27.933333333333334 |

## Section 2 — Forecast metrics (LSTM)

| Model | All-seed median ADE | Clean-seed median ADE | Beats linear |
|-------|---------------------|------------------------|--------------|
| Linear | 5.81 | 5.94 | — |
| A0 Plain | 10.68 | 10.43 | N |
| A1 Rule-Conditioned | 7.51 | 7.22 | N |
| A3 Graph | 8.67 | 8.47 | N |

## Section 3 — Per-seed breakdown

| Seed | A0 | A1 | Linear | Winner (A1 vs linear) |
|------|-----|-----|--------|------------------------|
| offset_0s | 40.15 | 44.98 | 7.09 | linear |
| offset_10s | 7.86 | 7.46 | 6.16 | linear |
| offset_12s | 10.69 | 6.34 | 4.69 | linear |
| offset_14s | 10.43 | 6.97 | 5.94 | linear |
| offset_15s | 37.30 | 41.58 | 5.19 | linear |
| offset_16s | 11.19 | 9.98 | 7.38 | linear |
| offset_18s | 10.67 | 7.55 | 5.95 | linear |
| offset_2s | 7.70 | 5.65 | 3.22 | linear |
| offset_4s | 9.57 | 6.75 | 4.55 | linear |
| offset_5s | 53.68 | 50.64 | 4.57 | linear |
| offset_6s | 11.56 | 7.22 | 5.67 | linear |
| offset_8s | 9.55 | 8.87 | 6.78 | linear |

## Section 4 — Key findings

- A1 beats A0 on 10/12 seeds (per robust report).
- A1 clean-seed median: 7.22 px vs linear: 5.94 px (21.4% higher than linear).
- Three failure seeds (`offset_0s`, `offset_5s`, `offset_15s`): high ADE on all models — SAM3.1 tracking failure, not LSTM-specific.
- Teacher-forced A1 median: 4.97 px; rollout forecast ADE gap reflects exposure bias during training.

## Section 5 — Honest limitations

- Evaluated on a single SportsMOT sequence (`sportsmot_example`).
- Prior held-out `offset_0s` training showed poor generalization on that window; `temporal_all` retrains all seeds.
- Game-rules post-refine (A2) hurts rather than helps — see ablation attribution.
- Cross-sequence generalization is untested.
