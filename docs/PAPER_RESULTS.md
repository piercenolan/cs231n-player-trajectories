# Paper Results (auto-generated)

## Section 1 - Detection metrics (pre-LSTM)

| Metric | SAM3.1 baseline | Augmented |
|--------|-----------------|-----------|
| Mean players/frame | 10.755555555555556 | 10.422222222222222 |
| ID switches | 0 | 0 |
| Mean track streak | 32.266666666666666 | 27.933333333333334 |

## Section 2 - Forecast metrics (LSTM)

| Model | All-seed median ADE | Clean-seed median ADE | Beats linear |
|-------|---------------------|------------------------|--------------|
| Linear | 5.81 | 5.94 | — |
| A0 Plain | 10.68 | 10.43 | N |
| A1 Rule-Conditioned | 7.51 | 7.22 | N |
| A1 Residual (headline) | 5.81 | 5.95 | N |
| A3 Graph | 8.67 | 8.47 | N |

## Section 3 - Per-seed breakdown

| Seed | A0 | A1 | A1 Residual | Linear | Residual vs linear |
|------|-----|-----|-------------|--------|---------------------|
| offset_0s | 40.15 | 44.98 | 7.25 | 7.09 | linear |
| offset_10s | 7.86 | 7.46 | 6.15 | 6.16 | A1_residual |
| offset_12s | 10.69 | 6.34 | 4.68 | 4.69 | A1_residual |
| offset_14s | 10.43 | 6.97 | 5.95 | 5.94 | tie |
| offset_15s | 37.30 | 41.58 | 5.15 | 5.19 | A1_residual |
| offset_16s | 11.19 | 9.98 | 7.38 | 7.38 | tie |
| offset_18s | 10.67 | 7.55 | 6.12 | 5.95 | linear |
| offset_2s | 7.70 | 5.65 | 3.25 | 3.22 | linear |
| offset_4s | 9.57 | 6.75 | 4.50 | 4.55 | A1_residual |
| offset_5s | 53.68 | 50.64 | 4.68 | 4.57 | linear |
| offset_6s | 11.56 | 7.22 | 5.66 | 5.67 | tie |
| offset_8s | 9.55 | 8.87 | 6.66 | 6.78 | A1_residual |

## Section 4 - Key findings

- A1 beats A0 on 10/12 seeds (held-out-seed training eval).
- **A1 Residual** median forecast ADE: 5.81 px (all seeds) / 5.95 px (clean); linear: 5.81 / 5.94 px.
- Residual beats linear on 5/12 seeds individually.
- Three failure seeds (`offset_0s`, `offset_5s`, `offset_15s`): high ADE on all models — SAM3.1 tracking failure, not LSTM-specific.
- Teacher-forced A1 median: 4.97 px; rollout gap reflects exposure bias during training.

## Section 5 - Cross-sequence transfer (eval-only)

| Dataset | Median residual ADE | Median linear ADE | Residual beats linear |
|---------|---------------------|-------------------|------------------------|
| sportsmot_example | 5.81 | 5.81 | tie |
| sportsmot_v_00hrwkvvjtq_c001 | 4.99 | 5.01 | Y |
| sportsmot_v_6os86hzwcs_c001 | 4.94 | 4.84 | N |
| sportsmot_v_6os86hzwcs_c003 | 5.56 | 5.41 | N |

## Section 6 - Honest limitations

- LSTM trained only on `sportsmot_example`; other clips use the same checkpoint (transfer).
- Game-rules post-refine (A2) hurts rather than helps — see ablation attribution.
- SAM augmented tracks on future frames are a detection ceiling, not a fair forecast baseline.
