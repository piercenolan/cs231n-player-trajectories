# Project status (report-ready)

**Last updated:** sprint completion audit

## Direct answers

| Question | Answer |
|----------|--------|
| LSTM **fully trained** on all new downloaded clips? | **No.** Trained once on `sportsmot_example` only. New clips were **tested via transfer** (same checkpoint). No `checkpoint.pt` under `sportsmot_v_*`. |
| LSTM **fully tested** on all new data? | **Mostly yes.** SAM3 + tensors + transfer eval on 3 clips (47 Modal seeds total). Some end-of-clip seeds failed tensor validation. |
| Is this the **best** the LSTM can get? | **No**, but best **within this sprint.** Per-clip retraining or pooled multi-sequence training (~2–5 min/clip locally) could improve; not run. |
| **Updated results** available? | **Yes.** `docs/PAPER_RESULTS.md`, `data/runs/multiseq_transfer_summary.csv`, per-dataset `lstm_ablation_robust.json`. |

## Is the project "as complete as possible" for this deadline?

**Yes for the agreed sprint scope:** one training clip + three transfer clips + report artifacts.

**Not** the theoretical maximum (30 GT clips, per-clip retrain, pooled training).

## What the results actually say (do not over-claim)

**Rule / basketball features (A1) DO help** vs plain LSTM (A0: ~10.7 px → A1: ~7.5 px on `sportsmot_example`).

**Residual LSTM + rule features** matches **linear** on the training clip (median ~5.8 px) and beats linear on **5/12** seeds there.

**Cross-sequence transfer** (same weights, no retrain):

| Dataset | Residual median | Linear median | Winner |
|---------|-----------------|---------------|--------|
| sportsmot_example (train) | 5.81 | 5.81 | tie |
| sportsmot_v_6os86hzwcs_c001 | 4.94 | 4.84 | linear |
| sportsmot_v_6os86hzwcs_c003 | 5.56 | 5.41 | linear |
| sportsmot_v_00hrwkvvjtq_c001 (holdout) | 4.99 | 5.01 | **residual** |

**Correct report claim:** Rule-conditioned **residual** LSTM closes most of the gap to linear on the training clip; **transfer** is mixed (holdout slightly favors residual; two train-split clips slightly favor linear). This is **not** "rule features never help."

**Incorrect claim:** "Adding basketball rule weights does not improve predictions at all."

## Report writing — start here

1. **Numbers:** [PAPER_RESULTS.md](PAPER_RESULTS.md) (Sections 2, 5)
2. **Narrative:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md)
3. **Figure (training clip):** `data/runs/sportsmot_example/figures/lstm_comparison.png`
4. **Figure (transfer):** `data/runs/figures/multiseq_transfer_bar.png`
5. **Checklist:** [MILESTONE_CHECKLIST.md](MILESTONE_CHECKLIST.md)

## Regenerate artifacts

```powershell
py scripts/aggregate_multiseq_eval.py
py scripts/generate_paper_results.py --multiseq-csv data/runs/multiseq_transfer_summary.csv
py scripts/plot_lstm_vs_baselines.py
py scripts/plot_multiseq_transfer.py
py scripts/plot_pre_lstm_gauge.py
```
