# Project status (report-ready)

**Last updated:** after per-clip LSTM retrain on 3 new basketball clips

## Direct answers

| Question | Answer |
|----------|--------|
| LSTM **fully trained** on all new downloaded clips? | **Yes (per-clip).** Residual LSTM retrained on each of the 3 new clips + original `sportsmot_example`. Checkpoints: `data/runs/{dataset}/lstm/lstm_rule_features_residual/checkpoint.pt`. |
| LSTM **fully tested** on all new data? | **Yes.** Multi-seed eval on all 4 clips (59 seeds total across new clips + 12 on example). |
| Is this the **best** the LSTM can get? | **Best within this sprint scope.** Pooled multi-sequence training or scaling to 30 GT clips could still help; not run. |
| **Updated results** available? | **Yes.** `docs/PAPER_RESULTS.md` Sections 5–6, `multiseq_perclip_summary.csv`, comparison figure. |

## Headline results (per-clip trained residual vs linear)

| Dataset | Residual median | Linear median | Winner |
|---------|-----------------|---------------|--------|
| sportsmot_example | 5.81 | 5.81 | tie |
| sportsmot_v_6os86hzwcs_c001 | **4.82** | 4.84 | **residual** |
| sportsmot_v_6os86hzwcs_c003 | 5.48 | 5.41 | linear |
| sportsmot_v_00hrwkvvjtq_c001 (holdout) | 5.16 | 5.01 | linear |

**Per-clip training vs transfer:** `c001` improves (−0.12 px residual); `c003` improves slightly (−0.08 px) but still loses to linear; holdout **worsens** (+0.17 px) vs transfer-from-example — transfer happened to generalize better on that clip.

## What the results say (do not over-claim)

- **Rule features help A1 vs A0** on the primary clip (~7.5 vs ~10.7 px).
- **Residual + rules ties linear** on `sportsmot_example` (~5.8 px median).
- **Per-clip retrain** flips `c001` to beat linear; **does not** universally beat linear across all clips.
- **Incorrect:** "Basketball rule weights never help."

## Report writing — start here

1. **Numbers:** [PAPER_RESULTS.md](PAPER_RESULTS.md) (Sections 2, 5, 6)
2. **Narrative:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md) §4.3
3. **Figures:**
   - `data/runs/sportsmot_example/figures/lstm_comparison.png`
   - `data/runs/figures/multiseq_perclip_bar.png`
   - `data/runs/figures/multiseq_train_vs_transfer.png`

## Regenerate artifacts

```powershell
py scripts/aggregate_multiseq_eval.py --output data/runs/multiseq_perclip_summary.csv --training-mode per_clip
py scripts/generate_paper_results.py
py scripts/plot_lstm_vs_baselines.py
py scripts/plot_multiseq_transfer.py
py scripts/plot_pre_lstm_gauge.py
```
