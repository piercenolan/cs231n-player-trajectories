# Deferred: pooled multi-sequence training

> **AI-assisted documentation (Cursor Agent).** Attribution: [GENERATIVE_AI_USE.md](GENERATIVE_AI_USE.md).

Per the 36-hour sprint plan, the following are **explicitly out of scope** until after the report deadline:

- `registry.json` with full manifest-driven training
- `build_dataloaders_from_manifest` pooling tensors across sequences
- Retraining residual LSTM on 4+ clips jointly
- Scaling to all 30 basketball GT sequences

**Current approach (completed for sprint):** Per-clip residual LSTM training on all 4 clips; transfer baseline preserved in `multiseq_transfer_baseline.csv`.

**Still deferred:**
