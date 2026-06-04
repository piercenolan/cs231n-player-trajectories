# Deferred: pooled multi-sequence training

Per the 36-hour sprint plan, the following are **explicitly out of scope** until after the report deadline:

- `registry.json` with full manifest-driven training
- `build_dataloaders_from_manifest` pooling tensors across sequences
- Retraining residual LSTM on 4+ clips jointly
- Scaling to all 30 basketball GT sequences

**Current approach:** Train on `sportsmot_example` only; evaluate the same residual checkpoint on additional clips via transfer (`--a1-residual-checkpoint`).

**When to revisit:** After `data/sportsmot_publish.zip` is re-downloaded and three sprint sequences are extracted (see `data/datasets/EXTRACTION_STATUS.md`).
