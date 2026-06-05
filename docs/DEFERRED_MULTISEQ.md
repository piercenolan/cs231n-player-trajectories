# Deferred: pooled multi-sequence training

Per the 36-hour sprint plan, the following are **explicitly out of scope** until after the report deadline:

- `registry.json` with full manifest-driven training
- `build_dataloaders_from_manifest` pooling tensors across sequences
- Retraining residual LSTM on 4+ clips jointly
- Scaling to all 30 basketball GT sequences

**Current approach:** Train on `sportsmot_example` only; evaluate the same residual checkpoint on additional clips via transfer (`--a1-residual-checkpoint`).

**When to revisit:** After the report deadline, if you want pooled training across the four sprint clips or scaling to all 30 basketball GT sequences (extraction for the sprint clips is already complete — see `data/datasets/EXTRACTION_STATUS.md`).
