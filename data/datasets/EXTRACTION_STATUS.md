# SportsMOT extraction status (sprint)

**Date:** 2026-05-28

## Zip integrity

`data/sportsmot_publish.zip` (~16.5 GB) fails per-sequence `tar` extraction:

```text
tar: Truncated input file (needed 169656 bytes, only 114837 available)
```

The archive is **incomplete** (download interrupted). `zipfile` central directory is also unavailable.

## Planned sprint sequences

| Sequence ID | Split | Role |
|-------------|-------|------|
| `v_-6Os86HzwCs_c001` | train | Transfer eval |
| `v_-6Os86HzwCs_c003` | train | Transfer eval |
| `v_00HRwkvvjtQ_c001` | val | **Sequence holdout** (eval only) |

## Fix (required before Modal on new clips)

1. Re-download full `sportsmot_publish.zip` from [SportsMOT OneDrive](https://1drv.ms/u/s!AtjeLq7YnYGRgQRrmqGr4B-k-xsC?e=7PndU8) or Codalab competition data.
2. Verify: `py scripts/extract_sportsmot_basketball.py --zip data/sportsmot_publish.zip --sequences v_-6Os86HzwCs_c001 --list-only` then extract without `--list-only`.
3. Register: `py scripts/register_sportsmot_sequence.py <seq_id>` (see `docs/MODAL_SPRINT_RUNBOOK.md`).

## What works without re-download

- Full pipeline on **`sportsmot_example`** (500 frames, 12 seeds @ 2s, residual LSTM).
- All scripts and `extra_datasets.json` registration path are ready.
