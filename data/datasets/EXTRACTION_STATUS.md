# SportsMOT extraction status

## Sprint complete

All 3 basketball sequences extracted, registered, Modal-tracked, and transfer-evaluated.

| Sequence | Dataset key | Modal seeds | Tensor export (rule features) |
|----------|-------------|-------------|-------------------------------|
| `v_-6Os86HzwCs_c001` | `sportsmot_v_6os86hzwcs_c001` | 16 | 14 passed validation |
| `v_-6Os86HzwCs_c003` | `sportsmot_v_6os86hzwcs_c003` | 8 | 7 passed validation |
| `v_00HRwkvvjtQ_c001` (holdout) | `sportsmot_v_00hrwkvvjtq_c001` | 23 | 22 passed validation |

Transfer results: `data/runs/multiseq_transfer_summary.csv`

## Windows extract tip

If extraction fails with `PermissionError`, use `--force` or delete partial folders under `data/datasets/sportsmot_basketball/`.
