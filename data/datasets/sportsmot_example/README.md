# SportsMOT example dataset (primary evaluation set)

Use this folder for **all LSTM-bound work** — it has matched frames and `gt.txt`.

## What to upload (from the SportsMOT example zip)

Copy files from the zip into these exact locations:

| From zip | Into repo |
|----------|-----------|
| `img1/000001.jpg` … `img1/000500.jpg` | `data/datasets/sportsmot_example/frames/` |
| `gt/gt.txt` | `data/datasets/sportsmot_example/gt/gt.txt` |
| `seqinfo.ini` (if present) | `data/datasets/sportsmot_example/seqinfo.ini` |

After copying you should have:

```text
data/datasets/sportsmot_example/
├── frames/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ... (500 files)
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

**Do not** commit JPEGs or GT to git if the repo is public (SportsMOT license).

## Frame ↔ GT alignment

- SportsMOT `gt.txt` frame index **1** = `frames/000001.jpg`
- Multi-seed SAM3 uses 45-frame windows at **resize_scale 0.5** (640×360 in tracks meta)
- Per-seed evaluation uses `data/runs/sportsmot_example/seeds/{seed_id}/gt_aligned.json`

## Pipeline outputs

All run artifacts:

```text
data/runs/sportsmot_example/
├── baseline_tracks.json          # optional run-root (offset_0s canonical)
├── ablations/                    # detection ablations + ADE
├── sanitize_grid/
├── seeds/
│   ├── seed_manifest.json        # 12 seeds @ 2s step
│   ├── offset_0s/ ... offset_18s/
│   │   ├── baseline_tracks.json
│   │   ├── gt_aligned.json
│   │   └── trajectory_tensors.json   # + rule_features when exported
│   └── multi_seed_summary.json
├── lstm/
│   ├── lstm_plain/               # A0 checkpoint
│   ├── lstm_rule_features/       # A1
│   ├── lstm_graph/               # A3
│   ├── lstm_ablation_summary.csv
│   ├── lstm_ablation_robust.json
│   └── lstm_per_seed_delta.csv
└── figures/
    ├── PRE_LSTM_GAUGE.md
    ├── lstm_rule_ablation_bar.png
    └── lstm_per_rule_delta_ade.png
```

## Quick start (after data upload)

```powershell
py scripts/run_all_seeds_modal.py --dataset sportsmot_example --step-sec 2 --skip-existing
py scripts/export_lstm_tensors.py --dataset sportsmot_example --all-seeds --with-rule-features
py scripts/train_lstm.py --model rule_features --split held_out_seed --val-seed offset_0s --epochs 80
py scripts/eval_lstm_ablations.py --dataset sportsmot_example --all-seeds --diagnose-seeds
```

Full documentation: [README.md](../../../README.md), [docs/MILESTONE_CHECKLIST.md](../../../docs/MILESTONE_CHECKLIST.md).

Legacy outputs: `data/outputs/`, `data/archive/`.
