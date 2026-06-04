# Multi-seed Modal commands (SportsMOT example)

## One-command pipeline (recommended)

Generates seeds every **2s** (12 windows) or **5s** across 500 frames, runs Modal sequentially, downloads from volume, aligns GT, runs augmentation metrics, and exports LSTM tensors:

```powershell
py scripts/run_all_seeds_modal.py --dataset sportsmot_example --step-sec 2 --skip-existing
```

Denser windows (~12 seeds) for LSTM training:

```powershell
py scripts/run_all_seeds_modal.py --step-sec 2 --skip-existing
```

Coarser (~4 seeds):

```powershell
py scripts/run_all_seeds_modal.py --step-sec 5 --skip-existing
```

Plan without Modal:

```powershell
py scripts/run_all_seeds_modal.py --dry-run
```

After seeds complete, export tensors **with rule features** (required for A1):

```powershell
py scripts/export_lstm_tensors.py --dataset sportsmot_example --all-seeds --with-rule-features
```

Then retrain and refresh ablation tables:

```powershell
py scripts/train_lstm.py --model plain --split temporal_all --epochs 80
py scripts/train_lstm.py --model rule_features --split temporal_all --epochs 80
py scripts/train_lstm.py --model graph --split temporal_all --epochs 80
py scripts/eval_lstm_ablations.py --dataset sportsmot_example --all-seeds
```

Use `py -m modal` if the `modal` CLI is not on PATH.

---

## Outputs per seed

```text
data/runs/sportsmot_example/seeds/{seed_id}/
├── baseline_tracks.json      # SAM3.1 output
├── gt_aligned.json           # MOT GT in track coordinates
├── augmented_tracks.json     # sanitize_plus_velocity_cap (from orchestrator)
├── trajectory_tensors.json   # positions + visibility + rule_features
└── trajectory_validation.json
```

Aggregate: `seeds/seed_manifest.json`, `seeds/multi_seed_summary.json`, `seeds/lstm_tensor_export_summary.json`

---

## Manual steps (legacy)

Run **one Modal job at a time**. Wait until each finishes before starting the next (reduces CUDA OOM from warm GPU reuse).

Multi-seed policy: **`--resize-scale 0.5`** and **`--max-frames 45`** (matches current `seed_manifest.json`).

---

## 1) Download seed 1 (offset_0s)

```powershell
mkdir -Force data\runs\sportsmot_example\seeds\offset_0s
py -m modal volume get sports-data runs/sportsmot_example/seeds/offset_0s/baseline_tracks.json data/runs/sportsmot_example/seeds/offset_0s/baseline_tracks.json
```

---

## 2) Additional seeds (example: offset_10s)

```powershell
py -m modal run scripts/run_modal.py --dataset sportsmot_example --skip-extract --max-frames 45 --resize-scale 0.5 --max-num-objects 12 --start-time-sec 10 --seed-id offset_10s --skip-upload
```

If OOM: wait 2–5 minutes and retry, or lower `--max-num-objects`.

---

## 3) Download all seeds

```powershell
py -m modal volume get sports-data runs/sportsmot_example/seeds data/runs/sportsmot_example/seeds --force
```

---

## 4) Per-seed GT alignment (local)

```powershell
py scripts/align_seed_gt.py --dataset sportsmot_example
```

---

## 5) Aggregate multi-seed detection ADE

```powershell
py scripts/run_multi_seed.py --dataset sportsmot_example --align-gt
```

Output: `data/runs/sportsmot_example/seeds/multi_seed_summary.json`

---

## 6) LSTM tensor export + training

See [README.md](../README.md) § Rule-aware LSTM.

---

## First-time frame upload

```powershell
py -m modal run scripts/run_modal.py --dataset sportsmot_example --skip-extract --max-frames 45 --resize-scale 0.5 --start-time-sec 0 --seed-id offset_0s
```

(Omit `--skip-upload` on the first run so frames upload to the volume.)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Could not connect to Modal server | Retry when online; `--skip-existing` keeps finished seeds |
| CUDA OOM on 2nd seed | One job at a time; `release_sam3_predictor` in pipeline |
| `offset_20s` past clip end | Use 2s step schedule (max ~18s on 500 frames) |
