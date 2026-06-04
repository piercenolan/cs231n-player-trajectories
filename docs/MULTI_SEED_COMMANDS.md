# Multi-seed Modal commands (SportsMOT example)

Run **one Modal job at a time**. Wait until each finishes before starting the next (reduces CUDA OOM from warm GPU reuse).

Use the **same** `--max-frames 45 --resize-scale 0.67` on every seed unless you document otherwise.

---

## 1) Download seed 1 (offset_0s) — already ran on Modal

If you used `--seed-id offset_0s`:

```powershell
mkdir -Force data\runs\sportsmot_example\seeds\offset_0s
py -m modal volume get sports-data runs/sportsmot_example/seeds/offset_0s/baseline_tracks.json data/runs/sportsmot_example/seeds/offset_0s/baseline_tracks.json
```

If you ran **without** `--seed-id` (main baseline only on volume):

```powershell
py -m modal volume get sports-data runs/sportsmot_example/baseline_tracks.json data/runs/sportsmot_example/baseline_tracks.json
copy data\runs\sportsmot_example\baseline_tracks.json data\runs\sportsmot_example\seeds\offset_0s\baseline_tracks.json
```

---

## 2) Seed 2 — offset_10s (10 seconds into clip)

```powershell
py -m modal run scripts/run_modal.py --dataset sportsmot_example --skip-extract --max-frames 45 --resize-scale 0.67 --max-num-objects 12 --start-time-sec 10 --seed-id offset_10s --skip-upload
```

If OOM persists, wait 2–5 minutes and retry, or use `--resize-scale 0.5` (same on all seeds).

---

## 3) Seed 3 — offset_15s

```powershell
py -m modal run scripts/run_modal.py --dataset sportsmot_example --skip-extract --max-frames 45 --resize-scale 0.67 --max-num-objects 12 --start-time-sec 15 --seed-id offset_15s --skip-upload
```

---

## 4) Download seeds 2 and 3

```powershell
mkdir -Force data\runs\sportsmot_example\seeds\offset_10s
mkdir -Force data\runs\sportsmot_example\seeds\offset_15s

py -m modal volume get sports-data runs/sportsmot_example/seeds/offset_10s/baseline_tracks.json data/runs/sportsmot_example/seeds/offset_10s/baseline_tracks.json

py -m modal volume get sports-data runs/sportsmot_example/seeds/offset_15s/baseline_tracks.json data/runs/sportsmot_example/seeds/offset_15s/baseline_tracks.json
```

Optional: download all seeds at once:

```powershell
py -m modal volume get sports-data runs/sportsmot_example/seeds data/runs/sportsmot_example/seeds --force
```

---

## 5) Per-seed GT alignment (local)

```powershell
py scripts/align_seed_gt.py --dataset sportsmot_example
```

Or one seed:

```powershell
py scripts/align_seed_gt.py --dataset sportsmot_example --seed-id offset_10s
```

Writes `data/runs/sportsmot_example/seeds/{seed_id}/gt_aligned.json` matched to each baseline window.

---

## 6) Aggregate multi-seed ADE

```powershell
py scripts/run_multi_seed.py --dataset sportsmot_example --align-gt
```

Output: `data/runs/sportsmot_example/seeds/multi_seed_summary.json`

---

## First-time frame upload (only if not on volume yet)

```powershell
py -m modal run scripts/run_modal.py --dataset sportsmot_example --skip-extract --max-frames 45 --resize-scale 0.67 --start-time-sec 0 --seed-id offset_0s
```

(Omit `--skip-upload` on the first run so frames upload to the volume.)
