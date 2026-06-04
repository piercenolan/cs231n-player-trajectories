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
- For a first SAM run, use the **first 45–100 frames** (1:1 index match):
  - `frame_number: 1` in tracks JSON ↔ `000001.jpg` ↔ GT line with frame `1`
- Default Modal/local settings: `max_frames=45`, `resize_scale=0.67` (matches prior runs)

If you subsample later (e.g. every 25th frame for 1 FPS from 25 FPS video), re-run
`scripts/setup_sportsmot_gt.py` with the correct `--extract-fps` and `--start-time-sec`.

## Pipeline outputs

All new runs write under:

```text
data/runs/sportsmot_example/
├── baseline_tracks.json
├── augmented_tracks.json
├── ablations/
├── seeds/
└── figures/
```

Legacy outputs remain in `data/outputs/` and `data/archive/`.
