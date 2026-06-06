# Archive (legacy layout)

Legacy pipeline artifacts were **removed from the repository** during the final audit (June 2026). Do not recreate these paths.

| Former path | Notes |
|-------------|--------|
| `data/outputs/` | Pre-reorg SAM3 + augmentation runs (`video_1` era) |
| `data/frames/`, `data/frames_dir/` | Duplicate / unpadded extracted JPEGs |
| `data/videos/video_1.mp4` | Unknown source — **do not use for paper ADE** |
| Root `tracks.json`, `output.log` | Early single-file SAM outputs |

**Canonical layout:**

| Role | Location |
|------|----------|
| Static inputs | `data/datasets/{dataset}/` |
| Experiment outputs | `data/runs/{dataset}/` |
| Multi-clip summaries | `data/runs/multiseq_*.csv` |

See [README.md](../../README.md) and [CONTEXT.md](../../CONTEXT.md).
