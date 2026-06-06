# SportsMOT ground truth (legacy layout)

**Do not use this tree for reporting.**

Canonical GT for all current experiments lives under each dataset:

```
data/datasets/{dataset}/gt/gt.txt
data/datasets/{dataset}/gt/gt.json   # after setup_sportsmot_gt.py / align_seed_gt.py
```

Per-seed aligned GT for evaluation:

```
data/runs/{dataset}/seeds/{seed_id}/gt_aligned.json
```

The old proxy GT under `video_1/` was removed during the final repository audit.
