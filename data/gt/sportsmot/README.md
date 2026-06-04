# SportsMOT ground truth

## Preferred (paper-quality)

Place MOT-format labels for the source sequence:

```text
data/gt/sportsmot/video_1/gt/gt.txt
```

Then align to track coordinates:

```bash
python scripts/setup_sportsmot_gt.py \
  --tracks data/outputs/baseline_tracks.json \
  --raw-gt path/to/SportsMOT/.../gt/gt.txt \
  --seqinfo path/to/seqinfo.ini \
  --start-time-sec 0 \
  --extract-fps 1
```

## Local dev (proxy)

When raw SportsMOT GT is unavailable:

```bash
python scripts/setup_sportsmot_gt.py \
  --tracks data/outputs/baseline_tracks.json \
  --proxy-smooth
```

This writes `data/gt/sportsmot/video_1/gt/gt.json` (smoothed baseline proxy).
Replace with real GT before paper ADE numbers.

## Evaluate

```bash
python utils/trajectory_metrics.py --tracks data/outputs/augmented_tracks.json --sequence video_1
```
