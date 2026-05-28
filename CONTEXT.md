# CS231N Project Context

## Project
Research project evaluating SAM3 (Segment Anything Model 3) on basketball video.
We run SAM3 on basketball clips, measure where it fails (ID switches, tracking loss),
then add basketball-domain rules to improve it.

## Stack
- Python 3.11
- PyTorch
- OpenCV for video processing
- SAM2/SAM3 from Meta (pip install segment-anything-3)
- Modal for cloud GPU compute
- Weights & Biases for experiment tracking

## Data
- Basketball video clips in data/clips/
- Three clips from NBA game footage at 360p
- Files: 60.0_clip.mp4, 690.0_clip.mp4, 2700.0_clip.mp4

## Project Structure
cs231n-player-trajectories/
├── data/clips/          # video clips, not committed to git
├── models/              # model definitions
├── utils/               # helper functions
├── scripts/             # runnable scripts
└── notebooks/           # visualization

## Phase 1 Goal
Run SAM3 on basketball clips, extract player tracks frame by frame,
measure ID switch rate and tracking loss, produce visualizations.