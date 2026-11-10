# Portfolio media pack

Curated figures for a personal website / project page. Regenerated 2026-08-08.

**Folder:** `portfolio_media/` (repo root)

## Recommended page layout

### Hero / above the fold
| File | Why |
|------|-----|
| `01_summary_sam_vs_augmented.png` | Shows tracking on real basketball frames (SAM vs cleaned tracks) |
| `10_forecast_qualitative_clean.png` | Predicted vs GT vs linear overlays — most “wow” visual |

### Headline metrics (callout chips)
Pull from `METRICS_PAPER_RESULTS.md`:
- **5.81 px** median forecast ADE (residual = linear)
- **10/12** seeds: rule-conditioned LSTM beats plain LSTM
- **5/12** seeds: residual beats linear
- Detection ADE **~7.2 px**; players/frame **10.76 → 10.42**
- **4** SportsMOT clips; **12** primary temporal windows

### Results section (charts)
| File | Caption idea |
|------|----------------|
| `05_lstm_forecast_comparison.png` | All-seed vs clean-seed forecast ADE |
| `06_lstm_variant_ade_bar.png` | A0 / A1 / residual / A3 / linear |
| `07_lstm_rule_ablation_bar.png` | Model ablation including A2 |
| `08_a2_per_rule_delta_ade.png` | Why hard post-refine hurts (every rule +ADE) |
| `03_augmentation_ablation_ade.png` | Track-cleaning ablations vs GT |
| `12_multiseq_train_vs_transfer.png` | Per-clip train vs cross-clip transfer |

### Optional / supporting
| File | Use |
|------|-----|
| `02_baseline_tracking_metrics.png` | Tracking quality (players/frame, streak) |
| `04_multi_seed_detection_ade.png` | Multi-seed detection robustness |
| `09_forecast_qualitative.png` | Alternate forecast windows (`offset_2s`) |
| `11_multiseq_perclip_bar.png` | Per-clip residual vs linear only |
| `13_multiseq_transfer_bar.png` | Transfer-only residual vs linear |
| `14_sample_frame.jpg` | Raw SportsMOT frame for mood / hero crop |

## Priority if you only pick 5 images
1. `01_summary_sam_vs_augmented.png`
2. `10_forecast_qualitative_clean.png`
3. `05_lstm_forecast_comparison.png`
4. `08_a2_per_rule_delta_ade.png`
5. `12_multiseq_train_vs_transfer.png`

## Regenerate
```powershell
py scripts/plot_pre_lstm_gauge.py --dataset sportsmot_example
py scripts/plot_lstm_vs_baselines.py --dataset sportsmot_example
py scripts/plot_multiseq_transfer.py
py scripts/plot_forecast_qualitative.py --dataset sportsmot_example --seed-id offset_2s
py scripts/plot_forecast_qualitative.py --dataset sportsmot_example --seed-id offset_4s --window-starts 4,11,22 --output data/runs/sportsmot_example/figures/forecast_qualitative_clean.png
py utils/visualize.py --frames data/runs/sportsmot_example/frames --baseline data/runs/sportsmot_example/baseline_tracks.json --augmented data/runs/sportsmot_example/ablations/sanitize_plus_velocity_cap/augmented_tracks.json --output data/runs/sportsmot_example/figures/summary_figure.png --summary --n-frames 4
```
