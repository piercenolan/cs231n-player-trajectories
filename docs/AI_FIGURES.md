# AI-assisted figures

> **AI-generated documentation.** Every report figure produced by AI-assisted scripts is listed here.

Figures are **rendered from experimental outputs** (tracks, tensors, metrics). The PNG pixels are not LLM-generated; the **plotting code** was AI-assisted via Cursor Agent.

## Primary clip (`sportsmot_example`)

| Figure | Generator | Description |
|--------|-----------|-------------|
| `data/runs/sportsmot_example/figures/summary_figure.png` | `utils/visualize.py --summary` | SAM3.1 raw vs augmentation |
| `data/runs/sportsmot_example/figures/baseline_metrics.png` | `scripts/plot_pre_lstm_gauge.py` | Players/frame + track streak |
| `data/runs/sportsmot_example/figures/ablation_ade_bar.png` | `scripts/run_ablations.py` | Augmentation detection ADE |
| `data/runs/sportsmot_example/figures/multi_seed_ade_bar.png` | `scripts/plot_pre_lstm_gauge.py` | Multi-seed detection ADE |
| `data/runs/sportsmot_example/figures/lstm_comparison.png` | `scripts/plot_lstm_vs_baselines.py` | Forecast ADE all vs clean seeds |
| `data/runs/sportsmot_example/figures/lstm_ade_bar.png` | `scripts/plot_lstm_vs_baselines.py` | LSTM variant bars |
| `data/runs/sportsmot_example/figures/lstm_rule_ablation_bar.png` | `scripts/eval_lstm_ablations.py` | A0–A3 ablation |
| `data/runs/sportsmot_example/figures/lstm_per_rule_delta_ade.png` | `scripts/eval_lstm_ablations.py` | A2 per-rule ΔADE |
| `data/runs/sportsmot_example/figures/forecast_qualitative.png` | `scripts/plot_forecast_qualitative.py` | LSTM vs GT trajectory overlays |

## Multi-clip

| Figure | Generator | Description |
|--------|-----------|-------------|
| `data/runs/figures/multiseq_perclip_bar.png` | `scripts/plot_multiseq_transfer.py` | Per-clip residual vs linear |
| `data/runs/figures/multiseq_train_vs_transfer.png` | `scripts/plot_multiseq_transfer.py` | Train vs transfer |
| `data/runs/figures/multiseq_transfer_bar.png` | `scripts/plot_multiseq_transfer.py` | Transfer baseline |

## Per-clip LSTM bars

| Pattern | Generator |
|---------|-----------|
| `data/runs/sportsmot_v_*/figures/lstm_rule_ablation_bar*.png` | `scripts/eval_lstm_ablations.py` |

## Regenerate commands

```powershell
py utils/visualize.py --frames data/runs/sportsmot_example/frames --baseline data/runs/sportsmot_example/baseline_tracks.json --augmented data/runs/sportsmot_example/ablations/sanitize_plus_velocity_cap/augmented_tracks.json --output data/runs/sportsmot_example/figures/summary_figure.png --summary --n-frames 4
py scripts/plot_pre_lstm_gauge.py --dataset sportsmot_example
py scripts/plot_lstm_vs_baselines.py --dataset sportsmot_example
py scripts/plot_multiseq_transfer.py
py scripts/plot_forecast_qualitative.py --dataset sportsmot_example --seed-id offset_0s
```

## Hub

Full file-level registry: **[AI_ARTIFACT_REGISTRY.md](AI_ARTIFACT_REGISTRY.md)** · **[GENERATIVE_AI_USE.md](GENERATIVE_AI_USE.md)**
