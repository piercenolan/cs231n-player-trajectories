# Modal sprint runbook (36h)

> **AI-assisted documentation (Cursor Agent).** Attribution: [GENERATIVE_AI_USE.md](GENERATIVE_AI_USE.md).

If extraction fails with `PermissionError` on Windows/OneDrive, use `--force` or delete the partial folder under `data/datasets/sportsmot_basketball/train/`.

Run these in an **external terminal** where `py -m modal` is authenticated (Cursor terminal may not connect).

## 1. Extract and register sequences

```powershell
cd C:\Users\63npi\OneDrive\Desktop\CS231N\cs231n-player-trajectories

py scripts/extract_sportsmot_basketball.py --zip data/sportsmot_publish.zip `
  --sequences v_-6Os86HzwCs_c001 v_-6Os86HzwCs_c003 v_00HRwkvvjtQ_c001

py scripts/register_sportsmot_sequence.py v_-6Os86HzwCs_c001
py scripts/register_sportsmot_sequence.py v_-6Os86HzwCs_c003
py scripts/register_sportsmot_sequence.py v_00HRwkvvjtQ_c001 --holdout
```

Dataset keys are written to `data/datasets/extra_datasets.json` (see printed names).

## 2. Modal batch (~30–60 min per dataset)

```powershell
py scripts/run_batch_sportsmot_modal.py --step-sec 2 --skip-existing `
  --datasets sportsmot_v_6os86hzwcs_c001 sportsmot_v_6os86hzwcs_c003 sportsmot_v_00hrwkvvjtq_c001
```

The first Modal job per dataset **uploads local frames** to the `sports-data` volume automatically. Re-deploy Modal after pulling dataset-registry fixes (`utils/datasets.py`, `scripts/run_modal.py`).

(~30–60 min per dataset; 3 clips ≈ 2–3 hours total.)

## 3. Tensors + transfer eval

```powershell
$CKPT = "data/runs/sportsmot_example/lstm/lstm_rule_features_residual/checkpoint.pt"

foreach ($ds in @("sportsmot_v_6os86hzwcs_c001", "sportsmot_v_6os86hzwcs_c003", "sportsmot_v_00hrwkvvjtq_c001")) {
  py scripts/export_lstm_tensors.py --dataset $ds --all-seeds --with-rule-features
  py scripts/eval_lstm_ablations.py --dataset $ds --all-seeds --skip-attribution `
    --a1-residual-checkpoint $CKPT
}

py scripts/aggregate_multiseq_eval.py
py scripts/generate_paper_results.py --multiseq-csv data/runs/multiseq_transfer_summary.csv
```

## 4. Refresh example clip artifacts

```powershell
py scripts/eval_lstm_ablations.py --dataset sportsmot_example --all-seeds --skip-attribution
py scripts/plot_lstm_vs_baselines.py
py scripts/plot_pre_lstm_gauge.py
py scripts/generate_paper_results.py --multiseq-csv data/runs/multiseq_transfer_summary.csv
```
