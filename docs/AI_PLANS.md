# AI-assisted project plans

> **AI-generated documentation.** Work plans drafted with Cursor Agent; executed and verified by project authors.

## Consolidated living plan (authoritative)

| Document | Role |
|----------|------|
| **[PROJECT_PLAN.md](PROJECT_PLAN.md)** | Phase 1–4 pipeline: SAM3.1 → augmentation → LSTM → multi-seed eval |
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | Report-ready status, headline numbers, figure pointers |
| **[DEFERRED_MULTISEQ.md](DEFERRED_MULTISEQ.md)** | Deferred pooled multi-clip training scope |
| **[MODAL_SPRINT_RUNBOOK.md](MODAL_SPRINT_RUNBOOK.md)** | Modal GPU batch commands (external terminal) |
| **[MULTI_SEED_COMMANDS.md](MULTI_SEED_COMMANDS.md)** | 12-seed orchestration copy-paste |

## Plan history (from agent sessions)

The following plans were created in Cursor Agent chat and consolidated into `PROJECT_PLAN.md` (not all original `.plan.md` files are committed):

| Plan topic | Evidence in transcript | Consolidated in |
|------------|------------------------|-----------------|
| SAM3 baseline + grid tracking | Turn 1 | `PROJECT_PLAN.md` Phase 1 |
| Augmentation layer + pre-LSTM validation | Turns ~15–30 | Phase 2–3 |
| SAM3.1 Multiplex migration | Turns ~30–45 | Phase 1, `run_sam3.py` |
| Rule-aware LSTM A0–A3 + residual | Turns ~50–90 | Phase 4 |
| Beat-linear / rule attribution sprint | Turns ~80–100 | `eval_lstm_ablations.py` |
| Multi-clip Modal sprint (4 basketball clips) | Turns ~100–120 | `MODAL_SPRINT_RUNBOOK.md` |
| Per-clip retrain + transfer eval | Turns ~120–132 | `PROJECT_STATUS.md`, `PAPER_RESULTS.md` |
| Forecast qualitative overlays | Latest sessions | `plot_forecast_qualitative.py` |

Cursor-native plan files (if present locally) live under:

```
%USERPROFILE%\.cursor\plans\
```

Example referenced in chat: `lstm_beat_linear_plan_02c4b5ba.plan.md` — content merged into Phase 4 of `PROJECT_PLAN.md`.

## Milestone checklist

Report one-pager: **[MILESTONE_CHECKLIST.md](MILESTONE_CHECKLIST.md)**

## Hub

**[GENERATIVE_AI_USE.md](GENERATIVE_AI_USE.md)**
