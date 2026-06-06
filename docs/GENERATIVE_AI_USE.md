# Generative AI use — attribution hub

> **AI-assisted documentation.** Index for CS231N generative-AI policy compliance within this repository.
> The **formal AI use statement** for the course final report is written **outside** this repo by the authors.

## Tool

| Field | Value |
|-------|-------|
| Product | **Cursor Agent (Cursor IDE)** |
| Version note | Cursor IDE 3.6.31+; underlying LLM model not logged per turn in repo |
| Sessions | May–June 2026 (see transcripts) |
| Authors' responsibility | All artifacts were reviewed, executed, and verified by project members before submission. |

## Required materials (in this repo)

| Material | Location | Purpose |
|----------|----------|---------|
| **Transcripts** | [AI_TRANSCRIPTS.md](AI_TRANSCRIPTS.md) | Canonical vs raw chat exports |
| **Prompts** | [AI_PROMPT_INDEX.md](AI_PROMPT_INDEX.md) | 148 indexed user prompts |
| **Plans** | [AI_PLANS.md](AI_PLANS.md) | Phased work plans (AI-assisted) |
| **Artifact registry** | [AI_ARTIFACT_REGISTRY.md](AI_ARTIFACT_REGISTRY.md) | Every AI-generated / AI-assisted file |
| **Figure attribution** | [AI_FIGURES.md](AI_FIGURES.md) | PNG figures + generator scripts |
| **Machine manifest** | [ai_manifest.json](ai_manifest.json) | JSON inventory for tooling |

## Quick counts (auto-generated)

- **50** Python modules (`scripts/`, `utils/`, `models/`)
- **21** Markdown docs (including this file)
- **16** figure paths (AI-assisted plot scripts)

## What is NOT generative-AI authorship

Experimental outputs from running the pipeline are **not** listed as AI-generated artifacts:

- LSTM / SAM **checkpoints** (`checkpoint.pt`)
- **Tracks**, **tensors**, **CSVs**, **metrics JSON** under `data/runs/`
- **SportsMOT** frames and GT under `data/datasets/`

See `not_ai_generated` in [ai_manifest.json](ai_manifest.json).

## Regenerate attribution docs

```powershell
py scripts/export_conversation_transcript.py
py scripts/generate_ai_attribution_docs.py
```

Last manifest update: **2026-06-05T22:28:27Z**
