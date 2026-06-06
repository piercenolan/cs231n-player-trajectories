#!/usr/bin/env python3
"""
Regenerate CS231N generative-AI attribution artifacts:

  docs/ai_manifest.json
  docs/AI_ARTIFACT_REGISTRY.md
  docs/AI_PROMPT_INDEX.md

Also refreshes headers in docs/GENERATIVE_AI_USE.md (inventory sections only).
Run after code changes or transcript updates:

  py scripts/export_conversation_transcript.py
  py scripts/generate_ai_attribution_docs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_export_spec = importlib.util.spec_from_file_location(
    "export_conversation_transcript",
    ROOT / "scripts" / "export_conversation_transcript.py",
)
_export_mod = importlib.util.module_from_spec(_export_spec)
_export_spec.loader.exec_module(_export_mod)
DEFAULT_JSONL = _export_mod.DEFAULT_JSONL
load_turns = _export_mod.load_turns
published_turns = _export_mod.published_turns

DOCS = ROOT / "docs"
MANIFEST_PATH = DOCS / "ai_manifest.json"
REGISTRY_PATH = DOCS / "AI_ARTIFACT_REGISTRY.md"
PROMPT_INDEX_PATH = DOCS / "AI_PROMPT_INDEX.md"
HUB_PATH = DOCS / "GENERATIVE_AI_USE.md"

TOOL = "Cursor Agent (Cursor IDE)"
TOOL_VERSION_NOTE = "Cursor IDE 3.6.31+; underlying LLM model not logged per turn in repo"

CODE_DIRS = ("scripts", "utils", "models")
CODE_ATTRIBUTION_MARKERS = (
    "scripts/CODE_AI_ATTRIBUTION.md",
    "utils/CODE_AI_ATTRIBUTION.md",
    "models/CODE_AI_ATTRIBUTION.md",
)
DATA_ATTRIBUTION_MARKERS = ("data/runs/FIGURES_AI_ATTRIBUTION.md",)
README_FILES = ("README.md", "GENERATIVE_AI_USE.md")

FIGURES = [
    {
        "path": "data/runs/sportsmot_example/figures/summary_figure.png",
        "script": "utils/visualize.py",
        "description": "SAM3.1 raw vs augmentation side-by-side",
    },
    {
        "path": "data/runs/sportsmot_example/figures/baseline_metrics.png",
        "script": "scripts/plot_pre_lstm_gauge.py",
        "description": "Pre-LSTM tracking continuity gauges",
    },
    {
        "path": "data/runs/sportsmot_example/figures/ablation_ade_bar.png",
        "script": "scripts/run_ablations.py / utils/metrics.py",
        "description": "Augmentation ablation detection ADE",
    },
    {
        "path": "data/runs/sportsmot_example/figures/lstm_comparison.png",
        "script": "scripts/plot_lstm_vs_baselines.py",
        "description": "Forecast ADE all seeds vs clean seeds",
    },
    {
        "path": "data/runs/sportsmot_example/figures/lstm_ade_bar.png",
        "script": "scripts/plot_lstm_vs_baselines.py",
        "description": "LSTM variant ADE bar chart",
    },
    {
        "path": "data/runs/sportsmot_example/figures/lstm_rule_ablation_bar.png",
        "script": "scripts/eval_lstm_ablations.py",
        "description": "A0–A3 forecast ablation bars",
    },
    {
        "path": "data/runs/sportsmot_example/figures/lstm_per_rule_delta_ade.png",
        "script": "scripts/eval_lstm_ablations.py",
        "description": "Per-rule post-refine ΔADE",
    },
    {
        "path": "data/runs/sportsmot_example/figures/multi_seed_ade_bar.png",
        "script": "scripts/plot_pre_lstm_gauge.py",
        "description": "Multi-seed detection ADE",
    },
    {
        "path": "data/runs/sportsmot_example/figures/forecast_qualitative.png",
        "script": "scripts/plot_forecast_qualitative.py",
        "description": "LSTM forecast vs GT trajectory overlays",
    },
    {
        "path": "data/runs/figures/multiseq_perclip_bar.png",
        "script": "scripts/plot_multiseq_transfer.py",
        "description": "Per-clip residual vs linear",
    },
    {
        "path": "data/runs/figures/multiseq_train_vs_transfer.png",
        "script": "scripts/plot_multiseq_transfer.py",
        "description": "Per-clip train vs transfer baseline",
    },
    {
        "path": "data/runs/figures/multiseq_transfer_bar.png",
        "script": "scripts/plot_multiseq_transfer.py",
        "description": "Transfer eval bar chart",
    },
]

PER_CLIP_FIGURE_GLOB = "data/runs/sportsmot_v_*/figures/lstm_rule_ablation_bar*.png"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_code_artifacts() -> list[dict]:
    items = []
    for dirname in CODE_DIRS:
        base = ROOT / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            items.append(
                {
                    "path": rel(path),
                    "category": "code",
                    "ai_generated": True,
                    "tool": TOOL,
                    "notes": "Drafted/refactored via Cursor Agent; reviewed and executed by project authors.",
                }
            )
    return items


def collect_doc_artifacts() -> list[dict]:
    items = []
    seen: set[str] = set()
    for path in sorted(DOCS.glob("*.md")):
        rp = rel(path)
        if rp in seen:
            continue
        seen.add(rp)
        items.append(
            {
                "path": rp,
                "category": "documentation",
                "ai_generated": True,
                "tool": TOOL,
                "notes": "AI-assisted drafting; authors responsible for factual claims.",
            }
        )
    for marker in CODE_ATTRIBUTION_MARKERS + DATA_ATTRIBUTION_MARKERS:
        path = ROOT / marker
        if path.is_file() and marker not in seen:
            seen.add(marker)
            items.append(
                {
                    "path": marker,
                    "category": "documentation",
                    "ai_generated": True,
                    "tool": TOOL,
                    "notes": "Directory-level AI attribution marker.",
                }
            )
    for name in README_FILES:
        path = ROOT / name
        if path.is_file():
            items.append(
                {
                    "path": rel(path),
                    "category": "documentation",
                    "ai_generated": True,
                    "tool": TOOL,
                    "notes": "AI-assisted drafting; authors responsible for factual claims.",
                }
            )
    return items


def collect_figure_artifacts() -> list[dict]:
    items = []
    for fig in FIGURES:
        path = ROOT / fig["path"]
        items.append(
            {
                "path": fig["path"],
                "category": "figure",
                "ai_generated": True,
                "tool": TOOL,
                "generator": fig["script"],
                "description": fig["description"],
                "on_disk": path.is_file(),
                "notes": "Rendered by AI-assisted script from experimental outputs.",
            }
        )
    for path in sorted(ROOT.glob(PER_CLIP_FIGURE_GLOB)):
        items.append(
            {
                "path": rel(path),
                "category": "figure",
                "ai_generated": True,
                "tool": TOOL,
                "generator": "scripts/eval_lstm_ablations.py",
                "description": "Per-clip LSTM rule ablation bar",
                "on_disk": True,
                "notes": "Rendered by AI-assisted script from experimental outputs.",
            }
        )
    return items


def collect_excluded_artifacts() -> list[dict]:
    """Artifacts explicitly NOT generative-AI outputs (listed for clarity)."""
    patterns = [
        ("data/runs/**/checkpoint.pt", "model_weights", "PyTorch checkpoints trained on SportsMOT-derived tensors"),
        ("data/runs/**/*.csv", "experiment_table", "Metrics tables from pipeline runs"),
        ("data/runs/**/baseline_tracks.json", "experiment_data", "SAM3.1 tracking output"),
        ("data/runs/**/augmented_tracks.json", "experiment_data", "Rule-augmented tracks"),
        ("data/runs/**/trajectory_tensors.json", "experiment_data", "LSTM training tensors"),
        ("data/runs/**/predicted_*.json", "experiment_data", "LSTM rollout predictions"),
        ("data/datasets/**", "external_dataset", "SportsMOT frames and ground truth"),
    ]
    items = []
    for pattern, category, notes in patterns:
        items.append(
            {
                "glob": pattern,
                "category": category,
                "ai_generated": False,
                "notes": notes,
            }
        )
    return items


def build_manifest() -> dict:
    code = collect_code_artifacts()
    docs = collect_doc_artifacts()
    figures = collect_figure_artifacts()
    excluded = collect_excluded_artifacts()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": TOOL,
        "tool_version_note": TOOL_VERSION_NOTE,
        "policy": "CS231N Generative AI Use Policy — all AI-assisted artifacts listed; authors responsible for verification.",
        "summary": {
            "ai_generated_code_files": len(code),
            "ai_generated_doc_files": len(docs),
            "ai_generated_figures": len(figures),
            "excluded_glob_patterns": len(excluded),
        },
        "artifacts": {
            "code": code,
            "documentation": docs,
            "figures": figures,
        },
        "not_ai_generated": excluded,
        "transcripts": {
            "canonical": "docs/CONVERSATION_TRANSCRIPT.md",
            "raw_jsonl": str(DEFAULT_JSONL),
        },
        "plans": {
            "consolidated": "docs/AI_PLANS.md",
            "living_plan": "docs/PROJECT_PLAN.md",
        },
    }


def write_registry(manifest: dict) -> None:
    lines = [
        "# AI artifact registry",
        "",
        "> **AI-generated documentation.** Machine-maintained inventory for CS231N generative-AI attribution.",
        f"> Regenerate: `py scripts/generate_ai_attribution_docs.py` · Last updated: {manifest['generated_at_utc']}",
        "",
        "See [GENERATIVE_AI_USE.md](GENERATIVE_AI_USE.md) for the full attribution hub.",
        "",
        f"**Tool:** {manifest['tool']}",
        "",
        "## Summary",
        "",
        f"| Category | Count |",
        f"|----------|------:|",
        f"| Python code (`scripts/`, `utils/`, `models/`) | {manifest['summary']['ai_generated_code_files']} |",
        f"| Documentation (Markdown) | {manifest['summary']['ai_generated_doc_files']} |",
        f"| Figures (PNG, AI-assisted scripts) | {manifest['summary']['ai_generated_figures']} |",
        "",
        "## Code (AI-generated)",
        "",
        "| Path | Notes |",
        "|------|-------|",
    ]
    for item in manifest["artifacts"]["code"]:
        lines.append(f"| `{item['path']}` | {item['notes']} |")

    lines += ["", "## Documentation (AI-assisted)", "", "| Path | Notes |", "|------|-------|"]
    for item in manifest["artifacts"]["documentation"]:
        lines.append(f"| `{item['path']}` | {item['notes']} |")

    lines += [
        "",
        "## Figures (AI-assisted rendering scripts)",
        "",
        "| Path | Generator | Description | On disk |",
        "|------|-----------|-------------|---------|",
    ]
    for item in manifest["artifacts"]["figures"]:
        on_disk = "yes" if item.get("on_disk") else "no"
        lines.append(
            f"| `{item['path']}` | `{item.get('generator', '')}` | {item.get('description', '')} | {on_disk} |"
        )

    lines += [
        "",
        "## Not generative-AI artifacts (experimental / external)",
        "",
        "These outputs are produced by running the pipeline on data; they are **not** treated as generative-AI authorship.",
        "",
        "| Glob pattern | Category | Notes |",
        "|--------------|----------|-------|",
    ]
    for item in manifest["not_ai_generated"]:
        lines.append(f"| `{item['glob']}` | {item['category']} | {item['notes']} |")

    lines.append("")
    REGISTRY_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_prompt_index(jsonl_path: Path) -> int:
    if not jsonl_path.is_file():
        PROMPT_INDEX_PATH.write_text(
            "# AI prompt index\n\n_(Transcript JSONL missing — run export first.)_\n",
            encoding="utf-8",
        )
        return 0

    turns = published_turns(load_turns(jsonl_path))
    lines = [
        "# AI prompt index",
        "",
        "> **AI-generated documentation.** User prompts from Cursor Agent sessions (CS231N project).",
        f"> Source: `{jsonl_path}` · Regenerate: `py scripts/generate_ai_attribution_docs.py`",
        "",
        "Full responses and tool traces: [CONVERSATION_TRANSCRIPT.md](CONVERSATION_TRANSCRIPT.md).",
        "",
        "| Turn | Prompt (excerpt) |",
        "|-----:|------------------|",
    ]
    for i, turn in enumerate(turns, start=1):
        user = (turn.get("user") or "").strip().replace("\n", " ")
        if not user:
            user = "_(empty)_"
        if len(user) > 160:
            user = user[:157] + "..."
        user = user.replace("|", "\\|")
        lines.append(f"| {i} | {user} |")

    lines += [
        "",
        "## Representative prompts by topic",
        "",
        "| Topic | Turn(s) |",
        "|-------|---------|",
        "| Initial SAM3 tracking script | 1 |",
        "| Metrics / augmentation / visualize | ~10–25 |",
        "| SAM3.1 Multiplex migration | ~30–40 |",
        "| LSTM pipeline (A0–A3, residual) | ~50–80 |",
        "| Multi-seed Modal sprint | ~90–110 |",
        "| Forecast qualitative overlays / AI attribution | ~130+ |",
        "",
    ]
    PROMPT_INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    return len(turns)


def write_hub(manifest: dict, n_prompts: int) -> None:
    """Write GENERATIVE_AI_USE.md hub (not the final report AI statement)."""
    body = f"""# Generative AI use — attribution hub

> **AI-assisted documentation.** Index for CS231N generative-AI policy compliance within this repository.
> The **formal AI use statement** for the course final report is written **outside** this repo by the authors.

## Tool

| Field | Value |
|-------|-------|
| Product | **{manifest['tool']}** |
| Version note | {manifest['tool_version_note']} |
| Sessions | May–June 2026 (see transcripts) |
| Authors' responsibility | All artifacts were reviewed, executed, and verified by project members before submission. |

## Required materials (in this repo)

| Material | Location | Purpose |
|----------|----------|---------|
| **Transcripts** | [AI_TRANSCRIPTS.md](AI_TRANSCRIPTS.md) | Canonical vs raw chat exports |
| **Prompts** | [AI_PROMPT_INDEX.md](AI_PROMPT_INDEX.md) | {n_prompts} indexed user prompts |
| **Plans** | [AI_PLANS.md](AI_PLANS.md) | Phased work plans (AI-assisted) |
| **Artifact registry** | [AI_ARTIFACT_REGISTRY.md](AI_ARTIFACT_REGISTRY.md) | Every AI-generated / AI-assisted file |
| **Figure attribution** | [AI_FIGURES.md](AI_FIGURES.md) | PNG figures + generator scripts |
| **Machine manifest** | [ai_manifest.json](ai_manifest.json) | JSON inventory for tooling |

## Quick counts (auto-generated)

- **{manifest['summary']['ai_generated_code_files']}** Python modules (`scripts/`, `utils/`, `models/`)
- **{manifest['summary']['ai_generated_doc_files']}** Markdown docs (including this file)
- **{manifest['summary']['ai_generated_figures']}** figure paths (AI-assisted plot scripts)

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

Last manifest update: **{manifest['generated_at_utc']}**
"""
    HUB_PATH.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate AI attribution documentation")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    args = parser.parse_args()

    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_registry(manifest)
    n_prompts = write_prompt_index(Path(args.jsonl))
    write_hub(manifest, n_prompts)

    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {REGISTRY_PATH}")
    print(f"Wrote {PROMPT_INDEX_PATH} ({n_prompts} prompts)")
    print(f"Wrote {HUB_PATH}")


if __name__ == "__main__":
    main()
