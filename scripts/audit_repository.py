#!/usr/bin/env python3
"""Validate repo layout, attribution coverage, and pipeline hygiene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_gen_spec = importlib.util.spec_from_file_location(
    "generate_ai_attribution_docs",
    ROOT / "scripts" / "generate_ai_attribution_docs.py",
)
_gen = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen)

CODE_DIRS = _gen.CODE_DIRS
LEGACY_PATHS = (
    "data/outputs",
    "data/frames",
    "data/frames_dir",
    "data/videos/video_1.mp4",
    "data/gt/sportsmot/video_1",
    "tracks.json",
    "output.log",
    "CURSOR_TRANSCRIPT.md",
)
DUPLICATE_FIGURE_SUFFIXES = ("-LaptopStudio", "-Copy", " (1)")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def discover_python_modules() -> set[str]:
    found: set[str] = set()
    for dirname in CODE_DIRS:
        base = ROOT / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            found.add(rel(path))
    return found


def discover_docs_md() -> set[str]:
    docs = {rel(p) for p in (ROOT / "docs").glob("*.md")}
    for name in ("README.md", "GENERATIVE_AI_USE.md", "CONTEXT.md"):
        path = ROOT / name
        if path.is_file():
            docs.add(name)
    for marker in (
        "scripts/CODE_AI_ATTRIBUTION.md",
        "utils/CODE_AI_ATTRIBUTION.md",
        "models/CODE_AI_ATTRIBUTION.md",
        "data/runs/FIGURES_AI_ATTRIBUTION.md",
    ):
        if (ROOT / marker).is_file():
            docs.add(marker)
    generated = ROOT / "data/runs/sportsmot_example/figures/PRE_LSTM_GAUGE.md"
    if generated.is_file():
        docs.add(rel(generated))
    return docs


def discover_figures() -> set[str]:
    figures: set[str] = set()
    for path in (ROOT / "data" / "runs").rglob("*.png"):
        name = path.name
        if any(suffix in name for suffix in DUPLICATE_FIGURE_SUFFIXES):
            continue
        figures.add(rel(path))
    return figures


def manifest_paths(manifest: dict) -> dict[str, set[str]]:
    code = {a["path"] for a in manifest["artifacts"]["code"]}
    docs = {a["path"] for a in manifest["artifacts"]["documentation"]}
    figures = {a["path"] for a in manifest["artifacts"]["figures"]}
    return {"code": code, "documentation": docs, "figures": figures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repository attribution and layout")
    parser.add_argument("--json", action="store_true", help="Print machine-readable report")
    args = parser.parse_args()

    manifest = _gen.build_manifest()
    listed = manifest_paths(manifest)
    py_on_disk = discover_python_modules()
    docs_on_disk = discover_docs_md()
    figures_on_disk = discover_figures()

    issues: list[str] = []
    warnings: list[str] = []

    for category, on_disk, in_manifest in (
        ("code", py_on_disk, listed["code"]),
        ("documentation", docs_on_disk, listed["documentation"]),
        ("figures", figures_on_disk, listed["figures"]),
    ):
        missing = sorted(on_disk - in_manifest)
        stale = sorted(in_manifest - on_disk)
        if missing:
            issues.append(f"{category}: {len(missing)} on disk but not in manifest: {missing[:5]}")
        if stale:
            warnings.append(f"{category}: {len(stale)} in manifest but missing on disk: {stale[:5]}")

    for legacy in LEGACY_PATHS:
        path = ROOT / legacy
        if path.exists():
            issues.append(f"legacy artifact still present: {legacy}")

    dupes = sorted(
        rel(p)
        for p in (ROOT / "data" / "runs").rglob("*.png")
        if any(suffix in p.name for suffix in DUPLICATE_FIGURE_SUFFIXES)
    )
    if dupes:
        issues.append(f"duplicate figure copies: {dupes}")

    report = {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "counts": {
            "python_modules": len(py_on_disk),
            "docs_markdown": len(docs_on_disk),
            "figures_png": len(figures_on_disk),
            "manifest_code": len(listed["code"]),
            "manifest_docs": len(listed["documentation"]),
            "manifest_figures": len(listed["figures"]),
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Repository audit")
        print("=" * 40)
        for key, val in report["counts"].items():
            print(f"  {key}: {val}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        if issues:
            print("\nIssues:")
            for i in issues:
                print(f"  - {i}")
            print("\nFAILED")
        else:
            print("\nPASSED")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
