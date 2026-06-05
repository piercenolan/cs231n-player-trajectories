#!/usr/bin/env python3
"""Export Cursor agent JSONL transcript to readable Markdown."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = (
    Path.home()
    / ".cursor"
    / "projects"
    / "c-Users-63npi-OneDrive-Desktop-CS231N-cs231n-player-trajectories"
    / "agent-transcripts"
    / "c9cd51f6-def0-4582-b4bd-b9e6d7fde87d"
    / "c9cd51f6-def0-4582-b4bd-b9e6d7fde87d.jsonl"
)


def strip_user_query(text: str) -> str:
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def clean_assistant_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"`?\[REDACTED\]`?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_substantive(text: str) -> bool:
    t = clean_assistant_text(text)
    if not t:
        return False
    if len(t) < 40 and t.endswith("..."):
        return False
    return True


def summarize_tools(content: list) -> list[str]:
    names = []
    for part in content:
        if part.get("type") == "tool_use":
            name = part.get("name", "tool")
            if name not in names:
                names.append(name)
    return names


def load_turns(jsonl_path: Path) -> list[dict]:
    turns: list[dict] = []
    current: dict | None = None

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            role = row.get("role")
            content = row.get("message", {}).get("content", [])

            if role == "user":
                if current:
                    turns.append(current)
                user_text = ""
                for part in content:
                    if part.get("type") == "text":
                        user_text = strip_user_query(part.get("text", ""))
                        break
                current = {
                    "user": user_text,
                    "assistant_parts": [],
                    "tools": [],
                }
                continue

            if role == "assistant" and current is not None:
                for part in content:
                    if part.get("type") == "text":
                        cleaned = clean_assistant_text(part.get("text", ""))
                        if cleaned:
                            current["assistant_parts"].append(cleaned)
                for tool in summarize_tools(content):
                    if tool not in current["tools"]:
                        current["tools"].append(tool)

    if current:
        turns.append(current)
    return turns


def merge_assistant(parts: list[str]) -> str:
    """Keep unique paragraphs; prefer longer final summaries over repeated intros."""
    if not parts:
        return ""
    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        key = part[:200]
        if key in seen:
            continue
        seen.add(key)
        merged.append(part)
    if len(merged) > 1:
        # Often the last chunk is the user-facing summary; if it's long, use it plus prior unique blocks
        last = merged[-1]
        if len(last) > 400:
            body = merged[:-1]
            body = [b for b in body if len(b) > 80]
            if body:
                return "\n\n".join(body + [last])
    return "\n\n".join(merged)


def format_turn(idx: int, turn: dict) -> str:
    lines = [f"## Turn {idx}", ""]
    lines.append("### User")
    lines.append("")
    lines.append(turn["user"] or "_(empty message)_")
    lines.append("")

    assistant = merge_assistant(turn["assistant_parts"])
    if assistant:
        lines.append("### Assistant")
        lines.append("")
        lines.append(assistant)
        lines.append("")

    if turn["tools"]:
        tool_note = ", ".join(turn["tools"][:12])
        if len(turn["tools"]) > 12:
            tool_note += f", … (+{len(turn['tools']) - 12} more)"
        lines.append(f"_Tools used: {tool_note}_")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Export agent JSONL to Markdown transcript")
    p.add_argument("--input", default=str(DEFAULT_JSONL))
    p.add_argument(
        "--output",
        default=str(ROOT / "docs" / "CONVERSATION_TRANSCRIPT.md"),
    )
    args = p.parse_args()

    jsonl_path = Path(args.input)
    if not jsonl_path.is_file():
        print(f"Missing transcript: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    turns = load_turns(jsonl_path)
    substantive = sum(1 for t in turns if t["user"] or merge_assistant(t["assistant_parts"]))

    header = [
        "# CS231N Project — Agent Conversation Transcript",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Source:** `{jsonl_path}`",
        f"**Turns:** {len(turns)} user messages ({substantive} with content)",
        "",
        "This is a cleaned export of the Cursor Agent chat. Internal reasoning marked",
        "`[REDACTED]` in the source file is omitted. Tool call details are summarized, not full outputs.",
        "",
        "---",
        "",
    ]

    body = []
    for i, turn in enumerate(turns, start=1):
        if not turn["user"] and not merge_assistant(turn["assistant_parts"]):
            continue
        body.append(format_turn(i, turn))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + body), encoding="utf-8")
    print(f"Wrote {out_path} ({len(turns)} turns, {out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
