#!/usr/bin/env python3
"""
Extract only basketball sequences from SportsMOT publish archive.

Uses the official splits_txt/basketball.txt list (from the zip, an extracted
tree, or Hugging Face). Does not extract football/volleyball clips.

Typical layout inside sportsmot_publish.zip:
  sportsmot_publish/dataset/{train,val,test}/<sequence>/img1|gt|seqinfo.ini
  sportsmot_publish/splits_txt/basketball.txt

Example:
  py scripts/extract_sportsmot_basketball.py --zip data/sportsmot_publish.zip
  py scripts/extract_sportsmot_basketball.py --source-dir data/sportsmot_publish --list-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "data" / "sportsmot_publish.zip"
DEFAULT_OUT = ROOT / "data" / "datasets" / "sportsmot_basketball"
HF_SPLITS_BASE = "https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main/splits_txt"
ZIP_ROOT_PREFIX = "sportsmot_publish/"
DATASET_PREFIX = f"{ZIP_ROOT_PREFIX}dataset/"
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Extract basketball-only SportsMOT data")
    p.add_argument(
        "--zip",
        type=Path,
        default=None,
        help=f"Path to sportsmot_publish.zip (default: {DEFAULT_ZIP})",
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Already-unzipped sportsmot_publish/ folder (skips zip)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--splits",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which dataset splits to extract (default: all)",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Include test split (no public GT; useful for frames only)",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap number of basketball sequences (debug)",
    )
    p.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Extract only these sequence IDs (must appear in basketball.txt)",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Print planned sequences and sizes; do not extract",
    )
    p.add_argument(
        "--use-tar",
        action="store_true",
        help="Use Windows tar for zip I/O (default when zipfile cannot open archive)",
    )
    p.add_argument(
        "--verify-in-zip",
        action="store_true",
        help="Scan full zip listing to verify members (slow; fails on truncated zips)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --list-only",
    )
    return p.parse_args()


def load_sequence_list(
    source_dir: Path | None,
    zip_path: Path | None,
    use_tar: bool,
) -> list[str]:
    """Load basketball sequence IDs from local tree, zip, or Hugging Face."""
    candidates = []
    if source_dir:
        for rel in (
            "splits_txt/basketball.txt",
            "sportsmot_publish/splits_txt/basketball.txt",
        ):
            p = source_dir / rel
            if p.is_file():
                candidates.append(p)
        if source_dir.name == "splits_txt":
            p = source_dir / "basketball.txt"
            if p.is_file():
                candidates.append(p)

    if zip_path and zip_path.is_file():
        for member in (
            "sportsmot_publish/splits_txt/basketball.txt",
            "splits_txt/basketball.txt",
        ):
            text = _read_text_from_zip(zip_path, member, use_tar)
            if text:
                return _parse_sequence_lines(text)

    for p in candidates:
        return _parse_sequence_lines(p.read_text(encoding="utf-8"))

    print("Fetching basketball.txt from Hugging Face...", file=sys.stderr)
    return _fetch_split_file("basketball.txt")


def _fetch_split_file(name: str) -> list[str]:
    url = f"{HF_SPLITS_BASE}/{name}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return _parse_sequence_lines(resp.read().decode("utf-8"))


def load_split_assignment(sequences: set[str]) -> dict[str, str]:
    """Map sequence_id -> train|val|test using official split lists."""
    assignment = {}
    for split in SPLITS:
        for seq in _fetch_split_file(f"{split}.txt"):
            if seq in sequences:
                assignment[seq] = split
    return assignment


def _parse_sequence_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _read_text_from_zip(zip_path: Path, member: str, use_tar: bool) -> str | None:
    if use_tar or not _zipfile_ok(zip_path):
        try:
            proc = subprocess.run(
                ["tar", "-xOf", str(zip_path), member],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
        except FileNotFoundError:
            pass
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.read(member).decode("utf-8")
    except (KeyError, zipfile.BadZipFile):
        return None


def _zipfile_ok(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False


def _iter_tar_members(zip_path: Path):
    proc = subprocess.run(
        ["tar", "-tf", str(zip_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"tar -tf failed: {proc.stderr.strip()}\n"
            "Zip may be incomplete — re-download sportsmot_publish.zip or use --source-dir."
        )
    for line in proc.stdout.splitlines():
        line = line.replace("\\", "/").strip()
        if line:
            yield line


def plan_sequences(
    sequences: list[str],
    split_assignment: dict[str, str],
    allowed_splits: set[str],
) -> dict:
    """Build extraction plan without scanning the zip."""
    found = {}
    for seq in sequences:
        split = split_assignment.get(seq)
        if not split or split not in allowed_splits:
            continue
        found[seq] = {
            "split": split,
            "prefix": f"{DATASET_PREFIX}{split}/{seq}",
            "members": [],
        }
    return found


def verify_seq_in_zip(zip_path: Path, prefix: str, use_tar: bool) -> bool:
    """Quick check: can we read seqinfo.ini from the archive?"""
    member = f"{prefix}/seqinfo.ini"
    if use_tar or not _zipfile_ok(zip_path):
        proc = subprocess.run(
            ["tar", "-xOf", str(zip_path), member],
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.read(member)
        return True
    except (KeyError, zipfile.BadZipFile):
        return False


def scan_zip_members(zip_path: Path, sequences: set[str], allowed_splits: set[str]) -> dict:
    """Optional full zip scan (slow; may fail if zip is truncated)."""
    found: dict[str, dict] = {}
    seq_re = re.compile(r"^sportsmot_publish/dataset/(train|val|test)/([^/]+)/")
    for member in _iter_tar_members(zip_path):
        m = seq_re.match(member)
        if not m:
            continue
        split, seq = m.group(1), m.group(2)
        if seq not in sequences or split not in allowed_splits:
            continue
        if seq not in found:
            found[seq] = {"split": split, "prefix": f"{DATASET_PREFIX}{split}/{seq}", "members": []}
        if member.endswith("/"):
            continue
        found[seq]["members"].append(member)
    return found


def find_sequences_in_tree(source_dir: Path, sequences: set[str], allowed_splits: set[str]) -> dict:
    root = source_dir
    if (source_dir / "dataset").is_dir():
        dataset_root = source_dir / "dataset"
    elif (source_dir / "sportsmot_publish" / "dataset").is_dir():
        dataset_root = source_dir / "sportsmot_publish" / "dataset"
    else:
        raise FileNotFoundError(
            f"No dataset/ under {source_dir}. Expected sportsmot_publish/dataset/{{train,val,test}}/."
        )

    found = {}
    for split in allowed_splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name not in sequences:
                continue
            members = []
            for f in seq_dir.rglob("*"):
                if f.is_file():
                    members.append(str(f.relative_to(source_dir)).replace("\\", "/"))
            found[seq_dir.name] = {"split": split, "members": members, "src_dir": seq_dir}
    return found


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def extract_from_tree(found: dict, source_dir: Path, out_dir: Path):
    for seq, info in sorted(found.items()):
        src = info.get("src_dir") or (source_dir / "dataset" / info["split"] / seq)
        if not src.is_dir() and (source_dir / "sportsmot_publish" / "dataset").is_dir():
            src = source_dir / "sportsmot_publish" / "dataset" / info["split"] / seq
        dest = out_dir / info["split"] / seq
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)


def extract_from_zip(zip_path: Path, found: dict, out_dir: Path, use_tar: bool):
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for seq, info in found.items():
        by_split[info["split"]].append(seq)

    for split, seqs in by_split.items():
        if not seqs:
            continue
        for seq in sorted(seqs):
            prefix = found[seq].get("prefix") or f"{DATASET_PREFIX}{split}/{seq}"
            dest_parent = out_dir / split
            dest_parent.mkdir(parents=True, exist_ok=True)
            tmp_extract = out_dir / "_tmp_extract" / ZIP_ROOT_PREFIX.rstrip("/")
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract.parent)
            tmp_extract.parent.mkdir(parents=True, exist_ok=True)

            cmd = ["tar", "-xf", str(zip_path), "-C", str(tmp_extract.parent), prefix]
            print(f"  extracting {split}/{seq} ...")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                err = proc.stderr.strip() or proc.stdout.strip()
                raise RuntimeError(
                    f"Failed to extract {prefix}:\n{err}\n"
                    "Zip archive may be truncated. Re-download or use --source-dir."
                )

            src = tmp_extract / "dataset" / split / seq
            dest = out_dir / split / seq
            if dest.exists():
                shutil.rmtree(dest)
            if src.is_dir():
                shutil.move(str(src), str(dest))

        if (out_dir / "_tmp_extract").exists():
            shutil.rmtree(out_dir / "_tmp_extract")


def write_manifest(out_dir: Path, sequences: list[str], found: dict, zip_path, source_dir):
    entries = []
    for seq in sequences:
        info = found.get(seq)
        if not info:
            entries.append({"sequence": seq, "status": "missing"})
            continue
        seq_dir = out_dir / info["split"] / seq
        n_frames = len(list((seq_dir / "img1").glob("*.jpg"))) if (seq_dir / "img1").is_dir() else 0
        entries.append(
            {
                "sequence": seq,
                "split": info["split"],
                "status": "ok",
                "path": str(seq_dir.relative_to(ROOT)).replace("\\", "/"),
                "num_frames": n_frames,
                "has_gt": (seq_dir / "gt" / "gt.txt").is_file(),
            }
        )
    manifest = {
        "sport": "basketball",
        "source_zip": str(zip_path) if zip_path else None,
        "source_dir": str(source_dir) if source_dir else None,
        "output": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "num_sequences": len(sequences),
        "num_extracted": sum(1 for e in entries if e.get("status") == "ok"),
        "sequences": entries,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    args = parse_args()
    list_only = args.list_only or args.dry_run

    zip_path = args.zip or (DEFAULT_ZIP if DEFAULT_ZIP.is_file() else None)
    source_dir = args.source_dir
    if not zip_path and not source_dir:
        raise SystemExit(
            "Provide --zip path/to/sportsmot_publish.zip or --source-dir path/to/sportsmot_publish"
        )

    allowed_splits = set(SPLITS)
    if args.splits != "all":
        allowed_splits = {args.splits}
    if not args.include_test and args.splits == "all":
        allowed_splits.discard("test")

    sequences = load_sequence_list(source_dir, zip_path, args.use_tar or (zip_path and not _zipfile_ok(zip_path)))
    if args.sequences:
        wanted = set(args.sequences)
        sequences = [s for s in sequences if s in wanted]
        missing_req = wanted - set(sequences)
        if missing_req:
            raise SystemExit(f"Unknown sequence IDs (not in basketball list): {sorted(missing_req)}")
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    print(f"Basketball sequences in list: {len(sequences)}")

    seq_set = set(sequences)
    split_assignment = load_split_assignment(seq_set)
    unassigned = [s for s in sequences if s not in split_assignment]
    if unassigned:
        print(f"Warning: {len(unassigned)} basketball sequences not in train/val/test lists.")

    if source_dir:
        source_dir = source_dir.resolve()
        found = find_sequences_in_tree(source_dir, seq_set, allowed_splits)
    else:
        zip_path = zip_path.resolve()
        if not zip_path.is_file():
            raise SystemExit(f"Zip not found: {zip_path}")
        use_tar = args.use_tar or not _zipfile_ok(zip_path)
        if use_tar:
            print("Using tar for zip extraction (zipfile central directory unavailable).")
        if args.verify_in_zip:
            print("Scanning full zip index (--verify-in-zip)...")
            found = scan_zip_members(zip_path, seq_set, allowed_splits)
        else:
            found = plan_sequences(sequences, split_assignment, allowed_splits)

    present = [s for s in sequences if s in found]
    excluded_test = [
        s for s in sequences if split_assignment.get(s) == "test" and "test" not in allowed_splits
    ]
    missing = [s for s in sequences if s not in found and s not in excluded_test]
    n_files = sum(len(found[s].get("members") or []) for s in present)

    by_split = {s: [] for s in SPLITS}
    for sid in present:
        by_split[found[sid]["split"]].append(sid)
    for sid in excluded_test:
        by_split["test"].append(sid)

    print("\nPlanned extraction:")
    for split in SPLITS:
        n = len(by_split[split])
        if n:
            note = " (excluded; use --include-test)" if split == "test" and "test" not in allowed_splits else ""
            print(f"  {split}: {n} sequences{note}")
    if n_files:
        print(f"  total files (indexed): {n_files}")
    if missing:
        print(f"  unassigned: {len(missing)} (first 5: {missing[:5]})")

    if list_only and not source_dir and zip_path and not args.verify_in_zip:
        print("\nZip scan skipped (use --verify-in-zip to probe archive).")
        print("Run without --list-only to extract via per-sequence tar paths.")
        return

    if list_only:
        return

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits_txt"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "basketball.txt").write_text("\n".join(sequences) + "\n", encoding="utf-8")

    print(f"\nExtracting to {out_dir} ...")
    if source_dir:
        extract_from_tree(found, source_dir, out_dir)
    else:
        extract_from_zip(zip_path, found, out_dir, args.use_tar or not _zipfile_ok(zip_path))

    manifest_path = write_manifest(out_dir, sequences, found, zip_path, source_dir)
    print(f"\nDone. {len(present)} sequences extracted.")
    print(f"Manifest: {manifest_path}")
    print(
        "\nNext: register a clip in utils/datasets.py or copy one sequence into "
        "data/datasets/sportsmot_example/ for the existing pipeline."
    )


if __name__ == "__main__":
    main()
