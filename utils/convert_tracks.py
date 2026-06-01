"""
Convert alternate tracks.json schemas to the pipeline canonical format.

Canonical format (run_sam3.py / augmentation / metrics / visualize):
  - frame_number (not "frame")
  - each player: id, bbox, mask_center
"""

import argparse
import json
from pathlib import Path


def normalize_player(player):
    """Normalize one player dict; derive mask_center from bbox if missing."""
    bbox = dict(player.get("bbox", {}))
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    w = float(bbox.get("w", 0))
    h = float(bbox.get("h", 0))

    if "mask_center" in player:
        mc = player["mask_center"]
        mask_center = {
            "x": round(float(mc.get("x", x + w / 2.0)), 1),
            "y": round(float(mc.get("y", y + h / 2.0)), 1),
        }
    else:
        mask_center = {
            "x": round(x + w / 2.0, 1),
            "y": round(y + h / 2.0, 1),
        }

    out = {
        "id": int(player["id"]),
        "bbox": {
            "x": int(round(x)),
            "y": int(round(y)),
            "w": int(round(w)),
            "h": int(round(h)),
        },
        "mask_center": mask_center,
    }
    if player.get("predicted"):
        out["predicted"] = True
    return out


def normalize_tracks(data):
    """Return canonical tracks dict from raw or partially canonical input."""
    frames_out = []
    for frame in data.get("frames", []):
        if "frame_number" in frame:
            fnum = int(frame["frame_number"])
        elif "frame" in frame:
            fnum = int(frame["frame"])
        else:
            raise ValueError("Each frame must have 'frame_number' or 'frame'")

        players = [normalize_player(p) for p in frame.get("players", [])]
        frames_out.append({"frame_number": fnum, "players": players})

    frames_out.sort(key=lambda f: f["frame_number"])
    return {"frames": frames_out}


def convert_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    normalized = normalize_tracks(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)

    n_frames = len(normalized["frames"])
    n_players = sum(len(fr["players"]) for fr in normalized["frames"])
    print(f"Converted {input_path} -> {output_path}")
    print(f"  Frames: {n_frames}, total player entries: {n_players}")
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Normalize tracks.json for the project pipeline.")
    parser.add_argument("--input", required=True, help="Input tracks.json path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite input)",
    )
    args = parser.parse_args()
    out = args.output or args.input
    convert_file(args.input, out)


if __name__ == "__main__":
    main()
