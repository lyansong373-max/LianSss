#!/usr/bin/env python3
"""Make first/last-frame contact sheets for human rollout review."""

import argparse
import csv
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]


def frame(path: Path, seconds: float) -> Image.Image | None:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(seconds),
             "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            capture_output=True, check=False,
        )
        if proc.returncode or not proc.stdout:
            return None
        return Image.open(BytesIO(proc.stdout)).convert("RGB").resize((400, 300))
    except (OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=ROOT)
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in csv.DictReader(args.audit.open()):
        if row["status"].startswith("technical_pass"):
            groups[(row["model"], row["task"])].append(row)
    chosen = [
        max(rows, key=lambda row: (float(row["seconds_min"]), float(row["arm_span_rad"])))
        for rows in groups.values()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in sorted({row["model"] for row in chosen}):
        model_rows = sorted((row for row in chosen if row["model"] == model), key=lambda row: row["task"])
        sheet = Image.new("RGB", (800, 330 * len(model_rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(model_rows):
            duration = float(row["seconds_min"])
            path = args.archive_root / row["camera1"]
            beginning = frame(path, 1.0)
            end = frame(path, max(1.0, duration - 1.0))
            for x, image in ((0, beginning), (400, end)):
                if image is not None:
                    sheet.paste(image, (x, index * 330 + 30))
            draw.text((8, index * 330 + 7), f"{row['task']}  {row['run'].split('/')[-1]}  beginning / end", fill="black")
        output = args.output_dir / f"{model}_candidates.png"
        sheet.save(output)
        print(output)


if __name__ == "__main__":
    main()
