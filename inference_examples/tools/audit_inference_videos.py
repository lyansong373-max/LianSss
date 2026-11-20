#!/usr/bin/env python3
"""Read-only inventory of real-robot rollout videos; never controls hardware.

This checks recording integrity and measured arm motion, not grasp success.
Run with the LeRobot Python environment (pyarrow required). Output is CSV.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("camera1", "camera2", "camera3")
FIELDS = (
    "model", "task", "run", "status", "reasons", "frames", "seconds_min",
    "arm_span_rad", "camera1", "camera2", "camera3",
)


def probe(path: Path) -> float | None:
    if not path.is_file() or path.stat().st_size < 1024:
        return None
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=False,
    )
    if proc.returncode:
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return None


def measured_arm_span(paths: list[Path]) -> tuple[int, float]:
    states = []
    frames = 0
    for path in paths:
        table = pq.read_table(path, columns=["observation.state"])
        frames += table.num_rows
        states.extend(table["observation.state"].to_pylist())
    if not states:
        return frames, 0.0
    # The 16D order is left 7 joints + gripper, right 7 joints + gripper.
    joints = [i for i in range(16) if i not in (7, 15)]
    span = max(max(s[i] for s in states) - min(s[i] for s in states) for i in joints)
    return frames, span


def audit_run(model: str, dataset: Path, archive_root: Path) -> dict[str, str]:
    run = dataset.parent
    reasons = []
    videos = {}
    durations = []
    for camera in CAMERAS:
        matches = sorted((dataset / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
        if len(matches) != 1:
            reasons.append(f"{camera}_count_{len(matches)}")
        if matches:
            duration = probe(matches[0])
            if duration is None:
                reasons.append(f"{camera}_unreadable")
            else:
                durations.append(duration)
            videos[camera] = str(matches[0].relative_to(archive_root))
        else:
            videos[camera] = ""
    data_files = sorted((dataset / "data").rglob("*.parquet"))
    try:
        frames, span = measured_arm_span(data_files) if data_files else (0, 0.0)
    except (OSError, ValueError, KeyError):
        frames, span = 0, 0.0
        reasons.append("data_unreadable")
    if frames < 300:
        reasons.append("under_300_frames")
    if span < 0.05:
        reasons.append("arm_span_under_0.05_rad")
    if not durations or min(durations) < 10.0:
        reasons.append("camera_under_10_seconds")
    if durations and max(durations) - min(durations) > 2.0:
        reasons.append("camera_duration_mismatch")
    curl_result = run / "curl" / "result.json"
    if curl_result.is_file():
        try:
            status = json.loads(curl_result.read_text())["status"]
        except (OSError, ValueError, KeyError):
            status = "invalid"
        if status != "curl_preparation_completed":
            reasons.append("curl_not_completed")
    else:
        reasons.append("curl_result_missing")
    if not (run / "prestart_check.json").is_file():
        reasons.append("prestart_check_missing")
    return {
        "model": model,
        "task": run.parent.name if model == "openvla-oft" else run.name.rsplit("_", 1)[-1],
        "run": str(run.relative_to(archive_root)),
        "status": "technical_pass_not_success_label" if not reasons else "reject",
        "reasons": ";".join(reasons),
        "frames": str(frames),
        "seconds_min": f"{min(durations):.2f}" if durations else "",
        "arm_span_rad": f"{span:.3f}",
        **videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=ROOT,
                        help="Local archive containing work/pi0 and work/openvla-oft")
    args = parser.parse_args()
    rows = []
    for model in ("pi0", "openvla-oft"):
        base = args.archive_root / "work" / model / "inference_videos"
        for dataset in sorted(base.rglob("rollout_dataset")):
            rows.append(audit_run(model, dataset, args.archive_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} rollouts: {sum(r['status'].startswith('technical_pass') for r in rows)} technical passes")
    print("A technical pass does not establish a successful grasp.")


if __name__ == "__main__":
    main()
