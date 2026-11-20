#!/usr/bin/env python3
"""Guarded six-object π₀ test with a local three-camera video for each run.

Without --execute, only checks the checkpoint and demonstration plans.
With --execute, each object requires a checked curl and a separate /start.
This does not start hardware, recalibrate encoders, or replace an E-stop.
"""

import sys

from interactive_object_grasp import PROJECT, main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--policy-kind", "pi0",
        "--output-root", str(PROJECT / "work/pi0/inference_videos"),
    ]
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n人工中断；程序不是硬件急停，请核对真机状态。", file=sys.stderr)
        raise SystemExit(130)
