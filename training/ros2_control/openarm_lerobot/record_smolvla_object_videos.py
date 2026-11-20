#!/usr/bin/env python3
"""Record locally saved, three-camera SmolVLA rollouts, one object at a time.

Run without --execute to check configuration only. With --execute, each object
still requires a typed /start after the checked curl preparation. No automatic
success label or video trimming is performed here.
"""

import sys

from interactive_object_grasp import ROOT, main


if __name__ == "__main__":
    # Keep the existing guarded six-object controller as the single source of
    # truth for pose checks and bridge safety settings. Only enable its local
    # video-recording mode from this dedicated entry point.
    sys.argv[1:1] = ["--record-videos", "--output-root", str(ROOT / "outputs/smolvla_object_videos")]
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n人工中断；程序不是硬件急停，请核对真机状态。", file=sys.stderr)
        raise SystemExit(130)
