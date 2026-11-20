#!/usr/bin/env python3
"""Verify that this is a complete OpenVLA-OFT/OpenArm training bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_DATASET = ROOT.parents[2] / "data" / "lerobot" / "six_objects_60"
BUNDLED_DATASET = ROOT / "training_data" / "six_objects_60"
DATASET = PROJECT_DATASET if PROJECT_DATASET.is_dir() else BUNDLED_DATASET
PLUGIN = ROOT / "third_party" / "lerobot-policy-openvla-oft"


def validate_layout() -> None:
    required = [
        ROOT / "train_openvla_oft.sh",
        ROOT / "start_server_training.sh",
        DATASET / "meta" / "info.json",
        DATASET / "meta" / "stats.json",
        DATASET / "meta" / "tasks.parquet",
        DATASET / "data" / "chunk-000" / "file-000.parquet",
        PLUGIN / "pyproject.toml",
        PLUGIN / "src" / "lerobot_policy_openvla_oft" / "modeling_openvla_oft.py",
        PLUGIN / "src" / "lerobot_policy_openvla_oft" / "convert_checkpoint.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("训练包缺少文件：" + ", ".join(missing))

    info = json.loads((DATASET / "meta" / "info.json").read_text())
    features = info["features"]
    image_keys = sorted(key for key in features if key.startswith("observation.images."))
    expected_images = [f"observation.images.camera{i}" for i in range(1, 4)]
    checks = {
        "episodes": info.get("total_episodes") == 60,
        "frames": info.get("total_frames") == 36000,
        "tasks": info.get("total_tasks") == 6,
        "fps": info.get("fps") == 30,
        "action": features.get("action", {}).get("shape") == [16],
        "state": features.get("observation.state", {}).get("shape") == [16],
        "cameras": image_keys == expected_images,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError("数据集结构不符合 OpenArm OpenVLA-OFT 训练配置：" + ", ".join(failed))

    plugin_init = (PLUGIN / "src" / "lerobot_policy_openvla_oft" / "__init__.py").read_text()
    if "openvla_oft" not in plugin_init.lower():
        raise RuntimeError("插件不是 OpenVLA-OFT")

    videos = sorted((DATASET / "videos").rglob("*.mp4"))
    if len(videos) != 13 or any(path.stat().st_size == 0 for path in videos):
        raise RuntimeError(f"视频文件不完整：预期 13 个，实际 {len(videos)} 个")

    print("结构检查通过：OpenVLA-OFT 插件、60 episodes、36000 frames、3 cameras、16-D state/action")


def validate_hashes() -> None:
    manifest_path = ROOT / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise RuntimeError("缺少 MANIFEST.sha256")
    checked = 0
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"缺少校验文件：{relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"SHA256 不匹配：{relative}")
        checked += 1
    print(f"完整 SHA256 检查通过：{checked} 个文件")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="校验所有文件内容，约需数分钟")
    args = parser.parse_args()
    try:
        validate_layout()
        if args.full:
            validate_hashes()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"校验失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
