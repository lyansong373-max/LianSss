#!/usr/bin/env python3
"""Run LeRobot rollout with the FP32 action head saved by full-data training.

The normal PEFT rollout loader builds the action head in BF16 before loading the
adapter. This entry point replaces only that loader and the tokenizer path in
the saved processor; the ordinary rollout and robot safety paths stay in use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))

import torch
import lerobot_policy_openvla_oft  # noqa: F401 - register policy and processors
import lerobot_robot_openarm_ros2  # noqa: F401 - register robot
from lerobot.rollout import context as rollout_context
from lerobot.scripts import lerobot_rollout
from train_openvla_oft_fp32 import configure_training_memory, fp32_modules, load_fp32_adapter


_original_make_processors = rollout_context.make_pre_post_processors


def _local_paths() -> tuple[Path, Path]:
    base = Path(os.environ["OPENVLA_BASE_MODEL"]).expanduser().resolve(strict=True)
    tokenizer = Path(os.environ["OPENVLA_TOKENIZER"]).expanduser().resolve(strict=True)
    return base, tokenizer


def _verified_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve(strict=True)
    metadata_path = checkpoint / "precision_requirements.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("head_dtype") != "torch.float32":
        raise ValueError(f"Expected FP32 OpenVLA-OFT action head: {metadata_path}")
    if metadata.get("loader") != "train_openvla_oft_fp32.load_fp32_adapter":
        raise ValueError(f"Unexpected adapter loader: {metadata_path}")
    return checkpoint


def _load_fp32_policy(policy_config):
    if policy_config.type != "openvla_oft" or not policy_config.use_peft:
        raise ValueError("This entry point accepts only an OpenVLA-OFT FP32 adapter")
    checkpoint = _verified_checkpoint(policy_config.pretrained_path)
    base_path, tokenizer_path = _local_paths()
    policy, base = load_fp32_adapter(checkpoint, base_path, tokenizer_path)
    configure_training_memory(base, gradient_checkpointing=False)
    for name, module in fp32_modules(base).items():
        active = module.modules_to_save["default"]
        if any(parameter.dtype != torch.float32 for parameter in active.parameters()):
            raise RuntimeError(f"{name} was rounded during adapter loading")
    print(f"FP32 OpenVLA-OFT adapter loaded: {checkpoint}; FiLM={base.config.use_film}", flush=True)
    return policy


def _make_processors_with_local_tokenizer(*args, **kwargs):
    policy_config = kwargs.get("policy_cfg", args[0] if args else None)
    if policy_config is None or policy_config.type != "openvla_oft":
        raise ValueError("Expected OpenVLA-OFT processor configuration")
    _verified_checkpoint(kwargs["pretrained_path"])
    _, tokenizer_path = _local_paths()
    overrides = dict(kwargs.get("preprocessor_overrides") or {})
    overrides["openvla_oft_tokenizer"] = {"tokenizer_name": str(tokenizer_path)}
    kwargs["preprocessor_overrides"] = overrides
    return _original_make_processors(*args, **kwargs)


def main() -> None:
    _local_paths()
    rollout_context._load_pretrained_policy = _load_fp32_policy
    rollout_context.make_pre_post_processors = _make_processors_with_local_tokenizer
    lerobot_rollout.main()


if __name__ == "__main__":
    main()
