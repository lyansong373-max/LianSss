#!/usr/bin/env python3
"""Run one OpenVLA-OFT action inference from the local dataset; never connects to a robot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this 7B smoke test is intended for the local RTX 4090")

    import lerobot_policy_openvla_oft  # noqa: F401 — registers policy and processor classes
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_batch_keys

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    fp32_adapter = (checkpoint / "precision_requirements.json").is_file()
    cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    if cfg.policy is None or cfg.policy.type != "openvla_oft" or not cfg.policy.use_peft:
        raise ValueError(f"Not an OpenVLA-OFT LoRA checkpoint: {checkpoint}")
    cfg.policy.device = "cuda"
    cfg.dataset.image_transforms.enable = False
    if fp32_adapter:
        project = Path(__file__).resolve().parents[3]
        data_path = Path(os.environ.get("OPENVLA_DATA", project / "data/lerobot/six_objects_60"))
        base_path = Path(os.environ["OPENVLA_BASE_MODEL"]).expanduser().resolve(strict=True)
        tokenizer_path = Path(os.environ["OPENVLA_TOKENIZER"]).expanduser().resolve(strict=True)
        cfg.dataset.root = data_path.expanduser().resolve(strict=True)
        cfg.dataset.video_backend = "pyav"
        cfg.policy.pretrained_path = base_path
        cfg.policy.tokenizer_name = str(tokenizer_path)

    train_dataset, eval_dataset = make_train_eval_datasets(cfg)
    dataset = eval_dataset if eval_dataset is not None else train_dataset
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample index {args.sample_index} outside dataset length {len(dataset)}")

    print("Loading OpenVLA-OFT base model and LoRA adapter...", flush=True)
    if fp32_adapter:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
        from train_openvla_oft_fp32 import configure_training_memory, fp32_modules, load_fp32_adapter

        policy, base = load_fp32_adapter(checkpoint, base_path, tokenizer_path)
        configure_training_memory(base, gradient_checkpointing=False)
        for name, module in fp32_modules(base).items():
            active = module.modules_to_save["default"]
            if any(weight.dtype != torch.float32 for weight in active.parameters()):
                raise RuntimeError(f"{name} was rounded during adapter loading")
    else:
        policy = make_policy(
            cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
            pretrained_path=checkpoint,
        )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
            **({"openvla_oft_tokenizer": {"tokenizer_name": str(tokenizer_path)}} if fp32_adapter else {}),
        },
    )

    sample = dataset[args.sample_index]
    target = sample["action"][0].clone()
    for key in dataset.meta.camera_keys:
        if key in sample and sample[key].dtype == torch.uint8:
            sample[key] = sample[key].to(dtype=torch.float32) / 255.0
    sample = preprocessor(rename_batch_keys(sample, cfg.rename_map))

    policy.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        predicted = policy.select_action(sample)
        predicted = postprocessor(predicted)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    predicted = predicted.squeeze(0).float().cpu()

    report = {
        "checkpoint": str(checkpoint),
        "sample_index": args.sample_index,
        "task": dataset[args.sample_index]["task"],
        "action_shape": list(predicted.shape),
        "inference_seconds": round(elapsed, 4),
        "peak_inference_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "predicted_action": predicted.tolist(),
        "target_first_action": target.float().cpu().tolist(),
        "first_action_mae": float((predicted - target.float().cpu()).abs().mean()),
        "robot_connected": False,
        "fp32_action_head": fp32_adapter,
        "use_film": bool(cfg.policy.use_film),
        "film_fp32_verified": bool(fp32_adapter and cfg.policy.use_film),
        "kv_cache_enabled": bool(base.model.llm.model.config.use_cache) if fp32_adapter else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
