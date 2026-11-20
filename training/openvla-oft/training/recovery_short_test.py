"""Isolated OpenVLA-OFT learning diagnostic; never imports robot/ROS drivers.

Starts NEW LoRA/head weights from the base model. An old checkpoint supplies only
feature configuration and normalization, never its learned weights. This is a
small-sample overfit test, NOT a production training or robot deployment command.
Saved adapters require load_trial_adapter() to preserve FP32 trainable heads.
"""

import argparse
import gc
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import default_collate

import lerobot_policy_openvla_oft  # noqa: F401
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot_policy_openvla_oft.configuration_openvla_oft import OpenVLAOFTConfig
from lerobot_policy_openvla_oft.modeling_openvla_oft import OpenVLAOFTPolicy, cast_parameters
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def emit(path, event, **fields):
    record = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()


def select_indices(dataset, frames_per_task):
    """Choose separated stages of the first episode of EACH task in this split."""
    table = dataset.hf_dataset.select_columns(["task_index", "episode_index", "frame_index"])
    columns = {key: np.asarray(table[key], dtype=np.int64) for key in table.column_names}
    indices = []
    for task in np.unique(columns["task_index"]):
        candidates = np.flatnonzero(columns["task_index"] == task)
        episode = columns["episode_index"][candidates].min()
        episode_rows = candidates[columns["episode_index"][candidates] == episode]
        positions = np.linspace(0.10, 0.85, frames_per_task) * (len(episode_rows) - 1)
        indices.extend(episode_rows[np.rint(positions).astype(int)].tolist())
    if len(indices) != len(set(indices)):
        raise ValueError("Not enough distinct frames for the requested test")
    return indices


def cache_samples(dataset, indices, pre, config):
    samples, records = [], []
    for index in indices:
        sample = dataset[index]
        for key in dataset.meta.camera_keys:
            if sample[key].dtype == torch.uint8:
                sample[key] = sample[key].float() / 255.0
        samples.append(sample)
        records.append({"dataset_index": index, "episode": int(sample["episode_index"]),
                        "frame": int(sample["frame_index"]), "task": sample["task"]})
    batch = pre(default_collate(samples))
    keys = [*config.image_features, "observation.state", ACTION,
            OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK]
    # Tokenize the entire subset together, so padding is fixed for all batches.
    tensors = {key: batch[key].detach().cpu().contiguous() for key in keys}
    if f"{ACTION}_is_pad" in samples[0]:
        tensors[f"{ACTION}_is_pad"] = torch.stack([s[f"{ACTION}_is_pad"] for s in samples]).bool()
    assert tensors[ACTION].shape[1:] == (config.chunk_size, config.action_feature.shape[0])
    for key, value in tensors.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite input: {key}")
    return tensors, records


def gpu_batch(cached, indices):
    return {key: value[indices].to("cuda") for key, value in cached.items()}


def lr_at_step(step, peak, warmup):
    return peak * min(1.0, step / max(1, warmup))


def load_trial_adapter(checkpoint, base_path, tokenizer):
    """Load this diagnostic's FP32 heads without silently rounding them to BF16."""
    cfg = OpenVLAOFTConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device, cfg.pretrained_path, cfg.tokenizer_name = "cuda", str(base_path), str(tokenizer)
    base = OpenVLAOFTPolicy.from_pretrained(base_path, config=cfg, local_files_only=True)
    for module in (base.model.action_head, base.model.proprio_projector):
        if module is not None:
            cast_parameters(module, torch.float32)
    adapter_cfg = PeftConfig.from_pretrained(checkpoint, local_files_only=True)
    adapter_cfg.base_model_name_or_path = str(base_path)
    policy = PeftModel.from_pretrained(base, checkpoint, config=adapter_cfg, is_trainable=False)
    policy.eval()
    return policy, base


@torch.inference_mode()
def evaluate(base, cached, constant, post, batch_size):
    base.eval()
    head = base.model.action_head.modules_to_save["default"]
    zero_counts = []
    hook = head.act.register_forward_hook(
        lambda _module, _args, output: zero_counts.append(
            (int((output == 0).sum()), output.numel())))
    try:
        predictions = torch.cat([
            base._predict(gpu_batch(cached, slice(i, i + batch_size))).float().cpu()
            for i in range(0, len(cached[ACTION]), batch_size)
        ])
    finally:
        hook.remove()
    targets = cached[ACTION].float()
    zero_fraction = sum(n for n, _ in zero_counts) / sum(n for _, n in zero_counts)
    return prediction_metrics(predictions, targets, constant, post, zero_fraction), predictions


def prediction_metrics(predictions, targets, constant, post, zero_fraction=None):
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Nonfinite evaluation predictions")
    raw_predictions, raw_targets = post(predictions).float(), post(targets).float()
    arm_ids = list(range(7)) + list(range(8, 15))
    if predictions.shape[-1] != 16:
        raise ValueError("This diagnostic expects the OpenArm 16D action layout")
    metrics = {
        "normalized_l1": float((predictions - targets).abs().mean()),
        "constant_median_l1": float((constant - targets).abs().mean()),
        "normalized_input_span": float((predictions.amax(0) - predictions.amin(0)).max()),
        "first_action_arm_span_rad": float((raw_predictions[:, 0, arm_ids].amax(0)
                                            - raw_predictions[:, 0, arm_ids].amin(0)).max()),
        "arm_mae_rad": float((raw_predictions[..., arm_ids] - raw_targets[..., arm_ids]).abs().mean()),
        "within_chunk_arm_span_rad": float((raw_predictions[..., arm_ids].amax(1)
                                            - raw_predictions[..., arm_ids].amin(1)).max()),
    }
    if zero_fraction is not None:
        metrics["head_relu_zero_fraction"] = zero_fraction
    return metrics


def full_training_constant(dataset, pre, config):
    """A second baseline fitted on ALL training actions, with no validation labels."""
    actions = torch.stack([torch.as_tensor(row) for row in dataset.hf_dataset[ACTION]]).float()
    median = actions.median(dim=0).values
    sample = dataset[0]
    for key in dataset.meta.camera_keys:
        if sample[key].dtype == torch.uint8:
            sample[key] = sample[key].float() / 255.0
    sample[ACTION] = median.expand(config.chunk_size, -1).clone()
    return pre(sample)[ACTION].squeeze(0).detach().cpu()


@torch.inference_mode()
def evaluate_full_dataset(base, dataset, pre, post, constant, full_train_constant,
                          batch_size, output, events, limit=None):
    """Stream 32-frame image batches; retain only small action tensors on CPU."""
    total = len(dataset) if limit is None else min(limit, len(dataset))
    phase = "full_validation" if limit is None else "full_validation_smoke"
    emit(events, phase + "_start", total=total)
    predictions, targets, records, padding = [], [], [], []
    zero_sum = 0.0
    started = time.monotonic()
    for start in range(0, total, 32):
        cache, sample_records = cache_samples(dataset, range(start, min(start + 32, total)), pre, base.config)
        metrics, predicted = evaluate(base, cache, constant, post, batch_size)
        predictions.append(predicted)
        targets.append(cache[ACTION])
        if f"{ACTION}_is_pad" in cache:
            padding.append(cache[f"{ACTION}_is_pad"])
        records.extend(sample_records)
        zero_sum += metrics["head_relu_zero_fraction"] * len(sample_records)
        done = len(records)
        if start == 0 or done % 320 == 0 or done == total:
            emit(events, phase + "_progress", completed=done, total=total,
                 elapsed_s=time.monotonic() - started)
        del cache
    predicted, target = torch.cat(predictions), torch.cat(targets)
    result = prediction_metrics(predicted, target, constant, post, zero_sum / total)
    result.update(samples=total, episodes=sorted({r["episode"] for r in records}),
                  full_training_median_l1=float((full_train_constant - target).abs().mean()),
                  include_repeated_end_padding=True, robot_deployment_approved=False,
                  elapsed_s=time.monotonic() - started)
    tensors = {"normalized_predictions": predicted, "normalized_targets": target,
               "episode_index": torch.tensor([r["episode"] for r in records]),
               "frame_index": torch.tensor([r["frame"] for r in records])}
    if padding:
        mask = torch.cat(padding)
        if len(mask) != total:
            raise ValueError("Inconsistent padding metadata")
        valid = ~mask
        result["valid_steps_normalized_l1"] = float((predicted - target).abs()[valid].mean())
        result["valid_steps_full_training_median_l1"] = float((full_train_constant - target).abs()[valid].mean())
        tensors["action_is_pad"] = mask
    per_task = {}
    for task in sorted({r["task"] for r in records}):
        selected = torch.tensor([r["task"] == task for r in records])
        task_metrics = prediction_metrics(predicted[selected], target[selected], constant, post)
        task_metrics.update(samples=int(selected.sum()),
                            full_training_median_l1=float((full_train_constant - target[selected]).abs().mean()))
        per_task[task] = task_metrics
    result["per_task"] = per_task
    result["beats_full_training_median_by_5pct"] = result["normalized_l1"] < 0.95 * result["full_training_median_l1"]
    write_json(output / f"{phase}.json", result)
    if limit is None:
        save_file(tensors, str(output / "full_validation_predictions.safetensors"))
        write_json(output / "full_validation_samples.json", records)
    emit(events, phase + "_complete", **result)
    return result


def save_trial_checkpoint(policy, config, pre, post, checkpoint, step, events):
    checkpoint.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint, safe_serialization=True)
    config.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        head_dtype = str(saved.get_tensor("base_model.model.model.action_head.input_proj.weight").dtype)
    assert head_dtype == "torch.float32"
    write_json(checkpoint / "precision_requirements.json", {
        "loader": "recovery_short_test.load_trial_adapter", "head_dtype": head_dtype, "step": step,
        "diagnostic_only": True, "robot_deployment_approved": False,
        "note": "Standard BF16 plugin loading may round FP32 head weights. Do not deploy this tiny-sample model.",
    })
    emit(events, "checkpoint_saved", step=step, checkpoint=str(checkpoint), head_dtype=head_dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reference", "data", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("This is a bounded diagnostic: steps must be in [1, 1000]")
    if min(args.batch_size, args.eval_every, args.lr, args.clip) <= 0 or min(args.warmup, args.save_every) < 0:
        parser.error("Invalid optimization settings")
    for path in (args.base, args.reference, args.data, args.tokenizer):
        if not path.is_dir():
            parser.error(f"Missing input directory: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing to start a CPU training run")
    args.output.mkdir(parents=True, exist_ok=False)
    events = args.output / "metrics.jsonl"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    diagnostic_only=True, robot_deployment_approved=False,
                    trainable_dtype="float32", forward_dtype="bfloat16",
                    augmentation=False, crop_scale=1.0, rank=32)
    write_json(args.output / "run_config.json", settings)
    emit(events, "start", **settings)

    cfg = TrainPipelineConfig.from_pretrained(args.reference, local_files_only=True)
    cfg.dataset.root = args.data
    cfg.dataset.image_transforms.enable = False
    cfg.dataset.video_backend = "pyav"
    cfg.policy.chunk_size = cfg.policy.n_action_steps = 8
    train, validation = make_train_eval_datasets(cfg)
    if validation is None:
        raise ValueError("A separate episode-level validation split is required")
    pcfg = OpenVLAOFTConfig.from_pretrained(args.reference, local_files_only=True)
    pcfg.device, pcfg.pretrained_path, pcfg.tokenizer_name = "cuda", str(args.base), str(args.tokenizer)
    pcfg.use_peft, pcfg.proprio_projector_fp32 = False, True
    pcfg.image_crop_scale, pcfg.compile_model = 1.0, False
    pcfg.chunk_size = pcfg.n_action_steps = 8
    pcfg.optimizer_lr, pcfg.optimizer_grad_clip_norm = args.lr, args.clip
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=args.reference,
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "openvla_oft_tokenizer": {"tokenizer_name": str(args.tokenizer)}},
    )
    cached_train, train_records = cache_samples(train, select_indices(train, 4), pre, pcfg)
    cached_val, val_records = cache_samples(validation, select_indices(validation, 2), pre, pcfg)
    assert not {s["episode"] for s in train_records} & {s["episode"] for s in val_records}
    if args.batch_size > len(train_records):
        raise ValueError("Batch size exceeds the small training subset")
    write_json(args.output / "samples.json", {"train": train_records, "validation": val_records})
    constant = cached_train[ACTION].median(dim=0).values
    all_train_constant = full_training_constant(train, pre, pcfg) if args.full_validation else None
    emit(events, "data_ready", train_samples=len(train_records), validation_samples=len(val_records),
         full_validation_samples=len(validation) if args.full_validation else 0)
    del train
    if not args.full_validation:
        del validation
    gc.collect()

    base = OpenVLAOFTPolicy.from_pretrained(args.base, config=pcfg, local_files_only=True)
    # Fresh initialization of NEW modules in FP32; never reuse the collapsed head.
    for module in (base.model.action_head, base.model.proprio_projector):
        if module is not None:
            cast_parameters(module, torch.float32)
            for layer in module.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
    policy = base.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 32})
    for param in policy.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    # The unused original copy inside modules_to_save does not need FP32 storage.
    for module in (base.model.action_head, base.model.proprio_projector):
        if module is not None:
            cast_parameters(module.original_module, torch.bfloat16)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in parameters)
    emit(events, "precision", trainable_params=sum(p.numel() for p in parameters),
         trainable_dtypes=dict(Counter(str(p.dtype) for p in parameters)))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01, foreach=False)
    initial_norm = base.model.action_head.modules_to_save["default"].output_norm.weight.detach().clone()
    initial = {}
    for split, cache in (("train", cached_train), ("validation", cached_val)):
        metrics, _ = evaluate(base, cache, constant, post, args.batch_size)
        initial[split] = metrics
        emit(events, "eval", step=0, split=split, **metrics)
    if args.full_validation:
        evaluate_full_dataset(base, validation, pre, post, constant, all_train_constant,
                              args.batch_size, args.output, events, limit=4)

    started = time.monotonic()
    generator = torch.Generator().manual_seed(args.seed)
    order, cursor = torch.randperm(len(train_records), generator=generator), 0
    final = {}
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order, cursor = torch.randperm(len(train_records), generator=generator), 0
        indices = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        base.train()
        lr = lr_at_step(step, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = gpu_batch(cached_train, indices)
        optimizer.zero_grad(set_to_none=True)
        predicted = base._predict(batch)
        loss = (predicted.float() - batch[ACTION].float()).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1:
            moments = [value for state in optimizer.state.values() for key, value in state.items()
                       if key in ("exp_avg", "exp_avg_sq")]
            assert moments and all(m.dtype == torch.float32 for m in moments)
            emit(events, "optimizer_precision", dtypes=dict(Counter(str(m.dtype) for m in moments)))
            del moments
        if step == 1 or step % 10 == 0:
            emit(events, "train", step=step, loss=float(loss.detach()), lr=lr,
                 grad_norm_before_clip=float(grad_norm), elapsed_s=time.monotonic() - started,
                 peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        del batch, predicted, loss, grad_norm
        if step % args.eval_every == 0 or step == args.steps:
            optimizer.zero_grad(set_to_none=True)
            for split, cache in (("train", cached_train), ("validation", cached_val)):
                metrics, prediction = evaluate(base, cache, constant, post, args.batch_size)
                final[split] = metrics
                emit(events, "eval", step=step, split=split, **metrics)
                if split == "validation":
                    before_reload = prediction
        if args.save_every and step % args.save_every == 0 and step != args.steps:
            save_trial_checkpoint(policy, pcfg, pre, post,
                                  args.output / "checkpoints" / f"{step:06d}" / "pretrained_model", step, events)

    checkpoint = args.output / "pretrained_model"
    save_trial_checkpoint(policy, pcfg, pre, post, checkpoint, args.steps, events)
    norm_change = float((base.model.action_head.modules_to_save["default"].output_norm.weight
                         .detach() - initial_norm).abs().max())
    emit(events, "saved", checkpoint=str(checkpoint), head_dtype="torch.float32",
         output_norm_max_change=norm_change)
    # Recreate processors as well as weights: test the actual saved artifact.
    del optimizer, parameters, group, param, module, layer, policy, base, initial_norm, pre, post
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, reloaded_base = load_trial_adapter(checkpoint, args.base, args.tokenizer)
    reloaded_pre, reloaded_post = make_pre_post_processors(
        policy_cfg=reloaded_base.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "openvla_oft_tokenizer": {"tokenizer_name": str(args.tokenizer)}},
    )
    reloaded_metrics, after_reload = evaluate(reloaded_base, cached_val, constant, reloaded_post, args.batch_size)
    reload_delta = float((before_reload - after_reload).abs().max())
    checks = {
        "train_beats_constant_by_50pct": final["train"]["normalized_l1"] < 0.5 * final["train"]["constant_median_l1"],
        "validation_beats_constant_by_5pct": final["validation"]["normalized_l1"] < 0.95 * final["validation"]["constant_median_l1"],
        "input_span_above_0_01rad": final["validation"]["first_action_arm_span_rad"] > 0.01,
        "fp32_norm_weights_updated": norm_change > 0,
        "reload_exact": reload_delta == 0,
    }
    summary = {"initial": initial, "final": final, "reloaded": reloaded_metrics,
               "reload_max_abs_delta": reload_delta, "checks": checks,
               "all_short_test_gates_pass": all(checks.values()),
               "robot_deployment_approved": False,
               "scope": "24 training frames / 12 held-out frames; not a full-validation or robot success test"}
    write_json(args.output / "summary.json", summary)
    emit(events, "short_training_complete", **summary)
    if args.full_validation:
        full = evaluate_full_dataset(reloaded_base, validation, reloaded_pre, reloaded_post,
                                     constant, all_train_constant, args.batch_size, args.output, events)
        summary["full_validation"] = full
        summary["checks"]["full_validation_beats_training_median_by_5pct"] = full["beats_full_training_median_by_5pct"]
        summary["all_short_test_gates_pass"] = all(summary["checks"].values())
        summary["scope"] = f"24 training frames; 12 fixed validation probes; full {len(validation)}-frame held-out validation; no robot success test"
        write_json(args.output / "summary.json", summary)
    emit(events, "complete", **summary)


if __name__ == "__main__":
    main()
