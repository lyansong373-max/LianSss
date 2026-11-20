"""Full-dataset OpenVLA-OFT training with FP32 trainable weights and Adam state.

No robot imports or motion commands. Uses the validated diagnostic's model and
metrics, but samples from ALL training frames rather than a cached tiny subset.
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import time
import traceback
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import recovery_short_test as diagnostic  # Sets offline environment before HF imports.
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import functional as image_functional
from tqdm import tqdm
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_pre_post_processors
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot_policy_openvla_oft.configuration_openvla_oft import OpenVLAOFTConfig
from lerobot_policy_openvla_oft.modeling_openvla_oft import OpenVLAOFTPolicy, cast_parameters
from lerobot_policy_openvla_oft.projectors import ProprioProjector
from peft import PeftConfig, PeftModel
from safetensors import safe_open


class StepBatchSampler(Sampler):
    """Shuffle every epoch; resume from completed steps, not the prefetch cursor."""
    def __init__(self, size, batch_size, steps, seed, start_step=0):
        if not 0 < batch_size <= size or not 0 <= start_step <= steps:
            raise ValueError("Invalid sampler configuration")
        self.size, self.batch_size, self.steps = size, batch_size, steps
        self.seed, self.start_step = seed, start_step

    def __len__(self):
        return self.steps - self.start_step

    def __iter__(self):
        batches_per_epoch = self.size // self.batch_size
        epoch, offset = divmod(self.start_step, batches_per_epoch)
        remaining = len(self)
        while remaining:
            order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch))
            for batch_index in range(offset, batches_per_epoch):
                if not remaining:
                    return
                begin = batch_index * self.batch_size
                yield order[begin:begin + self.batch_size].tolist()
                remaining -= 1
            epoch, offset = epoch + 1, 0


def spread_indices(dataset, per_task=10):
    task_ids = np.asarray(dataset.hf_dataset["task_index"], dtype=np.int64)
    result = []
    for task in np.unique(task_ids):
        candidates = np.flatnonzero(task_ids == task)
        positions = np.rint(np.linspace(0.025, 0.975, per_task) * (len(candidates) - 1)).astype(int)
        result.extend(candidates[positions].tolist())
    if len(result) != len(set(result)):
        raise ValueError("Probe indices must be unique")
    return result


def worker_init(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, args, start_step=0, max_steps=None):
    options = {}
    if args.num_workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_sampler=StepBatchSampler(len(dataset), args.batch_size, max_steps or args.steps,
                                      args.seed, start_step),
        num_workers=args.num_workers, worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(args.seed + 100_000), **options,
    )


ARM_STATE_INDICES = (*range(7), *range(8, 15))
GRIPPER_STATE_INDICES = (7, 15)
EXPECTED_STATE_NAMES = [
    *[f"left_joint{i}.pos" for i in range(1, 8)], "left_gripper.pos",
    *[f"right_joint{i}.pos" for i in range(1, 8)], "right_gripper.pos",
]
AUGMENTATION_DEFAULTS = {
    "image_augmentation": False, "random_crop": False, "random_crop_min_scale": 0.9,
    "state_noise_std_rad": 0.0, "state_noise_gripper": 0.0, "state_dropout_prob": 0.0,
}


def augmentation_options(args):
    return {key: getattr(args, key, default) for key, default in AUGMENTATION_DEFAULTS.items()}


def verify_all_objects_split(train, validation, manifest):
    """Cross-check the actual datasets, not only the preparation configuration."""
    if manifest.get("scope") != "all_six_objects" or len(manifest["task_ids"]) != 6:
        raise ValueError("Invalid all-object dataset manifest")
    counts = {}
    for name, dataset in (("train", train), ("validation", validation)):
        if sorted(dataset.episodes) != manifest[f"{name}_episodes"] or len(dataset) != manifest[f"{name}_frames"]:
            raise ValueError(f"Actual {name} dataset differs from all-object manifest")
        actual = Counter(int(i) for i in dataset.hf_dataset["task_index"])
        expected = {task_id: manifest["per_task"][task][f"{name}_frames"]
                    for task, task_id in manifest["task_ids"].items()}
        if dict(actual) != expected:
            raise ValueError(f"Actual {name} frame tasks differ from all-object manifest")
        counts[name] = {task: actual[task_id] for task, task_id in manifest["task_ids"].items()}
    return counts


def validate_augmentation(options):
    for key, low, high in (("state_noise_std_rad", 0, 0.05), ("state_noise_gripper", 0, 0.05),
                           ("state_dropout_prob", 0, 0.5),
                           ("random_crop_min_scale", 0, 1)):
        value = options[key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be finite and in [{low}, {high}]")
    if options["random_crop_min_scale"] == 0:
        raise ValueError("random_crop_min_scale must be greater than zero")


def validate_state_layout(features):
    feature = features["observation.state"]
    if list(feature["shape"]) != [16] or feature.get("names") != EXPECTED_STATE_NAMES:
        raise ValueError("State augmentation requires the verified 16-D OpenArm joint/gripper order")


def validate_resume_augmentation(current, previous):
    for key, default in AUGMENTATION_DEFAULTS.items():
        if current.get(key, default) != previous.get(key, default):
            raise ValueError(f"Resume setting changed: {key}")


def state_range_from_preprocessor(pre, dataset_stats):
    """Read actual checkpoint MIN_MAX bounds, and verify the dataset agrees."""
    normalizers = [step for step in pre.steps if isinstance(step, NormalizerProcessorStep)]
    if len(normalizers) != 1 or normalizers[0].norm_map.get(FeatureType.STATE) != NormalizationMode.MIN_MAX:
        raise ValueError("State noise requires exactly one MIN_MAX state normalizer")
    stats = normalizers[0].stats["observation.state"]
    bounds = {key: torch.as_tensor(stats[key], dtype=torch.float32).cpu() for key in ("min", "max")}
    for key, value in bounds.items():
        expected = torch.as_tensor(dataset_stats["observation.state"][key], dtype=torch.float32).cpu()
        if value.shape != (16,) or expected.shape != (16,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid state normalization {key} statistics")
        if not torch.allclose(value, expected, rtol=0, atol=1e-6):
            raise ValueError(f"Checkpoint/dataset state {key} statistics differ; do not guess noise scaling")
    state_range = bounds["max"] - bounds["min"]
    if not (state_range > 0).all():
        raise ValueError("State noise requires strictly positive normalization ranges")
    return state_range


def perturb_normalized_state(state, state_range, std_rad, gripper_std, keep=None, metrics=None):
    """Physical-unit Gaussian noise converted by 2*sigma/(max-min), then clipped.

    Only kept samples are perturbed. The input is already normalized; action
    targets never enter this function. Gripper units are opening fraction, not rad.
    """
    if state.ndim != 2 or state.shape[1] != 16 or not state.is_floating_point():
        raise ValueError("Expected floating point [B, 16] normalized joint states")
    ranges = state_range.to(device=state.device, dtype=state.dtype)
    if ranges.shape != (16,) or not torch.isfinite(ranges).all() or not (ranges > 0).all():
        raise ValueError("Expected 16 positive finite state ranges")
    sigma = torch.full_like(ranges, std_rad)
    sigma[list(GRIPPER_STATE_INDICES)] = gripper_std
    physical_noise = torch.randn_like(state) * sigma
    physical_noise = physical_noise.clamp(-3 * sigma, 3 * sigma)
    if keep is not None:
        physical_noise = physical_noise * keep.to(device=state.device)[:, None]
    noise = 2 * physical_noise / ranges
    result = (state + noise).clamp(-1, 1)
    if metrics is not None:
        effective = (result - state) * ranges / 2
        metrics.update(state_noise_realized_rms_rad=float(effective[:, ARM_STATE_INDICES].square().mean().sqrt()),
                       state_noise_max_abs_rad=float(effective[:, ARM_STATE_INDICES].abs().max()),
                       gripper_noise_max_abs=float(effective[:, GRIPPER_STATE_INDICES].abs().max()),
                       effective_noise_rms_by_dimension=effective.square().mean(dim=0).sqrt().tolist(),
                       normalized_sigma_by_dimension=(2 * sigma / ranges).tolist(),
                       noise_clipped_elements=int(((state + noise).abs() > 1).sum()))
    return result


def augment_training_state(batch, state_range, options, metrics=None):
    """Called after process_batch, only by the train/preflight paths."""
    keep = sample_state_keep(batch["observation.state"], options["state_dropout_prob"], metrics)
    result = dict(batch)
    if options["state_noise_std_rad"] or options["state_noise_gripper"]:
        result["observation.state"] = perturb_normalized_state(
            batch["observation.state"], state_range, options["state_noise_std_rad"],
            options["state_noise_gripper"], keep, metrics)
    return result, keep


def sample_state_keep(state, probability, metrics=None):
    """Sample the projector-output token mask; never overwrite normalized states."""
    keep = torch.rand(state.shape[0], device=state.device) >= probability if probability else None
    if metrics is not None:
        metrics.update(state_dropped_samples=int((~keep).sum()) if keep is not None else 0,
                       state_samples=len(state),
                       state_dropout_realized_fraction=float((~keep).float().mean()) if keep is not None else 0.0)
    return keep


class ProprioTokenDropout:
    """Training-scoped hook on the OUTER (possibly PEFT-wrapped) state projector.

    Keeps token count and checkpoint tensors unchanged. No inverted-dropout
    scaling: retained tokens must be bit-identical. Context must cover backward
    too, so activation recomputation cannot observe a different mask.
    """
    def __init__(self, projector):
        if projector is None:
            raise ValueError("State token dropout requires a proprio projector")
        self.keep = None
        self.handle = projector.register_forward_hook(self)

    def __call__(self, module, inputs, output):
        if self.keep is None or not module.training:
            return output
        if output.ndim != 3 or output.shape[:2] != (len(self.keep), 1):
            raise ValueError("Expected state projector output [B, 1, hidden_size]")
        return output * self.keep.to(device=output.device, dtype=output.dtype)[:, None, None]

    @contextmanager
    def masked(self, keep):
        if self.keep is not None:
            raise RuntimeError("Nested state dropout contexts are not supported")
        if keep is not None and (keep.ndim != 1 or keep.dtype != torch.bool):
            raise ValueError("State keep mask must be a boolean [B] tensor")
        self.keep = keep
        try:
            yield
        finally:
            self.keep = None

    def close(self):
        self.keep = None
        self.handle.remove()


def preflight_token_dropout(state, keep):
    """Exercise the real projector class without loading any base model weights."""
    # Isolate toy initialization from the augmentation/sampler RNG stream.
    with torch.random.fork_rng(devices=[]):
        projector = ProprioProjector(16, 32).float().train()
    hook = ProprioTokenDropout(projector)
    try:
        with torch.no_grad():
            clean = projector(state.cpu().float())
            with hook.masked(keep):
                masked = projector(state.cpu().float())
            expected = clean if keep is None else clean * keep.cpu()[:, None, None]
            if not torch.equal(masked, expected) or not torch.equal(projector(state.cpu().float()), clean):
                raise RuntimeError("CPU projector token dropout preflight failed")
    finally:
        hook.close()
    return {"token_dropout_cpu_check": True, "preflight_projector": "random 16-to-32 CPU test fixture"}


def augment_training_images(batch, image_keys, *, color=True, random_crop=False, min_scale=0.9):
    """Mild crop/color jitter with shared per-sample parameters across cameras.

    Keep aspect ratio, resize back to input dimensions; no flips/rotations. Crop
    scale is retained AREA. This is not the policy's saved inference crop scale.
    """
    result = dict(batch)
    image_keys = list(image_keys)
    count = batch[image_keys[0]].shape[0]
    factors = torch.empty(count, 3).uniform_(0.9, 1.1) if color else torch.ones(count, 3)
    crops = torch.rand(count, 3) if random_crop else None
    for key in image_keys:
        images = batch[key]
        if images.ndim != 4 or images.shape[:2] != (count, 3) or not images.is_floating_point():
            raise ValueError(f"Expected float BCHW RGB images: {key}")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError(f"Augmentation requires images in [0, 1]: {key}")
        augmented = []
        for index, (image, (brightness, contrast, saturation)) in enumerate(
                zip(images, factors.tolist(), strict=True)):
            if crops is not None:
                area, top_fraction, left_fraction = crops[index].tolist()
                side = math.sqrt(min_scale + (1 - min_scale) * area)
                height, width = image.shape[-2:]
                crop_h, crop_w = max(1, round(height * side)), max(1, round(width * side))
                top = min(height - crop_h, int(top_fraction * (height - crop_h + 1)))
                left = min(width - crop_w, int(left_fraction * (width - crop_w + 1)))
                image = image_functional.resized_crop(
                    image, top, left, crop_h, crop_w, [height, width], antialias=True)
            if color:
                image = image_functional.adjust_brightness(image, brightness)
                image = image_functional.adjust_contrast(image, contrast)
                image = image_functional.adjust_saturation(image, saturation)
            augmented.append(image)
        result[key] = torch.stack(augmented)
    return result


def training_progress(step, total_steps, elapsed_s, start_step=0):
    seconds_per_step = elapsed_s / max(1, step - start_step)
    remaining_s = max(0, total_steps - step) * seconds_per_step
    seconds = int(round(remaining_s))
    return {"progress_percent": round(100 * step / total_steps, 2),
            "remaining_steps": max(0, total_steps - step), "remaining_s": remaining_s,
            "remaining_time": f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}",
            "estimated_training_finish": datetime.fromtimestamp(time.time() + remaining_s).astimezone().isoformat(timespec="seconds"),
            "eta_scope": "training loop only; final reload and full validation excluded"}


def process_batch(raw, pre, config, device="cuda", image_augmentation=False, *,
                  random_crop=False, random_crop_min_scale=0.9):
    raw = dict(raw)
    for key in config.image_features:
        if raw[key].dtype == torch.uint8:
            raw[key] = raw[key].float() / 255.0
    processed = pre(raw)
    if image_augmentation or random_crop:
        processed = augment_training_images(processed, config.image_features,
                                            color=image_augmentation, random_crop=random_crop,
                                            min_scale=random_crop_min_scale)
    keys = [*config.image_features, "observation.state", ACTION,
            OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK]
    batch = {key: processed[key].to(device) for key in keys}
    if batch[ACTION].shape[1:] != (config.chunk_size, 16):
        raise ValueError(f"Unexpected action shape: {batch[ACTION].shape}")
    for key, value in batch.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite batch input: {key}")
    return batch


def fp32_modules(base):
    """New full-rank modules, including both FiLM generators when enabled."""
    modules = {"action_head": base.model.action_head,
               "proprio_projector": base.model.proprio_projector}
    vision = getattr(base.model, "vision", None)
    for name in ("dinov2", "siglip"):
        film = getattr(getattr(vision, name, None), "film", None)
        if film is not None:
            modules[f"vision.{name}.film"] = film
    return {name: module for name, module in modules.items() if module is not None}


def configure_film(config, use_film=None, mask_padding=None, resume=False):
    for name, value in (("use_film", use_film), ("film_mask_padding", mask_padding)):
        if value is not None:
            if resume and value != getattr(config, name):
                raise ValueError(f"Cannot change {name} while resuming an optimizer checkpoint")
            setattr(config, name, value)


def configure_chunk(config, chunk_size, resume=False):
    if not 1 <= chunk_size <= 50:
        raise ValueError("Action chunk size must be in 1..50")
    if resume and (config.chunk_size != chunk_size or config.n_action_steps != chunk_size):
        raise ValueError("Cannot change action chunk size while resuming an optimizer checkpoint")
    config.chunk_size = config.n_action_steps = chunk_size


def configure_training_memory(base, gradient_checkpointing):
    """Recompute Llama activations without changing weights, batch, or attention."""
    llama = base.model.llm.model
    # Parallel action prediction never reuses a preceding forward's KV cache.
    llama.config.use_cache = False
    if gradient_checkpointing:
        # Non-reentrant mode preserves gradients with frozen embeddings and LoRA.
        llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        llama.gradient_checkpointing_disable()
    if bool(llama.is_gradient_checkpointing) != gradient_checkpointing:
        raise RuntimeError("Llama gradient checkpointing configuration did not take effect")
    return {"llm_gradient_checkpointing": bool(llama.is_gradient_checkpointing),
            "checkpoint_use_reentrant": False if gradient_checkpointing else None,
            "llm_use_cache": llama.config.use_cache,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}


def film_trainable_parameters(base):
    result = {}
    for name, module in fp32_modules(base).items():
        if not name.endswith(".film"):
            continue
        if not hasattr(module, "modules_to_save") or "default" not in module.modules_to_save:
            raise RuntimeError(f"FiLM is not registered in PEFT modules_to_save: {name}")
        for suffix, parameter in module.modules_to_save["default"].named_parameters():
            if not parameter.requires_grad or parameter.dtype != torch.float32:
                raise RuntimeError(f"FiLM parameter is frozen or not FP32: {name}.{suffix}")
            result[f"{name}.{suffix}"] = parameter
    if base.config.use_film and len(result) != 8:
        raise RuntimeError(f"Expected 8 trainable FiLM tensors, found {len(result)}")
    return result


def verify_checkpoint_precision(model_dir, use_film):
    with safe_open(model_dir / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        head = "base_model.model.model.action_head.input_proj.weight"
        if saved.get_slice(head).get_dtype() != "F32":
            raise RuntimeError("Saved action head lost FP32 precision")
        film_keys = [key for key in saved.keys() if ".film." in key]
        if use_film:
            expected = {f"base_model.model.model.vision.{encoder}.film.{projection}.{parameter}"
                        for encoder in ("dinov2", "siglip")
                        for projection in ("scale", "shift") for parameter in ("weight", "bias")}
            if set(film_keys) != expected:
                raise RuntimeError("Checkpoint is missing FiLM parameters or contains unexpected FiLM keys")
            if any(saved.get_slice(key).get_dtype() != "F32" for key in film_keys):
                raise RuntimeError("Saved FiLM weights lost FP32 precision")
        elif film_keys:
            raise RuntimeError("FiLM tensors are present but use_film is disabled")
    return {"head_dtype": "torch.float32", "film_tensor_count": len(film_keys),
            "film_dtype": "torch.float32" if film_keys else None}


def swap_probe_language(cache, records):
    """Keep images/state fixed and replace each prompt with another task's prompt."""
    tasks = sorted({record["task"] for record in records})
    if len(tasks) < 2:
        raise ValueError("Language probe requires at least two tasks")
    first = {task: next(i for i, record in enumerate(records) if record["task"] == task)
             for task in tasks}
    alternative = dict(zip(tasks, tasks[1:] + tasks[:1]))
    indices = [first[alternative[record["task"]]] for record in records]
    swapped = dict(cache)
    for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
        swapped[key] = cache[key][indices].clone()
    if any(torch.equal(cache[OBS_LANGUAGE_TOKENS][i], swapped[OBS_LANGUAGE_TOKENS][i])
           for i in range(len(records))):
        raise ValueError("Language probe did not change every tokenized prompt")
    return swapped


@torch.inference_mode()
def evaluate_language_conditioning(base, cache, records, post, batch_size, reference):
    base.eval()
    def predict(inputs):
        return torch.cat([base._predict(diagnostic.gpu_batch(inputs, slice(i, i + batch_size)))
                          .float().cpu() for i in range(0, len(records), batch_size)])
    repeated = predict(cache)
    swapped = predict(swap_probe_language(cache, records))
    raw_delta = (post(swapped).float() - post(reference).float()).abs()
    arm_ids = list(range(7)) + list(range(8, 15))
    return {"samples": len(records),
            "repeat_same_input_max_abs_normalized": float((repeated - reference).abs().max()),
            "task_swap_mean_abs_normalized": float((swapped - reference).abs().mean()),
            "task_swap_max_arm_rad": float(raw_delta[..., arm_ids].max()),
            "task_swap_mean_arm_rad": float(raw_delta[..., arm_ids].mean()),
            "task_swap_max_gripper_normalized": float(raw_delta[..., [7, 15]].max()),
            "correct_task_l1": float((reference - cache[ACTION]).abs().mean()),
            "swapped_task_l1_to_original_target": float((swapped - cache[ACTION]).abs().mean()),
            "note": "Input sensitivity diagnostic, not semantic correctness or robot success"}


def fresh_model(config, base_path):
    base = OpenVLAOFTPolicy.from_pretrained(base_path, config=config, local_files_only=True)
    for module in fp32_modules(base).values():
        cast_parameters(module, torch.float32)
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    policy = base.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 32})
    normalize_trainable_precision(policy, base)
    return policy, base


def normalize_trainable_precision(policy, base):
    for param in policy.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    for module in fp32_modules(base).values():
        cast_parameters(module.original_module, torch.bfloat16)


def load_fp32_adapter(checkpoint, base_path, tokenizer, is_trainable=False):
    """Required loader: preserve saved FP32 action/proprio heads on load."""
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "precision_requirements.json").read_text())
    if metadata.get("head_dtype") != "torch.float32":
        raise ValueError("Checkpoint is not a verified FP32-head adapter")
    cfg = OpenVLAOFTConfig.from_pretrained(checkpoint, local_files_only=True)
    verify_checkpoint_precision(checkpoint, cfg.use_film)
    cfg.device, cfg.pretrained_path, cfg.tokenizer_name = "cuda", str(base_path), str(tokenizer)
    base = OpenVLAOFTPolicy.from_pretrained(base_path, config=cfg, local_files_only=True)
    # Cast BEFORE PEFT copies tensors into the modules; casting after loading
    # cannot recover precision already rounded to BF16, including FiLM weights.
    for module in fp32_modules(base).values():
        cast_parameters(module, torch.float32)
    adapter_cfg = PeftConfig.from_pretrained(checkpoint, local_files_only=True)
    adapter_cfg.base_model_name_or_path = str(base_path)
    # Stage the checkpoint tensors on CPU instead of holding a second complete
    # FP32 adapter on GPU while copying into the already-allocated modules.
    # This reduces loading peak memory without casting or changing any weights.
    policy = PeftModel.from_pretrained(base, checkpoint, config=adapter_cfg,
                                       is_trainable=is_trainable, torch_device="cpu")
    normalize_trainable_precision(policy, base)
    if not is_trainable:
        policy.eval()
    return policy, base


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(policy, cfg, pre, post, optimizer, step, args, events):
    if shutil.disk_usage(args.output).free < 10 * 2**30:
        raise RuntimeError("Less than 10 GiB free; refusing a potentially partial checkpoint")
    parent = args.output / "checkpoints"
    parent.mkdir(exist_ok=True)
    target, staging = parent / f"{step:06d}", parent / f".incomplete-{step:06d}"
    if target.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    model_dir = staging / "pretrained_model"
    model_dir.mkdir(parents=True)
    policy.save_pretrained(model_dir, safe_serialization=True)
    cfg.policy.save_pretrained(model_dir)
    cfg.save_pretrained(model_dir)
    pre.save_pretrained(model_dir)
    post.save_pretrained(model_dir)
    precision = verify_checkpoint_precision(model_dir, getattr(cfg.policy, "use_film", False))
    diagnostic.write_json(model_dir / "precision_requirements.json", {
        "loader": "train_openvla_oft_fp32.load_fp32_adapter", **precision,
        "use_film": getattr(cfg.policy, "use_film", False),
        "step": step, "training_scope": "full_training_split", "robot_deployment_approved": False,
        "note": "Use the FP32-head loader; standard old loading can round head weights to BF16.",
    })
    trainable_names = [name for name, p in policy.named_parameters() if p.requires_grad]
    torch.save({"step": step, "optimizer": cpu_tree(optimizer.state_dict()), "rng": rng_state(),
                "trainable_names": trainable_names}, staging / "training_state.pt")
    settings = json.loads((args.output / "run_config.json").read_text())
    diagnostic.write_json(staging / "run_config.json", settings)
    diagnostic.write_json(staging / "complete.json", {"step": step, "checkpoint_complete": True})
    staging.rename(target)
    diagnostic.write_json(args.output / "latest.json", {"step": step, "checkpoint": str(target)})
    diagnostic.emit(events, "checkpoint_saved", step=step, checkpoint=str(target), optimizer_saved=True)
    return target


def unvalidated_training_summary(step, checkpoint, frames_seen):
    return {"status": "training_complete_without_validation", "step": step,
            "checkpoint": str(checkpoint), "unique_training_frames_seen": frames_seen,
            "offline_validation": "skipped_by_user", "all_offline_checks_pass": None,
            "robot_deployment_approved": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "reference", "data", "tokenizer", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Action chunk and execution length, 1..50 frames (default 8)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resume", type=Path, help="A complete checkpoint directory, not pretrained_model")
    parser.add_argument("--use-film", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--film-mask-padding", action=argparse.BooleanOptionalAction, default=None,
                        help="Exclude padding tokens from FiLM's language condition")
    parser.add_argument("--image-augmentation", action=argparse.BooleanOptionalAction, default=False,
                        help="Train-only brightness/contrast/saturation jitter, each 0.9 to 1.1")
    parser.add_argument("--random-crop", action=argparse.BooleanOptionalAction, default=False,
                        help="Train-only random resized crop, shared relative crop across cameras")
    parser.add_argument("--random-crop-min-scale", type=float, default=0.9,
                        help="Minimum retained crop AREA; preserve aspect ratio, no flips")
    parser.add_argument("--state-noise-std-rad", "--state-noise-rad", type=float, default=0.0,
                        help="Train-only arm state Gaussian sigma in radians, 0..0.05; clip at 3 sigma")
    parser.add_argument("--state-noise-gripper", type=float, default=0.0,
                        help="Train-only gripper noise sigma in opening-fraction units, 0..0.05")
    parser.add_argument("--state-dropout-prob", "--state-dropout", type=float, default=0.0,
                        help="Train-only probability (0..0.5) of zeroing the projected state token")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
                        help="Recompute Llama activations to reduce memory, preserving physical batch size")
    parser.add_argument("--offline-validation", action=argparse.BooleanOptionalAction, default=True,
                        help="Run probe, language-swap, reload-output and full-dataset evaluations")
    parser.add_argument("--preflight-only", action="store_true", help="Decode/process four batches, no model or training")
    args = parser.parse_args()
    try:
        validate_augmentation(augmentation_options(args))
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.steps <= 15000:
        parser.error("This authorized run is bounded to at most 15000 steps")
    if not 1 <= args.chunk_size <= 50:
        parser.error("--chunk-size must be in 1..50")
    if min(args.batch_size, args.lr, args.clip, args.eval_every, args.save_every) <= 0:
        parser.error("Training sizes, intervals and learning rate must be positive")
    if min(args.num_workers, args.warmup) < 0:
        parser.error("Worker count and warmup cannot be negative")
    for path in (args.base, args.reference, args.data, args.tokenizer):
        if not path.is_dir():
            parser.error(f"Missing input directory: {path}")
    return args


def main(args):
    augmentation = augmentation_options(args)
    validate_augmentation(augmentation)
    batch_augmentation = {k: augmentation[k] for k in ("image_augmentation", "random_crop", "random_crop_min_scale")}
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    args._owns_output = True
    events = args.output / "metrics.jsonl"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if not k.startswith("_")}
    settings.update(gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU preflight",
                    torch=torch.__version__, crop_scale=1.0,
                    image_augmentation=args.image_augmentation,
                    image_augmentation_config={"brightness": [0.9, 1.1], "contrast": [0.9, 1.1],
                                               "saturation": [0.9, 1.1], "shared_across_cameras": True,
                                               "validation_augmented": False,
                                               "spatial_transforms": args.random_crop,
                                               "crop_area_range": [args.random_crop_min_scale, 1.0],
                                               "preserve_aspect_ratio": True, "flips": False},
                    state_augmentation_config={"std_rad": args.state_noise_std_rad,
                                               "noise_clip_sigma": 3.0, "noise_before_normalization": False,
                                               "noise_after_normalization": True,
                                               "sigma_conversion": "2 * physical_sigma / checkpoint MIN_MAX range",
                                               "noisy_indices": list(ARM_STATE_INDICES),
                                               "gripper_std_opening_fraction": args.state_noise_gripper,
                                               "dropout_prob": args.state_dropout_prob,
                                               "dropout_scope": "projector output [B,1,D] token, not input state",
                                               "dropout_fill": 0.0, "labels_modified": False,
                                               "validation_augmented": False},
                    rank=32, chunk_size=args.chunk_size, trainable_dtype="float32",
                    forward_dtype="bfloat16", lr_schedule="linear warmup then constant",
                    robot_deployment_approved=False)
    diagnostic.write_json(args.output / "run_config.json", settings)
    diagnostic.emit(events, "start", **settings)

    reference = args.resume / "pretrained_model" if args.resume else args.reference
    cfg = TrainPipelineConfig.from_pretrained(reference, local_files_only=True)
    cfg.dataset.root, cfg.dataset.video_backend = args.data, "pyav"
    cfg.dataset.image_transforms.enable = False
    cfg.dataset.return_uint8 = True
    configure_chunk(cfg.policy, args.chunk_size, resume=bool(args.resume))
    train, validation = make_train_eval_datasets(cfg)
    if validation is None or set(train.episodes) & set(validation.episodes):
        raise ValueError("An episode-disjoint validation split is required")
    scope_manifest = reference / "all_objects_manifest.json"
    if scope_manifest.is_file():
        manifest = json.loads(scope_manifest.read_text())
        task_counts = verify_all_objects_split(train, validation, manifest)
        settings.update(dataset_scope="all_six_objects", task_frame_counts=task_counts)
        diagnostic.write_json(args.output / "all_objects_manifest.json", manifest)
        diagnostic.emit(events, "all_objects_split_verified", tasks=6, frame_counts=task_counts)
    if args.state_noise_std_rad or args.state_noise_gripper or args.state_dropout_prob:
        validate_state_layout(train.meta.features)
        diagnostic.emit(events, "state_layout_verified", names=EXPECTED_STATE_NAMES,
                        arm_indices=list(ARM_STATE_INDICES), gripper_indices=list(GRIPPER_STATE_INDICES))
    pcfg = OpenVLAOFTConfig.from_pretrained(reference, local_files_only=True)
    configure_film(pcfg, args.use_film, args.film_mask_padding, resume=bool(args.resume))
    settings.update(use_film=pcfg.use_film, film_mask_padding=pcfg.film_mask_padding,
                    initialization="resume" if args.resume else "fresh_base",
                    film_initialization="pytorch_linear_default" if pcfg.use_film else None)
    pcfg.device, pcfg.pretrained_path, pcfg.tokenizer_name = "cuda", str(args.base), str(args.tokenizer)
    pcfg.use_peft, pcfg.proprio_projector_fp32 = False, True
    pcfg.image_crop_scale, pcfg.compile_model = 1.0, False
    configure_chunk(pcfg, args.chunk_size, resume=bool(args.resume))
    pcfg.optimizer_lr, pcfg.optimizer_grad_clip_norm = args.lr, args.clip
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=reference,
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "openvla_oft_tokenizer": {"tokenizer_name": str(args.tokenizer)}},
    )
    state_range = None
    if args.state_noise_std_rad or args.state_noise_gripper:
        state_range = state_range_from_preprocessor(pre, train.meta.stats)
        settings["state_augmentation_config"]["state_range"] = state_range.tolist()
        diagnostic.emit(events, "state_noise_scaling_verified", state_range=state_range.tolist(),
                        source="checkpoint MIN_MAX normalizer", dataset_stats_match=True)
    cfg.policy, cfg.steps, cfg.batch_size, cfg.num_workers = pcfg, args.steps, args.batch_size, args.num_workers
    cfg.output_dir, cfg.save_freq, cfg.eval_steps = args.output, args.save_every, args.eval_every
    cfg.optimizer = pcfg.get_optimizer_preset()
    cfg.scheduler = None  # Actual schedule is explicitly recorded in run_config.json.
    split = {"train_frames": len(train), "validation_frames": len(validation),
             "train_episodes": train.episodes, "validation_episodes": validation.episodes,
             "planned_frame_draws": args.steps * args.batch_size,
             "planned_epoch_equivalent": args.steps * args.batch_size / len(train)}
    settings.update(split)
    diagnostic.write_json(args.output / "run_config.json", settings)
    diagnostic.emit(events, "data_ready", **split)
    if args.preflight_only:
        loader = make_loader(train, args, max_steps=4)
        for index, raw in enumerate(loader):
            augmentation_metrics = {}
            batch = process_batch(raw, pre, pcfg, device="cpu", **batch_augmentation)
            batch, keep = augment_training_state(batch, state_range, augmentation, augmentation_metrics)
            if args.state_dropout_prob:
                augmentation_metrics.update(preflight_token_dropout(batch["observation.state"], keep))
            diagnostic.emit(events, "preflight_batch", batch=index + 1,
                            action_shape=list(batch[ACTION].shape),
                            augmentation=augmentation, augmentation_metrics=augmentation_metrics,
                            episodes=raw["episode_index"].tolist(), tasks=raw["task"])
        diagnostic.write_json(args.output / "summary.json", {"preflight_pass": True, **split})
        diagnostic.emit(events, "preflight_complete", batches=4, model_loaded=False, robot_connected=False)
        return

    if args.offline_validation:
        train_probe, train_records = diagnostic.cache_samples(train, spread_indices(train), pre, pcfg)
        val_probe, val_records = diagnostic.cache_samples(validation, spread_indices(validation), pre, pcfg)
        diagnostic.write_json(args.output / "probe_samples.json", {"train": train_records, "validation": val_records})
        constant = diagnostic.full_training_constant(train, pre, pcfg)
    else:
        diagnostic.emit(events, "offline_validation_disabled", train_split_unchanged=True,
                        checkpoint_saving_enabled=True, robot_deployment_approved=False)
    start_step, resumed_state = 0, None
    if args.resume:
        if not (args.resume / "complete.json").is_file():
            raise ValueError("Resume requires a completed checkpoint")
        previous = json.loads((args.resume / "run_config.json").read_text())
        for key in ("batch_size", "chunk_size", "lr", "warmup", "clip", "seed", "train_frames", "train_episodes", "validation_episodes"):
            if settings[key] != previous[key]:
                raise ValueError(f"Resume setting changed: {key}")
        validate_resume_augmentation(settings, previous)
        policy, base = load_fp32_adapter(reference, args.base, args.tokenizer, is_trainable=True)
        cfg.policy = base.config
        resumed_state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        start_step = resumed_state["step"]
        if start_step >= args.steps:
            raise ValueError("Resume target must exceed the saved step")
    else:
        policy, base = fresh_model(pcfg, args.base)
    memory_settings = configure_training_memory(base, args.gradient_checkpointing)
    token_dropout = ProprioTokenDropout(base.model.proprio_projector) if args.state_dropout_prob else None
    settings.update(memory_settings)
    diagnostic.write_json(args.output / "run_config.json", settings)
    diagnostic.emit(events, "memory_configuration", batch_size=args.batch_size,
                    image_augmentation=args.image_augmentation, **memory_settings)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if not all(p.dtype == torch.float32 for p in parameters):
        raise RuntimeError("Trainable weights must all be FP32")
    film_parameters = film_trainable_parameters(base)
    film_initial = {name: value.detach().flatten()[:1024].clone()
                    for name, value in film_parameters.items()}
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01, foreach=False)
    if resumed_state:
        names = [name for name, p in policy.named_parameters() if p.requires_grad]
        if names != resumed_state["trainable_names"]:
            raise ValueError("Resume parameter ordering changed")
        optimizer.load_state_dict(resumed_state["optimizer"])
    diagnostic.emit(events, "precision", trainable_params=sum(p.numel() for p in parameters),
                    trainable_dtypes=dict(Counter(str(p.dtype) for p in parameters)),
                    use_film=base.config.use_film, film_tensors=len(film_parameters),
                    film_trainable_params=sum(p.numel() for p in film_parameters.values()))
    if args.offline_validation:
        for split_name, cache in (("train_probe", train_probe), ("validation_probe", val_probe)):
            metrics, prediction = diagnostic.evaluate(base, cache, constant, post, args.batch_size)
            diagnostic.emit(events, "eval", step=start_step, split=split_name, **metrics)
            if base.config.use_film and split_name == "validation_probe":
                diagnostic.emit(events, "language_probe", step=start_step,
                                **evaluate_language_conditioning(base, cache, val_records, post,
                                                                 args.batch_size, prediction))
        diagnostic.evaluate_full_dataset(base, validation, pre, post, constant, constant,
                                         args.batch_size, args.output, events, limit=4)
    if resumed_state:
        restore_rng(resumed_state["rng"])
        del resumed_state
    loader = make_loader(train, args, start_step)
    initial_norm = base.model.action_head.modules_to_save["default"].output_norm.weight.detach().clone()
    started, consecutive_collapses = time.monotonic(), 0
    best = float("inf")
    final_probe = {}
    last_step = start_step
    seen_indices, seen_tasks = set(), Counter()
    wait_started = time.monotonic()
    progress = tqdm(loader, total=args.steps, initial=start_step, desc="Training",
                    unit="step", dynamic_ncols=True, mininterval=2)
    for step, raw in enumerate(progress, start=start_step + 1):
        data_s = time.monotonic() - wait_started
        seen_indices.update(int(i) for i in raw["index"].tolist())
        seen_tasks.update(raw["task"])
        augmentation_metrics = {} if step == start_step + 1 else None
        batch = process_batch(raw, pre, base.config, **batch_augmentation)
        batch, keep = augment_training_state(batch, state_range, augmentation, augmentation_metrics)
        if augmentation_metrics is not None:
            diagnostic.emit(events, "training_augmentation", step=step,
                            configuration=augmentation, **augmentation_metrics)
        base.train()
        lr = diagnostic.lr_at_step(step, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with token_dropout.masked(keep) if token_dropout is not None else nullcontext():
            predicted = base._predict(batch)
            loss = (predicted.float() - batch[ACTION].float()).abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
        if step == start_step + 1 and film_parameters:
            gradients = {name: float(p.grad.detach().float().norm()) if p.grad is not None else None
                         for name, p in film_parameters.items()}
            if any(value is None or not np.isfinite(value) for value in gradients.values()):
                raise RuntimeError("FiLM has missing or nonfinite gradients")
            if not any(value > 0 for value in gradients.values()):
                raise RuntimeError("FiLM gradients are all zero")
            diagnostic.emit(events, "film_gradients", step=step, norms=gradients)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True)
        optimizer.step()
        last_step = step
        if step == start_step + 1:
            moments = [value for state in optimizer.state.values() for key, value in state.items()
                       if key in ("exp_avg", "exp_avg_sq")]
            if not moments or not all(m.dtype == torch.float32 for m in moments):
                raise RuntimeError("Adam moments must be FP32")
            diagnostic.emit(events, "optimizer_precision", dtypes=dict(Counter(str(m.dtype) for m in moments)))
            if film_parameters:
                delta = {name: float((p.detach().flatten()[:1024] - film_initial[name]).abs().max())
                         for name, p in film_parameters.items()}
                if not any(value > 0 for value in delta.values()):
                    raise RuntimeError("Sampled FiLM weights did not update")
                diagnostic.emit(events, "film_update", step=step, sampled_max_abs_delta=delta)
            del moments
        if step == start_step + 1 or step % 20 == 0:
            elapsed = time.monotonic() - started
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
            diagnostic.emit(events, "train", step=step, total_steps=args.steps,
                            loss=float(loss.detach()), lr=lr, grad_norm_before_clip=float(grad_norm),
                            elapsed_s=elapsed, mean_s_per_step=elapsed / (step - start_step),
                            data_wait_s=data_s, unique_frames_seen=len(seen_indices), tasks_seen=dict(seen_tasks),
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                            gpu_free_gib=torch.cuda.mem_get_info()[0] / 2**30,
                            **training_progress(step, args.steps, elapsed, start_step))
        del raw, batch, predicted, loss, grad_norm
        if not args.offline_validation:
            if step % args.save_every == 0 or step == args.steps:
                optimizer.zero_grad(set_to_none=True)
                checkpoint = save_checkpoint(policy, cfg, pre, post, optimizer, step, args, events)
            wait_started = time.monotonic()
            continue
        if step % args.eval_every == 0 or step % args.save_every == 0 or step == args.steps:
            optimizer.zero_grad(set_to_none=True)
            for split_name, cache in (("train_probe", train_probe), ("validation_probe", val_probe)):
                metrics, prediction = diagnostic.evaluate(base, cache, constant, post, args.batch_size)
                final_probe[split_name] = metrics
                diagnostic.emit(events, "eval", step=step, split=split_name, **metrics)
                if split_name == "validation_probe":
                    before_reload = prediction
                    if base.config.use_film:
                        diagnostic.emit(events, "language_probe", step=step,
                                        **evaluate_language_conditioning(base, cache, val_records, post,
                                                                         args.batch_size, prediction))
            collapsed = step >= 1000 and final_probe["validation_probe"]["normalized_input_span"] < 1e-4
            consecutive_collapses = consecutive_collapses + 1 if collapsed else 0
            if step % args.save_every == 0 or step == args.steps or consecutive_collapses >= 3:
                checkpoint = save_checkpoint(policy, cfg, pre, post, optimizer, step, args, events)
                if final_probe["validation_probe"]["normalized_l1"] < best:
                    best = final_probe["validation_probe"]["normalized_l1"]
                    diagnostic.write_json(args.output / "best_probe_checkpoint.json", {
                        "step": step, "checkpoint": str(checkpoint), "probe_normalized_l1": best,
                        "selection_scope": "60-frame validation probe, not full validation or robot success",
                    })
            if consecutive_collapses >= 3:
                diagnostic.write_json(args.output / "summary.json", {
                    "status": "stopped_for_constant_outputs", "step": step, "final_probe": final_probe,
                    "robot_deployment_approved": False,
                })
                diagnostic.emit(events, "stopped_for_constant_outputs", step=step, checkpoint=str(checkpoint))
                if token_dropout is not None:
                    token_dropout.close()
                return
        wait_started = time.monotonic()
    if last_step != args.steps:
        raise RuntimeError(f"Unexpected end of loader at {last_step}/{args.steps}")
    norm_change = float((base.model.action_head.modules_to_save["default"].output_norm.weight
                         .detach() - initial_norm).abs().max())
    diagnostic.emit(events, "training_complete", step=last_step, checkpoint=str(checkpoint),
                    unique_frames_seen=len(seen_indices), output_norm_max_change=norm_change)
    if token_dropout is not None:
        token_dropout.close()
    del progress, loader, optimizer, parameters, group, policy, base, initial_norm, pre, post
    del film_parameters, film_initial
    gc.collect()
    torch.cuda.empty_cache()
    if not args.offline_validation:
        summary = unvalidated_training_summary(last_step, checkpoint, len(seen_indices))
        diagnostic.write_json(args.output / "summary.json", summary)
        diagnostic.emit(events, "complete", **summary)
        return
    reloaded, base = load_fp32_adapter(checkpoint / "pretrained_model", args.base, args.tokenizer)
    configure_training_memory(base, gradient_checkpointing=False)
    pre, post = make_pre_post_processors(
        policy_cfg=base.config, pretrained_path=checkpoint / "pretrained_model",
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "openvla_oft_tokenizer": {"tokenizer_name": str(args.tokenizer)}},
    )
    reloaded_metrics, after_reload = diagnostic.evaluate(base, val_probe, constant, post, args.batch_size)
    delta = float((before_reload - after_reload).abs().max())
    diagnostic.emit(events, "reload_check", max_abs_delta=delta, exact=delta == 0)
    summary = {"status": "training_complete_validation_pending", "step": last_step,
               "final_probe": final_probe, "reloaded_probe": reloaded_metrics,
               "reload_max_abs_delta": delta, "checkpoint": str(checkpoint),
               "unique_training_frames_seen": len(seen_indices), "robot_deployment_approved": False}
    diagnostic.write_json(args.output / "summary.json", summary)
    full = diagnostic.evaluate_full_dataset(base, validation, pre, post, constant, constant,
                                           args.batch_size, args.output, events)
    checks = {"reload_exact": delta == 0,
              "beats_constant_by_5pct": full["beats_full_training_median_by_5pct"],
              "input_span_above_0_01rad": full["first_action_arm_span_rad"] > 0.01}
    summary.update(status="complete", full_validation=full, offline_checks=checks,
                   all_offline_checks_pass=all(checks.values()))
    diagnostic.write_json(args.output / "summary.json", summary)
    diagnostic.emit(events, "complete", **summary)


if __name__ == "__main__":
    arguments = parse_args()
    arguments._owns_output = False
    try:
        main(arguments)
    except Exception as error:
        if arguments._owns_output:
            diagnostic.write_json(arguments.output / "failure.json", {
                "error": repr(error), "traceback": traceback.format_exc(), "robot_connected": False})
        raise
