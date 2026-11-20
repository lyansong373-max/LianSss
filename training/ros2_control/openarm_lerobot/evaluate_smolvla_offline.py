#!/usr/bin/env python3
"""Offline held-out SmolVLA loss evaluation; never connects to ROS or a robot."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def select_indices(task_ids, per_task_limit):
    """Sample throughout each task, not just the beginning of its first episode."""
    indices = []
    for task_id in np.unique(task_ids):
        rows = np.flatnonzero(task_ids == task_id)
        if per_task_limit and len(rows) > per_task_limit:
            rows = rows[np.linspace(0, len(rows) - 1, per_task_limit, dtype=int)]
        indices.extend(rows.tolist())
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Checkpoint pretrained_model directory')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--max-samples-per-task', type=int, default=0,
                        help='0 = all held-out frames; positive = evenly spaced subset')
    parser.add_argument('--list-split', action='store_true',
                        help='Check dataset split only; do not load model or use GPU')
    parser.add_argument('--output', type=Path,
                        help='New JSON report; existing files are not overwritten')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples_per_task < 0:
        parser.error('Invalid batch size, worker count or sample limit')
    if args.output and args.output.exists():
        parser.error(f'Report already exists: {args.output}')

    # Resolve "last" once, so a training process cannot change our checkpoint mid-load.
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    for filename in ['config.json', 'train_config.json', 'model.safetensors',
                     'policy_preprocessor.json']:
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_batch_keys
    from lerobot.utils.collate import lerobot_collate_fn

    cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    if cfg.policy is None or cfg.policy.type != 'smolvla':
        raise ValueError('This script only supports SmolVLA policy checkpoints')
    if cfg.dataset.streaming or not 0 < cfg.dataset.eval_split < 1:
        raise ValueError('Need a non-streaming dataset with the original held-out eval_split')
    if cfg.dataset.root is None or not Path(cfg.dataset.root).is_dir():
        raise FileNotFoundError(f'Original local dataset missing: {cfg.dataset.root}')
    cfg.policy.device = args.device
    train_dataset, dataset = make_train_eval_datasets(cfg)
    if dataset is None or not len(dataset):
        raise ValueError('Empty held-out dataset')
    train_episodes = list(map(int, train_dataset.episodes))
    eval_episodes = list(map(int, dataset.episodes))
    if set(train_episodes) & set(eval_episodes):
        raise ValueError('Training and validation episodes overlap')
    del train_dataset
    task_ids = dataset.hf_dataset.data.column('task_index').to_numpy()
    indices = select_indices(task_ids, args.max_samples_per_task)
    report = {
        'checkpoint': str(checkpoint), 'dataset_root': str(cfg.dataset.root),
        'train_episodes': train_episodes, 'eval_episodes': eval_episodes,
        'held_out_frames': len(dataset), 'evaluated_frames': len(indices),
        'seed': args.seed, 'device': args.device,
        'precision': args.precision if args.device == 'cuda' else 'fp32',
        'max_samples_per_task': args.max_samples_per_task,
        'metric': 'flow_matching_loss_weighted_by_valid_action_entries',
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.list_split:
        print('Split check passed. No model loaded, no robot connected.')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or check your GPU environment')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta,
                         rename_map=cfg.rename_map, pretrained_path=checkpoint)
    policy.to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={
            'device_processor': {'device': args.device},
            'rename_observations_processor': {'rename_map': cfg.rename_map},
        },
    )
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
    )
    totals = defaultdict(lambda: {'frames': 0, 'weighted_sum': 0.0, 'weight': 0})
    with torch.inference_mode():
        for batch in tqdm(loader, desc='Offline validation'):
            tasks = list(batch['task'])
            for key in dataset.meta.camera_keys:
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            batch = preprocessor(rename_batch_keys(batch, cfg.rename_map))
            autocast = (torch.autocast('cuda', dtype=torch.bfloat16)
                        if args.device == 'cuda' and args.precision == 'bf16'
                        else nullcontext())
            with autocast:
                losses, _ = policy(batch, reduction='none')
            if not torch.isfinite(losses).all():
                raise RuntimeError('Non-finite validation loss; no successful report written')
            actions = batch['action']
            pad = batch.get('action_is_pad')
            weights = ((~pad).sum(dim=1) * actions.shape[-1] if pad is not None
                       else torch.full((len(tasks),), actions.shape[1] * actions.shape[2],
                                       device=losses.device))
            for task, loss, weight in zip(tasks, losses.float().cpu().tolist(),
                                          weights.cpu().tolist(), strict=True):
                item = totals[task]
                item['frames'] += 1
                item['weighted_sum'] += loss * weight
                item['weight'] += weight
    total_weight = sum(item['weight'] for item in totals.values())
    if total_weight == 0:
        raise ValueError('No valid action entries in validation set')
    report['loss'] = sum(item['weighted_sum'] for item in totals.values()) / total_weight
    report['per_task'] = {
        task: {'frames': item['frames'], 'loss': item['weighted_sum'] / item['weight']
               if item['weight'] else None}
        for task, item in totals.items()
    }
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
