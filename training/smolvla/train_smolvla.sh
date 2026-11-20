#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_BIN="${LEROBOT_TRAIN:-lerobot-train}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smolvla_six_objects}"

"${TRAIN_BIN}" \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/openarm_six_objects \
  "--dataset.root=${PROJECT_ROOT}/data/lerobot/six_objects_60" \
  "--output_dir=${OUTPUT_DIR}" \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --accelerator.mixed_precision=bf16 \
  --num_workers=4 \
  --wandb.enable=false
