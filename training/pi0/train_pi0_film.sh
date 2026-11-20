#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PI0_BASE_CHECKPOINT:?Set PI0_BASE_CHECKPOINT to an existing Pi0 pretrained_model directory; weights are not included here}"
[[ -f "${PI0_BASE_CHECKPOINT}/config.json" ]] || {
  printf 'Missing Pi0 base checkpoint config: %s/config.json\n' "${PI0_BASE_CHECKPOINT}" >&2
  exit 1
}
TRAIN_BIN="${LEROBOT_TRAIN:-lerobot-train}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/pi0_film_state15_noise03}"

"${TRAIN_BIN}" \
  --dataset.repo_id=local/openarm_six_objects \
  "--dataset.root=${PROJECT_ROOT}/data/lerobot/six_objects_60" \
  --dataset.eval_split=0.1 \
  "--policy.path=${PI0_BASE_CHECKPOINT}" \
  --policy.device=cuda \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.use_film=true \
  --policy.state_dropout_prob=0.15 \
  --policy.state_noise_rad=0.03 \
  --policy.state_noise_gripper=0.02 \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --accelerator.gradient_accumulation.steps=4 \
  --steps=20000 \
  --eval_steps=4000 \
  --max_eval_samples=300 \
  --save_freq=10000 \
  --log_freq=20 \
  --wandb.enable=false \
  "--output_dir=${OUTPUT_DIR}"
