#!/usr/bin/env bash
# Independent OpenVLA-OFT + FiLM experiment; never starts a robot process.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
MODE="${1:-train}"
if (( $# )); then shift; fi
case "$MODE" in
  preflight|smoke|train) ;;
  *) printf 'Usage: bash %s [preflight|smoke|train] [trainer options]\n' "$0" >&2; exit 2 ;;
esac

BUNDLE_ROOT="${BUNDLE_ROOT:-/root/gpufree-data/openvla_oft_upload/openvla_oft_training_bundle}"
if [[ -d "$BUNDLE_ROOT/models/openvla_oft_base_7b" ]]; then
  BASE="${OPENVLA_BASE_MODEL:-$BUNDLE_ROOT/models/openvla_oft_base_7b}"
  DATA="${OPENVLA_DATASET_ROOT:-$BUNDLE_ROOT/training_data/six_objects_60}"
  TOKENIZER="${OPENVLA_TOKENIZER:-/root/gpufree-data/openvla_oft_upload/raw_openvla_7b}"
  REFERENCE="${OPENVLA_REFERENCE:-$BUNDLE_ROOT/outputs/openvla_oft_openarm_20261103_190536/checkpoints/020000/pretrained_model}"
  PYTHON="${OPENVLA_PYTHON:-$BUNDLE_ROOT/.venv/bin/python}"
  OUTPUT_ROOT="${OPENVLA_OUTPUT_ROOT:-$BUNDLE_ROOT/outputs/fp32_film_training}"
else
  BASE="${OPENVLA_BASE_MODEL:-$SCRIPT_DIR/../models/openvla_oft_base_7b}"
  DATA="${OPENVLA_DATASET_ROOT:-$PROJECT_ROOT/data/lerobot/six_objects_60}"
  TOKENIZER="${OPENVLA_TOKENIZER:-$SCRIPT_DIR/../models/tokenizer}"
  REFERENCE="${OPENVLA_REFERENCE:-$PROJECT_ROOT/work/openvla-oft/checkpoints/fp32_full_005000/pretrained_model}"
  PYTHON="${OPENVLA_PYTHON:-/home/yundu1/miniconda3/envs/lerobot/bin/python}"
  OUTPUT_ROOT="${OPENVLA_OUTPUT_ROOT:-$PROJECT_ROOT/work/openvla-oft/film_training}"
fi

[[ -x "$PYTHON" ]] || { printf 'Python not found: %s\n' "$PYTHON" >&2; exit 1; }
for INPUT_DIR in "$BASE" "$DATA" "$TOKENIZER" "$REFERENCE"; do
  [[ -d "$INPUT_DIR" ]] || { printf 'Missing input: %s\n' "$INPUT_DIR" >&2; exit 1; }
done
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.training.lock"
flock -n 9 || { printf 'Another FiLM run already holds the training lock.\n' >&2; exit 1; }
EXTRA=()
STEPS=5000
if [[ "$MODE" == preflight ]]; then
  EXTRA+=(--preflight-only)
else
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name())'
  FREE_KIB="$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
  MIN_FREE_GIB=35
  if [[ "$MODE" == smoke ]]; then STEPS=20; MIN_FREE_GIB=20; fi
  (( FREE_KIB >= MIN_FREE_GIB * 1024 * 1024 )) || {
    printf 'Need at least %s GiB free for FiLM checkpoints.\n' "$MIN_FREE_GIB" >&2; exit 1;
  }
fi
FILM_RUN="$(mktemp -d "$OUTPUT_ROOT/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
printf 'FiLM run: %s\n' "$FILM_RUN"
"$PYTHON" -u "$SCRIPT_DIR/train_openvla_oft_fp32.py" \
  --base "$BASE" --reference "$REFERENCE" --data "$DATA" --tokenizer "$TOKENIZER" \
  --output "$FILM_RUN/result" --steps "$STEPS" --batch-size 2 --num-workers 2 \
  --use-film --film-mask-padding --no-image-augmentation --gradient-checkpointing \
  --no-offline-validation \
  --lr 0.00005 --warmup 100 --clip 1.0 --eval-every 500 --save-every 2500 \
  "${EXTRA[@]}" "$@" 2>&1 | tee "$FILM_RUN/console.log"
