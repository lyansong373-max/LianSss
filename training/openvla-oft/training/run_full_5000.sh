#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/root/gpufree-data/openvla_oft_upload/openvla_oft_training_bundle}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$BUNDLE_ROOT/.venv/bin/python}"
MODE="${1:-train}"
if [[ "$MODE" != train && "$MODE" != preflight ]]; then
    printf 'Usage: bash %s [preflight|train]\n' "$0" >&2
    exit 2
fi
mkdir -p "$BUNDLE_ROOT/outputs/fp32_full_training"
FULL_RUN="$(mktemp -d "$BUNDLE_ROOT/outputs/fp32_full_training/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
EXTRA=()
if [[ "$MODE" == preflight ]]; then EXTRA+=(--preflight-only); fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
printf 'Full-dataset run: %s\n' "$FULL_RUN"
"$PYTHON" -u "$SCRIPT_DIR/train_openvla_oft_fp32.py" \
  --base "$BUNDLE_ROOT/models/openvla_oft_base_7b" \
  --reference "$BUNDLE_ROOT/outputs/openvla_oft_openarm_20261103_190536/checkpoints/020000/pretrained_model" \
  --data "$BUNDLE_ROOT/training_data/six_objects_60" \
  --tokenizer /root/gpufree-data/openvla_oft_upload/raw_openvla_7b \
  --output "$FULL_RUN/result" --steps 5000 --batch-size 2 --num-workers 2 \
  --lr 0.00005 --warmup 100 --clip 1.0 --eval-every 500 --save-every 1000 \
  "${EXTRA[@]}" 2>&1 | tee "$FULL_RUN/console.log"
