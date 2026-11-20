#!/usr/bin/env bash
set -euo pipefail

# Run from the rental server bundle root; does not touch the old training run.
BUNDLE_ROOT="${BUNDLE_ROOT:-/root/gpufree-data/openvla_oft_upload/openvla_oft_training_bundle}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$BUNDLE_ROOT/.venv/bin/python}"
TRIAL_PARENT="${TRIAL_PARENT:-$BUNDLE_ROOT/outputs/recovery_short_tests}"
mkdir -p "$TRIAL_PARENT"
TRIAL_RUN="$(mktemp -d "$TRIAL_PARENT/trial_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
printf 'Short test output: %s\n' "$TRIAL_RUN"
"$PYTHON" -u "$SCRIPT_DIR/recovery_short_test.py" \
  --base "$BUNDLE_ROOT/models/openvla_oft_base_7b" \
  --reference "$BUNDLE_ROOT/outputs/openvla_oft_openarm_20261103_190536/checkpoints/020000/pretrained_model" \
  --data "$BUNDLE_ROOT/training_data/six_objects_60" \
  --tokenizer /root/gpufree-data/openvla_oft_upload/raw_openvla_7b \
  --output "$TRIAL_RUN/result" --steps "${STEPS:-1000}" --batch-size 2 \
  --lr 0.00005 --warmup 100 --clip 1.0 --eval-every 50 \
  --save-every 500 --full-validation \
  2>&1 | tee "$TRIAL_RUN/console.log"
