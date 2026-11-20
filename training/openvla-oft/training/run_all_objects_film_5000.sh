#!/usr/bin/env bash
# All-six-object OpenVLA-OFT+FiLM. Default is CPU data preflight, NOT training.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-preflight}"
if (( $# > 1 )); then printf 'Only one mode argument is accepted.\n' >&2; exit 2; fi
case "$MODE" in
  preflight|smoke|train) ;;
  -h|--help|help)
    printf '%s\n' 'Usage: bash run_all_objects_film_5000.sh [preflight|smoke|train]' \
      'preflight: four CPU batches, no model loading or training (default)' \
      'smoke: 20 optimizer steps; train: 5000 steps unless OPENVLA_STEPS is set' \
      'All six objects, episode-disjoint train/holdout split. FiLM, batch=2, gradient checkpointing, ETA.' \
      'Defaults: arm sigma=0.03 rad, gripper sigma=0.02, token dropout=0.3, color ON, crop OFF.' \
      'No offline validation, resume, robot access or zero correction.' \
      'Overrides: OPENVLA_STATE_NOISE_STD_RAD, OPENVLA_STATE_NOISE_GRIPPER, OPENVLA_STATE_DROPOUT_PROB,' \
      'OPENVLA_RANDOM_CROP=0|1, OPENVLA_CROP_MIN_SCALE, OPENVLA_IMAGE_AUGMENTATION=0|1,' \
      'OPENVLA_NUM_WORKERS (default 2), OPENVLA_STEPS (1..5000).'
    exit 0 ;;
  *) printf 'Unknown mode: %s\n' "$MODE" >&2; exit 2 ;;
esac
BUNDLE_ROOT="${BUNDLE_ROOT:-/root/gpufree-data/openvla_oft_upload/openvla_oft_training_bundle}"
if [[ -d "$BUNDLE_ROOT/models/openvla_oft_base_7b" ]]; then
  BASE="${OPENVLA_BASE_MODEL:-$BUNDLE_ROOT/models/openvla_oft_base_7b}"
  DATA="${OPENVLA_DATASET_ROOT:-$BUNDLE_ROOT/training_data/six_objects_60}"
  TOKENIZER="${OPENVLA_TOKENIZER:-/root/gpufree-data/openvla_oft_upload/raw_openvla_7b}"
  REFERENCE="${OPENVLA_REFERENCE:-$BUNDLE_ROOT/outputs/fp32_film_training/train_20261104_155949_5HYZpI/result/checkpoints/005000/pretrained_model}"
  PYTHON="${OPENVLA_PYTHON:-$BUNDLE_ROOT/.venv/bin/python}"
  OUTPUT_ROOT="${OPENVLA_OUTPUT_ROOT:-$BUNDLE_ROOT/outputs/all_objects_film_training}"
  LOCK_ROOT="$BUNDLE_ROOT/outputs/fp32_film_training"
else
  PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
  BASE="${OPENVLA_BASE_MODEL:-$SCRIPT_DIR/../models/openvla_oft_base_7b}"
  DATA="${OPENVLA_DATASET_ROOT:-$PROJECT_ROOT/data/lerobot/six_objects_60}"
  TOKENIZER="${OPENVLA_TOKENIZER:-$SCRIPT_DIR/../models/tokenizer}"
  REFERENCE="${OPENVLA_REFERENCE:-$PROJECT_ROOT/work/openvla-oft/checkpoints/fp32_film_005000/pretrained_model}"
  PYTHON="${OPENVLA_PYTHON:-/home/yundu1/miniconda3/envs/lerobot/bin/python}"
  OUTPUT_ROOT="${OPENVLA_OUTPUT_ROOT:-$PROJECT_ROOT/work/openvla-oft/all_objects_film_training}"
  LOCK_ROOT="$PROJECT_ROOT/work/openvla-oft/film_training"
fi
[[ -x "$PYTHON" ]] || { printf 'Python missing: %s\n' "$PYTHON" >&2; exit 1; }
for INPUT in "$BASE" "$DATA" "$TOKENIZER" "$REFERENCE"; do
  [[ -d "$INPUT" ]] || { printf 'Missing input: %s\n' "$INPUT" >&2; exit 1; }
done
STEPS="${OPENVLA_STEPS:-5000}"
[[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS >= 1 && STEPS <= 5000 )) || {
  printf 'OPENVLA_STEPS must be 1..5000.\n' >&2; exit 2;
}
EXTRA=()
NUM_WORKERS="${OPENVLA_NUM_WORKERS:-2}"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { printf 'Invalid worker count.\n' >&2; exit 2; }
case "${OPENVLA_IMAGE_AUGMENTATION:-1}" in
  1) EXTRA+=(--image-augmentation) ;;
  0) EXTRA+=(--no-image-augmentation) ;;
  *) printf 'OPENVLA_IMAGE_AUGMENTATION must be 0 or 1.\n' >&2; exit 2 ;;
esac
case "${OPENVLA_RANDOM_CROP:-0}" in
  1) EXTRA+=(--random-crop) ;;
  0) EXTRA+=(--no-random-crop) ;;
  *) printf 'OPENVLA_RANDOM_CROP must be 0 or 1.\n' >&2; exit 2 ;;
esac
if [[ "$MODE" == preflight ]]; then EXTRA+=(--preflight-only); fi
if [[ "$MODE" == smoke ]]; then STEPS=20; fi
mkdir -p "$OUTPUT_ROOT" "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/.training.lock"
flock -n 9 || { printf 'Another FiLM run is active; refusing concurrent training.\n' >&2; exit 1; }
if [[ "$MODE" != preflight ]]; then
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name())'
  FREE_KIB="$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
  MIN_FREE_GIB=35
  if [[ "$MODE" == smoke ]]; then MIN_FREE_GIB=20; fi
  (( FREE_KIB >= MIN_FREE_GIB * 1024 * 1024 )) || { printf 'Not enough checkpoint space.\n' >&2; exit 1; }
fi
RUN_DIR="$(mktemp -d "$OUTPUT_ROOT/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$OUTPUT_ROOT/.cache/datasets}"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
printf 'All-six-object run: %s\n' "$RUN_DIR"
printf '%s\n' 'WARNING: Original dataset coordinates retained; right-joint-3 zero compatibility NOT verified.'
"$PYTHON" "$SCRIPT_DIR/prepare_all_objects_reference.py" \
  --data "$DATA" --reference "$REFERENCE" --output "$RUN_DIR/reference_config" \
  2>&1 | tee "$RUN_DIR/selection.log"
"$PYTHON" -u "$SCRIPT_DIR/train_openvla_oft_fp32.py" \
  --base "$BASE" --reference "$RUN_DIR/reference_config" --data "$DATA" --tokenizer "$TOKENIZER" \
  --output "$RUN_DIR/result" --steps "$STEPS" --batch-size 2 --num-workers "$NUM_WORKERS" \
  --use-film --film-mask-padding --gradient-checkpointing \
  --state-noise-std-rad "${OPENVLA_STATE_NOISE_STD_RAD:-0.03}" \
  --state-noise-gripper "${OPENVLA_STATE_NOISE_GRIPPER:-0.02}" \
  --state-dropout-prob "${OPENVLA_STATE_DROPOUT_PROB:-0.3}" \
  --random-crop-min-scale "${OPENVLA_CROP_MIN_SCALE:-0.9}" \
  --no-offline-validation --lr 0.00005 --warmup 100 --clip 1.0 \
  --eval-every 500 --save-every 2500 "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/console.log"
