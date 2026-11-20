#!/usr/bin/env bash
# End-to-end OpenVLA-OFT setup and training for a freshly rented GPU instance.

set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$BUNDLE_DIR/train_openvla_oft.sh"
MODE="${1:-all}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

fail() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

[[ -x "$LAUNCHER" ]] || chmod +x "$LAUNCHER"
command -v nvidia-smi >/dev/null || fail "找不到 nvidia-smi；实例没有正确挂载 NVIDIA GPU"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || fail "无法读取 GPU 显存"

AVAILABLE_KIB="$(df -Pk "$BUNDLE_DIR" | awk 'NR==2 {print $4}')"
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || fail "无法读取剩余磁盘"
(( AVAILABLE_KIB >= 80 * 1024 * 1024 )) || fail "剩余磁盘不足 80 GiB；建议扩容到 150 GB"

# Select conservative per-GPU defaults. Explicit environment variables always win.
if (( GPU_MEMORY_MIB >= 70 * 1024 )); then
    PRESET_BATCH=4
    PRESET_ACCUM=1
    PRESET_RANK=32
    PRESET_CHUNK=8
elif (( GPU_MEMORY_MIB >= 40 * 1024 )); then
    # Verified on a 46,068 MiB L40S: ~39.8 GiB peak with three cameras.
    PRESET_BATCH=3
    PRESET_ACCUM=1
    PRESET_RANK=32
    PRESET_CHUNK=8
elif (( GPU_MEMORY_MIB >= 23 * 1024 )); then
    PRESET_BATCH=1
    PRESET_ACCUM=1
    PRESET_RANK=16
    PRESET_CHUNK=4
else
    fail "OpenVLA-OFT 训练至少需要约 24 GiB 显存；当前为 ${GPU_MEMORY_MIB} MiB"
fi

export OPENVLA_PLUGIN_DIR="$BUNDLE_DIR/third_party/lerobot-policy-openvla-oft"
PROJECT_DATASET_ROOT="$BUNDLE_DIR/../../../data/lerobot/six_objects_60"
BUNDLED_DATASET_ROOT="$BUNDLE_DIR/training_data/six_objects_60"
if [[ -f "$PROJECT_DATASET_ROOT/meta/info.json" ]]; then
    export OPENVLA_DATASET_ROOT="${OPENVLA_DATASET_ROOT:-$PROJECT_DATASET_ROOT}"
else
    export OPENVLA_DATASET_ROOT="${OPENVLA_DATASET_ROOT:-$BUNDLED_DATASET_ROOT}"
fi
export OPENVLA_DATASET_ID="local/openarm_six_objects"
PROJECT_BASE_DIR="$BUNDLE_DIR/../models/openvla_oft_base_7b"
if [[ -f "$PROJECT_BASE_DIR/config.json" ]]; then
    export OPENVLA_BASE_DIR="${OPENVLA_BASE_DIR:-$PROJECT_BASE_DIR}"
else
    export OPENVLA_BASE_DIR="${OPENVLA_BASE_DIR:-$BUNDLE_DIR/models/openvla_oft_base_7b}"
fi
export OPENVLA_CACHE_DIR="$BUNDLE_DIR/.cache/openvla_oft"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BUNDLE_DIR/.cache/pip}"
export OPENVLA_BATCH_SIZE="${OPENVLA_BATCH_SIZE:-$PRESET_BATCH}"
export OPENVLA_GRAD_ACCUM="${OPENVLA_GRAD_ACCUM:-$PRESET_ACCUM}"
export OPENVLA_LORA_RANK="${OPENVLA_LORA_RANK:-$PRESET_RANK}"
export OPENVLA_CHUNK_SIZE="${OPENVLA_CHUNK_SIZE:-$PRESET_CHUNK}"
export OPENVLA_STEPS="${OPENVLA_STEPS:-30000}"
export OPENVLA_DECAY_STEPS="${OPENVLA_DECAY_STEPS:-20000}"
export OPENVLA_SAVE_FREQ="${OPENVLA_SAVE_FREQ:-5000}"

python_is_312_or_newer() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1
}

select_python() {
    if [[ -n "${OPENVLA_PYTHON:-}" ]]; then
        python_is_312_or_newer "$OPENVLA_PYTHON" || \
            fail "OPENVLA_PYTHON 必须是 Python 3.12+，当前为：$($OPENVLA_PYTHON --version 2>&1)"
        return
    fi

    local current_python
    current_python="$(command -v python || true)"
    if [[ -n "$current_python" ]] && python_is_312_or_newer "$current_python"; then
        export OPENVLA_PYTHON="$current_python"
        return
    fi

    local bootstrap_python
    bootstrap_python="$(command -v python3.12 || true)"
    [[ -n "$bootstrap_python" ]] || \
        fail "镜像缺少 Python 3.12；OpenVLA-OFT/LeRobot 0.6.1 不支持当前 Python"

    local venv_dir="${OPENVLA_VENV_DIR:-$BUNDLE_DIR/.venv}"
    if [[ ! -x "$venv_dir/bin/python" ]]; then
        printf '创建 Python 3.12 虚拟环境：%s\n' "$venv_dir"
        "$bootstrap_python" -m venv "$venv_dir"
    fi
    export OPENVLA_PYTHON="$venv_dir/bin/python"
}

ensure_pytorch() {
    "$OPENVLA_PYTHON" -m pip install --upgrade pip setuptools wheel
    if ! "$OPENVLA_PYTHON" -c \
        'import torch, torchvision; assert torch.__version__.startswith("2.7.1"); assert torch.version.cuda == "12.8"' \
        >/dev/null 2>&1; then
        printf '安装 PyTorch 2.7.1 + CUDA 12.8 到 Python 3.12 环境……\n'
        "$OPENVLA_PYTHON" -m pip install \
            --index-url https://download.pytorch.org/whl/cu128 \
            'torch==2.7.1' 'torchvision==0.22.1'
    fi
}

select_python

printf 'GPU：%s（%s MiB）\n' "$GPU_NAME" "$GPU_MEMORY_MIB"
printf 'OpenVLA-OFT：batch=%s，grad_accum=%s，LoRA rank=%s，chunk=%s\n' \
    "$OPENVLA_BATCH_SIZE" "$OPENVLA_GRAD_ACCUM" "$OPENVLA_LORA_RANK" "$OPENVLA_CHUNK_SIZE"

prepare() {
    ensure_pytorch
    "$LAUNCHER" setup
    "$LAUNCHER" convert
    "$LAUNCHER" check
}

smoke() {
    OPENVLA_OUTPUT_DIR="$BUNDLE_DIR/outputs/smoke_$RUN_STAMP" "$LAUNCHER" smoke
}

train() {
    export OPENVLA_OUTPUT_DIR="${OPENVLA_OUTPUT_DIR:-$BUNDLE_DIR/outputs/openvla_oft_openarm_$RUN_STAMP}"
    "$LAUNCHER" train
}

case "$MODE" in
    prepare)
        prepare
        ;;
    smoke)
        smoke
        ;;
    train)
        train
        ;;
    all)
        prepare
        smoke
        train
        ;;
    *)
        fail "用法：$0 {prepare|smoke|train|all}"
        ;;
esac
