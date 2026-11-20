#!/usr/bin/env bash
# Train OpenVLA-OFT on the local OpenArm LeRobot dataset with one RTX 4090.
#
# The launcher deliberately uses OpenVLA-OFT instead of autoregressive OpenVLA:
# this dataset has three cameras, 16-D bimanual state/action, and runs at 30 Hz.
# OpenVLA-OFT supports multiple images, proprioception, bimanual actions, and
# parallel action chunks while retaining the OpenVLA-7B backbone.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${OPENVLA_PYTHON:-}" ]]; then
    PYTHON_BIN="$OPENVLA_PYTHON"
elif [[ -x /home/yundu1/miniconda3/envs/lerobot/bin/python ]]; then
    PYTHON_BIN=/home/yundu1/miniconda3/envs/lerobot/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python || true)"
fi
PLUGIN_DIR="${OPENVLA_PLUGIN_DIR:-$SCRIPT_DIR/third_party/lerobot-policy-openvla-oft}"
PROJECT_DATASET_ROOT="$SCRIPT_DIR/../../../data/lerobot/six_objects_60"
BUNDLED_DATASET_ROOT="$SCRIPT_DIR/training_data/six_objects_60"
if [[ -f "$PROJECT_DATASET_ROOT/meta/info.json" ]]; then
    DEFAULT_DATASET_ROOT="$PROJECT_DATASET_ROOT"
else
    DEFAULT_DATASET_ROOT="$BUNDLED_DATASET_ROOT"
fi
DATASET_ROOT="${OPENVLA_DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
DATASET_ID="${OPENVLA_DATASET_ID:-local/openarm_six_objects}"
BASE_DIR="${OPENVLA_BASE_DIR:-$SCRIPT_DIR/outputs/openvla_oft_base_7b}"
OUTPUT_DIR="${OPENVLA_OUTPUT_DIR:-$SCRIPT_DIR/outputs/openvla_oft_openarm_4090}"
CACHE_DIR="${OPENVLA_CACHE_DIR:-$SCRIPT_DIR/.cache/openvla}"

# 4090-safe defaults. LeRobot 0.6.1 does not expose gradient accumulation.
BATCH_SIZE="${OPENVLA_BATCH_SIZE:-1}"
GRAD_ACCUM="${OPENVLA_GRAD_ACCUM:-1}"
LORA_RANK="${OPENVLA_LORA_RANK:-16}"
CHUNK_SIZE="${OPENVLA_CHUNK_SIZE:-4}"
STEPS="${OPENVLA_STEPS:-30000}"
DECAY_STEPS="${OPENVLA_DECAY_STEPS:-20000}"
SAVE_FREQ="${OPENVLA_SAVE_FREQ:-5000}"
NUM_WORKERS="${OPENVLA_NUM_WORKERS:-2}"

export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_DIR/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
    cat <<'EOF'
用法：
  ./train_openvla_4090.sh setup    # 克隆并安装 LeRobot OpenVLA-OFT 插件
  ./train_openvla_4090.sh convert  # 下载并转换 openvla/openvla-7b（约 15 GB）
  ./train_openvla_4090.sh check    # 检查数据集、插件和 CUDA
  ./train_openvla_4090.sh smoke    # 20 步显存/链路冒烟测试
  ./train_openvla_4090.sh train    # 正式训练
  ./train_openvla_4090.sh print    # 只打印正式训练命令

常用覆盖变量：
  OPENVLA_STEPS=50000 OPENVLA_OUTPUT_DIR=/path/to/run ./train_openvla_4090.sh train
  OPENVLA_LORA_RANK=8 OPENVLA_CHUNK_SIZE=2 ./train_openvla_4090.sh smoke

若 24 GB 显存仍 OOM，先把 LORA_RANK 调到 8、CHUNK_SIZE 调到 2；不要提高 batch。
EOF
}

fail() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

"$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1 || \
    fail "OpenVLA-OFT 需要 Python 3.12+；当前为：$($PYTHON_BIN --version 2>&1)"

positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 必须是正整数，当前为：$2"
}

check_scalars() {
    positive_int BATCH_SIZE "$BATCH_SIZE"
    positive_int GRAD_ACCUM "$GRAD_ACCUM"
    positive_int LORA_RANK "$LORA_RANK"
    positive_int CHUNK_SIZE "$CHUNK_SIZE"
    positive_int STEPS "$STEPS"
    positive_int DECAY_STEPS "$DECAY_STEPS"
    positive_int SAVE_FREQ "$SAVE_FREQ"
    positive_int NUM_WORKERS "$NUM_WORKERS"
    (( DECAY_STEPS < STEPS )) || fail "DECAY_STEPS 必须小于 STEPS"
}

check_dataset() {
    [[ -f "$DATASET_ROOT/meta/info.json" ]] || fail "找不到数据集：$DATASET_ROOT/meta/info.json"
    DATASET_ROOT="$DATASET_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
action = features.get("action", {})
state = features.get("observation.state", {})
images = sorted(key for key in features if key.startswith("observation.images."))
problems = []
if action.get("shape") != [16]:
    problems.append(f"action 不是 16 维：{action.get('shape')}")
if state.get("shape") != [16]:
    problems.append(f"observation.state 不是 16 维：{state.get('shape')}")
if len(images) != 3:
    problems.append(f"相机不是 3 路：{images}")
if info.get("total_episodes", 0) < 2:
    problems.append("episode 数不足 2，无法划分训练/验证集")
if problems:
    raise SystemExit("；".join(problems))
print(
    f"数据集通过：{info['total_episodes']} episodes，{info['total_frames']} frames，"
    f"16-D 双臂动作，3 cameras，{info['fps']} Hz"
)
PY
}

check_runtime() {
    [[ -x "$PYTHON_BIN" ]] || fail "Python 不存在：$PYTHON_BIN"
    "$PYTHON_BIN" -c 'import lerobot_policy_openvla_oft' 2>/dev/null || \
        fail "未安装 lerobot_policy_openvla_oft；先运行 setup"
    "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用；请检查 NVIDIA 驱动与 PyTorch CUDA 安装")
props = torch.cuda.get_device_properties(0)
total_gib = props.total_memory / 1024**3
print(f"CUDA 通过：{props.name}，{total_gib:.1f} GiB，bf16={torch.cuda.is_bf16_supported()}")
if total_gib < 23:
    raise SystemExit("显存少于 23 GiB，不适用本启动配置")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("GPU 不支持 bfloat16")
PY
}

setup_plugin() {
    if [[ ! -f "$PLUGIN_DIR/pyproject.toml" ]]; then
        command -v git >/dev/null || fail "缺少 git，且训练包内没有 OpenVLA-OFT 插件源码"
        mkdir -p "$(dirname -- "$PLUGIN_DIR")"
        git clone --depth 1 https://github.com/tc-huang/lerobot-policy-openvla-oft.git "$PLUGIN_DIR"
    fi
    "$PYTHON_BIN" -m pip install -e "$PLUGIN_DIR[training]"
    printf '插件安装完成：%s\n' "$PLUGIN_DIR"
}

convert_base() {
    check_runtime
    if [[ -f "$BASE_DIR/config.json" ]]; then
        printf '基座已经存在，跳过转换：%s\n' "$BASE_DIR"
        return
    fi
    mkdir -p "$(dirname -- "$BASE_DIR")"
    "$PYTHON_BIN" -m lerobot_policy_openvla_oft.convert_checkpoint \
        --base --repo-id openvla/openvla-7b --output-dir "$BASE_DIR"
}

build_train_command() {
    local run_output="$1"
    local run_steps="$2"
    local run_save_freq="$3"
    TRAIN_COMMAND=(
        "$PYTHON_BIN" -m lerobot.scripts.lerobot_train
        "--policy.path=$BASE_DIR"
        --policy.push_to_hub=false
        "--policy.chunk_size=$CHUNK_SIZE"
        "--policy.n_action_steps=$CHUNK_SIZE"
        --policy.dtype=bfloat16
        '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX"}'
        --policy.proprio_projector_fp32=false
        --policy.use_film=false
        --policy.compile_model=false
        --policy.optimizer_lr=5e-4
        "--policy.scheduler_decay_steps=$DECAY_STEPS"
        --policy.scheduler_decay_factor=0.1
        --peft.method_type=LORA
        "--peft.r=$LORA_RANK"
        "--dataset.repo_id=$DATASET_ID"
        "--dataset.root=$DATASET_ROOT"
        --dataset.eval_split=0.1
        --dataset.video_backend=pyav
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.max_num_transforms=4
        '--dataset.image_transforms.tfs={"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}}}'
        "--batch_size=$BATCH_SIZE"
        "--steps=$run_steps"
        "--save_freq=$run_save_freq"
        --log_freq=20
        --eval_steps=1000
        --max_eval_samples=600
        "--num_workers=$NUM_WORKERS"
        --wandb.enable=false
        "--output_dir=$run_output"
        --job_name=openvla_oft_openarm_4090
    )
    if (( GRAD_ACCUM > 1 )); then
        if "$PYTHON_BIN" -c 'import dataclasses; from lerobot.configs.train import TrainPipelineConfig; raise SystemExit(not any(f.name == "accelerator" for f in dataclasses.fields(TrainPipelineConfig)))' >/dev/null 2>&1; then
            TRAIN_COMMAND+=("--accelerator.gradient_accumulation.steps=$GRAD_ACCUM")
        else
            fail "当前 LeRobot 不支持梯度累积参数；请设置 OPENVLA_GRAD_ACCUM=1"
        fi
    fi
}

print_command() {
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
}

run_training() {
    local run_output="$1"
    local run_steps="$2"
    local run_save_freq="$3"
    check_scalars
    check_dataset
    check_runtime
    [[ -f "$BASE_DIR/config.json" ]] || fail "未找到转换后的基座：$BASE_DIR；先运行 convert"
    [[ ! -e "$run_output" ]] || fail "输出目录已存在：$run_output；请换 OPENVLA_OUTPUT_DIR"
    build_train_command "$run_output" "$run_steps" "$run_save_freq"
    printf '训练配置：batch=%s，grad_accum=%s，effective_batch=%s，LoRA r=%s，chunk=%s\n' \
        "$BATCH_SIZE" "$GRAD_ACCUM" "$((BATCH_SIZE * GRAD_ACCUM))" "$LORA_RANK" "$CHUNK_SIZE"
    print_command
    exec "${TRAIN_COMMAND[@]}"
}

MODE="${1:-}"
case "$MODE" in
    setup)
        setup_plugin
        ;;
    convert)
        convert_base
        ;;
    check)
        check_scalars
        check_dataset
        check_runtime
        [[ -f "$BASE_DIR/config.json" ]] || fail "未转换基座；运行 convert"
        printf 'OpenVLA-OFT 4090 训练前检查全部通过。\n'
        ;;
    smoke)
        SMOKE_OUTPUT="${OPENVLA_OUTPUT_DIR:-$SCRIPT_DIR/outputs/openvla_oft_4090_smoke}"
        run_training "$SMOKE_OUTPUT" 20 20
        ;;
    train)
        run_training "$OUTPUT_DIR" "$STEPS" "$SAVE_FREQ"
        ;;
    print)
        check_scalars
        build_train_command "$OUTPUT_DIR" "$STEPS" "$SAVE_FREQ"
        print_command
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        usage >&2
        fail "未知模式：$MODE"
        ;;
esac
