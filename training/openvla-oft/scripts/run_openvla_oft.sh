#!/usr/bin/env bash
# OpenVLA-OFT inference launcher for the local RTX 4090 and OpenArm ROS2 bridge.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd -- "$OPENVLA_DIR/../.." && pwd)"
OPENARM_DIR="$PROJECT_DIR/training/ros2_control/openarm_lerobot"
CHECKPOINT="${OPENVLA_CHECKPOINT:-$PROJECT_DIR/work/openvla-oft/checkpoints/fp32_full_005000/pretrained_model}"
BASE_MODEL="${OPENVLA_BASE_MODEL:-$OPENVLA_DIR/models/openvla_oft_base_7b}"
TOKENIZER="${OPENVLA_TOKENIZER:-$OPENVLA_DIR/models/tokenizer}"
PYTHON_BIN="${OPENVLA_PYTHON:-/home/yundu1/miniconda3/envs/lerobot/bin/python}"
PLUGIN_SRC="$OPENVLA_DIR/training/third_party/lerobot-policy-openvla-oft/src"
ROBOT_PLUGIN="$OPENARM_DIR/lerobot_robot_openarm_ros2"

CAMERA1_DEVICE="${OPENVLA_CAMERA1:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0}"
CAMERA2_DEVICE="${OPENVLA_CAMERA2:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0}"
CAMERA3_DEVICE="${OPENVLA_CAMERA3:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0}"
STATE_PORT="${OPENVLA_STATE_PORT:-50110}"
CMD_PORT="${OPENVLA_CMD_PORT:-50111}"
MAX_STATE_AGE_S="${OPENVLA_MAX_STATE_AGE_S:-1.5}"

MODE="${1:-check}"

export PYTHONPATH="$OPENVLA_DIR/training:$PLUGIN_SRC:$ROBOT_PLUGIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENVLA_BASE_MODEL="$BASE_MODEL"
export OPENVLA_TOKENIZER="$TOKENIZER"
export HF_HOME="${HF_HOME:-$OPENVLA_DIR/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

fail() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
用法：
  ./run_openvla_oft.sh check
  ./run_openvla_oft.sh smoke [数据帧索引]
  ./run_openvla_oft.sh run [每次抓取持续秒数]

物品名：screwdriver、transparent_tape、double_sided_tape、tissue、black_tape、red_tape
默认模型：新训练的 fp32_full_005000；本机基座和 tokenizer 不会重复下载。

推荐顺序：
  1. ./run_openvla_oft.sh check
  2. ./run_openvla_oft.sh smoke 0       # 只跑模型，不连接真机
  3. ./run_openvla_oft.sh run 20   # 真机会先缓慢移动到示教起始位
  4. 程序等待你输入物品；安全准备和起始姿态检查完成后输入 /start。
  5. 三路录像保存并校验后，可继续输入下一件物品；输入 /quit 才退出程序。

可覆盖环境变量：
  OPENVLA_STATE_PORT=50110 OPENVLA_CMD_PORT=50111
  OPENVLA_MAX_STATE_AGE_S=1.5
  OPENVLA_CAMERA1=/dev/... OPENVLA_CAMERA2=/dev/... OPENVLA_CAMERA3=/dev/...
EOF
}

check_files_and_cuda() {
    [[ -x "$PYTHON_BIN" ]] || fail "找不到 Python：$PYTHON_BIN"
    [[ -f "$CHECKPOINT/adapter_model.safetensors" ]] || fail "缺少 LoRA 检查点：$CHECKPOINT"
    [[ -f "$CHECKPOINT/policy_preprocessor.json" ]] || fail "缺少策略预处理配置"
    [[ -f "$BASE_MODEL/model.safetensors" ]] || fail "缺少 OpenVLA-OFT 7B 基座：$BASE_MODEL"
    [[ -f "$TOKENIZER/tokenizer.json" ]] || fail "缺少 tokenizer：$TOKENIZER"
    [[ "$STATE_PORT" =~ ^[0-9]+$ && "$CMD_PORT" =~ ^[0-9]+$ ]] || fail "UDP 端口必须是整数"
    OPENVLA_CHECKPOINT_PATH="$CHECKPOINT" \
    OPENVLA_BASE_MODEL_PATH="$BASE_MODEL" \
    OPENVLA_TOKENIZER_PATH="$TOKENIZER" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import torch
import lerobot
import lerobot_policy_openvla_oft
import lerobot_robot_openarm_ros2

checkpoint = Path(os.environ["OPENVLA_CHECKPOINT_PATH"])
config = json.loads((checkpoint / "config.json").read_text())
if config.get("type") != "openvla_oft" or not config.get("use_peft"):
    raise RuntimeError("检查点不是 OpenVLA-OFT LoRA")
if os.environ.get("OPENVLA_REQUIRE_FILM") == "1" and not config.get("use_film"):
    raise RuntimeError("FiLM 专用入口拒绝加载未启用 FiLM 的模型")
precision_path = checkpoint / "precision_requirements.json"
if precision_path.exists():
    precision = json.loads(precision_path.read_text())
    if (precision.get("head_dtype") != "torch.float32"
            or precision.get("loader") != "train_openvla_oft_fp32.load_fp32_adapter"):
        raise RuntimeError("新检查点的 FP32 加载要求不匹配")
    from train_openvla_oft_fp32 import verify_checkpoint_precision
    verified = verify_checkpoint_precision(checkpoint, bool(config.get("use_film", False)))
    if config.get("use_film") and (not precision.get("use_film") or precision.get("film_tensor_count") != 8):
        raise RuntimeError("FiLM 检查点精度声明不匹配")
    print(f"FiLM={bool(config.get('use_film', False))}，FiLM FP32 张量数={verified['film_tensor_count']}")
    print("FP32 检查点：推理时使用本机基座和 tokenizer 路径覆盖服务器路径")
else:
    if Path(config["pretrained_path"]).resolve() != Path(os.environ["OPENVLA_BASE_MODEL_PATH"]).resolve():
        raise RuntimeError("config.json 的基座路径与当前基座不一致")
    if Path(config["tokenizer_name"]).resolve() != Path(os.environ["OPENVLA_TOKENIZER_PATH"]).resolve():
        raise RuntimeError("config.json 的 tokenizer 路径与当前 tokenizer 不一致")
if not torch.cuda.is_available():
    raise RuntimeError("当前终端看不到 CUDA；检查 NVIDIA 驱动、CUDA_VISIBLE_DEVICES 和运行环境")
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
if total < 22:
    raise RuntimeError(f"显存不足：{total:.1f} GiB；请使用24GB级显卡并先运行 smoke 实测峰值")
print(f"检查通过：{name}，{total:.1f} GiB，LeRobot {getattr(lerobot, '__version__', '?')}")
print(f"策略插件：{lerobot_policy_openvla_oft.__file__}")
print(f"机器人插件：{lerobot_robot_openarm_ros2.__file__}")
PY
}

check_cameras() {
    [[ -e "$CAMERA1_DEVICE" ]] || fail "找不到 camera1：$CAMERA1_DEVICE"
    [[ -e "$CAMERA2_DEVICE" ]] || fail "找不到 camera2：$CAMERA2_DEVICE"
    [[ -e "$CAMERA3_DEVICE" ]] || fail "找不到 camera3：$CAMERA3_DEVICE"
}

case "$MODE" in
    check)
        check_files_and_cuda
        check_cameras
        printf '三路相机路径存在。下一步执行：bash %s smoke 0\n' "${OPENVLA_ENTRYPOINT:-$0}"
        ;;
    smoke)
        SAMPLE_INDEX="${2:-0}"
        [[ "$SAMPLE_INDEX" =~ ^[0-9]+$ ]] || fail "数据帧索引必须是非负整数"
        check_files_and_cuda
        exec "$PYTHON_BIN" "$SCRIPT_DIR/smoke_infer_openvla_oft.py" \
            --checkpoint "$CHECKPOINT" \
            --sample-index "$SAMPLE_INDEX"
        ;;
    run)
        DURATION="${2:-20}"
        [[ "$DURATION" =~ ^[0-9]+$ && "$DURATION" -gt 0 ]] || fail "持续时间必须是正整数秒"
        check_files_and_cuda
        check_cameras
        [[ -f "$CHECKPOINT/precision_requirements.json" ]] || fail "当前真机入口需要新训练的 FP32 检查点"
        export CAMERA_TOP="$CAMERA1_DEVICE"
        export CAMERA_LEFT_WRIST="$CAMERA2_DEVICE"
        export CAMERA_RIGHT_WRIST="$CAMERA3_DEVICE"
        printf 'OpenVLA-OFT 六物品交互程序已启动；当前不会移动，等待你的物品指令。\n'
        printf '本程序会先弯举再自行启动桥接；不要同时运行独立的 LeRobot 桥接。\n'
        printf '确认现场硬件急停有人看护；每次安全准备完成后仍需手动输入 /start。\n'
        exec "$PYTHON_BIN" "$OPENARM_DIR/interactive_openvla_oft_grasp.py" \
            --execute \
            --record-videos \
            --output-root "$PROJECT_DIR/work/openvla-oft/inference_videos" \
            --duration "$DURATION" \
            --policy-path "$CHECKPOINT" \
            --rollout-script "$SCRIPT_DIR/rollout_openvla_oft_fp32.py" \
            --state-port "$STATE_PORT" \
            --cmd-port "$CMD_PORT" \
            --max-state-age-s "$MAX_STATE_AGE_S"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        fail "未知模式：$MODE"
        ;;
esac
