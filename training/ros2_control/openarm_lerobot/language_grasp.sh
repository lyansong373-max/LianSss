#!/usr/bin/env bash
# Five-object language-conditioned grasping with the existing OpenArm ROS2 plugins.
# This script starts LeRobot only; start ROS2 hardware, exo retargeting and bridge separately.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/lerobot_robot_openarm_ros2:$SCRIPT_DIR/lerobot_teleoperator_openarm_exo${PYTHONPATH:+:$PYTHONPATH}"
LEROBOT_BIN="${LEROBOT_BIN:-/home/yundu1/miniconda3/envs/lerobot/bin}"
DATASET_ID="${DATASET_ID:-local/openarm_language_grasp_v1}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/datasets/language_grasp_v1}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$SCRIPT_DIR/outputs/train/language_grasp_smolvla_v1}"
POLICY_PATH="${POLICY_PATH:-$TRAIN_OUTPUT/checkpoints/last/pretrained_model}"
START_POSE_PATH="${START_POSE_PATH:-$SCRIPT_DIR/config/curl_start_pose.json}"
# Verified 2026-11-01: video4=global (USB 10), video2=left/yellow (USB 9), video0=right/white (USB 3).
# USB by-path names follow physical USB ports, so keep these connections fixed.
CAMERA_TOP="${CAMERA_TOP:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0}"
CAMERA_LEFT_WRIST="${CAMERA_LEFT_WRIST:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0}"
CAMERA_RIGHT_WRIST="${CAMERA_RIGHT_WRIST:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0}"

fail() { printf '%s\n' "$*" >&2; exit 2; }

help_text() {
  printf '%s\n' \
    '用法：bash language_grasp.sh <模式> [参数]' \
    '' \
    '  tasks                         显示五个物品和对应英文指令' \
    '  config                        显示实际传给 LeRobot 的相机配置' \
    '  cameras                       检测相机并保存预览帧（不连接机械臂）' \
    '  teach-start                   外骨骼示教弯举姿态，输入 s 回车保存' \
    '  show-start                    查看已保存的实际起始姿态' \
    '  home                          自动缓慢回到已保存的弯举姿态' \
    '  demo-tissue [录制秒数=120]     15秒后接管，跟随5秒后录制；右箭头结束并保存' \
    '  demo-screwdriver [秒数=120]    抓取螺丝刀，独立数据目录；同样按右箭头保存' \
    '  demo-white-tape [秒数=120]     抓取双面胶（原白色胶带），右箭头保存' \
    '  demo-double-sided-tape [秒数]  同上，沿用原数据目录，任务名为抓取双面胶' \
    '  demo-clear-tape [秒数=120]     抓取透明胶带，独立数据目录；右箭头保存' \
    '  demo-red-tape [秒数=120]       红色胶带放入蓝盒，独立目录；右箭头保存' \
    '  demo-black-tape [秒数=120]     黑色胶带放入蓝盒，独立目录；右箭头保存' \
    '  record <物品ID> [条数=10]      每条先回位并等待外骨骼对齐，再采集' \
    '  inspect [episode=0]           在 Rerun 中查看已保存的数据' \
    '  train [步数=20000] [batch=8]   微调 SmolVLA，使用 CUDA 和 bf16' \
    '  test [秒数=30]                启动后先回位，等你输入物品后再抓取' \
    '  rollout <物品ID> [秒数=30]     先回位，再执行命令行指定的抓取任务' \
    '' \
    '已配置：全局=video4，左腕/黄色=video2，右腕/白色=video0。' \
    '默认使用已核对的 USB by-path 路径；更换 USB 端口后需重新核对。' \
    '如需覆盖相机对应关系：' \
    '  export CAMERA_TOP=/dev/video...' \
    '  export CAMERA_LEFT_WRIST=/dev/video...' \
    '  export CAMERA_RIGHT_WRIST=/dev/video...' \
    '也可使用 /dev/v4l/by-path/...-video-index0 等稳定路径。' \
    'camera1=全局；camera2=左腕；camera3=右腕。' \
    '' \
    '可选环境变量：LEROBOT_BIN、DATASET_ID、DATASET_ROOT、TRAIN_OUTPUT、POLICY_PATH、START_POSE_PATH。' \
    '录制前需已启动真实 ROS2 硬件、外骨骼输入、重定向和 LeRobot 桥接。' \
    'teach-start/home/test 前关闭其他 LeRobot 遥操作、录制或推理进程。' \
    '自动回位是关节空间插值；执行前确保回位路径无障碍。' \
    '正式录制使用真实图像；不要向此数据集追加旧的黑图测试数据。'
}

task_text() {
  case "${1:-}" in
    white_tape|白色胶带|抓取白色胶带) printf '%s' 'Pick up the white tape.' ;;
    black_tape|黑色胶带|抓取黑色胶带) printf '%s' 'Pick up the black tape.' ;;
    tissue|纸巾|抓取纸巾) printf '%s' 'Pick up the tissue.' ;;
    water_bottle|水瓶|抓取水瓶) printf '%s' 'Pick up the water bottle.' ;;
    clear_tape|透明胶带|抓取透明胶带) printf '%s' 'Pick up the clear tape.' ;;
    *) fail '未知物品ID。可用：white_tape black_tape tissue water_bottle clear_tape' ;;
  esac
}

positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] || fail "需要正整数：$1"
}

nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]] || fail "需要非负整数：$1"
}

need_binary() {
  [[ -x "$LEROBOT_BIN/$1" ]] || fail "找不到 $LEROBOT_BIN/$1；请设置 LEROBOT_BIN。"
}

need_dataset() {
  [[ -f "$DATASET_ROOT/meta/info.json" ]] || fail "未找到数据集：$DATASET_ROOT/meta/info.json；请先采集。"
}

need_pose() {
  [[ -f "$START_POSE_PATH" ]] || fail "请先运行 teach-start 保存弯举姿态：$START_POSE_PATH"
}

run_policy() {
  exec "$LEROBOT_BIN/lerobot-rollout" \
    --strategy.type=base "--policy.path=$POLICY_PATH" \
    --robot.type=openarm_ros2 --robot.id=openarm_bi \
    "--robot.cameras=$CAMERAS" --robot.dummy_cameras=0 \
    "--task=$TASK" --fps=30 "--duration=$DURATION" \
    --return_to_initial_position=false --display_data=true --display_mode=rerun
}

camera_config() {
  [[ -n "${CAMERA_TOP:-}" ]] || fail '请先设置 CAMERA_TOP（全局相机）。'
  [[ -n "${CAMERA_LEFT_WRIST:-}" ]] || fail '请先设置 CAMERA_LEFT_WRIST（左腕相机）。'
  [[ -n "${CAMERA_RIGHT_WRIST:-}" ]] || fail '请先设置 CAMERA_RIGHT_WRIST（右腕相机）。'
  need_binary python
  "$LEROBOT_BIN/python" - "$CAMERA_TOP" "$CAMERA_LEFT_WRIST" "$CAMERA_RIGHT_WRIST" <<'PY'
import json
import sys
from pathlib import Path

devices = [int(value) if value.isdecimal() else value for value in sys.argv[1:]]
resolved = [str(Path(f'/dev/video{value}' if isinstance(value, int) else value).resolve())
            for value in devices]
if len(set(resolved)) != 3:
    raise SystemExit('全局、左腕、右腕必须是三个不同的视频设备。')
print(json.dumps({
    f'camera{i}': {
        'type': 'opencv', 'index_or_path': device,
        'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG',
    }
    for i, device in enumerate(devices, 1)
}))
PY
}

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  help|-h|--help)
    help_text
    ;;
  tasks)
    for task_id in white_tape black_tape tissue water_bottle clear_tape; do
      printf '%-14s %s\n' "$task_id" "$(task_text "$task_id")"
    done
    ;;
  config)
    [[ $# -eq 0 ]] || fail '用法：config'
    camera_config
    ;;
  cameras)
    [[ $# -eq 0 ]] || fail '用法：cameras'
    need_binary lerobot-find-cameras
    exec "$LEROBOT_BIN/lerobot-find-cameras" opencv
    ;;
  teach-start|show-start|home)
    [[ $# -eq 0 ]] || fail "用法：$MODE"
    need_binary python
    case "$MODE" in
      teach-start) POSE_MODE=teach ;;
      show-start) need_pose; POSE_MODE=show ;;
      home) need_pose; POSE_MODE=home ;;
    esac
    exec "$LEROBOT_BIN/python" "$SCRIPT_DIR/start_pose.py" "$POSE_MODE" --pose "$START_POSE_PATH"
    ;;
  record)
    [[ $# -ge 1 && $# -le 2 ]] || fail '用法：record <物品ID> [条数=10]'
    TASK="$(task_text "$1")"
    EPISODES="${2:-10}"
    positive_int "$EPISODES"
    need_pose
    CAMERAS="$(camera_config)"
    need_binary lerobot-record
    RESUME=false
    if [[ -f "$DATASET_ROOT/meta/info.json" ]]; then
      RESUME=true
    elif [[ -e "$DATASET_ROOT" ]]; then
      fail "目录已存在但缺少数据集元数据，请检查或设置新的 DATASET_ROOT：$DATASET_ROOT"
    fi
    printf '任务：%s\n数据集：%s\n追加：%s；本次新增：%s 条\n' "$TASK" "$DATASET_ROOT" "$RESUME" "$EPISODES"
    exec "$LEROBOT_BIN/python" "$SCRIPT_DIR/record_with_start_pose.py" --pose "$START_POSE_PATH" -- \
      --robot.type=openarm_ros2 --robot.id=openarm_bi \
      "--robot.cameras=$CAMERAS" --robot.dummy_cameras=0 \
      --teleop.type=openarm_exo --teleop.id=qnbot_exo --teleop.fail_on_stale=true \
      "--dataset.repo_id=$DATASET_ID" "--dataset.root=$DATASET_ROOT" \
      --dataset.no_stamp=true --dataset.push_to_hub=false \
      "--dataset.single_task=$TASK" "--dataset.num_episodes=$EPISODES" \
      --dataset.fps=30 --dataset.episode_time_s=30 --dataset.reset_time_s=15 \
      --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
      "--resume=$RESUME" --display_data=true --display_mode=rerun
    ;;
  demo-tissue|demo-screwdriver|demo-white-tape|demo-double-sided-tape|demo-clear-tape|demo-red-tape|demo-black-tape)
    [[ $# -le 1 ]] || fail "用法：$MODE [最长秒数=120]"
    DURATION="${1:-120}"
    positive_int "$DURATION"
    CAMERAS="$(camera_config)"
    need_binary python
    if [[ "$MODE" == demo-screwdriver ]]; then
      DEMO_ROOT="${SCREWDRIVER_DATASET_ROOT:-$SCRIPT_DIR/datasets/screwdriver/demonstrations}"
      DEMO_ID="${SCREWDRIVER_DATASET_ID:-local/openarm_screwdriver_demonstrations}"
      DEMO_LABEL='抓取螺丝刀'
      TASK='Pick up the screwdriver.'
    elif [[ "$MODE" == demo-white-tape || "$MODE" == demo-double-sided-tape ]]; then
      DEMO_ROOT="${WHITE_TAPE_DATASET_ROOT:-$SCRIPT_DIR/datasets/white_tape/demonstrations}"
      DEMO_ID="${WHITE_TAPE_DATASET_ID:-local/openarm_white_tape_demonstrations}"
      DEMO_LABEL='抓取双面胶：夹紧后先向上提离挂钩，再向外撤回'
      TASK='抓取双面胶'
    elif [[ "$MODE" == demo-clear-tape ]]; then
      DEMO_ROOT="${CLEAR_TAPE_DATASET_ROOT:-$SCRIPT_DIR/datasets/clear_tape/demonstrations}"
      DEMO_ID="${CLEAR_TAPE_DATASET_ID:-local/openarm_clear_tape_demonstrations}"
      DEMO_LABEL='抓取透明胶带'
      TASK='Pick up the clear tape.'
    elif [[ "$MODE" == demo-red-tape ]]; then
      DEMO_ROOT="${RED_TAPE_DATASET_ROOT:-$SCRIPT_DIR/datasets/red_tape/demonstrations}"
      DEMO_ID="${RED_TAPE_DATASET_ID:-local/openarm_red_tape_demonstrations}"
      DEMO_LABEL='抓取红色胶带并放入蓝盒'
      TASK='Pick up the red tape and place it in the blue box.'
    elif [[ "$MODE" == demo-black-tape ]]; then
      DEMO_ROOT="${BLACK_TAPE_DATASET_ROOT:-$SCRIPT_DIR/datasets/black_tape/demonstrations}"
      DEMO_ID="${BLACK_TAPE_DATASET_ID:-local/openarm_black_tape_demonstrations}"
      DEMO_LABEL='抓取黑色胶带并放入蓝盒'
      TASK='Pick up the black tape and place it in the blue box.'
    else
      DEMO_ROOT="${DEMO_DATASET_ROOT:-$SCRIPT_DIR/datasets/tissue_to_blue_box_demonstrations}"
      DEMO_ID="${DEMO_DATASET_ID:-local/openarm_tissue_to_blue_box_demonstrations}"
      DEMO_LABEL='整包抽纸从托架放入蓝盒'
      TASK='Pick up the whole tissue package from the shelf and place it in the blue box.'
    fi
    RESUME=false
    if [[ -f "$DEMO_ROOT/meta/info.json" ]]; then
      RESUME=true
    elif [[ -e "$DEMO_ROOT" ]]; then
      fail "示范目录已存在但缺少元数据；请检查或为该模式设置新的数据目录：$DEMO_ROOT"
    fi
    printf '连续示范：%s\n任务文字：%s\n保存位置：%s\n本次新增 1 条，正式录制最长 %s 秒；设备就绪后15秒自动接管，跟随5秒后录制。\n结束方式：聚焦录制终端按右箭头→结束并保存；夹爪闭合不再触发结束，无自动回位。\n' "$DEMO_LABEL" "$TASK" "$DEMO_ROOT" "$DURATION"
    exec "$LEROBOT_BIN/python" "$SCRIPT_DIR/record_continuous_demo.py" \
      --robot.type=openarm_ros2 --robot.id=openarm_bi \
      "--robot.cameras=$CAMERAS" --robot.dummy_cameras=0 \
      --teleop.type=openarm_exo --teleop.id=qnbot_exo --teleop.fail_on_stale=true \
      "--dataset.repo_id=$DEMO_ID" "--dataset.root=$DEMO_ROOT" \
      --dataset.no_stamp=true --dataset.push_to_hub=false \
      "--dataset.single_task=$TASK" \
      --dataset.num_episodes=1 --dataset.fps=30 \
      "--dataset.episode_time_s=$DURATION" --dataset.reset_time_s=0 \
      --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
      "--resume=$RESUME" --display_data=true --display_mode=rerun
    ;;
  inspect)
    [[ $# -le 1 ]] || fail '用法：inspect [episode=0]'
    EPISODE="${1:-0}"
    nonnegative_int "$EPISODE"
    need_dataset
    need_binary lerobot-dataset-viz
    exec "$LEROBOT_BIN/lerobot-dataset-viz" \
      --repo-id "$DATASET_ID" --root "$DATASET_ROOT" --episode-index "$EPISODE"
    ;;
  train)
    [[ $# -le 2 ]] || fail '用法：train [步数=20000] [batch=8]'
    STEPS="${1:-20000}"
    BATCH_SIZE="${2:-8}"
    positive_int "$STEPS"
    positive_int "$BATCH_SIZE"
    need_dataset
    [[ ! -e "$TRAIN_OUTPUT" ]] || fail "训练目录已存在；新一轮训练请设置新的 TRAIN_OUTPUT：$TRAIN_OUTPUT"
    need_binary python
    "$LEROBOT_BIN/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch"; print(torch.cuda.get_device_name(0))'
    need_binary lerobot-train
    exec "$LEROBOT_BIN/lerobot-train" \
      --policy.path=lerobot/smolvla_base --policy.input_features=null \
      --policy.device=cuda --policy.push_to_hub=false \
      "--dataset.repo_id=$DATASET_ID" "--dataset.root=$DATASET_ROOT" \
      "--output_dir=$TRAIN_OUTPUT" --job_name=openarm_language_grasp \
      "--batch_size=$BATCH_SIZE" "--steps=$STEPS" --save_freq=5000 \
      --accelerator.mixed_precision=bf16 --num_workers=4 --wandb.enable=false
    ;;
  rollout)
    [[ $# -ge 1 && $# -le 2 ]] || fail '用法：rollout <物品ID> [秒数=30]'
    TASK="$(task_text "$1")"
    DURATION="${2:-30}"
    positive_int "$DURATION"
    CAMERAS="$(camera_config)"
    need_pose
    [[ -f "$POLICY_PATH/config.json" ]] || fail "未找到模型配置：$POLICY_PATH/config.json"
    need_binary lerobot-rollout
    "$LEROBOT_BIN/python" "$SCRIPT_DIR/start_pose.py" home --pose "$START_POSE_PATH"
    run_policy
    ;;
  test)
    [[ $# -le 1 ]] || fail '用法：test [秒数=30]'
    DURATION="${1:-30}"
    positive_int "$DURATION"
    [[ -t 0 ]] || fail 'test 模式请在交互终端运行'
    CAMERAS="$(camera_config)"
    need_pose
    [[ -f "$POLICY_PATH/config.json" ]] || fail "未找到模型配置：$POLICY_PATH/config.json"
    need_binary lerobot-rollout
    "$LEROBOT_BIN/python" "$SCRIPT_DIR/start_pose.py" home --pose "$START_POSE_PATH"
    printf '%s\n' '请输入本次抓取指令：白色胶带 / 黑色胶带 / 纸巾 / 水瓶 / 透明胶带。输入 q 退出。'
    while true; do
      read -r -p '抓取指令> ' OBJECT || exit 0
      [[ "$OBJECT" != q ]] || exit 0
      if TASK="$(task_text "$OBJECT")"; then break; fi
    done
    run_policy
    ;;
  *)
    fail "未知模式：$MODE。使用 --help 查看用法。"
    ;;
esac
