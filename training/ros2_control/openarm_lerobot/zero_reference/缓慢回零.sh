#!/usr/bin/env bash
# Return to the existing mechanical zero; NEVER recalibrate motor encoders.
# With no argument: read-only motion preflight. --execute: preflight + confirmation + movement.
set -eo pipefail

case "${1:-}" in
  '') ZERO_DO_EXECUTE=false ;;
  --execute) ZERO_DO_EXECUTE=true ;;
  -h|--help)
    printf '%s\n' '用法：bash 缓慢回零.sh [--execute]' \
      '默认只检查和拍摄，不发运动目标；--execute 预检通过并输入 ZERO 后缓慢回零。' \
      '不会启动硬件、停止别的程序、修改电机零点或自动恢复遥操作。'
    exit 0 ;;
  *) printf '%s\n' '未知参数；只允许 --execute 或 --help。' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { printf '%s\n' '参数过多。' >&2; exit 2; }

source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
ZERO_PROGRAM=/home/yundu1/ros2_ws/openarm_lerobot/slow_demo_replay.py
[[ -f "$ZERO_PROGRAM" ]] || { printf '缺少程序：%s\n' "$ZERO_PROGRAM" >&2; exit 1; }
ZERO_RUN_TAG="$(date +%Y%m%d_%H%M%S_%N)"
ZERO_PREFLIGHT_DIR="/home/yundu1/ros2_ws/openarm_lerobot/poses/zero_preflight_${ZERO_RUN_TAG}"
ZERO_EXECUTION_DIR="/home/yundu1/ros2_ws/openarm_lerobot/poses/standard_zero_${ZERO_RUN_TAG}"

printf '%s\n' '请先停止录制、自动回放和遥操作桥接，保留硬件控制器运行。' \
  '夹爪应为空，回零路径无人、无障碍，现场急停可用。' \
  '开始只读预检：检查反馈、竞争发布者、关节限位、模型自碰撞和相机。' \
  '模型检查不包含未建模的桌面、支架、盒子或人员。'
/usr/bin/python3 "$ZERO_PROGRAM" zero --out "$ZERO_PREFLIGHT_DIR"

if [[ "$ZERO_DO_EXECUTE" == false ]]; then
  printf '只读预检完成，未移动机械臂。记录：%s\n' "$ZERO_PREFLIGHT_DIR"
  printf '%s\n' '实际回零请带 --execute 运行，并在提示时输入 ZERO。'
  exit 0
fi

[[ -t 0 ]] || { printf '%s\n' '实际回零须在现场交互终端运行。' >&2; exit 1; }
printf '%s\n' '将缓慢回到标准机械零位，双臂到位后闭合夹爪。' \
  '这不是电机零点标定；不会运行 set_zero.sh。'
read -r -p '确认机械臂空载、路径清空且急停可用，输入 ZERO 执行：' ZERO_CONFIRM
[[ "$ZERO_CONFIRM" == ZERO ]] || { printf '%s\n' '已取消，没有发送运动目标。'; exit 0; }
/usr/bin/python3 "$ZERO_PROGRAM" zero --out "$ZERO_EXECUTION_DIR" --execute
printf '回零完成，本次零位记录：%s/zero_pose.json\n' "$ZERO_EXECUTION_DIR"
printf '%s\n' '电机仍使能；未恢复遥操作，也未覆盖桌面保存的原始零位记录。'
