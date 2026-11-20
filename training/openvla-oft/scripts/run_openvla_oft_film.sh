#!/usr/bin/env bash
# Explicit entry point for the downloaded FiLM checkpoint. No motion by default.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
MODE="${1:-check}"
case "$MODE" in
  -h|--help|help)
    printf '%s\n' \
      'OpenVLA-OFT + FiLM chunk 15 / 10000 步推理入口（不替换旧入口默认模型）' \
      'bash run_openvla_oft_film.sh check      # 检查模型、GPU和相机路径，不移动机器人' \
      'bash run_openvla_oft_film.sh smoke 0    # 数据集单帧推理，不连接机器人' \
      'bash run_openvla_oft_film.sh run 25     # 六物品交互抓取，每次最多25秒，三路录像' \
      '注意：run 模式输入物品名就会移动到准备位；输入 /start 才执行模型抓取。' \
      '本入口自行完成弯举、启动桥接和录像；不要另外启动 LeRobot 桥接。' \
      '退出用 /quit；软件退出不是硬件急停。' \
      '可用 OPENVLA_FILM_CHECKPOINT 显式覆盖本入口模型路径。'
    exit 0 ;;
  check|smoke|run) ;;
  *) printf '未知模式：%s\n' "$MODE" >&2; exit 2 ;;
esac
export OPENVLA_CHECKPOINT="${OPENVLA_FILM_CHECKPOINT:-$PROJECT_DIR/work/openvla-oft/checkpoints/fp32_film_chunk15_010000/pretrained_model}"
export OPENVLA_REQUIRE_FILM=1
export OPENVLA_ENTRYPOINT="$SCRIPT_DIR/run_openvla_oft_film.sh"
printf 'FiLM 模型：%s\n' "$OPENVLA_CHECKPOINT"
if [[ "$MODE" == run && $# -lt 2 ]]; then
  exec bash "$SCRIPT_DIR/run_openvla_oft.sh" run 25
fi
exec bash "$SCRIPT_DIR/run_openvla_oft.sh" "${@:-check}"
