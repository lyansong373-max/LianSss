#!/usr/bin/env bash
# 创建 LeRobot 运行环境（conda: lerobot, Python 3.12）并安装 SmolVLA 与 OpenArm 插件
# 可重复运行：已完成的步骤会自动跳过。
# 用法：./setup_lerobot_env.sh                 （默认走代理 127.0.0.1:10808）
#       PROXY= ./setup_lerobot_env.sh          （不走代理）
set -euo pipefail
PROXY="${PROXY-http://127.0.0.1:10808}"
if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
  echo "使用代理: $PROXY"
fi
CONDA=~/miniconda3/bin/conda
ENV=~/miniconda3/envs/lerobot
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -x "$ENV/bin/python" ] || $CONDA create -n lerobot python=3.12 -y
[ -x "$ENV/bin/ffmpeg" ] || $CONDA install -n lerobot -c conda-forge ffmpeg -y

[ -d ~/lerobot/.git ] || git clone https://github.com/huggingface/lerobot.git ~/lerobot
PIP="$ENV/bin/pip"
$PIP install --progress-bar off -e ~/lerobot"[smolvla,core_scripts,training]"
$PIP install --progress-bar off -e "$HERE/lerobot_robot_openarm_ros2" -e "$HERE/lerobot_teleoperator_openarm_exo"

"$ENV/bin/python" - <<'PY'
import torch, lerobot
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from lerobot_robot_openarm_ros2 import OpenArmROS2Config
from lerobot_teleoperator_openarm_exo import OpenArmExoConfig
print("plugins OK:", OpenArmROS2Config().type, OpenArmExoConfig().type)
PY
echo "=== DONE ==="
