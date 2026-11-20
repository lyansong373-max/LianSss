#!/usr/bin/env bash
# One supervised terminal. No hardware is started without --execute + ZERO.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
exec /usr/bin/python3 /home/yundu1/ros2_ws/openarm_lerobot/enable_and_zero.py "$@"
