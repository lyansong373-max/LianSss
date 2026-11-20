# 螺丝刀抓取示范

独立于抽纸示范。任务文字：`Pick up the screwdriver.`（抓取螺丝刀）。
不预设“放入蓝盒”；若要训练抓取并放置，应另定任务文字和独立数据集。

运行：

```bash
conda activate lerobot
cd /home/yundu1/ros2_ws/openarm_lerobot
bash language_grasp.sh demo-screwdriver 180
```

数据集根目录：本目录下的 `demonstrations/`，首次启动录制时由 LeRobot 创建。
不要手动预建该子目录或在其中放杂项文件。`data/`、`videos/`、`meta/` 由 LeRobot 管理，
每次成功录制追加一个 episode，不手动为每条示范移动视频或拆目录。
默认 repo_id：`local/openarm_screwdriver_demonstrations`；不自动上传。

启动前保持硬件、WebSocket、映射、目标RViz及LeRobot桥接正常；夹爪桥接量程0.03。
关闭旧录制、相机预览和旧exoskeleton_bridge，禁止重复控制。
设备就绪后15秒自动接管，跟随5秒后正式录制；聚焦录制终端按右箭头结束并保存。
不自动回弯举姿态，不通过夹爪闭合结束；最长录制时限仍有效，按键不是急停。
保存三个相机图像、实际状态、请求动作和任务文字；保存不代表抓取成功，训练前需人工检查。

专用覆盖变量：`SCREWDRIVER_DATASET_ROOT`、`SCREWDRIVER_DATASET_ID`。
本入口不使用抽纸入口的 `DEMO_DATASET_ROOT`、`DEMO_DATASET_ID`，防止旧环境变量串目录。
