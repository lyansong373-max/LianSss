# OpenVLA-OFT

这里放 OpenVLA-OFT 的数据适配、训练和推理脚本。实验在租用的 NVIDIA L40S 服务器上训练，使用 [六物品训练集](../../data/lerobot/six_objects_60/)。本机的基座和检查点需要另行指定。

OpenVLA-OFT 按实际数据接口适配三路 RGB、16 维双臂动作和六条语言指令；保留 episode 与任务标签的对应关系。适配后先做读回与动作范围检查，再训练和离线评估。

目录布局：

- `models/openvla_oft_base_7b/`：转换后的 OpenVLA-OFT 7B 基座。
- `../../work/openvla-oft/checkpoints/fp32_film_chunk15_010000/` 是脚本默认查找的本地 FiLM 检查点路径。
- `models/tokenizer/`：本地分词器文件。
- `training/` 存放训练脚本和 OpenVLA-OFT 插件源码。
- `scripts/`：本机推理与真机启动脚本。
- `configs/`：推理和部署配置。

## 本机 RTX 4090 推理

当前 FiLM 入口默认使用 `../../work/openvla-oft/checkpoints/fp32_film_chunk15_010000/pretrained_model/`。驱动和桥接参见 [ROS 2 控制说明](../ros2_control/openarm_lerobot/README.md)。
在另一个终端先检查设备并执行不连接机器人的模型推理，再进入短时真机交互：

```bash
source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
cd training/openvla-oft/scripts
./run_openvla_oft_film.sh check
./run_openvla_oft_film.sh smoke 0
./run_openvla_oft_film.sh run 5
```

`check` 和 `smoke` 都通过后再执行 `run`。新模型使用单独的 FP32 动作头加载入口和本地 tokenizer，不会修改保存的模型权重。真机命令启动一个六物品交互程序，初始状态只显示物品菜单并等待指令，不会自动选择物品或运动。输入 `螺丝刀`、`透明胶带`、`双面胶`、`抽纸`、`黑色胶带`、`红色胶带`（也支持对应英文 ID）后，程序先核对示教起始状态，慢速进入准备位；准备完成后仍须手动输入 `/start`。

安全阈值以当前脚本和桥接启动参数为准，不要依赖旧实验记录中的数值。示教范围检查不是环境碰撞规划，不能保证运动不会碰到未建模物体。该检查点未做离线验证，也尚未通过真机部署验证。

每次运行会把全局、左腕、右腕三路录像及动作状态保存到仓库根目录下的 `work/openvla-oft/inference_videos/`。结束时程序检查三路 MP4，随后等待下一条物品指令。输入 `/quit` 退出。默认 UDP 端口为 `50110/50111`。

首次使用新检查点前，请先运行 `check` 和 `smoke`；真机试验必须现场监护并确认路径无障碍。
