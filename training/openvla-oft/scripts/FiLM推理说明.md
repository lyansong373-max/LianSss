# OpenVLA-OFT + FiLM 推理

模型固定为 `work/openvla-oft/checkpoints/fp32_film_005000/pretrained_model`。
使用本机基模和 tokenizer；FiLM、动作头、状态投影层保持 FP32，不进行模型合并或量化。
加载适配器文件时先暂存到 CPU，再复制进 GPU 模块，避免在 4090 上额外占用一整份适配器显存。
关闭不需要的 KV cache；推理不启用梯度检查点。原机器人阈值及检查流程不变。

在本机终端准备环境：

```bash
source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
cd /home/yundu1/ros2_ws/openarm_vla/openvla-oft/scripts
```

先独立执行不连接机器人、也不访问相机画面的单帧模型检查：

```bash
bash run_openvla_oft_film.sh smoke 0
```

看到输出 `robot_connected: false`、`use_film: true`、`film_fp32_verified: true` 后，
根据现场情况决定是否进入真机交互。单帧检查不保证抓取成功。

```bash
bash run_openvla_oft_film.sh run 25
```

启动本身不运动；输入物品名会立即执行准备位运动并打开夹爪。
准备位完成后输入 `/start` 才运行最多 25 秒的模型抓取；`/quit` 退出。
支持螺丝刀、透明胶带、双面胶、抽纸、黑色胶带、红色胶带（也可输入菜单编号）。
每次抓取保存三路录像到 `work/openvla-oft/inference_videos/` 下对应物品目录。
异常时使用现场硬件急停；软件退出不能保证真机停止。

只检查模型、显卡及相机设备路径，可用 `bash run_openvla_oft_film.sh check`。
这个新入口不修改旧 `run_openvla_oft.sh` 的默认模型；不要再用已删除的旧模型路径。

## 本机单帧检查记录

已在 RTX 4090 上用验证集第 0 帧测试成功：FiLM FP32 校验通过，输出 16 维动作，
PyTorch 推理分配显存峰值约 18.873 GiB，单次模型调用约 0.349 秒。
这些数值不包含完整抓取循环的相机、通信、加载和录像开销，也不是持续控制频率承诺。
没有连接机器人或发布动作；安全阈值没有调整。
