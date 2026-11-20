# OpenVLA-OFT + FiLM：六种物品一起训练

这是用户最新确认的训练范围，取代此前的螺丝刀单任务方案。
螺丝刀、透明胶带、双面胶、抽纸、黑色胶带、红色胶带均参与训练。

- 使用现有 `six_objects_60`，选择全部 episode 0–59；不复制视频，不修改源数据。
- 每种物品 9 条示教训练、1 条留出：总计 54 条 / 32400 帧训练，6 条 / 3600 帧留出。
  留出 episode 为 9、19、29、39、49、59；“全部物品”不代表把留出集也用于拟合。
- 准备程序核对所有 episode 和逐帧任务标签；trainer 再核对实际加载数据的六类帧数，
  防止继承旧模型配置中的螺丝刀筛选。各类训练帧均为 5400，留出帧均为 600。
- 从基座重新训练 LoRA、动作头、状态投影和 FiLM，不是续训旧 5000 步权重。
  保留的旧模型仅提供配置和归一化统计。
- FiLM、语言 padding mask、梯度检查点开启；batch=2、chunk=8、LoRA rank=32。
- 臂状态噪声标准差 0.03 rad；夹爪噪声 0.02；归一化后逐维换算，动作标签不变。
- 按样本以 0.3 概率屏蔽投影后的状态 token，不把输入关节状态填零。
- 颜色增强开启（亮度/对比度/饱和度 0.9–1.1）；随机裁剪关闭。
- 保留此前的离线验证关闭设置；数据预检不等于模型效果验证。
- 按最新要求默认 **15000 步**，每 2500 步保存检查点（共六份）；显示百分比和剩余时间。
- 六类的全部训练帧均进入采样池，15000 步 × batch 2 抽取 30000 次起始帧，
  约为 32400 帧训练集的 0.926 遍；仍不等于遍历每一帧。
- 当前完整检查点约 9.7 GiB，15000 步入口按六份检查点预留至少 76 GiB 空闲空间，
  不自动删除旧检查点。每次保存前仍检查至少 10 GiB 可用空间。

## 服务器命令

```bash
cd /root/gpufree-data/openvla_oft_upload/openvla_oft_training_bundle
bash all_objects_training_fp32_15000/run_all_objects_film_15000.sh preflight
```

先做 1000 步实验（只有手动执行才会开始训练）：

```bash
OPENVLA_STEPS=1000 bash all_objects_training_fp32_15000/run_all_objects_film_15000.sh train
```

训练 15000 步（从基座重新训练总计 15000 步，不是旧 5000 步模型再续训 15000 步）：

```bash
bash all_objects_training_fp32_15000/run_all_objects_film_15000.sh train
```

默认 mode 为 preflight，不加载 7B 权重或启动训练。训练步数可设 1–15000，其他参数可沿用
`OPENVLA_STATE_NOISE_STD_RAD`、`OPENVLA_STATE_NOISE_GRIPPER`、`OPENVLA_STATE_DROPOUT_PROB`、
`OPENVLA_IMAGE_AUGMENTATION`、`OPENVLA_RANDOM_CROP`、`OPENVLA_NUM_WORKERS` 环境变量。
本地受限环境预检可设 `OPENVLA_NUM_WORKERS=0`，服务器默认 2。

新输出目录：`outputs/all_objects_film_training/train_<时间>_<随机串>/result/`。
最后权重位于 `checkpoints/015000/pretrained_model/`（1000 步实验则为 `001000`）。
每次独立目录，旧六物品模型和单任务脚本保留，但当前请使用本页的全物品入口。

状态噪声和丢弃只在训练生效；推理入口、零位及机器人安全阈值不变。
训练效果仍需检查图像/指令依赖，以及真机停住姿态下的离线动作推进和方向。
同一轨迹重复回放的问题仍存在；增强不能替代多样化示教和纠偏数据。
