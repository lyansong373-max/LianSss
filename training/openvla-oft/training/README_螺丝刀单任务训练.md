# OpenVLA-OFT + FiLM：只训练螺丝刀

只读取已有 `six_objects_60` 数据集中任务文字完全等于
`Pick up the screwdriver and place it in the receiving box.` 的 episode。
准备程序同时核对 episode 元数据和逐帧 task_index，不仅修改提示词。
当前数据选择 episode 0–9：0–8 训练，共 5400 帧；9 留出，共 600 帧。
其他五类物品不进入训练 batch。不会复制视频或修改源数据。

## 默认设置

- 从基座新建 LoRA、动作头、状态投影和 FiLM；不是继续训练现有 5000 步权重。
- 保留的 FiLM 5000 步模型只提供配置及原有归一化统计，不复制适配器权重。
- 5000 次优化器更新（不是 5000 遍），batch=2，三路图像，chunk=8，LoRA rank=32。
- 开启 FiLM、语言 padding mask 和 Llama 梯度检查点；可训练参数/Adam 为 FP32，前向 BF16。
- 按最新确认的方案：14 个臂关节噪声标准差 **0.03 rad**，两个夹爪标准差 **0.02**（开合比例单位，不是 rad）。
  随机噪声限制在 ±3 倍标准差，噪声按实际检查点统计逐维换算后加到归一化状态上，再截断到 [-1,1]。
- 以 **0.3 概率**按样本丢弃状态：将 ProprioProjector 输出的整个状态 token 置零，**不把输入状态填零**。
  被丢弃样本不加状态噪声；保留 token 不做 1/(1-p) 放大。动作标签保持不变。
- **关闭随机裁剪**，保留轻度颜色增强（亮度/对比度/饱和度各 0.9–1.1）；同一样本三路相机共享颜色系数。
  可选裁剪功能已经实现，但本轮不启用，避免同时增加变量；不翻转/旋转图像。
- 以上增强只用于训练输入，不修改视频、状态/动作源数据和归一化统计，不保存为推理增强。
  推理的 `image_crop_scale` 仍为 1.0；若运行验证，验证输入也不加增强。
- 按此前要求关闭离线验证；仍保留独立留出集，不将它并入训练。
- 学习率 5e-5，warmup=100，clip=1；2500/5000 步保存完整检查点及续训状态。
- 显示步数、百分比、速度、loss、剩余时间；每 20 步写结构化 ETA。
- 5000 步共抽取 10000 个训练起始帧，约 1.85 遍 5400 帧训练集，并非 10000 个独立样本。
- 单物品不会减少每步模型计算量，不能据此承诺比六物品训练更快。

## 服务器启动

文件位于 bundle 下 `screwdriver_training_fp32_5000/`。使用原有 `.venv`、基座和数据。
在服务器终端执行：

```bash
cd /root/gpufree-data/openvla_oft_upload/openvla_oft_training_bundle
bash screwdriver_training_fp32_5000/run_screwdriver_film_5000.sh preflight
```

preflight 只验证任务选择并解码/预处理四批数据，不加载模型，不更新权重，不连接机器人。
还会核对 16 维关节名称顺序、检查点与数据的归一化统计一致性，输出丢弃比例、各维换算 sigma、
截断后的实际噪声 RMS，并用一个小型 CPU ProprioProjector 测试 token 置零（不加载 7B 权重）。
确认 `train_episodes` 为 0–8、`validation_episodes` 为 9、每批 task 只有螺丝刀，再手动训练：

```bash
bash screwdriver_training_fp32_5000/run_screwdriver_film_5000.sh train
```

可选的 20 步显存检查（这会实际训练并保存一份检查点，正式训练仍重新初始化）：

```bash
bash screwdriver_training_fp32_5000/run_screwdriver_film_5000.sh smoke
```

不传 mode 时默认 preflight，不会意外启动训练。`OPENVLA_STEPS=1000` 可将正式训练改为
1000 步（允许 1–5000）；其他默认设置保持不变。此专用入口不接收任意 trainer 参数，
避免通过额外参数覆盖数据筛选或误加载旧模型。

## 可选增强参数

直接执行 `train` 即使用上面的默认值，不需要额外设置。可用环境变量改变：

```bash
OPENVLA_STATE_NOISE_STD_RAD=0.03 OPENVLA_STATE_NOISE_GRIPPER=0.02 OPENVLA_STATE_DROPOUT_PROB=0.3 \
OPENVLA_RANDOM_CROP=0 OPENVLA_IMAGE_AUGMENTATION=1 \
bash screwdriver_training_fp32_5000/run_screwdriver_film_5000.sh train
```

- `OPENVLA_STATE_NOISE_STD_RAD`：0–0.05，单位 rad，0 关闭。0.03 是标准差，不是最大扰动。
- `OPENVLA_STATE_NOISE_GRIPPER`：0–0.05，默认 0.02；0 表示夹爪不加噪声。
- `OPENVLA_STATE_DROPOUT_PROB`：0–0.5，默认 0.3；0 表示不丢弃状态 token。
- `OPENVLA_RANDOM_CROP=1` 可开启裁剪，`OPENVLA_CROP_MIN_SCALE=0.9` 保留约 90%–100% 面积。
  `OPENVLA_IMAGE_AUGMENTATION=0` 关闭颜色增强。
- `OPENVLA_NUM_WORKERS=0` 用于不允许多进程的本地预检；服务器默认 2。
- 完整参数记录在 `result/run_config.json` 和检查点的 `run_config.json` 中。
  通用 trainer 默认所有增强关闭，旧入口行为不变；断点续训不允许静默更换增强配置。

先做新方案的 1000 步试验可用下面的命令（需要用户自行执行）：

```bash
OPENVLA_STEPS=1000 bash screwdriver_training_fp32_5000/run_screwdriver_film_5000.sh train
```

`train` 不指定步数时仍为原定 5000 步。本次没有自动启动 A/B/C 三组试验，也没有重新打开离线验证。
若做 A/B/C：A 设置两项噪声与丢弃全为 0；B 设置两项噪声为 0、丢弃 0.3；C 使用本次默认值。
三组均保留 FiLM、相同 seed 和颜色增强、关闭裁剪。新训练每次使用独立目录，不会加载旧适配器权重。

## 具体代码接入点

- `train_openvla_oft_fp32.py::state_range_from_preprocessor` 读取已经加载的 MIN_MAX 归一化器，
  与数据统计核对；不相符就停止，不猜测换算范围。
- `process_batch()` 之后调用 `augment_training_state()`：`sigma_norm = 2 * sigma_physical / (max-min)`。
  仅训练和 CPU 预检调用，离线评估缓存、推理入口不调用。
- `ProprioTokenDropout` 的 hook 挂在 **PEFT 包装后的外层** `base.model.proprio_projector`，
  作用位置是两层 MLP 之后、与图像 token 拼接之前。
- mask 的上下文覆盖 `_predict()` 和 `loss.backward()`，异常也会清除；评估模式忽略 mask。
  这样不会在梯度检查点重算时使用错误的 mask，保存的权重键和推理格式不变。
- 日志中的实际扰动会因状态边界截断和样本丢弃而小于配置 sigma；batch=2 的某批丢弃率可能是
  0%、50% 或 100%，0.3 是长期采样概率，不要求每批恰好丢 30%。

交接说明第八节表明，重复回放示教易让模型过度依赖状态。增强是针对这个现象的训练实验，
不能保证解决真机停住，也不能替代位置/起始姿态变化的真人示教和失败纠偏数据。
单任务模型只训练螺丝刀，不据此宣称已经学会六物品的语言区分。

输出在 `outputs/screwdriver_film_training/train_<时间>_<随机串>/`，包含
`selection.log`、`reference_config/single_task_manifest.json`、`console.log` 和 `result/`。
最终模型在 `result/checkpoints/005000/pretrained_model/`，不会覆盖原六物品模型。
默认不运行离线验证，训练结束不等于验证通过或允许真机部署。

## 零位重要说明

2026-11-04 右臂 3 号电机已重设零点。本程序保留旧数据的状态、动作和归一化统计，
没有自动给右关节 3 加/减偏移，也没有改机器人安全阈值。
必须另行核对采集时的零位与当前物理零位，决定重新采集还是经过验证的坐标转换。
本次置零前反馈与数据采集时的偏移不一定相同，不能直接把约 1.091 rad 当作数据修正量。
核对前不要将训练结果直接用于真机。这里的程序完全不导入 ROS 或机器人控制驱动。
