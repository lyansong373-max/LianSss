# OpenArm OpenVLA-OFT 服务器训练包

这是 **OpenVLA-OFT** 训练包，不是普通 OpenVLA。它针对当前 OpenArm 数据配置：

- 60 个 demonstrations、36,000 帧、6 个语言任务；
- 三路 640×480 相机；
- 16 维双臂状态和 16 维双臂绝对关节动作；
- OpenVLA-7B 基座 + OpenVLA-OFT 连续动作头、proprio 输入和动作分块；
- 状态和绝对关节动作采用完整 min/max 范围归一化，避免 q01/q99 裁掉关键关节极值；
- 单卡 24/48/80GB GPU 自动选择保守 batch、LoRA rank 和 chunk size。

## 项目文件

- `../../data/lerobot/six_objects_60/`：共享的完整 LeRobot v3 数据集，训练目录不再保存重复副本；
- `third_party/lerobot-policy-openvla-oft/`：固定版本的 OpenVLA-OFT LeRobot 插件源码；
- `train_openvla_oft.sh`：底层安装、转换、检查与训练入口；
- `start_server_training.sh`：服务器一键入口；
- `verify_bundle.py`、`MANIFEST.sha256`：上传完整性检查。

OpenVLA-7B 基座约 15GB，未放进上传包。`convert` 阶段会从 Hugging Face 官方
`openvla/openvla-7b` 下载并转换到 `models/openvla_oft_base_7b/`。

## 上传

最简单的方式是只上传同级目录中的两个文件：

- `openvla_oft_training_bundle.tar.zst`
- `openvla_oft_training_bundle.tar.zst.sha256`

上传后在服务器执行：

```bash
sha256sum -c openvla_oft_training_bundle.tar.zst.sha256
tar --zstd -xf openvla_oft_training_bundle.tar.zst
```

如果镜像没有 `zstd`，先执行 `apt-get update && apt-get install -y zstd`。

也可以直接上传整个 `openarm_vla` 目录，这样共享数据路径保持不变。在本地执行，将地址和端口替换成算力平台提供的值：

```bash
rsync -avhP --partial -e "ssh -p SSH端口" \
  openarm_vla/ \
  root@服务器IP:/workspace/openarm_vla/
```

建议使用 150GB 持久化盘。训练包本身约 2.2GB；基座、缓存、Python 环境及检查点
还需要约 70–90GB。

## 上传后检查

```bash
cd /workspace/openvla_oft_training_bundle
python verify_bundle.py --full
```

完整校验通过后，直接运行：

```bash
chmod +x train_openvla_oft.sh start_server_training.sh
./start_server_training.sh all
```

`all` 会依次完成：

1. 从包内源码安装 OpenVLA-OFT 插件和 LeRobot 训练依赖；
2. 下载并转换 `openvla/openvla-7b` 基座；
3. 检查 CUDA、数据集和 OpenVLA-OFT 类型；
4. 运行 20 步显存冒烟测试；
5. 冒烟成功后启动 30,000 步正式训练。

训练结果位于 `outputs/openvla_oft_openarm_时间戳/`。不要在下载检查点前销毁实例。

## 分步执行

```bash
./start_server_training.sh prepare
./start_server_training.sh smoke
./start_server_training.sh train
```

48GB L40S/A6000 默认使用 batch 2、梯度累积 4、LoRA rank 32、8 帧动作块。
若需要覆盖：

```bash
OPENVLA_BATCH_SIZE=1 OPENVLA_GRAD_ACCUM=8 OPENVLA_CHUNK_SIZE=4 \
  ./start_server_training.sh all
```

## 下载训练结果

在本地执行：

```bash
rsync -avhP --partial -e "ssh -p SSH端口" \
  root@服务器IP:/workspace/openvla_oft_training_bundle/outputs/ \
  ./openvla_oft_server_outputs/
```

同时保留 `models/openvla_oft_base_7b/`；LoRA 检查点需要该 OpenVLA-OFT 基座。
