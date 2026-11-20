# ACT 训练

复用 [六物品训练集](../../data/lerobot/six_objects_60/)。六类各有 10 段，包含三路相机、16 维状态和动作。脚本按物品筛选 episode，训练六个独立模型。每类默认 8 段训练、2 段留出验证。

从仓库根目录预览命令。

```bash
conda activate lerobot
python training/act/scripts/train_act.py --object 抽纸
```

确认后训练单类或依次训练六类。

```bash
python training/act/scripts/train_act.py --object 抽纸 --execute --wandb
python training/act/scripts/train_act.py --all --execute --wandb
```

默认每类 10,000 步、batch 2、动作块 30 帧。可通过 `--steps`、`--batch-size`、`--chunk-size` 调整。小规模连通性测试可用 `--steps 100 --execute`。结果写到仓库根目录下的 `work/act/runs/`，该路径被 Git 忽略。首次运行可能需要下载 ResNet18 预训练主干权重。
