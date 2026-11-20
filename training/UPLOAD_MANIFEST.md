# GitHub 待上传清单

这是独立的待发布目录，磁盘占用约 2.9 GiB。模型权重和训练检查点保存在本机工作目录。

| 内容 | 路径 | 大小约 |
| --- | --- | ---: |
| 共享训练数据与精选原始示教 | `data/` | 2.5 GB |
| 四个模型、六种物品的 26 段真机推理录像，含失败和中断 | `inference_examples/` | 264 MiB |
| OpenArm/ROS2 相关源码（含第三方许可证） | `training/ros2_control/ros2_src/` | 123 MB |
| 采集、转换、训练、推理及审核代码 | `training/`、`inference_examples/tools/` | 数 MB（除上列 ROS2 源码） |

本机尚未安装 Git LFS，也未初始化 Git、提交或上传。上传前先安装 Git LFS 并核对账号容量；`.gitattributes` 已将 MP4、AVI、Parquet 标记为 LFS，`.gitignore` 排除常见权重格式与训练输出。GitHub [普通 Git 超过 100 MiB 会拒收](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)，浏览器单文件上传限制更低，因此不要直接拖拽整个目录到网页。

在执行 `git add` 前还需：逐段人工复核录像和训练数据中的人员/隐私、确认数据及第三方源码的公开授权、决定项目许可证，并检查模型训练脚本内的本机绝对路径。自动密钥模式扫描未发现明文密钥，但不能替代人工审查。不要把本机其他 `work/`、`models/` 或 `checkpoints/` 目录复制进来。
