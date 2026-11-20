# OpenArm（外骨骼遥操作）× LeRobot SmolVLA

## 一键使能并缓慢回零（单终端）

回零完成验收的夹爪归一化容差为 `0.10`（现场确认夹爪已闭合后授权调整），
监督器复核使用同一阈值。双臂关节容差仍为 `0.02 rad`，末段稳定性检查不变；
原始实测反馈保留，不重写编码器零点、不修改夹爪目标行程或运动限位。
旧失败记录保持原样，不会因调整阈值而自动改为成功。

2026-11-02 启动稳定性补丁：此入口在初始化 ROS 前加载
`config/fastdds_udp.xml`，让监督器及其驱动、规划、回零子进程使用 UDP-only，
规避共享内存传输路径；控制器查询限为最多 2 Hz，并检测启动器日志中实际工作节点的退出。
不修改系统 ROS 安装、用户 shell、ROS 域、零位、限位或运动速度，也不删除 `/dev/shm`。
如果环境已指定其他 RMW/XML 配置，会拒绝覆盖并提示。
直接运行旧 `ros2 launch ...` 不会自动获得这个入口的 UDP 配置。
此补丁已通过隔离假硬件测试，但旧配置在隔离域也能通过：不能据此确定共享内存就是根因，
实际硬件启动仍需现场验收，不能把“启动器还在”或“电机灯亮”当作控制器正常。
若驱动异常退出，电机可能仍保留使能；先可靠支撑并使用硬件急停，再结束旧启动器，
不要反复重启真机尝试恢复。成功回零后保持该终端运行，不要为下一条录制重复启动硬件。

替代分别启动底层驱动、模型检查服务、回零脚本的三个终端。入口本身不重写编码器零点，
标准零位仍为双臂关节目标全 0、夹爪闭合；不是弯举起始姿态。

先确保没有其他底层驱动、MoveIt 服务、遥操作桥接、录制/回放或 CAN 调试程序运行。
本入口遇到已知冲突进程或 ROS 控制来源会拒绝启动，不会替你关闭别的程序。
三个相机须连接并可独占使用，夹爪空载，运动路径清空，现场急停可用。

在普通交互终端执行（无需激活 Conda）：

```bash
bash /home/yundu1/ros2_ws/openarm_lerobot/enable_and_zero.sh --execute
```

启动前输入一次 `ZERO` 确认。随后自动：

1. 启动修复后的真机驱动，先验证新鲜反馈并保持当前姿态。
2. 等待双臂 16 关节反馈、唯一 `/joint_states` 来源和五个控制器 active。
3. 启动 `move_group`（`allow_trajectory_execution:=false`），等待模型检查服务。
4. 调用原 `slow_demo_replay.py zero --execute`：预检限位、自碰撞、控制来源和相机；
   预检通过后倒计时 5 秒，再缓慢回零，到位后闭合夹爪并保存验收记录。
5. 显示“回零验收成功”，继续保持驱动和规划服务运行。此时可在其他终端启动映射、桥接、录制。

**不要关闭这个终端，也不要再另起一套底层驱动。** Ctrl+C 会先停止回零进程，再关闭
本入口启动的规划服务和硬件驱动；机械臂可能失去保持力，需事先做好可靠支撑。
软件退出不是硬件急停。失败时不重试、不自动继续；若驱动启动器还活着，会留在终端
等待现场处理，避免仅因预检失败就突然停用驱动。启动器存活不保证硬件仍在线或已回零。

本次目录为 `poses/enable_zero_时间戳/`，包含三个进程日志；回零录像、状态日志和
`zero_pose.json` 位于其 `zero/` 子目录。不会覆盖桌面旧零位档案。
模型检查不保证避开未建模的洞洞板、支架、桌面或人员。

不带 `--execute` 或带 `--help` 仅显示说明，不启动 ROS 节点、驱动或运动。
离线测试使用假进程、假 ROS 和合成反馈，不连接 CAN、不使能电机。

## 黑色胶带独立示范入口

硬件、外骨骼接收和映射、LeRobot 桥接正常运行后，在现场录制终端执行：

```bash
conda activate lerobot
bash /home/yundu1/ros2_ws/openarm_lerobot/language_grasp.sh demo-black-tape 120
```

任务为 `Pick up the black tape and place it in the blue box.`。
每次新增一条，独立保存到 `/home/yundu1/ros2_ws/openarm_lerobot/datasets/black_tape/demonstrations`，
可用 `BLACK_TAPE_DATASET_ROOT`、`BLACK_TAPE_DATASET_ID` 覆盖；不混入红色胶带数据。
接管前对齐真机与外骨骼；15秒后自动接管、跟随5秒后录制，右箭头保存。
不自动回位、不失能、不判定成功；不要同时运行其他录制、回放或旧桥接。

## 红色胶带独立示范入口

现有硬件、外骨骼接收和映射、LeRobot ROS 桥接正常运行后，在现场录制终端执行：

```bash
conda activate lerobot
bash /home/yundu1/ros2_ws/openarm_lerobot/language_grasp.sh demo-red-tape 120
```

任务为 `Pick up the red tape and place it in the blue box.`，沿用取物放盒流程。
每次新增一条，独立保存到 `/home/yundu1/ros2_ws/openarm_lerobot/datasets/red_tape/demonstrations`，
不混入抽纸或其他胶带。可通过 `RED_TAPE_DATASET_ROOT` 和 `RED_TAPE_DATASET_ID` 覆盖。
设备就绪后15秒自动接管，再跟随5秒开始录制；120秒是上限，录制终端右箭头结束保存。
接管前对齐外骨骼与真机；不自动回位、不判定成功。不要同时运行旧桥接、其他录制或回放。

## 透明胶带独立示范入口

硬件、WebSocket接收、OpenArm外骨骼重定向和LeRobot ROS桥接就绪后，在现场交互终端运行：

```bash
conda activate lerobot
bash /home/yundu1/ros2_ws/openarm_lerobot/language_grasp.sh demo-clear-tape 120
```

任务为 `Pick up the clear tape.`。每次新增1条，独立保存到
`/home/yundu1/ros2_ws/openarm_lerobot/datasets/clear_tape/demonstrations`，不混入白色胶带或螺丝刀。
可通过 `CLEAR_TAPE_DATASET_ROOT`、`CLEAR_TAPE_DATASET_ID` 覆盖目录及数据集名。
设备就绪后15秒自动接管，再跟随5秒开始正式录制；120秒为录制上限，在录制终端按右箭头结束保存。
录制三路图像、双臂/夹爪状态和请求动作；不自动回零、回弯举或判定成功。
不要同时运行旧exoskeleton_bridge或自动回放。接管前对齐外骨骼与真机，保持急停可用。

## 抓取双面胶（原白色胶带）独立示范入口

先启动 WebSocket 接收、OpenArm 外骨骼重定向和 LeRobot ROS 桥接；硬件已运行时不要重复启动。
停止自动回放和旧 `exoskeleton_bridge`，避免多个来源控制真机。在现场交互终端执行：

```bash
conda activate lerobot
cd /home/yundu1/ros2_ws/openarm_lerobot
bash /home/yundu1/ros2_ws/openarm_lerobot/language_grasp.sh demo-double-sided-tape 120
```

每次新增一条，保存到 `/home/yundu1/ros2_ws/openarm_lerobot/datasets/white_tape/demonstrations`，
不覆盖或混入螺丝刀数据。可用 `WHITE_TAPE_DATASET_ROOT`、`WHITE_TAPE_DATASET_ID` 单独覆盖。
任务名称按用户要求改为中文 `抓取双面胶`；旧命令 `demo-white-tape` 仍可用，写入相同的新任务名。
为保持已有数据路径有效，未重命名 `white_tape` 目录。已有 episode 0 的任务表及 episode 标签
已同步修改，修改前元数据备份在 `backups/double_sided_task_20261102_01`；动作与视频不变。
设备就绪后15秒自动接管，再跟随5秒才开始录制；正式录制上限120秒，完成后在录制终端按右箭头保存。
示范顺序：接近、夹紧双面胶、向上提离挂钩、再向外撤回；若需要学习放盒，也应完整示范放置和松爪。
保存三路视频、双臂/夹爪状态及动作和任务文字。夹爪闭合不结束录制，不自动回位。
开始前把外骨骼目标与真机姿态对齐；没有起始误差门限，接管可能快速追赶目标。急停保持可用。

## 螺丝刀独立录制入口

在现有ROS程序正常运行、桥接夹爪缩放为0.03时执行：

```bash
conda activate lerobot
cd /home/yundu1/ros2_ws/openarm_lerobot
bash language_grasp.sh demo-screwdriver 180
```

任务为 `Pick up the screwdriver.`，保存到 `datasets/screwdriver/demonstrations/`，
不混入抽纸示范；每次运行追加1条。首次录制自动创建数据集目录。
三相机、15秒后接管、跟随5秒后录制、右箭头结束保存的行为与下方连续示范一致。
目前只描述抓取，不擅自增加放入蓝盒的任务。详细目录说明见 `datasets/screwdriver/README.md`。

## 一次连续示范：整包抽纸放入蓝盒

### 显示外骨骼映射后的 OpenArm 目标（当前使用）

先启动现有 WebSocket 接收和外骨骼 retargeting；关闭之前只显示实际状态的 RViz 窗口，
用下面这一条替换原 RViz 命令，不用重启硬件或编译：

```bash
source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
ros2 launch /home/yundu1/ros2_ws/openarm_lerobot/rviz/exo_target.launch.py
```

模型只显示 `/left_arm/joint_command`、`/right_arm/joint_command` 映射后的目标，
不是原始外骨骼模型，也不是真机反馈。录制程序的15秒等待期间也能随外骨骼更新。
夹爪归一化目标乘以当前桥接的 `0.03` 转为模型关节位移；与用户原来正常使用的
`exoskeleton_bridge.launch.py gripper_scaling_factor:=0.03` 一致，不再次添加偏移。
仅向 `/exo_target_view/joint_states` 及独立的描述、TF话题发布可视化数据，
不发布 `/joint_states`、公共TF或任何控制命令；不要另启旧的 `openarm_display.launch.py`。
双臂数据缺失、无效或超过0.5秒时停止更新并在终端警告；RViz可能保留最后画面，不能当实时反馈。
此画面不是桥接限幅后的最终动作，更不能证明真机已对齐；观察真实机械臂并保持急停可用。

### 只显示真机状态的 RViz

硬件控制已启动时，可另外打开一个普通 ROS 终端运行（无需重新启动硬件或编译）：

```bash
source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
rviz2 -d /home/yundu1/ros2_ws/openarm_lerobot/rviz/actual_state.rviz --ros-args -r __node:=openarm_actual_state_view
```

此配置仅包含网格、RobotModel 和视角移动工具，复用硬件发布的 `/robot_description`、
`/tf`、`/tf_static`；模型描述使用 Transient Local，支持硬件启动后再打开 RViz。
不启动任何关节状态发布器、关节合并节点、硬件控制器或 MoveIt，不发布运动目标。
运行 `ros2 topic info /joint_states --verbose` 应确认唯一发布者是 `joint_state_broadcaster`。
**不要用 `qnbot_teleoperator openarm_display.launch.py` 来查看采集中的真机**，它会引入显示状态发布者。
该 RViz 显示的是实际机械臂姿态，不是外骨骼目标；在15秒不下发动作的准备期，外骨骼变化不会让模型跟着动。
请调整外骨骼而不是强行扳动已使能的机械臂；倒计时结束后会自动接管。可将 RViz 和倒计时终端并排放置。
RViz 画面不能替代真实反馈新鲜度检查或现场安全观察；机械臂通信中断时可能停留在最后姿态。

### 连续示范操作

不必逐点保存姿态。此入口录制一条完整遥操作轨迹，保存三路图像、双臂及夹爪实际状态、
遥操作请求动作和任务文字；**不自动回位，不自动判定成功，不自动回放**。
首次使用不需要已有弯举姿态文件。

先按已验证的真机流程启动硬件、外骨骼目标发布和 LeRobot ROS 桥接。
真实示范需要桥接 `dry_run=false`；硬件 launch 自身可能使能/归零，不能依赖桥接空跑阻止它。
关闭相机预览、其他 LeRobot 录制/遥操作/推理进程，避免旧外骨骼桥接同时向机械臂发指令。
确认急停可用、运动空间无人，先低速操作。

```bash
cd /home/yundu1/ros2_ws/openarm_lerobot
conda activate lerobot
bash language_grasp.sh demo-tissue
# 如果 120 秒不够，可改为：bash language_grasp.sh demo-tissue 180
```

1. 设备连接就绪后倒计时15秒，期间不发送动作、不录制。无需按空格，也不要求零位或精确对齐。
   到时自动接管：移除起始姿态误差门限，但仍检查数据缺失/过期/非有限值和夹爪量程。
   按用户要求恢复最初的目标跟随：不再附加0.25rad/s关节限速或1.0/s夹爪目标限速。
   ROS桥接原有关节限位、单步限幅保持不变（默认 `max_step_rad=0.15`），夹爪目标缩放仍为
   `gripper_scale=0.03`（2026-11-02 根据用户实际使用的旧桥接命令，将之前的0.01纠正为0.03）。
   单步限幅不是时间意义上的速度限制或碰撞规划；
   接管时姿态不一致可能快速追赶目标，倒计时结束前应大致对齐并确保路径无障碍。
2. 接管后的前5秒可以遥操作，但不录制；然后自动提示“正式开始录制”。15秒和5秒均不计入
   命令行指定的正式录制时长。5秒不保证所有关节都已追上外骨骼。正式开始后可保持弯举约2秒再做任务。
3. 连续完成：接近托架上的整包抽纸、夹持、取离托架、移到蓝盒上方、放入、松爪、撤离。
   中间不必按键或停下来保存点位。避免包装被托架卡住，也不要让另一只手臂挡住路径。
4. 完成任务后，聚焦运行 `demo-tissue` 的录制终端，按键盘右箭头 `→` 结束本条并保存（`n`备用）。
   已删除右夹爪持续闭合5秒结束的判定；夹爪开合和持续夹持不会结束录制，无需先张开右爪。
   最长录制时限仍会结束本条；保存成功不是任务成功。结束后不回位、不失能，只停止本录制进程的跟随输出。
   空格无作用。按键仅在录制终端生效，不是在RViz/Rerun窗口：Esc/q退出，←/r重录；
   重录重新执行15+5秒准备。右箭头在准备阶段不生效。异常时先处理真机安全，这些按键不是急停。

默认保存在 `/home/yundu1/ros2_ws/openarm_lerobot/datasets/tissue_to_blue_box_demonstrations`，
再次执行会追加一条，不覆盖原示范。暂不上传，不直接混入已确认成功的训练集。
录好后可根据录像、夹爪变化和关节轨迹提取起始姿态，以及接近、夹持、取离、放置、松爪、
撤离的候选时刻；需检查真实夹持、取离和落盒情况，不能仅靠夹爪闭合判定成功。
此自动开始模式保存的 `action` 是重定向后的遥操作请求值（经原有遥操作处理器，不再附加上述限速），
不是桥接限幅后的实际下发值；`observation.state` 保留实际反馈。
单次示范可提供关节空间参考，但不替代视觉标定、碰撞检查或多位置训练数据，不能直接保证自动复现。

夹爪实际反馈范围检查（2026-11-02 用户确认）：连续示范入口（`demo-tissue`、`demo-screwdriver`）的15秒准备、5秒接管预热
和正式录制取消反馈上下界检查，替代此前的[-0.03, 1.25]容差。原始有限反馈即使小于0或
大于1也保留，不裁剪、不平移；缺失通道、NaN/Inf和数据过期仍拒绝。
外骨骼动作目标和保存姿态仍要求0～1，桥接缩放保持0.03；不修改硬件限位、速度、行程或控制器。
这不会抑制超调，也不会因为反馈超出物理范围而自动停止；异常开合、振荡或撞限位时须现场停止。
其他示教起始姿态/回位流程保持原校验；连续示范不再根据夹爪状态判定录制结束。

## GPT‑6 + 普通 RGB 相机 + 竖直洞洞板（准备阶段）

新增 `gpt_grasp_prepare.py` 和 `ros2_bridge/preview_pregrasp.py`。
当前完成静态图像采集、GPT‑6 目标候选识别、实测标定下的坐标换算和 MoveIt
预抓取规划请求。**尚未完成真实抓取执行、挂钩取离、自动成功判定或成功轨迹入库**。
这些工具不发送关节/夹爪命令，不切换控制器，也不自动启动硬件。
原有 `language_grasp.sh test` 仍依赖训练好的 SmolVLA，不能当作 GPT 抓取入口。

### 现在可以做：静态识别

先关闭占用相机的预览/录制程序，保持场景静止。在 LeRobot 环境运行：

```bash
cd /home/yundu1/ros2_ws/openarm_lerobot
conda activate lerobot
python gpt_grasp_prepare.py snapshot --out snapshots/pegboard_001
```

这一步只保存本机三张 PNG 和相机路径、时间、尺寸、校验值，不需要 API 密钥。
三路映射沿用 `language_grasp.sh config`：全局 camera1、黄色左腕 camera2、白色右腕 camera3。
相机顺序拍摄，用于静止场景识别，**不能当作同步训练数据**。
可重复使用新目录名采集棋盘格标定照片。

`locate` 默认使用**本机 ChatGPT 登录的 Codex CLI**，遵循该账户的 Codex 额度/计费方式，
不需要 API 密钥。已核对本机 Codex CLI 0.162.1 支持图像输入和结构化输出。
图像仍发送至模型服务，不是离线本地模型。每次识别发起独立的临时 Codex 请求，
复用登录账户，不复用当前桌面聊天的历史。

```bash
python gpt_grasp_prepare.py locate \
  --backend codex \
  --snapshot snapshots/pegboard_001 \
  --instruction '抓取白色胶带' \
  --model gpt-6-astra \
  --out proposals/white_tape_codex_001
```

程序检查 `codex login status`，只接受已确认的 ChatGPT 登录；清除子进程的 API 密钥
和接口地址覆盖变量，并强制 ChatGPT 认证。未登录、额度不足、模型无权限或连接失败会
停止，**不会自动切换 API 计费**。需要登录时自行运行 `codex login`，选择 ChatGPT 登录。
程序不读取、导出或打印 OAuth 令牌，不改写个人 Codex 配置。
识别子进程使用临时目录、只读沙箱，禁用 shell、apps、hooks、多代理和远程插件；
只返回 JSON 目标候选，不运行机器人命令。最多等待 180 秒。

旧 API 路径只在显式传入 `--backend api` 时启用，仍需要本机 `OPENAI_API_KEY`，使用 API
账户额度；当前 Codex 方案不需要设置它。参考 [Codex 非交互模式与登录复用](https://learn.chatgpt.com/docs/non-interactive-mode)。

输出 `selection.json` 和 `global_proposal.png`。只有 `found` 的唯一目标才可进入后续换算；
找不到、多个匹配或不支持的指令应返回 `not_found` / `ambiguous`。
即使返回 `found` 也必须检查识别预览，模型输出的格式检查不证明目标识别准确。
本地程序拒绝额外运动字段、越界目标框、未完成的识别返回。
默认模型为 `gpt-6-astra`，实际可用性以所选后端的账户模型权限为准。

### 真实抓取前需要的标定

已提供两个离线标定入口：

```bash
python gpt_grasp_prepare.py calibrate-intrinsics --help
python gpt_grasp_prepare.py calibrate-board --help
```

内参入口读取 `--snapshots` 目录下多组快照中的 `camera1.png`，至少需要 10 张能检测到
棋盘格的照片；`--columns` / `--rows` 填**内角点**数量，`--square-mm` 填实测格边长。
拍摄时改变棋盘格的位置、距离和倾角，保持相机焦距和对焦不变。
输出的内参 JSON 可传给外参入口的 `--intrinsics`。

外参入口还需要 `--snapshot` 对应的板面照片，以及 `--points` 实测点文件，格式见
`pegboard_points.template.json`。至少 6 个拟合点和 2 个不参与拟合的独立检查点，均在
板面上且分布覆盖工作区域；每点填写图像像素和相应的基座坐标（米）。当前程序会检查
共面性、点分布和重投影误差，但不会自动知道你量的坐标是否正确。
外参输出仍保留 `measured_and_verified=false` 与空工作空间，需要完成实际验证后填写。

请另存并填写 `pegboard_calibration.template.json` 和 `pegboard_object.template.json`。
模板中的 `null` 是缺失实测值，不是默认值。`measured_and_verified` 只有在实测与独立
检查完成后才能改为 `true`；这个标志不代替标定过程。

- 全局相机内参和畸变：固定 640×480、焦距和对焦，用不同位置、倾角的棋盘格图像标定。
- 外参 `T_base_camera`：4×4，**从 OpenCV 相机光学坐标变换到机械臂基座坐标**，平移单位米。
  用已知板面标记点与图像点、相机内参求 PnP；PnP 返回的 base→camera 变换需取逆。
  必须知道标记点在机械臂基座中的位置，仅知道孔距不能建立板面与基座的关系。
- 板面原点、单位外法向及经过验证的工作空间：外法向朝向机器人/相机；坐标系须核对 TF。
  移动相机、机器人基座或洞洞板之后需重新验证外参。
- 每个物品单独配置 `grasp_plane_offset_m`：实际抓取点所在平面突出于板面的距离。
  不应把所有物品深度设为零；圆柱瓶、胶带边缘的抓取点必须结合实际几何选择。
- 夹爪参数：`tcp_in_hand_m` 是抓取中心在 `openarm_left_hand` 或 `openarm_right_hand`
  下的位置；`hand_orientation_xyzw` 是该 hand 在基座坐标系中的实测目标姿态。
  `pregrasp_distance_m` 沿板面外法向后退。左右手参数不能直接互换。

在 `global_proposal.png` 上确认一个真正可夹取的原始图像像素 `(U,V)`，而不是盲用框中心。
第一阶段用人工确认像素来验证几何；全自动模式还需可靠的局部抓取点提取与腕部视觉校正。
标定完成后的命令（U、V 替换为实际像素值）：

```bash
python gpt_grasp_prepare.py target \
  --snapshot snapshots/pegboard_001 \
  --selection proposals/white_tape_codex_001/selection.json \
  --calibration pegboard_calibration.json --profile white_tape_profile.json \
  --pixel U V --out pregrasp_target.json
```

算法先去畸变，再将视线与**物品抓取平面**求交，并扣除旋转后的 TCP 偏移。
先投到裸板再加一个深度偏移会产生视差错误，尤其影响偏离画面中心的物品。
生成文件只代表预抓取手部位姿，不包含夹爪闭合、脱钩或取离动作。

### 仅请求 MoveIt 规划

需已有 MoveIt 服务，并按实测位置、尺寸将洞洞板、挂钩和周围障碍物加入规划场景。
工具会检查指定洞洞板 ID 的几何体存在，但不会自动证明建模正确或场景完整。
在已 source ROS2 工作空间的系统 Python 终端运行：

```bash
/usr/bin/python3 /home/yundu1/ros2_ws/openarm_lerobot/ros2_bridge/preview_pregrasp.py \
  --target /home/yundu1/ros2_ws/openarm_lerobot/pregrasp_target.json \
  --board-object-id pegboard \
  --out /home/yundu1/ros2_ws/openarm_lerobot/pregrasp_preview.json
```

只调用 `/get_planning_scene` 和 `/plan_kinematic_path`，保存规划响应，无执行接口。
不要为了获得规划服务而在已有硬件控制进程旁直接启动另一个 `demo.launch.py`：仓库的
demo 会另起控制器管理器，且默认 `use_fake_hardware=false`。2026-11-02 修改后的 V10
驱动已取消启动直接回零，改为验证新鲜反馈后保持当前位置，并加入同 CAN 接口的协作式
独占锁；旧二进制和其他 CAN 工具不受此锁保护，仍禁止并行启动。详见
`../src/openarm_ros2/openarm_hardware/STARTUP_SAFETY.md`。
后续真实执行需先统一 MoveIt 轨迹控制器与 LeRobot 位置控制器的控制权；已有 SRDF 还
禁用了若干左右臂之间的碰撞检测，需检查后再做双臂动作。初次验证选一只手，另一只保持。

### 成功数据入库的后续接入点

执行器完成后，将“回弯举 → 等待指令 → 识别/规划 → 开始记录 → 接近/夹取/脱钩/取离 →
结果确认”串为状态机。回位、等待和规划不进入训练 episode。洞洞板上取物可能需先抬升
脱钩再向外撤离，须逐物品验证，不能统一用水平拉出。

录制需以一致频率保存三路图像、16 维真实状态、16 维控制目标、语言任务、时间戳及
执行来源。MoveIt 的轨迹控制器执行路径不会经过现有 UDP `send_action`；需从控制器
参考轨迹获取与观测对齐的 action，不能把下一帧实测状态冒充控制目标。夹爪单位/尺度
沿用或显式转换并记录，当前桥接为 `gripper_scale=0.03`；之前0.01量程的数据不能直接混用。

每次尝试保留原始记录；第一阶段由操作者核对目标正确、已脱离挂钩、夹持稳定且未掉落，
再将成功 episode 纳入 LeRobot 数据集。GPT 视觉结果只能辅助判定，不能单凭闭爪或规划成功
自动标记成功。失败/不确定记录另存，不作为成功模仿样本；留出独立场景用于评估。

参考：[OpenAI GPT‑6](https://developers.openai.com/api/docs/guides/latest-model)、
[工具调用](https://developers.openai.com/api/docs/guides/function-calling)、
[OpenCV PnP](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)。

## 固定弯举起始姿态（采集和测试共用）

入口：`bash /home/yundu1/ros2_ws/openarm_lerobot/language_grasp.sh --help`。
已配置全局 video4、左腕/黄色 video2、右腕/白色 video0，使用对应 USB by-path 路径。

1. 启动 ROS2 硬件、外骨骼接收、重定向和 LeRobot 桥接；桥接使用 `dry_run:=false`。
   本功能更新后需重启桥接，才能启用实际关节状态超时检查。
2. 关闭其他 LeRobot 遥操作、录制或推理进程，运行 `bash language_grasp.sh teach-start`。
   该模式会跟随外骨骼；将双臂和夹爪摆到期望的弯举起始姿态，稳定后在终端输入 `s` 回车。
   程序保存实际测得的 16 维姿态至 `config/curl_start_pose.json`，不会猜测关节角度。
3. `bash language_grasp.sh record white_tape 10`：每条（包括重录）先自动缓慢回位，
   再等待外骨骼关节和夹爪目标对齐，然后开始录制。默认 15 秒复位阶段仍用于放回物品，
   不写入数据。自动回位和对齐阶段也不写入数据。
4. 模型训练完成后运行 `bash language_grasp.sh test`：先自动回弯举姿态，再等待输入
   `白色胶带`、`黑色胶带`、`纸巾`、`水瓶` 或 `透明胶带`（也支持 `抓取水瓶` 等固定指令）。
   输入后执行该次抓取，默认最多 30 秒。`q` 退出等待；下次测试重新运行 `test`。
   指令会映射到采集时对应的英文任务文字。

回位默认至少 5 秒，并按最大请求关节速度 0.25 rad/s 延长；跟随误差过大、状态超时、
外骨骼断流或对齐超时会中止。回位是关节空间插值，不做碰撞规划；需确保路径无障碍，
并在复位阶段先处理好手中物品。回位同时恢复保存的夹爪开度。
Ctrl+C 只停止继续发送新目标，不会让电机失能。
2026-11-02 修改后的底层驱动不再启动直接回零，而是检查反馈后保持当前姿态（含夹爪）。
标准机械零位定义不变，不会重写编码器零点。离线测试通过不代表真机验收通过；异常运动后
必须先完成现场静止启动、反馈与实物姿态核对，不能直接恢复 15 秒自动接管录制。

`bash language_grasp.sh home` 只执行回位；`show-start` 查看保存值。
设置 `START_POSE_PATH` 可使用另一份示教姿态。重新示教会备份旧文件。

## 架构

LeRobot 需要 Python ≥ 3.12，ROS2 Humble 的 rclpy 只能用 Python 3.10，二者不能在同一个进程里运行，所以用本机 UDP 桥接：

```
外骨骼 → websocket_teleoperator → exo_retargeting → /left|right_arm/joint_command ─┐
                                                                                    │ ROS2 (py3.10)
ros2_control (openarm.bimanual.launch.py) ← /…_forward_position_controller/commands │
                                         ← /…_gripper_controller/gripper_cmd       │
                                         → /joint_states                            │
                                                  ▲                                 ▼
                                       ros2_bridge/openarm_lerobot_bridge.py（限位裁剪、单步限幅、夹爪换算）
                                         UDP 50010 状态 │ 50011 指令 ▲ │ 50012 外骨骼目标
                                                       ▼            │ ▼
LeRobot (py3.12)   openarm_ros2 (Robot)  ←── lerobot-record / 策略推理 ──→  openarm_exo (Teleoperator)
```

* 本桥接**取代 `exoskeleton_bridge`**：录制和部署时都**不要**启动 exoskeleton_bridge（如果启动了，桥接节点会打警告）。
* 动作/状态共 16 维：`left_joint1..7.pos, left_gripper.pos, right_joint1..7.pos, right_gripper.pos`
  关节单位为弧度；夹爪目标为 0~1 的归一化值（桥接里乘以 `gripper_scale=0.03`，和用户原来的 `gripper_scaling_factor:=0.03` 一致）。
* 安全：关节限位从 `/robot_description` 中读取；每条指令每个关节最多变化 `max_step_rad`（默认 0.15）；还没拿到 URDF 时拒绝下发指令。

## 1. 安装（只需一次）

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
conda install ffmpeg -c conda-forge -y
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot && pip install -e ".[smolvla]"
cd <本目录> && pip install -e ./lerobot_robot_openarm_ros2 -e ./lerobot_teleoperator_openarm_exo
```

桥接节点不需要编译，直接用 ROS2 的系统 Python 运行即可。

查找相机编号：`lerobot-find-cameras opencv`

## 2. 采集数据

终端 A–D（ROS2 环境，`source ~/ros2_ws/install/setup.bash`），**按顺序**启动：

```bash
ros2 launch qnbot_teleoperator websocket_teleoperator.launch.py
# 采集时不启动 openarm_display：其关节合并节点会向 /joint_states 发布外骨骼目标，混入实际观测。
ros2 launch qnbot_teleoperator exo_retargeting.launch.py robot_type:=OpenArm
ros2 launch openarm_bringup openarm.bimanual.launch.py arm_type:=v10 robot_controller:=forward_position_controller launch_rviz:=false
# 不要启动 exoskeleton_bridge
```

终端 E（ROS2 环境）：

```bash
python3 ros2_bridge/openarm_lerobot_bridge.py --ros-args -p gripper_scale:=0.03
# 第一次先空跑，只打印、不下发：
python3 ros2_bridge/openarm_lerobot_bridge.py --ros-args -p gripper_scale:=0.03 -p dry_run:=true
```

终端 F（`conda activate lerobot`）——先测试遥操作：

```bash
lerobot-teleoperate \
  --robot.type=openarm_ros2 --robot.id=openarm_bi \
  --teleop.type=openarm_exo --teleop.id=qnbot_exo \
  --display_data=true
```

确认没问题后开始录制：

```bash
lerobot-record \
  --robot.type=openarm_ros2 --robot.id=openarm_bi \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --teleop.type=openarm_exo --teleop.id=qnbot_exo \
  --dataset.repo_id=<HF用户名>/openarm_exo_<任务名> \
  --dataset.single_task="Pick up the cup and put it on the plate" \
  --dataset.num_episodes=50 --dataset.fps=30 \
  --dataset.episode_time_s=30 --dataset.reset_time_s=15
```

录制时的键盘操作：`→` 提前结束当前条，`←` 重录当前条，`Esc` 结束录制。

## 3. 训练 SmolVLA

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<HF用户名>/openarm_exo_<任务名> \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}' \
  --batch_size=32 --steps=20000 \
  --output_dir=outputs/train/openarm_smolvla --job_name=openarm_smolvla \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false
```

## 4. 部署 / 评估

只启动硬件控制（不启动外骨骼的那几个节点）和桥接节点：

```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py arm_type:=v10 robot_controller:=forward_position_controller launch_rviz:=false
python3 ros2_bridge/openarm_lerobot_bridge.py --ros-args -p max_step_rad:=0.05   # 第一次部署建议把单步限幅调小
```

```bash
lerobot-record \
  --robot.type=openarm_ros2 --robot.id=openarm_bi \
  --robot.cameras="<与采集时完全相同>" \
  --policy.path=outputs/train/openarm_smolvla/checkpoints/last/pretrained_model \
  --dataset.repo_id=<HF用户名>/eval_openarm_exo_<任务名> \
  --dataset.single_task="Pick up the cup and put it on the plate" \
  --dataset.num_episodes=10
```

## 桥接参数（`--ros-args -p 参数名:=值`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gripper_scale` | 0.03 | 夹爪归一化值 → GripperCommand position 的系数；反馈除以同一系数 |
| `gripper_threshold` | 0.005 | 夹爪变化小于该值（归一化单位）时不发送新 goal |
| `max_step_rad` | 0.15 | 每条指令每个关节相对上一条的最大变化量 |
| `dry_run` | false | 只打印，不下发指令 |
| `state_port` / `cmd_port` / `teleop_port` | 50010 / 50011 / 50012 | 需要与 LeRobot 端的 `--robot.state_port`、`--robot.cmd_port`、`--teleop.teleop_port` 一致 |

## 已知需要你在真机上确认的点

1. **夹爪状态换算**：观测值 = `openarm_{side}_finger_joint1` 的位置 ÷ `gripper_scale`。请在真机上开合夹爪，确认读数大致落在 0~1 之间。
2. **关节顺序**：默认 `/left_arm/joint_command` 的前 7 个值依次对应 joint1..7（和原 exoskeleton_bridge 的假设相同）。
3. 首次上真机前，先用 `use_fake_hardware:=true` 加 `dry_run:=true` 把整条链路跑通。

## 连续选择物品抓取（六物品 SmolVLA）

先在一个终端运行 `bash /home/yundu1/ros2_ws/openarm_lerobot/enable_and_zero.sh --execute`，完成使能、回机械零位，并**保持该终端运行**。不要另开 `exoskeleton_bridge` 或 `openarm_lerobot_bridge.py`；交互脚本会在需要时自行启动/停止 LeRobot 桥接。

在另一个终端：

```bash
source /opt/ros/humble/setup.bash
source /home/yundu1/ros2_ws/install/setup.bash
conda activate lerobot
cd /home/yundu1/ros2_ws/openarm_lerobot
python interactive_object_grasp.py             # 只检查路径，不运动
python interactive_object_grasp.py --execute   # 真机交互运行
```

运行后输入 `抓取抽纸` 等物品名，脚本先从实际关节反馈慢速移动到该物品示教的弯举位并张开夹爪；待它提示准备完成后，输入 `/start` 才运行 20 秒抓取。结束后再次输入下一个物品名；输入 `/quit` 退出。切换任务前请移走夹爪中的物品、放回下个物品并清空接收盒。整个过程需人员监护硬件急停。

弯举准备先核对轨迹与原始示教的首帧动作是否唯一匹配；找不到对应的实测观测就不运动。弯举结束时，使用示教首帧的**实测关节姿态**验收（双臂每关节误差 ≤0.08 rad，最后一秒稳定），而非要求实际角度与当时的动作命令完全一致。验收失败不会启动抓取；记录保存在本次运行目录的 `curl_reference.json` 和 `curl/result.json`。

为防止切换程序时机械臂因重力下沉，输入 `/start` 后会从现有桥接状态流只读采样约 0.5 秒：双臂每关节相对本次弯举终点偏移不得超过 0.05 rad，状态波动不得超过 0.02 rad，夹爪归一化偏移不得超过 0.15。通过后记录到 `prestart_check.json`；模型第一条动作到达桥接时会再次核对同一弯举终点。任何检查失败都拒绝抓取，不自动追赶下沉位置；需现场确认姿态后重新准备。模型目标偏差熔断仍保持 0.25 rad，并在日志中显示具体关节、目标角、实测角和差值。

此交互流程的 LeRobot 桥接启用 `safe_mode`：抓取阶段关节最大目标速度 0.25 rad/s、单次最大命令变化 0.01 rad（此前分别为 0.12 rad/s、0.006 rad）；模型目标还必须留在该物品的示教关节范围 ±0.10 rad 内。模型目标与实际角度偏差超过 0.25 rad、控制目标跟踪误差超过 0.15 rad、反馈超时或出现竞争发布者时锁定故障，不再发送新目标。锁定后仍可能保持先前目标，**不是硬件急停**；需检查真机并重启流程。弯举阶段检查关节限位、模型自碰撞和三路相机，但这些检查**不包含**洞洞板、托架、盒子或人员；VLA 抓取阶段也没有环境碰撞规划。该程序不会重新标定电机零点，也不会在抓取后自动回机械零位。

交互抓取专用的模型夹爪命令验收范围为归一化 `[-0.25, 1.25]`（其他桥接流程仍默认 `[-0.05, 1.05]`）。通过验收的有限越界值只会裁剪到实际物理命令 `[0,1]`，不增加夹爪目标行程；NaN、缺失和更大越界仍熔断，并记录原始值。
