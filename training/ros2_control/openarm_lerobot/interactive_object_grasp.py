#!/usr/bin/env python3
"""Interactive six-object grasping: task -> checked curl -> /start -> video.

Run only while a human supervises the real robot and can use its hardware E-stop.
This does not calibrate motors, avoid unmodelled obstacles, or judge grasp success.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DEMONSTRATIONS = PROJECT / "data/demonstrations"
ACT_RUNS = PROJECT / "work/act/runs"
ACT_POLICIES = {
    name: ACT_RUNS / f"20261103_042148_{slug}/checkpoints/020000/pretrained_model"
    for name, slug in {
        "螺丝刀": "screwdriver",
        "透明胶带": "transparent_tape",
        "双面胶": "double_sided_tape",
        "抽纸": "tissue",
        "黑色胶带": "black_tape",
        "红色胶带": "red_tape",
    }.items()
}
PI0_POLICY = PROJECT / "work/pi0/run_01/checkpoints/008000/pretrained_model"
DEFAULT_ENVELOPE_MARGIN_RAD = 0.15
# Clear tape uses a larger supervised-trial envelope for both arms; its saved
# curl plan differs from the training episodes, especially on the right.
ENVELOPE_MARGIN_RAD = {"透明胶带": 0.30}
# Applies independently to every joint of both arms for all six ACT tasks.
# Velocity, per-step, tracking, demonstration-envelope and URDF guards remain.
MAX_TARGET_ERROR_RAD = 0.40
PI0_MAX_JOINT_VELOCITY_RAD_S = 0.50
PI0_MAX_STEP_RAD = 0.02
MAX_TRACKING_ERROR_RAD = 0.20
TASKS = {
    "螺丝刀": (
        "Pick up the screwdriver and place it in the receiving box.",
        DEMONSTRATIONS / "screwdriver/replay_0_to_11s.json",
    ),
    "透明胶带": (
        "Pick up the transparent tape and place it in the receiving box.",
        DEMONSTRATIONS / "clear_tape/replay_0_to_13p5s.json",
    ),
    "双面胶": (
        "Pick up the double-sided tape and place it in the receiving box.",
        DEMONSTRATIONS / "white_tape/replay_double_sided_0_to_12p5s.json",
    ),
    "抽纸": (
        "Pick up the whole tissue package and place it in the receiving box.",
        DEMONSTRATIONS / "tissue_to_blue_box_demonstrations/replay_grasp_down50_left50mm.json",
    ),
    "黑色胶带": (
        "Pick up the black tape and place it in the receiving box.",
        DEMONSTRATIONS / "black_tape/replay_episode0_0_to_9s.json",
    ),
    "红色胶带": (
        "Pick up the red tape and place it in the receiving box.",
        DEMONSTRATIONS / "red_tape/replay_episode1_0_to_12s.json",
    ),
}
VIDEO_NAMES = {
    "螺丝刀": "screwdriver",
    "透明胶带": "clear_tape",
    "双面胶": "double_sided_tape",
    "抽纸": "tissue",
    "黑色胶带": "black_tape",
    "红色胶带": "red_tape",
}
ALIASES = {
    "纸巾": "抽纸", "整包抽纸": "抽纸", "白色胶带": "双面胶",
    "screwdriver": "螺丝刀", "clear_tape": "透明胶带",
    "double_sided_tape": "双面胶", "tissue": "抽纸",
    "black_tape": "黑色胶带", "red_tape": "红色胶带",
}


def resolve_task(raw: str) -> str | None:
    name = raw.strip()
    if name.startswith("抓取"):
        name = name[2:].strip()
    name = ALIASES.get(name, name)
    return name if name in TASKS else None


def check_policy(policy: Path, expected_type: str) -> Path:
    """Reject an absent or incompatible checkpoint before commanding a curl."""
    if not (policy / "model.safetensors").is_file():
        raise ValueError(f"缺少 {expected_type} 权重：{policy / 'model.safetensors'}")
    try:
        config = json.loads((policy / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取 {expected_type} 配置：{policy / 'config.json'}") from exc
    features = config.get("input_features", {})
    expected = {f"observation.images.camera{i}": [3, 480, 640] for i in range(1, 4)}
    expected["observation.state"] = [16]
    if (config.get("type") != expected_type or
            any(features.get(key, {}).get("shape") != shape for key, shape in expected.items()) or
            config.get("output_features", {}).get("action", {}).get("shape") != [16]):
        raise ValueError(f"{expected_type} 模型与三路相机/16维机械臂配置不符：{policy}")
    if expected_type == "pi0" and any(not (policy / filename).is_file() for filename in
                                       ("policy_preprocessor.json", "policy_postprocessor.json")):
        raise ValueError(f"π₀ 模型缺少预处理配置：{policy}")
    return policy


def check_act_policy(task: str) -> Path:
    return check_policy(ACT_POLICIES[task], "act")


def check_pi0_policy(policy: Path = PI0_POLICY) -> Path:
    return check_policy(policy, "pi0")


def stop_bridge(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def new_run_dir(root: Path, task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{timestamp}_{task}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    plugins = [str(ROOT / "lerobot_robot_openarm_ros2"),
               str(ROOT / "lerobot_teleoperator_openarm_exo")]
    if env.get("PYTHONPATH"):
        plugins.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(plugins)
    return env


def require_command_port_free(port: int) -> None:
    """Reject a separately started bridge before the curl sends any motion."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(
            f"命令端口 {port} 已被占用；请关闭单独启动的 LeRobot 桥接，"
            "本程序会在弯举完成后自行启动桥接。未发送运动目标。"
        ) from exc


def reference_observation(plan: Path) -> dict:
    """Read the unique first measured frame behind this exact action plan.

    Run with the LeRobot Python (pyarrow available). Fail before any motion if
    the plan and source dataset cannot be matched unambiguously.
    """
    import pyarrow.parquet as pq

    data = json.loads(plan.read_text())
    first = data.get("actions", [None])[0]
    keys = data.get("keys")
    expected_keys = [key for side in ("left", "right") for key in
                     ([f"{side}_joint{i}.pos" for i in range(1, 8)] + [f"{side}_gripper.pos"])]
    if keys != expected_keys or not isinstance(first, list) or len(first) != 16:
        raise ValueError(f"轨迹关节顺序或首帧动作无效：{plan}")
    recorded_source = Path(data["source"])
    bundled_source = DEMONSTRATIONS / recorded_source.parent.name / recorded_source.name
    source = bundled_source if (bundled_source / "data").is_dir() else recorded_source
    files = sorted((source / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"找不到示教原始数据：{source}")
    matches = []
    for file in files:
        table = pq.read_table(file, columns=["episode_index", "timestamp", "action", "observation.state"])
        for row in table.to_pylist():
            if abs(row["timestamp"]) > 0.001:
                continue
            if data.get("episode_index") is not None and row["episode_index"] != data["episode_index"]:
                continue
            action = row["action"]
            if len(action) != 16 or max(abs(a - b) for a, b in zip(action, first)) > 1e-6:
                continue
            observation = row["observation.state"]
            if len(observation) != 16 or not all(math.isfinite(v) for v in observation):
                raise ValueError(f"示教初始观测无效：{file}")
            matches.append((row["episode_index"], observation))
    if len(matches) != 1:
        raise ValueError(f"轨迹首帧无法唯一匹配原始示教：{plan}（匹配 {len(matches)} 条）")
    return {"keys": keys, "first_action": first, "observation": matches[0][1],
            "source": str(source), "episode_index": matches[0][0]}


def checked_curl_pose(result_path: Path) -> list[float]:
    result = json.loads(result_path.read_text())
    pose = result.get("final_actual")
    if result.get("status") != "curl_preparation_completed" or not isinstance(pose, list) or \
            len(pose) != 16 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
        raise ValueError(f"弯举结果缺少有效的最终实测姿态：{result_path}")
    return pose


def check_start_pose(reference: list[float], samples: list[list[float]],
                     arm_tolerance: float = 0.05, gripper_tolerance: float = 0.15) -> dict:
    """Fail closed if the robot has sagged or is still moving before /start."""
    if len(reference) != 16 or not all(math.isfinite(v) for v in reference):
        raise ValueError("弯举参考姿态无效")
    if len(samples) < 20 or any(len(q) != 16 or not all(math.isfinite(v) for v in q)
                                for q in samples):
        raise RuntimeError("/start 前未收到足够的有效真机状态")
    arms = [i for i in range(16) if i not in (7, 15)]
    spread = max(max(q[i] for q in samples) - min(q[i] for q in samples) for i in arms)
    if spread > 0.02:
        raise RuntimeError(f"/start 前机械臂仍在移动：关节波动 {spread:.3f} rad")
    actual = samples[-1]
    worst = max(arms, key=lambda i: abs(actual[i] - reference[i]))
    drift = abs(actual[worst] - reference[worst])
    if drift > arm_tolerance:
        side = "左" if worst < 8 else "右"
        joint = worst + 1 if worst < 8 else worst - 7
        raise RuntimeError(f"{side}臂关节{joint}在弯举后偏移 {drift:.3f} rad，"
                           f"超过 {arm_tolerance:g} rad；拒绝抓取，请检查下沉后重新准备")
    grip_drift = max(abs(actual[i] - reference[i]) for i in (7, 15))
    if grip_drift > gripper_tolerance:
        raise RuntimeError(f"夹爪在弯举后偏移 {grip_drift:.3f}（归一化），拒绝抓取")
    return {"actual": actual, "max_arm_drift_rad": drift,
            "max_gripper_drift_normalized": grip_drift, "arm_spread_rad": spread}


def read_bridge_start_pose(port: int, keys: list[str], reference: list[float]) -> dict:
    """Read only the bridge's existing state UDP stream; never send a command."""
    samples = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", port))
        receiver.settimeout(0.3)
        deadline = time.monotonic() + 3
        first_at = None
        while time.monotonic() < deadline:
            try:
                payload = json.loads(receiver.recvfrom(65535)[0])
                state = payload["state"]
                sample = [float(state[key]) for key in keys]
            except (socket.timeout, ValueError, KeyError, TypeError):
                continue
            if not all(math.isfinite(v) for v in sample):
                continue
            now = time.monotonic()
            if first_at is None:
                first_at = now
            samples.append(sample)
            if now - first_at >= 0.5 and len(samples) >= 20:
                break
    return check_start_pose(reference, samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-kind", choices=("act", "pi0"), default="act",
                        help="Model family; ACT remains the default")
    parser.add_argument("--policy-path", type=Path, default=PI0_POLICY,
                        help="Pi0 checkpoint directory; ignored for ACT")
    parser.add_argument("--execute", action="store_true", help="Allow real motion; without this only list tasks and check paths")
    parser.add_argument("--duration", type=int, default=20, help="Seconds per /start rollout, default 20")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/interactive_object_grasp")
    parser.add_argument("--record-videos", action="store_true",
                        help="Save all three camera videos (already enabled by default)")
    parser.add_argument("--state-port", type=int, default=50110)
    parser.add_argument("--cmd-port", type=int, default=50111)
    parser.add_argument("--teleop-port", type=int, default=50112)
    args = parser.parse_args()
    if args.duration <= 0 or len({args.state_port, args.cmd_port, args.teleop_port}) != 3:
        parser.error("duration must be positive and UDP ports must differ")
    try:
        if args.policy_kind == "pi0":
            check_pi0_policy(args.policy_path)
        else:
            for name in TASKS:
                check_act_policy(name)
    except ValueError as exc:
        parser.error(str(exc))
    for name, (_, plan) in TASKS.items():
        if not plan.is_file():
            parser.error(f"Missing {name} curl plan: {plan}")
    print(f"模型：{args.policy_kind}；可用物品：" + "、".join(TASKS), flush=True)
    if not args.execute:
        print("只检查配置，未连接或驱动机械臂。确认现场安全后加 --execute 启动。", flush=True)
        return 0
    if not sys.stdin.isatty():
        parser.error("Real motion requires an interactive terminal")
    try:
        require_command_port_free(args.cmd_port)
    except RuntimeError as exc:
        parser.error(str(exc))

    env = make_env()
    ros_python = "/usr/bin/python3"
    lerobot_python = "/home/yundu1/miniconda3/envs/lerobot/bin/python"
    for executable in (ros_python, lerobot_python):
        if not Path(executable).is_file():
            parser.error(f"Missing Python interpreter: {executable}")
    cameras = subprocess.check_output(
        ["bash", str(ROOT / "language_grasp.sh"), "config"],
        env=env, text=True,
    ).strip()
    pi0 = args.policy_kind == "pi0"
    max_joint_velocity = PI0_MAX_JOINT_VELOCITY_RAD_S if pi0 else 0.30
    max_step = PI0_MAX_STEP_RAD if pi0 else 0.015
    soft_limit_description = (
        "π₀不检查示教范围、模型目标偏差和跟踪误差；"
        if pi0 else f"跟踪误差上限 {MAX_TRACKING_ERROR_RAD:.2f} rad；"
    )
    print(
        "本程序集成弯举、桥接、推理和录像；不要单独启动 LeRobot 桥接。\n"
        "启动后不运动。输入物品名即开始慢速弯举并张开夹爪；到位后输入 /start 才抓取。\n"
        "每次换物品前请确认夹爪为空、物品和接收盒已复位、路径无人且急停有人看护。\n"
        "输入 /start 后会检查弯举以来的下沉与状态稳定性；不通过就拒绝抓取。\n"
        f"抓取阶段关节限速 {max_joint_velocity:.2f} rad/s、单步上限 {max_step:g} rad；"
        f"{soft_limit_description}保留机械限位和反馈等检查，熔断不是急停。\n"
        "模型检查不包含托架、洞洞板、盒子或人员；任何异常立即使用现场硬件急停。\n"
        "不会自动回机械零位，也不会重写电机零点。输入 /quit 退出。\n"
        "录像模式：每次 /start 的三路视频分别保存在该物品目录的 rollout_dataset/videos，"
        "不上传云端；运行完成不代表抓取成功。",
        flush=True,
    )
    while True:
        try:
            raw = input("\n物品（或 /quit）> ").strip()
        except EOFError:
            return 0
        if raw in ("/quit", "/stop", "q", "退出"):
            print("已退出；控制器保持最后目标，本程序不是硬件急停。", flush=True)
            return 0
        task = resolve_task(raw)
        if task is None:
            print("未知物品；请输入：" + "、".join(TASKS), flush=True)
            continue
        instruction, plan = TASKS[task]
        policy_path = check_pi0_policy(args.policy_path) if pi0 else check_act_policy(task)
        envelope_margin = ENVELOPE_MARGIN_RAD.get(task, DEFAULT_ENVELOPE_MARGIN_RAD)
        run_dir = new_run_dir(args.output_root, task)
        try:
            reference = reference_observation(plan)
            reference_path = run_dir / "curl_reference.json"
            reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError) as exc:
            print(f"示教参考观测核对失败；未发送运动目标：{exc}", file=sys.stderr, flush=True)
            return 1
        soft_limit_status = (
            "示教范围、模型目标偏差、关节跟踪误差软阈值已关闭；" if pi0 else
            f"双臂示教范围余量 {envelope_margin:.2f} rad；"
            f"目标与实测差值上限 {MAX_TARGET_ERROR_RAD:.2f} rad；"
            f"关节跟踪误差上限 {MAX_TRACKING_ERROR_RAD:.2f} rad；"
        )
        print(f"准备 {task}：使用 {args.policy_kind} 权重 {policy_path}；将从当前关节反馈慢速到示教弯举位；"
              f"{soft_limit_status}记录：{run_dir}", flush=True)
        try:
            require_command_port_free(args.cmd_port)
        except RuntimeError as exc:
            print(f"弯举预检失败：{exc}", file=sys.stderr, flush=True)
            return 1
        # This command performs its own fresh joint-state, single-publisher,
        # URDF limit, MoveIt self-collision and three-camera preflight.
        try:
            subprocess.run(
                [ros_python, str(ROOT / "slow_demo_replay.py"), "curl",
                 "--plan", str(plan), "--out", str(run_dir / "curl"),
                 "--reference-observation", str(reference_path),
                 "--from-current", "--execute"],
                check=True, env=env,
            )
        except subprocess.CalledProcessError as exc:
            print(f"弯举失败（退出码 {exc.returncode}）；停止本程序，请检查实际姿态和日志。", file=sys.stderr)
            return 1

        try:
            curl_pose = checked_curl_pose(run_dir / "curl/result.json")
        except (OSError, ValueError, KeyError) as exc:
            print(f"弯举实测姿态无效；不会启动抓取桥接：{exc}", file=sys.stderr, flush=True)
            return 1

        bridge = None
        try:
            bridge_command = [
                 ros_python, str(ROOT / "ros2_bridge/openarm_lerobot_bridge.py"),
                 "--ros-args", "-p", f"state_port:={args.state_port}",
                 "-p", f"cmd_port:={args.cmd_port}",
                 "-p", f"teleop_port:={args.teleop_port}",
                 "-p", "gripper_scale:=0.03", "-p", f"max_step_rad:={max_step:g}",
                 "-p", "gripper_command_tolerance:=0.25",
                 "-p", f"max_joint_velocity_rad_s:={max_joint_velocity:.2f}",
                 "-p", f"enforce_policy_soft_limits:={'false' if pi0 else 'true'}",
                 "-p", f"startup_pose_path:={run_dir / 'curl/result.json'}",
                 "-p", "startup_arm_tolerance_rad:=0.05",
                 "-p", "startup_gripper_tolerance:=0.15",
                 "-p", "max_joint_state_age_s:=0.3",
                 "-p", "safe_mode:=true", "-p", "dry_run:=false",
            ]
            if not pi0:
                bridge_command.extend([
                    "-p", f"max_target_error_rad:={MAX_TARGET_ERROR_RAD:.2f}",
                    "-p", f"max_tracking_error_rad:={MAX_TRACKING_ERROR_RAD:.2f}",
                    "-p", f"safety_plan_path:={plan}",
                    "-p", f"envelope_margin_rad:={envelope_margin:.2f}",
                ])
            bridge = subprocess.Popen(bridge_command, env=env, start_new_session=True)
            time.sleep(2)
            if bridge.poll() is not None:
                raise RuntimeError("LeRobot ROS2 bridge did not start; check its log above")
            print(f"{task} 弯举准备完成。确认物品在位后输入 /start；输入 /quit 退出。", flush=True)
            while True:
                command = input("等待 /start > ").strip()
                if command == "/start":
                    break
                if command in ("/quit", "/stop", "q", "退出"):
                    print("未执行抓取；保持弯举位。", flush=True)
                    return 0
                print("此阶段只接受 /start 或 /quit。", flush=True)
            if bridge.poll() is not None:
                raise RuntimeError("Bridge exited before /start")
            start_check = read_bridge_start_pose(args.state_port, reference["keys"], curl_pose)
            (run_dir / "prestart_check.json").write_text(
                json.dumps(start_check, ensure_ascii=False, indent=2))
            print(f"/start 前状态检查通过：最大关节偏移 {start_check['max_arm_drift_rad']:.3f} rad；"
                  "抓取第一条动作到达时还会再次检查。", flush=True)
            print(f"执行 {task}，最多 {args.duration} 秒；运行结束后再输入下一个物品。", flush=True)
            rollout_command = [
                lerobot_python, "-m", "lerobot.scripts.lerobot_rollout",
                 "--robot.type=openarm_ros2", "--robot.id=openarm_bi",
                 f"--robot.state_port={args.state_port}", f"--robot.cmd_port={args.cmd_port}",
                 f"--robot.cameras={cameras}", f"--policy.path={policy_path}",
                 "--device=cuda", "--inference.type=sync",
                 f"--task={instruction}", "--fps=30", f"--duration={args.duration}",
                 "--interactive=false", "--return_to_initial_position=false",
                 "--display_data=false", "--play_sounds=false",
            ]
            rollout_command += [
                "--strategy.type=episodic", "--strategy.reset_to_initial_position=false",
                f"--dataset.repo_id=local/rollout_{VIDEO_NAMES[task]}",
                f"--dataset.root={run_dir / 'rollout_dataset'}",
                f"--dataset.single_task={instruction}",
                "--dataset.num_episodes=1", "--dataset.fps=30",
                "--dataset.video=true", "--dataset.push_to_hub=false",
                "--dataset.no_stamp=true",
            ]
            subprocess.run(rollout_command, check=True, env=env)
            if bridge.poll() is not None:
                raise RuntimeError("Bridge exited during grasp; inspect robot state")
            videos = sorted((run_dir / "rollout_dataset" / "videos").rglob("*.mp4"))
            missing = [camera for camera in ("camera1", "camera2", "camera3")
                       if not any(camera in str(video) for video in videos)]
            if missing:
                raise RuntimeError(f"缺少相机视频 {', '.join(missing)}；请检查 {run_dir / 'rollout_dataset'}")
            print(f"{task} 三路视频已保存（未判断是否成功）：", flush=True)
            for video in videos:
                print(f"  {video}", flush=True)
            print(f"{task} 运行结束；请确认是否成功、清空夹爪并复位下一件物品。", flush=True)
        except (subprocess.CalledProcessError, RuntimeError, EOFError, OSError, ValueError) as exc:
            print(f"抓取流程停止：{exc}。不能据此保证真机已停止；异常时使用现场硬件急停。",
                  file=sys.stderr, flush=True)
            return 1
        finally:
            stop_bridge(bridge)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n人工中断；本程序不是硬件急停，请核对真机状态。", file=sys.stderr)
        raise SystemExit(130)
