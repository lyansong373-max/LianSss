#!/usr/bin/env python3
"""Train one ACT policy per object from the shared, local LeRobot dataset.

Without --execute, this script only validates metadata and prints the command.
It never connects to or commands the robot.
"""

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[3]
DATASET = PROJECT / "data/lerobot/six_objects_60"
RUNS = PROJECT / "work/act/runs"
TASKS = {
    "screwdriver": "Pick up the screwdriver and place it in the receiving box.",
    "transparent_tape": "Pick up the transparent tape and place it in the receiving box.",
    "double_sided_tape": "Pick up the double-sided tape and place it in the receiving box.",
    "tissue": "Pick up the whole tissue package and place it in the receiving box.",
    "black_tape": "Pick up the black tape and place it in the receiving box.",
    "red_tape": "Pick up the red tape and place it in the receiving box.",
}
ALIASES = {
    "螺丝刀": "screwdriver",
    "透明胶带": "transparent_tape",
    "双面胶": "double_sided_tape",
    "抽纸": "tissue",
    "黑色胶带": "black_tape",
    "红色胶带": "red_tape",
}


def episode_map() -> dict[str, list[int]]:
    info = json.loads((DATASET / "meta/info.json").read_text())
    features = info["features"]
    assert features["observation.state"]["shape"] == [16]
    assert features["action"]["shape"] == [16]
    for camera in ("camera1", "camera2", "camera3"):
        assert f"observation.images.{camera}" in features

    result = defaultdict(list)
    files = sorted((DATASET / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No episode metadata found in the shared dataset")
    for file in files:
        table = pq.read_table(file, columns=["episode_index", "tasks"])
        for episode, tasks in zip(table["episode_index"].to_pylist(), table["tasks"].to_pylist()):
            if len(tasks) != 1:
                raise ValueError(f"Episode {episode} has {len(tasks)} tasks; expected exactly one")
            result[tasks[0]].append(episode)
    for task in TASKS.values():
        if len(result[task]) != 10:
            raise ValueError(f"Expected 10 episodes for {task!r}; found {len(result[task])}")
    return result


def command(name: str, episodes: list[int], args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--dataset.repo_id=local/openarm_six_objects",
        f"--dataset.root={DATASET}",
        f"--dataset.episodes={json.dumps(sorted(episodes))}",
        "--dataset.eval_split=0.2",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.chunk_size}",
        f"--output_dir={output}",
        f"--job_name=act_{name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--eval_steps={args.eval_steps}",
        f"--wandb.enable={str(args.wandb).lower()}",
        "--wandb.project=openarm_act",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--object", choices=[*TASKS, *ALIASES], help="One object, English or Chinese")
    selection.add_argument("--all", action="store_true", help="Train six models sequentially")
    parser.add_argument("--execute", action="store_true", help="Actually start GPU training")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.chunk_size, args.num_workers, args.save_freq, args.eval_steps) <= 0:
        parser.error("All numeric options must be positive")

    episodes_by_task = episode_map()
    names = list(TASKS) if args.all else [ALIASES.get(args.object, args.object)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in names:
        episodes = episodes_by_task[TASKS[name]]
        output = RUNS / f"{stamp}_{name}"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {output}")
        cmd = command(name, episodes, args, output)
        print(f"\n{name}: episodes={episodes}, train/eval=8/2, output={output}", flush=True)
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True)
    if not args.execute:
        print("\nOnly validated and printed commands. Add --execute to start training.")


if __name__ == "__main__":
    main()
