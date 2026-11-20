"""Validate ALL six tasks and build a metadata-only training reference. No robot access."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from prepare_screwdriver_reference import verify_frame_rows


def select_all_episodes(records, eval_split):
    if not math.isfinite(eval_split) or not 0 < eval_split < 1:
        raise ValueError("An episode-disjoint holdout split is required")
    groups, seen = defaultdict(list), set()
    for row in records:
        ep, tasks = int(row["episode_index"]), row["tasks"] or []
        if ep in seen or len(tasks) != 1 or not tasks[0] or int(row["length"]) <= 0:
            raise ValueError(f"Duplicate, mixed-task or invalid episode: {ep}")
        seen.add(ep)
        groups[tasks[0]].append(row)
    if len(groups) != 6:
        raise ValueError(f"All-object training requires all six tasks; found {len(groups)}")
    per_task, train, holdout = {}, [], []
    for task, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["episode_index"]))
        n_holdout = math.ceil(len(rows) * eval_split)
        if n_holdout >= len(rows):
            raise ValueError(f"Task lacks disjoint training/holdout episodes: {task}")
        training, validation = rows[:-n_holdout], rows[-n_holdout:]
        train.extend(training)
        holdout.extend(validation)
        per_task[task] = {
            "selected_episodes": [int(row["episode_index"]) for row in rows],
            "train_episodes": [int(row["episode_index"]) for row in training],
            "validation_episodes": [int(row["episode_index"]) for row in validation],
            "train_frames": sum(int(row["length"]) for row in training),
            "validation_frames": sum(int(row["length"]) for row in validation),
        }
    return {"scope": "all_six_objects", "task_count": 6, "tasks": sorted(groups),
            "selected_episodes": sorted(seen), "per_task": per_task,
            "train_episodes": sorted(int(row["episode_index"]) for row in train),
            "validation_episodes": sorted(int(row["episode_index"]) for row in holdout),
            "train_frames": sum(int(row["length"]) for row in train),
            "validation_frames": sum(int(row["length"]) for row in holdout)}


def make_config(source, data, split):
    config = copy.deepcopy(source)
    config["dataset"]["root"] = str(data)
    # Explicitly replace any inherited single-task episode filter.
    config["dataset"]["episodes"] = split["selected_episodes"]
    if "exclude_episodes" in config["dataset"]:
        config["dataset"]["exclude_episodes"] = None
    config["dataset"]["streaming"] = False
    config["dataset"]["image_transforms"]["enable"] = False
    config["job_name"] = "openvla_oft_film_all_six_objects_augmented"
    config["resume"], config["checkpoint_path"] = False, None
    return config


def prepare(data, reference, output):
    import pyarrow.parquet as pq
    data, reference = data.resolve(strict=True), reference.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = json.loads((reference / "train_config.json").read_text())
    policy = json.loads((reference / "config.json").read_text())
    if policy.get("type") != "openvla_oft" or not policy.get("use_film"):
        raise ValueError("Reference must be the retained OpenVLA-OFT FiLM checkpoint")
    episode_files = sorted((data / "meta/episodes").rglob("*.parquet"))
    data_files = sorted((data / "data").rglob("*.parquet"))
    if not episode_files or not data_files:
        raise ValueError("Expected existing LeRobot v3 parquet data")
    records = pq.read_table(episode_files, columns=["episode_index", "tasks", "length"]).to_pylist()
    split = select_all_episodes(records, float(config["dataset"]["eval_split"]))
    tasks = pq.read_table(data / "meta/tasks.parquet").to_pylist()
    task_ids = {row["task"]: int(row["task_index"]) for row in tasks}
    if len(task_ids) != 6 or len(set(task_ids.values())) != 6 or len(tasks) != 6 or set(task_ids) != set(split["tasks"]):
        raise ValueError("Task table does not match the six episode task groups")
    rows = pq.read_table(data_files, columns=["episode_index", "task_index", "frame_index"]).to_pylist()
    if {int(row["episode_index"]) for row in rows} != set(split["selected_episodes"]):
        raise ValueError("Frame data and episode metadata do not cover the same episodes")
    for task, group in split["per_task"].items():
        verify_frame_rows(rows, task_ids[task], records, group["selected_episodes"])
    files = {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in json.loads((reference / name).read_text())["steps"]:
            state_file = step.get("state_file")
            if state_file:
                if Path(state_file).name != state_file or not state_file.endswith(".safetensors"):
                    raise ValueError("Unexpected processor state path")
                files.add(state_file)
    for name in files:
        if not (reference / name).is_file() or (reference / name).stat().st_size > 1024**2:
            raise ValueError(f"Expected small configuration/normalizer file: {name}")
    output.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in sorted(files):
        shutil.copy2(reference / name, output / name)
        copied[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    (output / "train_config.json").write_text(json.dumps(make_config(config, data, split), indent=2) + "\n")
    manifest = {**split, "dataset": str(data), "source_reference": str(reference),
                "task_ids": task_ids, "copied_files_sha256": copied,
                "initialization": "fresh_base_not_resume", "normalization": "unchanged",
                "model_weights_copied": False, "videos_copied": False, "source_data_modified": False,
                "robot_deployment_approved": False, "joint_zero_conversion": False}
    (output / "all_objects_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.reference, args.output)
