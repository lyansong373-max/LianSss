"""Create a metadata-only single-task view; never copy videos or alter joint coordinates."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

TASK = "Pick up the screwdriver and place it in the receiving box."


def select_episodes(records, eval_split):
    if not 0 < eval_split < 1:
        raise ValueError("An episode-disjoint holdout split is required")
    selected = []
    seen = set()
    for row in records:
        ep = int(row["episode_index"])
        if ep in seen:
            raise ValueError(f"Duplicate episode metadata: {ep}")
        seen.add(ep)
        tasks = row["tasks"] or []
        if TASK in tasks:
            if set(tasks) != {TASK}:
                raise ValueError(f"Mixed-task episode cannot be used: {ep}")
            if int(row["length"]) <= 0:
                raise ValueError(f"Empty episode: {ep}")
            selected.append(row)
    selected.sort(key=lambda r: int(r["episode_index"]))
    n_holdout = math.ceil(len(selected) * eval_split)
    if len(selected) < 2 or n_holdout >= len(selected):
        raise ValueError("Need screwdriver episodes in both train and holdout splits")
    train, holdout = selected[:-n_holdout], selected[-n_holdout:]
    return {
        "task": TASK,
        "selected_episodes": [int(r["episode_index"]) for r in selected],
        "train_episodes": [int(r["episode_index"]) for r in train],
        "validation_episodes": [int(r["episode_index"]) for r in holdout],
        "train_frames": sum(int(r["length"]) for r in train),
        "validation_frames": sum(int(r["length"]) for r in holdout),
    }


def verify_frame_rows(rows, task_id, records, selected):
    chosen = set(selected)
    counts = Counter()
    seen = set()
    for row in rows:
        ep = int(row["episode_index"])
        if ep not in chosen:
            continue
        if int(row["task_index"]) != task_id:
            raise ValueError(f"Non-screwdriver frame in selected episode {ep}")
        key = ep, int(row["frame_index"])
        if key in seen:
            raise ValueError(f"Duplicate frame: {key}")
        seen.add(key)
        counts[ep] += 1
    expected = {int(r["episode_index"]): int(r["length"]) for r in records
                if int(r["episode_index"]) in chosen}
    if dict(counts) != expected:
        raise ValueError(f"Frame counts do not match metadata: {dict(counts)} vs {expected}")


def make_config(source, data, split):
    config = copy.deepcopy(source)
    config["dataset"]["root"] = str(data)
    config["dataset"]["episodes"] = split["selected_episodes"]
    # Older server LeRobot does not have exclude_episodes in its config schema.
    if "exclude_episodes" in config["dataset"]:
        config["dataset"]["exclude_episodes"] = None
    config["dataset"]["streaming"] = False
    config["dataset"]["image_transforms"]["enable"] = False
    config["job_name"] = "openvla_oft_film_screwdriver"
    config["resume"] = False
    config["checkpoint_path"] = None
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
        raise ValueError("Expected the existing LeRobot v3 parquet dataset")
    records = pq.read_table(episode_files, columns=["episode_index", "tasks", "length"]).to_pylist()
    split = select_episodes(records, float(config["dataset"]["eval_split"]))
    tasks = pq.read_table(data / "meta/tasks.parquet").to_pylist()
    task_ids = [int(row["task_index"]) for row in tasks if row["task"] == TASK]
    if len(task_ids) != 1:
        raise ValueError("Expected exactly one screwdriver task ID")
    rows = pq.read_table(data_files, columns=["episode_index", "task_index", "frame_index"]).to_pylist()
    verify_frame_rows(rows, task_ids[0], records, split["selected_episodes"])
    files = {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((reference / name).read_text())
        for step in processor["steps"]:
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
    manifest = {
        **split, "task_index": task_ids[0], "dataset": str(data), "source_reference": str(reference),
        "normalization": "Unchanged reference statistics; no recomputation or joint-zero conversion",
        "initialization": "Fresh base, NEW LoRA/action/proprio/FiLM weights; NOT a 5000-step resume",
        "copied_files_sha256": copied, "model_weights_copied": False,
        "videos_copied": False, "source_data_modified": False,
        "robot_deployment_approved": False,
        "calibration_warning": "Right joint 3 was rezeroed on 2026-11-04. This dataset remains in its original coordinates; validate calibration compatibility before deployment.",
    }
    (output / "single_task_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.reference, args.output)
