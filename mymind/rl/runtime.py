"""实验入口的输入、源码记录与结果写入；不参与策略决策。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from agents.customer_policy.configuration import load_configuration
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--resume-partial", help="接续指定任务已保存全部观察的消息边界")
    args = parser.parse_args()
    root, config, output = load_configuration(args.config)
    if (config.get("dataset_mode") == "formal" or config.get("run_kind")) and (output / "configuration.json").exists():
        previous = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("正式续跑必须使用原配置和输出目录")
        for key in ("tasks", "standards", "policy_support"):
            source = root / config[key]
            recorded = output / "inputs" / source.name
            if not recorded.exists() or recorded.read_bytes() != source.read_bytes():
                raise ValueError("正式续跑输入与原记录不一致")
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for key in ("tasks", "standards", "policy_support"):
        source = root / config[key]
        shutil.copy2(source, inputs / source.name)
    write_json(output / "configuration.json", config)
    invocation = {"argv": sys.argv, "cwd": str(Path.cwd()),
        "work_root": str(root), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "python": sys.version}
    write_json(output / "invocation.json", invocation)
    with (output / "invocations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(invocation, ensure_ascii=False) + "\n")
    source_root = Path(__file__).resolve().parents[2]
    manifest = source_root / config.get("source_manifest", "scripts/agentic_sync_files.txt")
    if manifest.exists():
        for relative in manifest.read_text(encoding="utf-8").splitlines():
            if not relative or relative.startswith("#"):
                continue
            source = source_root / relative
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    tasks = load_public_tasks(root / config["tasks"])
    standards = load_private_standards(root / config["standards"])
    clauses = load_policy_support(root / config["policy_support"])
    validate_task_pairs(tasks, standards, clauses)
    if config.get("run_purpose"):
        from rl.run_control import check_inputs
        check_inputs(root, config, tasks, standards, config["run_purpose"])
    elif config.get("dataset_mode") in {"calibration", "formal"}:
        training_groups = {"R01", "R02", "R03", "R04", "R05", "R06", "D01", "D02", "D05", "D06",
                           "B01", "T01", "P01", "P02", "P03", "P04", "P05", "P06", "F01", "F02"}
        if config.get("dataset_mode") == "calibration" and any(
                s.split != "train" or s.group_id not in training_groups for s in standards.values()):
            raise ValueError("校准仅允许既定训练侧组合")
        if config.get("dataset_mode") == "formal":
            if len(config["task_ids"]) != len(set(config["task_ids"])):
                raise ValueError("正式任务清单不能重复")
            if any(standards[t].split != "train" or standards[t].group_id not in training_groups
                   for t in config["task_ids"]):
                raise ValueError("正式教师采集及导出仅允许训练任务")
    config = apply_resume(root, config, output, args.resume_from_checkpoint)
    if args.resume_partial:
        if config.get("run_purpose") != "evaluate" or config.get("run_kind") != "formal":
            raise ValueError("partial恢复仅用于原正式评测作业")
        if args.resume_partial not in config["task_ids"]:
            raise ValueError("partial恢复任务不属于原作业")
        config = {**config, "resume_partial_task": args.resume_partial}
    return root, config, output, tasks, standards, clauses


def apply_resume(root, config, output, checkpoint):
    if not checkpoint:
        return config
    if config.get("run_purpose") not in {"sft", "grpo"}:
        raise ValueError("恢复参数仅用于训练入口")
    resolved = (root / checkpoint).resolve()
    if resolved.parent != output.resolve() or not (resolved / "trainer_state.json").is_file():
        raise ValueError("仅从同一作业输出目录的checkpoint恢复，不允许开发/预检起点替换")
    return {**config, "resume_from_checkpoint": checkpoint}
