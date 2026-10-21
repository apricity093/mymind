"""从冻结比较及正式选模记录生成阶段8输入，不调用模型或初始化账本。"""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "agentic_runs/stage8-20261016"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if RUN.exists():
        raise RuntimeError("阶段8目录已存在，不能重新准备覆盖")
    frozen = read(ROOT / "agentic_runs/stage7c-20261016/comparison-configurations.json")
    standards = read(ROOT / "agentic_runs/pre-stage7-fixes-20261014/standards/formal/private_standards.v5.json")
    test_ids = [s["task_id"] for s in standards if s["split"] == "test"]
    assert len(test_ids) == 120
    prefix = "runs/stage8-20261016"
    write(RUN / "authorization-request.json", {
        "date": "2026-10-16", "stage": "8A+8B",
        "source": "本会话用户明确授权连续完成阶段8A最终测试及8B独立演示、面试材料后交接停止",
        "additional_limits": {"student_trajectories": 600, "api_request_allowance": 7200,
                              "gpu_seconds_reserved": 43200, "sft_updates": 0, "grpo_updates": 0},
        "old_remaining_is_not_additional_authorization": True,
        "demo_trajectory_limit": 4, "diagnostic_trajectory_limit": 8,
        "command_timeouts_s": {"test-base": 7200, "test-sft": 7200, "test-rl": 7200,
                               "test-one_search": 3600, "regression": 1800, "demo": 900, "diagnostic": 900}})
    devdir = ROOT / "mymind/experiments/agentic_rl/resources"
    dev_ids = [f"S{i:02d}" for i in range(1, 14)]
    for name in ("dev_tasks.json", "dev_standards.json"):
        write(RUN / "inputs" / name, [r for r in read(devdir / name) if r["task_id"] in dev_ids])
    devbase = read(ROOT / "agentic_runs/stage7-preparation-20261014/baseline.json")
    devbase.update(public_tasks=f"{prefix}/inputs/dev_tasks.json", standards=f"{prefix}/inputs/dev_standards.json")
    write(RUN / "regression-baseline.json", devbase)
    configurations = {}
    for group, source in frozen["configurations"].items():
        cfg = deepcopy(source)
        if group in {"sft", "rl", "one_search"}:
            phase = "grpo" if group == "rl" else "sft"
            selection_path = f"runs/stage7-preparation-20261014/formal/{phase}-selection.json"
            selection = read(ROOT / selection_path.replace("runs/", "agentic_runs/", 1))
            assert selection["selected_step"] == (40 if group == "rl" else 120)
            cfg["adapter_path"] = selection["selected_adapter"]
            cfg["model_selection"] = selection_path
        cfg.update(task_ids=test_ids, evaluation_split="test", output_dir=f"{prefix}/test/{group}",
                   source_manifest="scripts/stage8_sync_files.txt", stage_authorization=f"{prefix}/authorization.json",
                   timeout_s=3600 if group == "one_search" else 7200)
        write(RUN / "configs" / f"test-{group}.json", cfg)
        configurations[group] = cfg
        regression = deepcopy(cfg)
        regression.update(tasks=devbase["public_tasks"], standards=devbase["standards"],
                          baseline=f"{prefix}/regression-baseline.json", task_ids=dev_ids,
                          evaluation_split="dev", output_dir=f"{prefix}/regression/{group}", timeout_s=1800)
        # 一搜仍读取同一正式SFT selection；开发基线仅改变公开任务和对应标准。
        regression.pop("adapter_selection", None)
        write(RUN / "configs" / f"regression-{group}.json", regression)
    write(RUN / "comparison-configurations.json", {"frozen_source": "runs/stage7c-20261016/comparison-configurations.json",
          "configurations": configurations, "test_task_ids": test_ids, "regression_task_ids": dev_ids})


if __name__ == "__main__":
    main()
