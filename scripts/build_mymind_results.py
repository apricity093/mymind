"""从保留的逐题记录生成可独立打开的 MyMind 展示页，不调用模型或服务。"""
import json
import math
from pathlib import Path
from statistics import mean

from summarize_grpo_comparison import disposition


ROOT = Path(__file__).resolve().parents[1]
FORMAL = "agentic_runs/stage7-preparation-20261014/formal"
TEST = "agentic_runs/stage8-20261016/test"
COMPARISON = "agentic_runs/grpo-standard-20261021"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def cohort(directory, name, key, expected_n, standards):
    path = ROOT / directory
    records = [json.loads(line) for line in (path / "first_records.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics = read(f"{directory}/metrics.json")
    assert len(records) == len({r["task_id"] for r in records}) == expected_n
    assert metrics["planned_tasks"] == metrics["finished_tasks"] == expected_n
    assert metrics["all_tasks_finished"] and metrics["infrastructure_failures"] == 0
    assert not any(r["infrastructure_failure"] for r in records)
    for row in records:
        assert math.isfinite(row["elapsed_s"]) and row["elapsed_s"] > 0
        for generation in row["generations"]:
            assert generation["input_tokens"] > 0
            assert generation["output_tokens"] == len(generation["output_token_ids"])
    counts = dict(
        success=sum(r["grade"]["complete_success"] for r in records),
        severe=sum(r["grade"]["severe_error"] for r in records),
        protocol=sum(r["grade"]["protocol_error"] for r in records),
        disposition=sum(disposition(r) in standards[r["task_id"]]["dispositions"] for r in records),
    )
    assert math.isclose(counts["success"] / expected_n, metrics["instance_success_rate"])
    assert counts["protocol"] == metrics["protocol_failures"]
    groups = []
    for group_id, values in metrics["breakdown"]["group"].items():
        group_records = [r for r in records if r["task_id"].split("-")[1] == group_id]
        group = dict(id=group_id, n=len(group_records), **{
            field: sum(r["grade"][grade] for r in group_records)
            for field, grade in (("success", "complete_success"), ("severe", "severe_error"), ("protocol", "protocol_error"))
        })
        assert group["n"] == values["tasks"]
        groups.append(group)
    assert math.isclose(mean(g["success"] / g["n"] for g in groups), metrics["combination_equal_success_rate"])
    result = dict(id=key, name=name, n=expected_n, **counts,
        weighted=metrics["combination_equal_success_rate"] * 100,
        groups=groups, source=f"{directory}/first_records.jsonl",
        efficiency=dict(
            seconds=mean(r["elapsed_s"] for r in records),
            input_tokens=mean(sum(g["input_tokens"] for g in r["generations"]) for r in records),
            output_tokens=mean(sum(g["output_tokens"] for g in r["generations"]) for r in records),
            searches=mean(len(r["trajectory"]["searches"]) for r in records),
            model_calls=mean(len(r["generations"]) for r in records),
        ))
    successes = {r["task_id"] for r in records if r["grade"]["complete_success"]}
    return result, successes, {r["task_id"] for r in records}


def build():
    historic = read("agentic_runs/stage8-20261016/RESULT.json")
    comparison = read(f"{COMPARISON}/RESULT.json")
    standards = {r["task_id"]: r for r in read(f"{TEST}/rl/inputs/private_standards.v5.json")}
    data = dict(formal={}, nodes=[], selection=historic["model_selections"],
        selected_a=comparison["selected_a"], selected_b=comparison["selected_b"],
        selected_pair={k: v for k, v in comparison["selected_pair"].items() if k != "details"},
        training=comparison["training"], environment=read(f"{COMPARISON}/environment.json")["versions"])
    success_sets = {}
    for split, paths, n in (
        ("final_test", [f"{TEST}/base", f"{TEST}/sft", f"{TEST}/rl"], 120),
        ("validation", [f"{FORMAL}/base", f"{FORMAL}/sft-validation-120", f"{FORMAL}/grpo-validation-40"], 80),
    ):
        rows = []
        previous_ids = None
        for key, name, directory in zip(("base", "sft", "rl"), ("Base", "SFT120", "当前 A40 · DAPO式"), paths):
            row, success, ids = cohort(directory, name, key, n, standards)
            assert previous_ids is None or ids == previous_ids
            previous_ids = ids
            rows.append(row)
            success_sets[split, key] = success
        data["formal"][split] = rows
    data["formal_pair"] = dict(
        gained=len(success_sets["final_test", "rl"] - success_sets["final_test", "sft"]),
        lost=len(success_sets["final_test", "sft"] - success_sets["final_test", "rl"]))
    selected = {}
    for step in (20, 40, 80):
        a, a_success, a_ids = cohort(f"{FORMAL}/grpo-validation-{step}", f"A{step}", "a", 80, standards)
        b, b_success, b_ids = cohort(f"{COMPARISON}/validation-{step}", f"B{step}", "b", 80, standards)
        assert a_ids == b_ids == previous_ids
        for arm, value in (("a", a), ("b", b)):
            expected = comparison["nodes"][str(step)][arm]
            for actual, source in (("success", "complete_success"), ("severe", "severe_errors"),
                                   ("protocol", "protocol_failures"), ("disposition", "primary_disposition_correct")):
                assert value[actual] == expected[source]
            assert math.isclose(value["weighted"], expected["combination_equal_success_rate"] * 100)
            if step == data[f"selected_{arm}"]:
                selected[arm] = value
        data["nodes"].append(dict(step=step, a=a, b=b, gained=len(b_success-a_success), lost=len(a_success-b_success)))
    data["formal"]["validation"].append(dict(selected["b"], id="b", name="对照 B20 · GRPO式"))
    pair = data["selected_pair"]
    assert selected["a"]["success"] == len(pair["shared_success"]) + len(pair["lost"])
    assert selected["b"]["success"] == len(pair["shared_success"]) + len(pair["gained"])
    data["configs"] = dict(
        base=read("agentic_runs/stage8-20261016/comparison-configurations.json")["configurations"]["base"],
        sft=read(f"{FORMAL}/sft/configuration.json"),
        rl=read(f"{FORMAL}/grpo/configuration.json"),
        b=read(f"{COMPARISON}/configs/train.json"))
    for key, expected in (
        ("rl", dict(loss_type="dapo", scale_rewards="none", epsilon=0.2, epsilon_high=0.28, beta=0.0)),
        ("b", dict(loss_type="grpo", scale_rewards="group", epsilon=0.2, epsilon_high=0.2, beta=0.04)),
    ):
        config = data["configs"][key]
        assert all(config["grpo"][field] == value for field, value in expected.items())
        assert config["training"]["max_steps"] == 80
        assert config["training"]["learning_rate"] == 5e-6 and config["seed"] == 42
    assert data["selection"]["grpo"]["selected_step"] == data["selected_a"] == 40
    assert data["selection"]["sft"]["selected_step"] == 120
    # 展示实际目标和路径；公开题目ID仍由对应原始配置提供。
    for config in data["configs"].values():
        config.pop("task_ids", None)
    fragment = (ROOT / "exports/showcase/mymind-results.fragment.html").read_text(encoding="utf-8")
    assert fragment.count("__MM_DATA__") == 1
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    head = '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta name="color-scheme" content="light">\n<title>MyMind · 配置、质量与推理开销</title>\n</head>\n<body>\n'
    output = head + fragment.replace("__MM_DATA__", payload) + "\n</body>\n</html>\n"
    (ROOT / "mymind-results.html").write_text(output, encoding="utf-8")
    print("Built mymind-results.html: 11 cohorts, 1000 first trajectories; quality, tokens and task IDs verified.")


if __name__ == "__main__":
    build()
