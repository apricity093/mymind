"""核对正式三组结果、选模和原故障证据，不采样、不训练、不重新判分。"""
import argparse
import json
from pathlib import Path
import sys

from verify_agentic_retained import verify as verify_inputs


def verify(root, host, expected=None):
    root = Path(root).resolve()
    runs = root / ("agentic_runs" if host == "local" else "runs")
    source = root if host == "local" else root / "src"
    sys.path.insert(0, str(source / "mymind"))
    from experiments.agentic_rl.stage7 import check_diagnostic_configuration

    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = lambda path: [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if expected:
        for name, value in read(expected).items():
            assert read(root / name) == value, name
    inputs = verify_inputs(root, host)
    active = read(runs / "active_run_baseline.json")
    prep = runs / active["directory"]
    baseline = read(prep / active["baseline"])
    ledger = read(prep / "formal/limits.json")
    preflight = read(prep / "preflight/limits.json")
    remaining = {k: v-ledger["consumed"].get(k, 0) for k, v in ledger["limits"].items()}
    assert remaining["sft_updates"] == remaining["grpo_updates"] == 0
    assert all(value >= 0 for value in remaining.values())
    assert preflight["consumed"]["student_trajectories"] == 22
    assert preflight["consumed"]["sft_updates"] == 2 and preflight["consumed"]["grpo_updates"] == 4
    counts = {}
    for phase, steps in (("sft", (40, 80, 120)), ("grpo", (20, 40, 80))):
        selection = read(prep / f"formal/{phase}-selection.json")
        candidates = []
        for step in steps:
            directory = prep / f"formal/{phase}-validation-{step}"
            config, metrics = read(directory / "configuration.json"), read(directory / "metrics.json")
            rows = records(directory / "first_records.jsonl")
            assert len(rows) == 80 and [r["task_id"] for r in rows] == config["task_ids"]
            assert metrics["all_tasks_finished"] and metrics.get("unresolved_infrastructure_failures", metrics["infrastructure_failures"]) == 0
            assert read(prep / f"formal/{phase}/checkpoint-{step}/trainer_state.json")["global_step"] == step
            candidates.append((step, metrics["combination_equal_success_rate"], metrics["combination_equal_severe_error_rate"]))
            counts[f"{phase}-{step}"] = len(rows)
        assert selection["selected_step"] == min(candidates, key=lambda c: (-c[1], c[2], c[0]))[0] == (120 if phase == "sft" else 40)
    original = prep / "formal/sft-validation-40"
    partial = read(original / "M6-P07-002.partial.json")
    recovered = read(original / "M6-P07-002.json")
    assert len(partial["trajectory"]["searches"]) == 2 and partial["messages"][-1]["role"] == "tool"
    assert recovered["recovery"] and recovered["generations"][:1] == partial["generations"]
    assert recovered["trajectory"]["searches"] == partial["trajectory"]["searches"]
    assert read(original / "M6-P07-003.json")["infrastructure_failure"]
    diagnostic_name = read(original / "infrastructure_resolution.json")["diagnostics"]["M6-P07-003"]
    diagnostic = root / (diagnostic_name.replace("runs/", "agentic_runs/", 1) if host == "local" else diagnostic_name)
    check_diagnostic_configuration(read(original / "configuration.json"), read(diagnostic / "configuration.json"))
    assert read(diagnostic / "execution.json")["exit_code"] == 0
    assert read(diagnostic / "metrics.json")["infrastructure_failures"] == 0
    assert len(records(diagnostic / "first_records.jsonl")) == 1
    result = read(runs / "stage8-20261016/RESULT.json")
    for scope, count in (("test", 120), ("regression", 13)):
        for group in ("base", "sft", "rl"):
            directory = runs / "stage8-20261016" / scope / group
            rows = records(directory / "first_records.jsonl")
            config = read(directory / "configuration.json")
            assert len(rows) == count and [r["task_id"] for r in rows] == config["task_ids"]
            metrics = result["final_test" if scope == "test" else "basic_regression"][group]
            assert sum(r["grade"]["complete_success"] for r in rows) == metrics["complete_success"]
            assert sum(r["grade"]["severe_error"] for r in rows) == metrics["severe_errors"]
    state = read(runs / "stage7-dispatch-20261014/state.json")
    assert state["automation_status"] == "PAUSED"
    if host == "server":
        assert len(list((root / baseline["model_dir"]).glob("model-*-of-00003.safetensors"))) == 3
        for phase, step in (("sft", 120), ("grpo", 40)):
            assert (prep / f"formal/{phase}/checkpoint-{step}/adapter_model.safetensors").stat().st_size > 0
        assert (root / "cache/minilm/onnx/model.onnx").is_file()
        assert (root / "data/agentic_rl/knowledge/chroma").is_dir()
    return dict(host=host, status="passed", inputs=inputs, validation_records=counts,
                final_test_per_group=120, basic_regression_per_group=13,
                selected_sft=120, selected_grpo=40, original_fault_and_recovery_retained=True,
                remaining=remaining, model_calls=0, parameter_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", choices=("local", "server"), required=True)
    parser.add_argument("--expected")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.host, args.expected), ensure_ascii=False, indent=2))
