"""SFT超时后的工具观察边界接续及原首次表、额度语义。"""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import parse_qwen
from agents.customer_policy.protocol import SearchObservation
from agents.customer_policy.runner import run_task, restore_progress
from experiments.agentic_rl import formal_evaluate as evaluation
from experiments.agentic_rl.runtime import write_json, initialize
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support
from experiments.agentic_rl.stage7 import reserve_launch
from tests.agentic_rl_samples import retained_path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("MYMIND_WORK_ROOT", str(ROOT)))
RUNS = WORK / ("runs" if os.environ.get("MYMIND_WORK_ROOT") else "agentic_runs")
PREP = RUNS / "stage7-preparation-20261014"
TID = "M6-P07-002"
FINAL = '{"disposition":"answer","reasons":[],"facts":[],"next_steps":[],"missing_info":[]}'


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def saved():
    return read(retained_path(PREP / f"formal/sft-validation-40/{TID}.partial.json"))


@pytest.fixture
def tasks():
    return load_public_tasks(WORK / "data/agentic_rl/formal/public_tasks.json")


class Generator:
    def __init__(self, saved, extra_search=False, failure=False):
        self.saved, self.inputs = saved, []
        self.extra_search, self.failure = extra_search, failure
        self.torch = SimpleNamespace(manual_seed=lambda _: None)
        self.load_info = {"cpu_fixture": True}

    def token_count(self, messages):
        return self.saved["initial_tokens"] if len(messages) == 2 else 6000

    async def generate(self, messages, tools_enabled=True, **options):
        self.inputs.append((deepcopy(messages), tools_enabled, options))
        if self.failure:
            raise RuntimeError("CPU生成故障")
        text = ('<tool_call>{"name":"search","arguments":{"query":"补检"}}</tool_call>'
                if self.extra_search and len(self.inputs) == 1 else FINAL)
        result = parse_qwen(text)
        result.diagnostics = {"latency_ms": 100, "output_tokens": 10}
        return result


def test_actual_partial_continues_with_one_remaining_search(saved, tasks):
    calls = []
    class Retrieval:
        async def search(self, query, number):
            calls.append((query, number))
            return SearchObservation(request_number=number,
                status={"reference_id": f"search:{number}:status", "state": "empty"}, snippets=[]), {}
    generation = Generator(saved, extra_search=True)
    row = asyncio.run(run_task(tasks[TID], generation, PolicyEnvironment(Retrieval()),
                              budget=saved["budget"], resume_state=saved))
    assert calls == [("补检", 3)]
    assert generation.inputs[0][0] == saved["messages"]
    assert generation.inputs[0][1:] == (True, {"max_new_tokens": 2048})
    assert generation.inputs[1][1] is False
    assert row["generations"][:1] == saved["generations"]
    assert row["trajectory"]["searches"][:2] == saved["trajectory"]["searches"]
    assert row["tool_diagnostics"][:2] == saved["tool_diagnostics"]
    assert row["initial_tokens"] == saved["initial_tokens"]
    assert row["runtime_error"] is None


@pytest.mark.parametrize("change", ["messages", "observation", "budget", "initial", "fault"])
def test_incomplete_or_changed_boundary_is_rejected(saved, tasks, change):
    state = deepcopy(saved)
    if change == "messages": state["messages"].pop()
    if change == "observation": state["trajectory"]["searches"].pop()
    if change == "budget": state["budget"]["completion_tokens"] += 1
    if change == "initial": state["initial_tokens"] += 1
    if change == "fault": state["trajectory"]["searches"][0]["observation"]["status"]["degraded"] = True
    with pytest.raises(ValueError):
        restore_progress(tasks[TID], Generator(saved), PolicyEnvironment(None), saved["budget"], state)


def test_remaining_completion_budget_is_not_reset(saved, tasks):
    class Tight(Generator):
        def token_count(self, messages):
            return saved["initial_tokens"] if len(messages) == 2 else saved["initial_tokens"] + 8192 - 100
    generation = Tight(saved)
    row = asyncio.run(run_task(tasks[TID], generation, PolicyEnvironment(None),
                              budget=saved["budget"], resume_state=saved))
    assert generation.inputs[0][1:] == (False, {"max_new_tokens": 100})
    assert row["runtime_error"] is None


@pytest.fixture
def job(tmp_path, monkeypatch, saved, tasks):
    cfg = read(PREP / "configs/formal-sft-validation-40.json")
    standards = load_private_standards(RUNS / "pre-stage7-fixes-20261014/standards/formal/private_standards.v5.json")
    clauses = load_policy_support(WORK / cfg["policy_support"])
    baseline = tmp_path / cfg["baseline"]
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PREP / "baseline.json", baseline)
    output = tmp_path / cfg["output_dir"]
    output.mkdir(parents=True)
    original = PREP / "formal/sft-validation-40/first_records.jsonl"
    (output / original.name).write_bytes(b"".join(original.read_bytes().splitlines(keepends=True)[:41]))
    partial = output / f"{TID}.partial.json"
    shutil.copy2(PREP / f"formal/sft-validation-40/{partial.name}", partial)
    recovery = output / "recovery" / TID
    recovery.mkdir(parents=True)
    shutil.copy2(partial, recovery / "original.partial.json")
    write_json(recovery / "source.json", {"task_id": TID, "original_command_exit_code": 124,
        "historical_cost": {"embedding_requests": 2, "rerank_http_requests": 2},
        "usage_evidence": "原84次账单增量与84条usage逐项核对，包括该题两次成功检索"})
    limits = tmp_path / cfg["limits_file"]
    write_json(limits, {"limits": {"student_trajectories": 1000, "api_request_allowance": 12000},
        "consumed": {"student_trajectories": 122, "api_request_allowance": 1464}, "events": []})
    async def close(): pass
    knowledge = SimpleNamespace(stats=lambda: {"pending_main": 0}, _main=SimpleNamespace(count=lambda: 7),
        close=lambda: None, embedding=SimpleNamespace(usage={"requests": 0}, close=lambda: None))
    client = SimpleNamespace(_client=SimpleNamespace(event_hooks={"request": []}), close=close)
    generator = Generator(saved)
    monkeypatch.setattr(evaluation, "rerank_gateway", lambda _: SimpleNamespace(client=client))
    monkeypatch.setattr(evaluation, "build_retrieval", lambda *_: SimpleNamespace(knowledge=knowledge))
    monkeypatch.setattr(evaluation, "QwenGeneration", lambda *_: generator)
    return tmp_path, {**cfg, "resume_partial_task": TID}, output, tasks, standards, clauses, generator


def test_41_records_untouched_recovery_cost_merged_and_reentry_no_double_charge(job, saved, monkeypatch):
    root, cfg, output, tasks, standards, clauses, generator = job
    first = output / "first_records.jsonl"
    prefix = first.read_bytes()
    graded = []
    original_grade = evaluation.grade_result
    def grade(standard, *args):
        graded.append(standard.task_id)
        return original_grade(standard, *args)
    monkeypatch.setattr(evaluation, "grade_result", grade)
    asyncio.run(evaluation.evaluate(root, cfg, output, tasks, standards, clauses))
    assert first.read_bytes().startswith(prefix)
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 80 and len(graded) == len(generator.inputs) == 39
    recovered = rows[41]
    assert recovered["task_id"] == TID and recovered["cost"]["embedding_requests"] == 2
    assert recovered["cost"]["rerank_http_requests"] == 2
    assert len(recovered["cost"]["usage"]) == 2
    assert recovered["cost"]["gpu_generation_s"] == pytest.approx(saved["generations"][0]["latency_ms"] / 1000 + .1)
    assert recovered["recovery"]["new_cost"]["usage"] == []
    assert recovered["recovery"]["new_cost"]["rerank_http_requests"] == 0
    assert recovered["recovery"]["interrupted_generation_output_tokens"] is None
    assert recovered["elapsed_s"] is None and read(output / "metrics.json")["all_tasks_finished"]
    assert read(output / f"{TID}.partial.json") == saved
    assert read(root / cfg["limits_file"])["consumed"] == {"student_trajectories": 160, "api_request_allowance": 1920}
    finished, ledger = first.read_bytes(), (root / cfg["limits_file"]).read_bytes()
    asyncio.run(evaluation.evaluate(root, cfg, output, tasks, standards, clauses))
    assert first.read_bytes() == finished and (root / cfg["limits_file"]).read_bytes() == ledger
    assert len(graded) == len(generator.inputs) == 39


def test_new_failure_stops_and_reentry_cannot_skip_it(job):
    root, cfg, output, tasks, standards, clauses, generator = job
    generator.failure = True
    with pytest.raises(RuntimeError, match="基础设施故障"):
        asyncio.run(evaluation.evaluate(root, cfg, output, tasks, standards, clauses))
    assert read(output / "metrics.json")["finished_tasks"] == 42
    assert read(output / "metrics.json")["infrastructure_failures"] == 1
    with pytest.raises(RuntimeError, match="未处理"):
        asyncio.run(evaluation.evaluate(root, cfg, output, tasks, standards, clauses))
    assert len(generator.inputs) == 1


@pytest.mark.parametrize("key,value", [("model_revision", "changed"), ("tasks", "other.json"),
    ("grading_revision", "other"), ("generation", {"temperature": 1}), ("budget", {"turn_tokens": 4096})])
def test_frozen_evaluation_changes_rejected(job, key, value):
    root, cfg, output, tasks, standards, clauses, _ = job
    with pytest.raises(ValueError):
        asyncio.run(evaluation.evaluate(root, {**cfg, key: value}, output, tasks, standards, clauses))


def test_initialize_refuses_config_change_before_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("MYMIND_WORK_ROOT", str(tmp_path))
    cfg = read(retained_path(PREP / "configs/formal-sft-validation-40.json"))
    output = tmp_path / cfg["output_dir"]
    output.mkdir(parents=True)
    write_json(output / "configuration.json", cfg)
    changed = {**cfg, "adapter_path": "other/checkpoint-40"}
    path = tmp_path / "changed.json"
    write_json(path, changed)
    monkeypatch.setattr("sys.argv", ["evaluate", "--config", str(path), "--resume-partial", TID])
    with pytest.raises(ValueError, match="原配置"):
        initialize()
    assert read(output / "configuration.json") == cfg


def test_launch_reserves_actual_10800_seconds_and_keeps_frozen_config(tmp_path):
    limits = tmp_path / "limits.json"
    write_json(limits, {"limits": {"gpu_seconds_reserved": 115200},
                       "consumed": {"gpu_seconds_reserved": 18000}, "events": []})
    config = {"timeout_s": 3600, "limits_file": "limits.json", "output_dir": "original"}
    frozen = deepcopy(config)
    for step in (40, 80, 120):
        assert reserve_launch(tmp_path, config, f"formal-sft-validation-{step}", 1, "10800") == 10800
    assert read(limits)["consumed"]["gpu_seconds_reserved"] == 50400
    assert config == frozen
    for name, override in (("formal-base", "10800"), ("formal-sft-validation-40", "21600")):
        with pytest.raises(ValueError):
            reserve_launch(tmp_path, config, name, 1, override)
    assert read(limits)["consumed"]["gpu_seconds_reserved"] == 50400
