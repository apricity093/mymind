"""正式有限采集的续跑、首条停止与逐轮原始记录。"""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest

from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import parse_qwen
from agents.customer_policy.protocol import PublicTask, SearchObservation, ToolStatus
from agents.customer_policy.runner import run_task
from experiments.agentic_rl import evaluate


def test_progress_keeps_real_observation_before_next_generation():
    states = []
    class Retrieval:
        async def search(self, query, number):
            return SearchObservation(request_number=number,
                status=ToolStatus(reference_id=f"search:{number}:status", state="empty"), snippets=[]), {"query": query}
    class Generation:
        async def generate(self, messages, tools_enabled=True):
            if len(messages) == 2:
                return parse_qwen('<tool_call>{"name":"search","arguments":{"query":"退款计时"}}</tool_call>')
            assert states[-1]["messages"][-1]["role"] == "tool"
            return parse_qwen('{"disposition":"request_info","reasons":[],"facts":[],"next_steps":[],"missing_info":[]}')
    row = asyncio.run(run_task(PublicTask(task_id="formal-progress", role="billing", user_text="退款只知受理"),
        Generation(), PolicyEnvironment(Retrieval()), on_progress=lambda s: states.append(deepcopy(s))))
    assert row["runtime_error"] is None
    assert len(states[-1]["generations"]) == 2
    assert json.loads(states[-2]["messages"][-1]["content"])["status"]["state"] == "empty"


def test_formal_resume_counts_interrupted_candidate_and_stops_on_first_success(tmp_path, monkeypatch):
    task = PublicTask(task_id="formal-resume", role="billing", user_text="退款只知受理")
    partial = dict(task_id=task.task_id, trajectory={"task_id": task.task_id,
        "materials": [m.model_dump() for m in task.materials()], "searches": []}, messages=task.messages(),
        generations=[], tool_diagnostics=[], raw_result="", rendered=None, budget=None, initial_tokens=None)
    (tmp_path / f"{task.task_id}-candidate-1.partial.json").write_text(json.dumps(partial), encoding="utf-8")
    config = dict(dataset_mode="formal", retrieval={}, generation={}, task_ids=[task.task_id], candidates_per_task=2, concurrency=2)
    class Closable:
        async def close(self): pass
    class Resource:
        def close(self): pass
        def stats(self): return {"pending_main": 0}
        def count(self): return 7
    resource = Resource()
    resource._main = resource
    resource.embedding = resource
    monkeypatch.setattr(evaluate, "initialize", lambda: (tmp_path, config, tmp_path, {task.task_id: task},
        {task.task_id: SimpleNamespace(split="train")}, {}))
    monkeypatch.setattr(evaluate, "rerank_gateway", lambda c: SimpleNamespace(client=Closable()))
    monkeypatch.setattr(evaluate, "build_retrieval", lambda *a: SimpleNamespace(knowledge=resource))
    monkeypatch.setattr(evaluate, "DeepSeekGeneration", lambda *a: Closable())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    calls = []
    async def generate(*args, **kwargs):
        calls.append(task.task_id)
        return {**partial, "raw_result": "success", "runtime_error": None, "infrastructure_failure": False}
    monkeypatch.setattr(evaluate, "run_task", generate)
    monkeypatch.setattr(evaluate, "grade_result", lambda s, result, *a: SimpleNamespace(
        reward=1 if result == "success" else 0, complete_success=result == "success",
        to_dict=lambda: {"complete_success": result == "success"}))
    asyncio.run(evaluate.main(teacher=True))
    assert len(calls) == 1
    first = json.loads((tmp_path / f"{task.task_id}-candidate-1.json").read_text(encoding="utf-8"))
    assert first["runtime_error"]["kind"] == "interrupted"
    accepted = json.loads((tmp_path / "accepted_trajectories.json").read_text(encoding="utf-8"))
    assert len(accepted) == 1 and accepted[0]["candidate"] == 2
    asyncio.run(evaluate.main(teacher=True))
    assert len(calls) == 1


def test_duplicate_query_rejection_requires_same_nonempty_body():
    def turn(query, content):
        return {"action": {"query": query}, "observation": {"status": {"state": "candidates"},
            "snippets": [{"source_id": "policy", "content": content}]}}
    row = {"trajectory": {"searches": [turn("退款", "起算规则"), turn("退款", "起算规则")]}}
    assert evaluate.process_errors(row)
    row["trajectory"]["searches"][1] = turn("退款", "完整起算与时效")
    assert not evaluate.process_errors(row)


@pytest.mark.parametrize("cached_failure", [True, False])
def test_balance_pause_uses_current_requests_not_recorded_402(tmp_path, monkeypatch, cached_failure):
    tasks = {name: PublicTask(task_id=name, role="billing", user_text="退款只知受理") for name in ("first", "next")}
    def row(task, result="", error=None):
        return dict(task_id=task.task_id, trajectory={"task_id": task.task_id,
            "materials": [m.model_dump() for m in task.materials()], "searches": []}, messages=task.messages(),
            generations=[], tool_diagnostics=[], raw_result=result, rendered=None, budget=None,
            initial_tokens=None, runtime_error=error, infrastructure_failure=bool(error), elapsed_s=1)
    balance_error = {"kind": "generation_failure", "status_code": 402}
    if cached_failure:
        old = row(tasks["first"], error=balance_error)
        old.update(candidate=1, grade={"complete_success": False})
        (tmp_path / "first-candidate-1.json").write_text(json.dumps(old), encoding="utf-8")
    config = dict(dataset_mode="formal", retrieval={}, generation={}, task_ids=list(tasks), candidates_per_task=2, concurrency=1)
    class Closable:
        async def close(self): pass
    class Resource:
        def close(self): pass
        def stats(self): return {"pending_main": 0}
        def count(self): return 7
    resource = Resource()
    resource._main = resource
    resource.embedding = resource
    monkeypatch.setattr(evaluate, "initialize", lambda: (tmp_path, config, tmp_path, tasks,
        {name: SimpleNamespace(split="train") for name in tasks}, {}))
    monkeypatch.setattr(evaluate, "rerank_gateway", lambda c: SimpleNamespace(client=Closable()))
    monkeypatch.setattr(evaluate, "build_retrieval", lambda *a: SimpleNamespace(knowledge=resource))
    monkeypatch.setattr(evaluate, "DeepSeekGeneration", lambda *a: Closable())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    calls = []
    async def generate(task, *args, **kwargs):
        calls.append(task.task_id)
        return row(task, result="success" if cached_failure else "", error=None if cached_failure else balance_error)
    monkeypatch.setattr(evaluate, "run_task", generate)
    monkeypatch.setattr(evaluate, "grade_result", lambda s, result, *a: SimpleNamespace(
        reward=1 if result == "success" else 0, complete_success=result == "success",
        to_dict=lambda: {"complete_success": result == "success"}))
    asyncio.run(evaluate.main(teacher=True))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["paused_for_service_balance"] is not cached_failure
    assert calls == (["first", "next"] if cached_failure else ["first"])
    if cached_failure:
        assert json.loads((tmp_path / "first-candidate-1.json").read_text(encoding="utf-8")) == old
        assert summary["accepted_task_ids"] == ["first", "next"]
    else:
        assert summary["unstarted_task_ids"] == ["next"]
        assert not (tmp_path / "first-candidate-2.json").exists()
