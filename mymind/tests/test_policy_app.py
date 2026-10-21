import asyncio
from copy import deepcopy
import json
import sys
from pathlib import Path
from threading import Event, Lock

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import parse_qwen
from agents.customer_policy.protocol import PublicTask, SearchObservation
from agents.customer_policy.runner import run_task
from api import policy_app

FINAL = json.dumps(dict(disposition="request_info", reasons=[], facts=[], next_steps=[], missing_info=[]))
SEARCH = '<tool_call>{"name":"search","arguments":{"query":"退款政策"}}</tool_call>'


class Retrieval:
    async def search(self, query, number):
        return SearchObservation(request_number=number,
            status={"reference_id": f"search:{number}:status", "state": "candidates"},
            snippets=[dict(reference_id=f"search:{number}:1", source_id="policy", title="政策",
                           section_path=[], content="实际正文" * 1200)]), {"query": query}


def test_http_requests_are_public_and_each_has_fresh_observations(tmp_path):
    class Generation:
        async def generate(self, messages, tools_enabled=True, **options):
            if not any(m["role"] == "tool" for m in messages):
                return parse_qwen(SEARCH)
            return parse_qwen(FINAL)
        def token_count(self, messages):
            return len(json.dumps(messages, ensure_ascii=False)) // 3
    app = policy_app.app
    app.state.generation, app.state.retrieval = Generation(), Retrieval()
    app.state.output = tmp_path
    (tmp_path / "requests").mkdir()
    app.state.lock = asyncio.Lock()
    app.state.config = {"budget": {"context_tokens": 16384, "completion_tokens": 8192, "turn_tokens": 2048}}
    app.state.requests_completed = 0
    client = TestClient(app)
    rows = []
    for role in ("billing", "technical"):
        response = client.post("/requests", json={"role": role, "user_text": "公开问题"})
        assert response.status_code == 200
        row = response.json()
        assert row["runtime_error"] is None and row["rendered"] is not None
        assert len(row["trajectory"]["searches"]) == 1
        assert len(row["generations"]) == 2
        assert not any(m["role"] == "tool" for m in row["generations"][0]["input_messages"])
        assert "实际正文" in row["generations"][1]["input_messages"][-1]["content"]
        assert client.get("/requests/" + row["request_id"]).json() == row
        rows.append(row)
    assert rows[0]["request_id"] != rows[1]["request_id"]
    for field in ("task_id", "standards", "reference_answer", "reward"):
        assert client.post("/requests", json={"role": "billing", "user_text": "问题", field: {}}).status_code == 422
    assert client.post("/requests", json={"role": "other", "user_text": "问题"}).status_code == 422


@pytest.mark.parametrize("cancel_first", [False, True])
def test_status_responds_during_blocking_generation_and_gpu_requests_stay_serial(tmp_path, cancel_first):
    class Generation:
        load_info = {"model_class": "BlockingGeneration"}

        def __init__(self):
            self.entered, self.release = Event(), Event()
            self.guard = Lock()
            self.calls, self.active, self.max_active = 0, 0, 0

        def token_count(self, messages):
            return len(json.dumps(messages, ensure_ascii=False)) // 3

        async def generate(self, messages, tools_enabled=True, **options):
            with self.guard:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.entered.set()
            try:
                assert self.release.wait(timeout=5), "测试未释放阻塞生成"
                return parse_qwen(FINAL)
            finally:
                with self.guard:
                    self.active -= 1

    async def check():
        app = policy_app.app
        generation = Generation()
        app.state.generation, app.state.retrieval = generation, Retrieval()
        app.state.output, app.state.lock = tmp_path, asyncio.Lock()
        app.state.started_at, app.state.requests_completed = "test", 0
        app.state.config = {"budget": {}, "retrieval": {}, "generation": {}}
        (tmp_path / "requests").mkdir()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/requests", json={"role": "billing", "user_text": "问题一"}))
            second = None
            try:
                assert await asyncio.to_thread(generation.entered.wait, 2)
                second = asyncio.create_task(client.post("/requests", json={"role": "technical", "user_text": "问题二"}))
                response = await asyncio.wait_for(client.get("/status"), timeout=1)
                assert response.status_code == 200 and response.json()["ready"]
                assert response.json()["requests_completed"] == 0
                assert generation.active == 1 and generation.calls == 1
                assert not first.done() and not second.done()
                if cancel_first:
                    first.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert generation.active == 1 and generation.calls == 1
                    assert not second.done()
            finally:
                generation.release.set()
                responses = await asyncio.gather(first, *([second] if second else []), return_exceptions=cancel_first)
            if cancel_first:
                assert isinstance(responses.pop(0), asyncio.CancelledError)
            assert responses and all(r.status_code == 200 for r in responses)
            assert generation.calls == 2 and generation.max_active == 1
            assert app.state.requests_completed == len(responses)
            assert len(list((tmp_path / "requests").glob("*.json"))) == len(responses)

    asyncio.run(check())


def test_http_generation_failure_is_preserved_and_saved(tmp_path):
    class Generation:
        async def generate(self, messages, tools_enabled=True, **options):
            raise RuntimeError("模型生成故障")

    app = policy_app.app
    app.state.generation, app.state.retrieval = Generation(), Retrieval()
    app.state.output, app.state.lock = tmp_path, asyncio.Lock()
    app.state.config, app.state.requests_completed = {"budget": {}}, 0
    (tmp_path / "requests").mkdir()
    client = TestClient(app)
    response = client.post("/requests", json={"role": "billing", "user_text": "问题"})
    assert response.status_code == 200
    row = response.json()
    assert row["runtime_error"] == {"kind": "generation_failure", "type": "RuntimeError", "detail": "模型生成故障"}
    assert row["infrastructure_failure"] and row["rendered"] is None
    assert client.get("/requests/" + row["request_id"]).json() == row
    assert app.state.requests_completed == 1


def test_budget_preserves_final_turn_and_omits_unseen_whole_snippets():
    class Generation:
        def __init__(self):
            self.inputs = []
        def token_count(self, messages):
            return len(json.dumps(messages, ensure_ascii=False))
        async def generate(self, messages, tools_enabled=True, **options):
            self.inputs.append((deepcopy(messages), tools_enabled, options))
            return parse_qwen(SEARCH if tools_enabled else FINAL)
    generation = Generation()
    row = asyncio.run(run_task(PublicTask(task_id="x", role="billing", user_text="退款"),
        generation, PolicyEnvironment(Retrieval()),
        budget={"context_tokens": 20000, "completion_tokens": 6000, "turn_tokens": 2048}))
    assert row["runtime_error"] is None
    assert generation.inputs[-1][1] is False
    assert generation.inputs[-1][2]["max_new_tokens"] == 2048
    assert any(d.get("observation_budget_applied") for d in row["tool_diagnostics"])
    for turn, diagnostic in zip(row["trajectory"]["searches"], row["tool_diagnostics"]):
        if diagnostic.get("observation_budget_applied"):
            assert len(turn["observation"]["snippets"]) < len(diagnostic["before_token_budget"]["snippets"])


def test_tool_failure_is_returned_with_existing_trajectory():
    class Broken:
        async def search(self, query, number):
            raise RuntimeError("检索服务故障")
    class Generation:
        async def generate(self, messages, tools_enabled=True):
            return parse_qwen(SEARCH)
    row = asyncio.run(run_task(PublicTask(task_id="x", role="billing", user_text="退款"),
                               Generation(), PolicyEnvironment(Broken())))
    assert row["runtime_error"]["kind"] == "tool_failure"
    assert row["infrastructure_failure"] and row["rendered"] is None
    assert row["generations"][0]["raw_text"] == SEARCH


def test_import_does_not_load_experiments_or_trainers():
    import subprocess
    script = "import sys; import api.policy_app; assert not any(n.startswith(('experiments.agentic_rl', 'trl', 'peft', 'transformers')) for n in sys.modules)"
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], check=True)
