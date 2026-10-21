import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import DeepSeekGeneration, Generation, parse_qwen
from agents.customer_policy.protocol import PublicTask, SearchObservation
from agents.customer_policy.retrieval import PolicyRetrieval
from agents.customer_policy.runner import run_task
from core.llm_gateway import CacheUsage, LLMResult, LLMRequest, DeepSeekGateway
from mcp.tool_manager import MCPToolManager, Tool, ToolResult


TASK = PublicTask(task_id="dev", role="billing", user_text="银行卡退款咨询")
FINAL = json.dumps(dict(disposition="request_info", reasons=[], facts=[], next_steps=[], missing_info=[]))


def test_teacher_rejects_other_models_before_creating_api_client():
    with pytest.raises(ValueError, match="deepseek-flash"):
        DeepSeekGeneration("unused", {"model": "deepseek-v4-pro"})


class Retriever:
    async def search(self, query, number):
        return SearchObservation(request_number=number,
            status={"reference_id": f"search:{number}:status", "state": "candidates"},
            snippets=[dict(reference_id=f"search:{number}:1", content="实际返回的银行卡政策正文",
                           source_id="billing-policy", title="退款", section_path=[])]), {"query": query}


def test_last_observation_has_final_generation_and_strings_reach_context():
    class Generator:
        def __init__(self):
            self.inputs = []

        async def generate(self, messages, tools_enabled=True):
            self.inputs.append((deepcopy(messages), tools_enabled))
            if len(self.inputs) <= 3:
                return parse_qwen('<tool_call>{"name":"search","arguments":{"query":"退款"}}</tool_call>')
            return parse_qwen(FINAL)

    generator = Generator()
    row = asyncio.run(run_task(TASK, generator, PolicyEnvironment(Retriever())))
    assert len(row["trajectory"]["searches"]) == 3 and len(generator.inputs) == 4
    assert generator.inputs[-1][1] is False
    assert len([m for m in generator.inputs[-1][0] if m["role"] == "tool"]) == 3
    assert all(isinstance(m["content"], str) and "实际返回" in m["content"]
               for m in generator.inputs[-1][0] if m["role"] == "tool")
    assert row["runtime_error"] is None
    assert row["messages"][0] == TASK.messages()[0]
    assert row["messages"][-2]["role"] == "system"
    assert "不再执行检索" in generator.inputs[-1][0][-1]["content"]


def test_invalid_arguments_consume_budget_without_fabricating_query():
    env = PolicyEnvironment(Retriever())
    env.reset(TASK)
    async def run():
        for _ in range(3):
            result = await env.step("search", {})
            assert result.status.state == "invalid_arguments" and not result.snippets
        try:
            await env.step("search", {"query": "第四次"})
        except ValueError:
            return
        assert False
    asyncio.run(run())
    assert env.trajectory().searches[0].action == {}


def test_deepseek_mode_is_explicit_and_actual_response_model_is_recorded():
    requests = []
    async def create(**kwargs):
        requests.append(kwargs)
        raw = {"model": "actual-service-model", "choices": [{"finish_reason": "stop"}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
        return SimpleNamespace(model_dump=lambda: raw,
            choices=[SimpleNamespace(message=SimpleNamespace(content="[0]", tool_calls=[]))])
    gateway = DeepSeekGateway("test", "deepseek-flash", thinking="disabled")
    gateway.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = asyncio.run(gateway.complete(LLMRequest(model="deepseek-flash", stable_prompt="排序")))
    assert requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result.metadata["response_model"] == "actual-service-model"


def test_policy_can_stop_without_forced_search():
    class Generator:
        async def generate(self, messages, tools_enabled=True):
            return parse_qwen(FINAL)
    row = asyncio.run(run_task(TASK, Generator(), PolicyEnvironment(Retriever())))
    assert row["trajectory"]["searches"] == [] and row["runtime_error"] is None


def test_parallel_search_calls_count_requests_and_reject_over_budget():
    class Generator:
        async def generate(self, messages, tools_enabled=True):
            return parse_qwen('\n'.join(['<tool_call>{"name":"search","arguments":{"query":"退款"}}</tool_call>']*4))
    row = asyncio.run(run_task(TASK, Generator(), PolicyEnvironment(Retriever())))
    assert row["runtime_error"]["kind"] == "protocol_error"
    assert row["trajectory"]["searches"] == []


def test_fallback_content_is_excluded_and_empty_keeps_this_call_route():
    class Manager:
        async def call(self, *args, **kwargs):
            args[2]["retrieval_diagnostics"]["index_route"] = "main"
            return ToolResult(True, [{"source_id": "fake", "content": "故障提示"}], "x", error="timeout", degraded=True)
    retrieval = PolicyRetrieval(None, Manager())
    observation, log = asyncio.run(retrieval.search("退款", 1))
    assert observation.status.state == "unavailable" and observation.snippets == []
    assert log["index_route"] == "main"
    class EmptyManager:
        async def call(self, *args, **kwargs):
            args[2]["retrieval_diagnostics"].update(index_route="backup", degraded=True)
            return ToolResult(True, [], "x")
    observation, log = asyncio.run(PolicyRetrieval(None, EmptyManager()).search("退款", 1))
    assert observation.status.state == "empty" and observation.status.degraded
    assert log["index_route"] == "backup"


def test_observation_omits_whole_over_budget_chunks_and_internal_scores():
    class Manager:
        async def call(self, *args, **kwargs):
            return ToolResult(True, [{"source_id": "policy", "content": "规则"*1000, "score": 0.9},
                                     {"source_id": "policy", "content": "短正文", "score": 0.2}], "x")
    observation, log = asyncio.run(PolicyRetrieval(None, Manager(), observation_chars=300).search("规则", 1))
    assert [s.content for s in observation.snippets] == ["短正文"]
    assert "score" not in observation.tool_content() and len(log["data"]) == 2


def test_rerank_reports_success_skip_fallback_and_usage_per_call():
    class Gateway:
        async def complete(self, request):
            return LLMResult("[2, 1, 0]", CacheUsage("deepseek", input_tokens=42))
    async def run():
        manager = MCPToolManager("unused", gateway=Gateway())
        items = [{"content": str(i), "chunk_id": str(i)} for i in range(3)]
        async def handler(params, context):
            return items[:params["count"]]
        manager.register(Tool("x", "x", handler, {"required": [], "properties": {}}, supports_rerank=True))
        result = await manager.call("x", {"count": 3}, rerank_top_k=2)
        assert result.reranked and result.diagnostics["rerank_status"] == "llm_success"
        assert result.diagnostics["rerank_usage"]["input_tokens"] == 42
        result = await manager.call("x", {"count": 1}, rerank_top_k=2)
        assert not result.reranked and result.diagnostics["rerank_status"].startswith("skipped")
        class FailedGateway:
            async def complete(self, request):
                raise RuntimeError("service failure")
        manager._gateway = FailedGateway()
        result = await manager.call("x", {"count": 3}, rerank_top_k=2)
        assert result.success and not result.reranked
        assert result.diagnostics["rerank_status"] == "failed_fallback"
        assert "rerank_usage" not in result.diagnostics
    asyncio.run(run())
