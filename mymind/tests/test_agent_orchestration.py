"""工具编排、请求隔离与策略 HTTP 接口。"""

import asyncio
import httpx
import json
import pytest
import sys
import time
import uuid
from agents.agent_orchestrator import (
    AgentFeatureConfig,
    AgentOrchestrator,
    AgentType,
    BillingAgent,
    GeneralAgent,
    Request,
    TechnicalAgent,
)
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import parse_qwen
from agents.customer_policy.protocol import PublicTask, SearchObservation
from agents.customer_policy.runner import run_task
from agents.tools import (
    billing_tools,
    build_shared_rag_tools,
    check_billing_fields,
    redact_arguments,
    validate_tool_input,
)
from agents.trace_store import InMemoryTraceStore, RedisTraceStore
from copy import deepcopy
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.llm_gateway import (
    AnthropicGateway,
    CacheUsage,
    DeepSeekGateway,
    LLMResult,
    OpenAIGateway,
    ToolCall,
)
from fastapi.testclient import TestClient
from mcp.tool_manager import ToolResult
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace


def test_chat_endpoint_keeps_legacy_and_diagnostic_fields(monkeypatch):
    import api.main as api
    from agents.agent_orchestrator import AgentType
    from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel

    class Orchestrator:
        def get_stats(self):
            return {"agents": 3}

        async def recognize_intent(self, message, history=None):
            return IntentResult(
                intent=IntentCategory.REFUND,
                confidence=0.9,
                urgency=UrgencyLevel.LOW,
                intent_group="billing",
                entities={"order_id": ["#123"]},
                reasoning="refund",
                latency_ms=2.0,
                source_scores={"pattern": 0.9},
            )

        async def run(self, request):
            return SimpleNamespace(
                response="已为您登记退款，款项将在 5-7 个工作日退回。",
                intent=IntentCategory.REFUND,
                agent_type=AgentType.BILLING,
                agent_types=[AgentType.BILLING],
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                routing_reason="intent=refund",
                routing_confidence=0.9,
                escalated=False,
                latency_ms=3.0,
                tools_used=["search_knowledge_base"],
                tool_traces=[{"tool_name": "search_knowledge_base", "knowledge_status": "used"}],
            )

    class Memory:
        async def get_context(self, user_id, conv_id, query=None):
            return SimpleNamespace(recent_messages=[])

        async def add_message(self, *args):
            return None

        async def update_profile(self, *args):
            return None

    monkeypatch.setattr(api, "_orchestrator", Orchestrator())
    monkeypatch.setattr(api, "_memory", Memory())
    monkeypatch.setattr(api, "_context_builder", SimpleNamespace(build=lambda *args: SimpleNamespace(text="ctx", metadata={})))
    response = TestClient(api.app).post("/chat", json={"message": "我要退款"})
    assert response.status_code == 200
    payload = response.json()
    for field in ("conv_id", "request_id", "response", "intent", "agent_type", "escalated", "latency_ms",
                  "knowledge_used", "knowledge_status", "knowledge_reason",
                  "intent_group", "entities", "agent_types", "primary_agent", "supporting_agents", "tools_used"):
        assert field in payload, field
    assert payload["knowledge_used"] is True
    assert payload["knowledge_status"] == "used"


class SequenceGateway:
    provider = "test"
    tool_protocol = "openai"

    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.results.pop(0)


def llm_result(text="", tool_calls=None, assistant_message=None):
    return LLMResult(
        text=text,
        usage=CacheUsage("test", input_tokens=10),
        metadata={"cache_status": "miss"},
        tool_calls=list(tool_calls or []),
        assistant_message=assistant_message,
    )


def technical_request(request_id="req-1"):
    return Request(
        message="登录时报 401",
        user_id="u1",
        conv_id="c1",
        intent=IntentCategory.TECHNICAL_LOGIN,
        intent_group="technical",
        urgency=UrgencyLevel.HIGH,
        intent_confidence=0.95,
        entities={"error_code": ["401"]},
        request_id=request_id,
    )


def e5_orchestrator(gateway):
    return AgentOrchestrator(
        api_key="test", model="test", gateway=gateway,
        features=AgentFeatureConfig.for_variant("E5"),
    )


def test_feature_variants_and_role_scopes_are_distinct():
    assert AgentFeatureConfig.for_variant("E0").profile_enabled is False
    assert AgentFeatureConfig.for_variant("E2").tool_use_enabled is True
    assert AgentFeatureConfig.for_variant("E5").trace_enabled is True
    assert set(GeneralAgent.profile.tool_scope) != set(TechnicalAgent.profile.tool_scope)
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "compare_amounts" in BillingAgent.profile.tool_scope
    request_id = Request("hello", "u", "c").request_id
    assert str(uuid.UUID(request_id)) == request_id


def test_tool_input_validation_and_trace_redaction():
    from agents.tools import technical_tools
    tool = technical_tools()["lookup_error_code"]
    try:
        validate_tool_input(tool, {"error_code": "401", "secret": "bad"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool field must fail")
    assert redact_arguments({"api_key": "abc", "query": "hello"}) == {
        "api_key": "[REDACTED]", "query": "hello"
    }
    try:
        validate_tool_input(billing_tools()["compare_amounts"], {"amount_a": True, "amount_b": 1})
    except ValueError as exc:
        assert "类型错误" in str(exc)
    else:
        raise AssertionError("JSON boolean must not satisfy a numeric schema")
    assert "sensitive-value" not in redact_arguments({
        "query": "请检查 token=sensitive-value 和 Bearer sensitive-value"
    })["query"]


def test_provider_tool_call_parsers_normalize_anthropic_and_openai():
    anthropic = AnthropicGateway._tool_calls([{
        "type": "tool_use", "id": "a1", "name": "lookup_error_code", "input": {"error_code": "401"}
    }])
    assert anthropic == [ToolCall("a1", "lookup_error_code", {"error_code": "401"})]
    openai = OpenAIGateway._tool_calls({"tool_calls": [{
        "id": "o1", "function": {"name": "compare_amounts", "arguments": '{"amount_a": 2, "amount_b": 1}'}
    }]})
    assert openai == [ToolCall("o1", "compare_amounts", {"amount_a": 2, "amount_b": 1})]
    assert DeepSeekGateway.tool_protocol == "openai"
    assert DeepSeekGateway._tool_calls({"tool_calls": []}) == []


def test_tool_round_trip_executes_whitelisted_tool_and_records_trace():
    gateway = SequenceGateway([
        llm_result(
            tool_calls=[ToolCall("call-1", "lookup_error_code", {"error_code": "401"})],
            assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        ),
        llm_result("请检查 Token 是否过期。"),
    ])

    async def run():
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(technical_request())
        assert result.tools_used == ["lookup_error_code"]
        assert result.tool_traces[0]["success"] is True
        assert result.tool_traces[0]["input"] == {"error_code": "401"}
        trace = orchestrator.get_tool_trace("req-1")
        assert trace["tool_calls"][0]["tool_name"] == "lookup_error_code"
        assert len(gateway.requests) == 2

    asyncio.run(run())


def test_unauthorized_tool_is_traced_but_never_executed():
    gateway = SequenceGateway([
        llm_result(
            tool_calls=[ToolCall("call-1", "compare_amounts", {"amount_a": 2, "amount_b": 1})],
            assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        ),
        llm_result("我无法执行账单工具。"),
    ])

    async def run():
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(technical_request())
        assert result.tools_used == []
        assert result.tool_traces[0]["success"] is False
        assert result.tool_traces[0]["error_code"] == "unauthorized_tool"

    asyncio.run(run())


def test_invalid_tool_arguments_are_standardized_and_do_not_abort_agent():
    gateway = SequenceGateway([
        llm_result(
            tool_calls=[ToolCall("call-1", "lookup_error_code", {"error_code": 401})],
            assistant_message={"role": "assistant", "content": ""},
        ),
        llm_result("请补充字符串形式的错误码。"),
    ])

    async def run():
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(technical_request())
        assert result.response.startswith("请补充")
        assert result.tools_used == []
        assert result.tool_traces[0]["error_code"] == "invalid_tool_arguments"

    asyncio.run(run())


def test_provider_without_tool_protocol_falls_back_to_plain_generation():
    class Gateway:
        provider = "legacy"
        tool_protocol = "openai"

        def __init__(self):
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            if request.tools:
                raise RuntimeError("tool calls are not supported by this provider")
            return llm_result("无工具降级回答")

    async def run():
        gateway = Gateway()
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(technical_request())
        assert result.response == "无工具降级回答"
        assert len(gateway.requests) == 2
        assert gateway.requests[-1].tools is None
        assert result.tool_traces[0]["error_code"] == "provider_tool_unsupported"

    asyncio.run(run())


def test_tool_loop_stops_after_three_rounds_with_explicit_degradation():
    call = ToolCall("loop", "lookup_error_code", {"error_code": "401"})
    gateway = SequenceGateway([
        llm_result(tool_calls=[call], assistant_message={"role": "assistant", "content": ""})
        for _ in range(3)
    ])

    async def run():
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(technical_request())
        assert len(gateway.requests) == 3
        assert len(result.tool_traces) == 3
        assert "轮数上限" in result.response

    asyncio.run(run())


def test_rag_tool_query_is_request_deduplicated():
    class Manager:
        calls = 0

        async def search_with_rewrite(self, *args, **kwargs):
            self.calls += 1
            return ToolResult(success=True, data=[{"content": "退款规则"}], tool_name="knowledge_search")

    async def run():
        manager = Manager()
        tool = build_shared_rag_tools(manager)["search_knowledge_base"]
        req = Request("退款规则", "u", "c")
        first = await tool.handler(req, {"query": "退款规则", "top_k": 5})
        second = await tool.handler(req, {"query": "退款规则", "top_k": 5})
        assert first["knowledge_status"] == second["knowledge_status"] == "used"
        assert second["results"] == first["results"]
        assert second["request_cached"] is True
        assert manager.calls == 1

    asyncio.run(run())


def test_composer_failure_preserves_all_successful_agent_results():
    gateway = SequenceGateway([llm_result("技术结论"), llm_result("账单结论")])

    async def run():
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(Request(
            message="登录报 401，而且退款还没到账", user_id="u", conv_id="c",
            intent=IntentCategory.TECHNICAL_LOGIN, intent_group="technical",
            urgency=UrgencyLevel.HIGH, intent_confidence=0.95,
        ))
        assert "技术结论" in result.response
        assert "账单结论" in result.response
        assert "补充说明" in result.response

    asyncio.run(run())


def test_escalation_agent_is_deterministic_and_does_not_call_gateway():
    gateway = SequenceGateway([])

    async def run():
        orchestrator = e5_orchestrator(gateway)
        result = await orchestrator.run(Request(
            message="我要转人工", user_id="u", conv_id="c",
            intent=IntentCategory.HUMAN_HANDOFF, intent_group="escalation",
            urgency=UrgencyLevel.HIGH, intent_confidence=0.99,
        ))
        assert result.agent_type == AgentType.ESCALATION
        assert result.escalated is True
        assert "人工升级" in result.response
        assert gateway.requests == []

    asyncio.run(run())


def test_request_trace_store_ttl_and_capacity():
    now = [100.0]
    store = InMemoryTraceStore(ttl_s=10, max_entries=2, clock=lambda: now[0])
    store.save({"request_id": "one"})
    store.save({"request_id": "two"})
    store.save({"request_id": "three"})
    assert store.get("one") is None
    assert [item["request_id"] for item in store.recent(20)] == ["three", "two"]
    now[0] = 111.0
    assert store.recent() == []


class SharedRedisBackend:
    def __init__(self):
        self.values = {}
        self.sorted_sets = {}

    def pipeline(self, transaction=True):
        backend = self

        class Pipeline:
            def __init__(self):
                self.ops = []

            def set(self, *args, **kwargs):
                self.ops.append((backend.set, args, kwargs)); return self

            def zadd(self, *args, **kwargs):
                self.ops.append((backend.zadd, args, kwargs)); return self

            def expire(self, *args, **kwargs):
                self.ops.append((backend.expire, args, kwargs)); return self

            def execute(self):
                return [operation(*args, **kwargs) for operation, args, kwargs in self.ops]

        return Pipeline()

    def set(self, key, value, ex=None):
        self.values[key] = value; return True

    def get(self, key):
        return self.values.get(key)

    def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update(mapping); return len(mapping)

    def zcard(self, key):
        return len(self.sorted_sets.get(key, {}))

    def zrem(self, key, member):
        return int(self.sorted_sets.get(key, {}).pop(member, None) is not None)

    def zremrangebyscore(self, key, minimum, maximum):
        maximum = float(maximum)
        members = [member for member, score in self.sorted_sets.get(key, {}).items() if score <= maximum]
        for member in members:
            self.sorted_sets[key].pop(member, None)
        return len(members)

    def zremrangebyrank(self, key, start, end):
        ordered = sorted(self.sorted_sets.get(key, {}).items(), key=lambda item: item[1])
        if end < 0:
            end = len(ordered) + end
        for member, _ in ordered[start:end + 1]:
            self.sorted_sets[key].pop(member, None)

    def zrevrange(self, key, start, end):
        ordered = sorted(self.sorted_sets.get(key, {}).items(), key=lambda item: item[1], reverse=True)
        return [member for member, _ in ordered[start:end + 1]]

    def expire(self, key, ttl):
        return True


def test_redis_trace_is_visible_across_store_instances_and_pruned():
    backend = SharedRedisBackend()
    writer = RedisTraceStore(backend, ttl_s=10, max_entries=2)
    reader = RedisTraceStore(backend, ttl_s=10, max_entries=2)
    now = time.time()
    for index in range(3):
        writer.save({"request_id": f"r{index}", "timestamp_epoch": now + index})
    assert reader.get("r2")["request_id"] == "r2"
    assert [item["request_id"] for item in reader.recent(20)] == ["r2", "r1"]


def test_two_hundred_concurrent_requests_do_not_share_trace_state():
    class Gateway:
        provider = "test"
        tool_protocol = "openai"

        async def complete(self, request):
            await asyncio.sleep(0)
            return llm_result("独立结果")

    async def run():
        orchestrator = e5_orchestrator(Gateway())
        requests = [technical_request(f"req-{index}") for index in range(200)]
        results = await asyncio.gather(*(orchestrator.run(req) for req in requests))
        assert len({item.request_id for item in results}) == 200
        assert all(item.tools_used == [] and item.tool_traces == [] for item in results)
        assert len(orchestrator.get_recent_tool_traces(100)) == 100
        assert all(orchestrator.get_tool_trace(f"req-{index}")["request_id"] == f"req-{index}" for index in range(200))

    asyncio.run(run())


def test_trace_api_is_hidden_by_default_and_available_when_enabled(monkeypatch):
    import api.main as api

    class Orchestrator:
        features = SimpleNamespace(trace_api_enabled=False)

        def get_tool_trace(self, request_id):
            return {"request_id": request_id}

        def get_recent_tool_traces(self, limit):
            return [{"request_id": "one"}]

    monkeypatch.setattr(api, "_orchestrator", Orchestrator())
    client = TestClient(api.app)
    assert client.get("/trace/tool/one").status_code == 404
    api._orchestrator.features.trace_api_enabled = True
    payload = client.get("/trace/tool/one").json()
    assert payload["found"] is True
    assert client.get("/trace/tools?limit=999").json()["items"][0]["request_id"] == "one"


@pytest.mark.parametrize("channel", ["未知", "unknown", "  未知  ", " UNKNOWN ", " ", ""])
def test_explicit_unknown_channel_is_missing(channel):
    request = SimpleNamespace(entities={})
    arguments = {"payment_channel": channel}
    validate_tool_input(billing_tools()["check_billing_fields"], arguments)
    result = check_billing_fields(request, arguments)
    assert result["fields"]["payment_channel"] is False
    assert "payment_channel" in result["missing_fields"]
    assert result["can_confirm_refund"] is False


@pytest.mark.parametrize("channel", ["银行卡", "线上支付渠道"])
def test_provided_channel_keeps_existing_field_check(channel):
    result = check_billing_fields(SimpleNamespace(entities={}), {"payment_channel": channel})
    assert result["fields"]["payment_channel"] is True
    assert result["missing_fields"] == ["order_id", "amount", "date"]
    assert result["can_confirm_refund"] is False


def test_unknown_channel_does_not_change_other_known_fields():
    request = SimpleNamespace(entities={"order_id": "ORD-1", "amount": "80元", "date": "2026-10-17"})
    result = check_billing_fields(request, {"payment_channel": "未知"})
    assert result["fields"] == {"order_id": True, "amount": True, "date": True, "payment_channel": False}
    assert result["missing_fields"] == ["payment_channel"]
    assert result["reason"] == "当前工具只做字段检查，不连接订单或支付系统"


FINAL = json.dumps(dict(disposition="request_info", reasons=[], facts=[], next_steps=[], missing_info=[]))


SEARCH = '<tool_call>{"name":"search","arguments":{"query":"退款政策"}}</tool_call>'


class Retrieval:
    async def search(self, query, number):
        return SearchObservation(request_number=number,
            status={"reference_id": f"search:{number}:status", "state": "candidates"},
            snippets=[dict(reference_id=f"search:{number}:1", source_id="policy", title="政策",
                           section_path=[], content="实际正文" * 1200)]), {"query": query}


def test_http_requests_are_public_and_each_has_fresh_observations(tmp_path, policy_service):
    class Generation:
        async def generate(self, messages, tools_enabled=True, **options):
            if not any(m["role"] == "tool" for m in messages):
                return parse_qwen(SEARCH)
            return parse_qwen(FINAL)
        def token_count(self, messages):
            return len(json.dumps(messages, ensure_ascii=False)) // 3
    app = policy_service
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
def test_status_responds_during_blocking_generation_and_gpu_requests_stay_serial(tmp_path, cancel_first, policy_service):
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
        app = policy_service
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


def test_http_generation_failure_is_preserved_and_saved(tmp_path, policy_service):
    class Generation:
        async def generate(self, messages, tools_enabled=True, **options):
            raise RuntimeError("模型生成故障")

    app = policy_service
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
    script = "import sys; import api.policy_app; assert not any(n.startswith(('rl', 'trl', 'peft', 'transformers')) for n in sys.modules)"
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], check=True)
