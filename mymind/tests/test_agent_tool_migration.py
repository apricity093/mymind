import asyncio
import time
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agents.agent_orchestrator import (
    AgentFeatureConfig,
    AgentOrchestrator,
    AgentType,
    BillingAgent,
    GeneralAgent,
    Request,
    TechnicalAgent,
)
from agents.tools import billing_tools, build_shared_rag_tools, redact_arguments, validate_tool_input
from agents.trace_store import InMemoryTraceStore, RedisTraceStore
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.llm_gateway import AnthropicGateway, CacheUsage, DeepSeekGateway, LLMResult, OpenAIGateway, ToolCall


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


def test_preloaded_rag_query_is_request_deduplicated():
    class Manager:
        calls = 0

        async def search_with_rewrite(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("preloaded query must not reach the knowledge backend")

    async def run():
        manager = Manager()
        tool = build_shared_rag_tools(manager)["search_knowledge_base"]
        req = Request("退款规则", "u", "c", knowledge_already_loaded=True)
        result = await tool.handler(req, {"query": "退款规则", "top_k": 5})
        assert result["preloaded"] is True
        assert result["request_cached"] is True
        assert manager.calls == 0

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


def test_trace_api_is_hidden_by_default_and_available_when_enabled():
    import api.main as api

    class Orchestrator:
        features = SimpleNamespace(trace_api_enabled=False)

        def get_tool_trace(self, request_id):
            return {"request_id": request_id}

        def get_recent_tool_traces(self, limit):
            return [{"request_id": "one"}]

    previous = api._orchestrator
    api._orchestrator = Orchestrator()
    try:
        client = TestClient(api.app)
        assert client.get("/trace/tool/one").status_code == 404
        api._orchestrator.features.trace_api_enabled = True
        payload = client.get("/trace/tool/one").json()
        assert payload["found"] is True
        assert client.get("/trace/tools?limit=999").json()["items"][0]["request_id"] == "one"
    finally:
        api._orchestrator = previous


def test_migration_dataset_and_offline_ablation_pass(tmp_path):
    from pathlib import Path
    from experiments.agent_migration import load_dataset, run_agent_migration

    dataset = Path(__file__).parents[1] / "data" / "eval" / "agent_migration_dataset.json"
    rows = load_dataset(dataset)
    assert len(rows) == 240
    assert sum(bool(row["turns"]) for row in rows) == 60
    assert sum(bool(row["adversarial"]) for row in rows) >= 48
    report = run_agent_migration(tmp_path, dataset)
    assert report["overall_passed"] is True
    assert report["variants"]["E5"]["tool_f1"] >= 0.90
    assert report["variants"]["E5"]["unauthorized_tool_executions"] == 0


def test_downloaded_echomind_examples_are_smoke_only():
    import json
    from pathlib import Path

    path = Path(__file__).parents[1] / "data" / "eval" / "echomind_smoke.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["formal_gate"] is False
    assert data["purpose"] == "smoke_only"
    assert len(data["intent_cases"]) == 11
    assert len(data["dialog_cases"]) == 5


def test_real_migration_experiment_requires_explicit_cost_confirmation(tmp_path):
    from pathlib import Path
    from experiments.agent_migration_real import run_real_agent_migration

    dataset = Path(__file__).parents[1] / "data" / "eval" / "agent_migration_dataset.json"
    try:
        run_real_agent_migration(tmp_path, dataset, confirm_cost=False)
    except RuntimeError as exc:
        assert "--confirm-cost" in str(exc)
    else:
        raise AssertionError("paid experiment must require explicit confirmation")


def test_real_experiment_gateway_enforces_budget_and_disables_hidden_retry():
    from experiments.agent_migration_real import BudgetedGateway
    from core.llm_gateway import LLMRequest

    class Delegate:
        provider = "deepseek"
        tool_protocol = "anthropic"

        def __init__(self):
            self.request = None

        async def complete(self, request):
            self.request = request
            return llm_result("ok")

    async def run():
        delegate = Delegate()
        gateway = BudgetedGateway(delegate, max_calls=1)
        await gateway.complete(LLMRequest(model="m", stable_prompt="s", max_tokens=10))
        assert delegate.request.max_tokens == 2048
        assert delegate.request.retry_incomplete is False
        try:
            await gateway.complete(LLMRequest(model="m", stable_prompt="s"))
        except RuntimeError as exc:
            assert "model_call_budget_exhausted" in str(exc)
        else:
            raise AssertionError("model call budget must be a hard limit")

    asyncio.run(run())
