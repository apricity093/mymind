import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agents.agent_orchestrator import AgentFeatureConfig, AgentOrchestrator, AgentType, TechnicalAgent
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from tests.test_agent_tool_migration import SequenceGateway, llm_result, technical_request
from core.llm_gateway import ToolCall


def test_default_configuration_enables_only_tool_use_and_rag(monkeypatch):
    for name in (
        "AGENT_PROFILE_ENABLED", "AGENT_TOOL_USE_ENABLED", "AGENT_ESCALATION_ENABLED",
        "AGENT_COMPOSER_ENABLED", "AGENT_TRACE_ENABLED", "TRACE_API_ENABLED", "KNOWLEDGE_TOOL_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = AgentFeatureConfig.from_env()
    assert config == AgentFeatureConfig()
    assert config.tool_use_enabled is True
    assert config.knowledge_tool_mode == "tool_only"
    assert not any((config.profile_enabled, config.escalation_enabled, config.composer_enabled,
                    config.trace_enabled, config.trace_api_enabled))
    orchestrator = AgentOrchestrator(api_key="test", gateway=SequenceGateway([]), features=config)
    assert all("search_knowledge_base" in agent.get_tools()
               for agents in orchestrator._pool.values() for agent in agents)


@pytest.mark.parametrize("kind", [AgentType.TECHNICAL, AgentType.BILLING])
def test_single_degraded_specialist_routes_to_general(kind, monkeypatch):
    monkeypatch.setenv("MYMIND_MONITOR_FALLBACK_PENALTY", "0.5")
    orchestrator = AgentOrchestrator(api_key="test", gateway=SequenceGateway([]))
    orchestrator._pool[kind][0].stats.monitor_penalty = 0.5
    req = technical_request()
    if kind == AgentType.BILLING:
        req.message = "退款还没到账"
        req.intent = IntentCategory.REFUND
        req.intent_group = "billing"
        req.entities = {}
    decision = orchestrator._route_decision(req)
    assert decision.primary_agent == AgentType.GENERAL
    assert decision.supporting_agents == []
    assert f"monitor_fallback=[{kind.value}]" in decision.reason


def test_degraded_specialist_cannot_return_as_supporting_agent(monkeypatch):
    monkeypatch.setenv("MYMIND_MONITOR_FALLBACK_PENALTY", "0.5")
    orchestrator = AgentOrchestrator(api_key="test", gateway=SequenceGateway([]))
    orchestrator._pool[AgentType.BILLING][0].stats.monitor_penalty = 0.8
    req = technical_request()
    req.message += "，同时重复扣款，需要退款"
    decision = orchestrator._route_decision(req)
    assert decision.primary_agent == AgentType.TECHNICAL
    assert decision.supporting_agents == []
    assert "monitor_fallback=[billing]" in decision.reason


def test_multi_instance_routing_uses_remaining_healthy_specialist(monkeypatch):
    monkeypatch.setenv("MYMIND_MONITOR_FALLBACK_PENALTY", "0.5")
    orchestrator = AgentOrchestrator(api_key="test", gateway=SequenceGateway([]))
    degraded = orchestrator._pool[AgentType.TECHNICAL][0]
    degraded.stats.monitor_penalty = 0.5
    healthy = TechnicalAgent(None, "test", gateway=SequenceGateway([]))
    healthy.stats.total, healthy.stats.success = 10, 1
    orchestrator._pool[AgentType.TECHNICAL].append(healthy)
    assert orchestrator._route_decision(technical_request()).primary_agent == AgentType.TECHNICAL
    assert orchestrator._best_agent(AgentType.TECHNICAL) is healthy
    healthy.stats.monitor_penalty = 0.5
    assert orchestrator._route_decision(technical_request()).primary_agent == AgentType.GENERAL
    assert orchestrator._best_agent(AgentType.TECHNICAL) is None


@pytest.mark.parametrize("env_name", ["MYMIND_MONITOR_FALLBACK_PENALTY", "ECHOMIND_MONITOR_FALLBACK_PENALTY"])
def test_monitor_fallback_threshold_is_configurable(env_name, monkeypatch):
    monkeypatch.delenv("MYMIND_MONITOR_FALLBACK_PENALTY", raising=False)
    monkeypatch.setenv(env_name, "0.7")
    orchestrator = AgentOrchestrator(api_key="test", gateway=SequenceGateway([]))
    agent = orchestrator._pool[AgentType.TECHNICAL][0]
    agent.stats.monitor_penalty = 0.5
    assert orchestrator._route_decision(technical_request()).primary_agent == AgentType.TECHNICAL
    agent.stats.monitor_penalty = 0.7
    assert orchestrator._route_decision(technical_request()).primary_agent == AgentType.GENERAL


def test_tool_trace_survives_model_failure_and_general_fallback():
    class Gateway(SequenceGateway):
        async def complete(self, request):
            if len(self.requests) == 1:
                self.requests.append(request)
                raise ConnectionError("model unavailable after tool execution")
            return await super().complete(request)

    async def run():
        gateway = Gateway([
            llm_result(tool_calls=[ToolCall("lookup-401", "lookup_error_code", {"error_code": "401"})],
                       assistant_message={"role": "assistant", "content": ""}),
            llm_result("通用客服回退回答"),
        ])
        orchestrator = AgentOrchestrator(api_key="test", gateway=gateway,
                                         features=AgentFeatureConfig(trace_enabled=True))
        result = await orchestrator.run(technical_request())
        assert result.agent_type == AgentType.GENERAL
        assert result.primary_agent == AgentType.TECHNICAL
        assert result.tools_used == ["lookup_error_code"]
        assert result.tool_traces[0]["input"] == {"error_code": "401"}
        assert result.tool_traces[0]["tool_call_id"] == "lookup-401"
        assert result.tool_traces[1]["error_code"] == "ConnectionError"
        assert orchestrator.get_tool_trace(result.request_id)["tool_calls"] == result.tool_traces
        assert orchestrator._pool[AgentType.TECHNICAL][0].stats.success == 0

    asyncio.run(run())


def test_same_agent_concurrent_tools_keep_inputs_and_trace_ids_isolated():
    class Gateway:
        provider = "test"
        tool_protocol = "openai"

        def __init__(self):
            self.started = 0
            self.both_started = asyncio.Event()

        async def complete(self, request):
            if request.messages[-1]["role"] == "tool":
                return llm_result("排障建议")
            code = "500" if "500" in str(request.messages) else "401"
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await self.both_started.wait()
            return llm_result(tool_calls=[ToolCall(f"lookup-{code}", "lookup_error_code", {"error_code": code})],
                              assistant_message={"role": "assistant", "content": ""})

    async def run():
        orchestrator = AgentOrchestrator(api_key="test", gateway=Gateway(),
                                         features=AgentFeatureConfig(trace_enabled=True))
        first = technical_request("request-401")
        second = technical_request("request-500")
        second.message = "页面报 500"
        second.entities = {"error_code": ["500"]}
        results = await asyncio.gather(orchestrator.run(first), orchestrator.run(second))
        for code, result in zip(("401", "500"), results):
            assert result.request_id == f"request-{code}"
            assert result.tools_used == ["lookup_error_code"]
            assert len(result.tool_traces) == 1
            trace = result.tool_traces[0]
            assert trace["input"] == {"error_code": code}
            assert trace["tool_call_id"] == f"lookup-{code}"
            assert trace["success"] is True
            assert trace["latency_ms"] >= 0
            assert orchestrator.get_tool_trace(result.request_id)["tool_calls"] == [trace]

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["used", "empty", "degraded", "error", "raised", "skipped"])
def test_chat_rag_is_agent_driven_and_reports_actual_tool_outcome(outcome, monkeypatch):
    import api.main as api

    class Manager:
        calls = 0

        async def search_with_rewrite(self, name, query, top_k=5):
            self.calls += 1
            assert name == "knowledge_search"
            assert query == "订单物流政策"
            if outcome == "raised":
                raise ConnectionError("knowledge unavailable")
            return SimpleNamespace(
                success=outcome in {"used", "empty", "degraded"},
                data=[{"content": "物流政策"}] if outcome in {"used", "degraded"} else [],
                degraded=outcome == "degraded", cached=False, reranked=outcome == "used",
                error="backend_unavailable" if outcome == "error" else None,
            )

    class Memory:
        async def get_context(self, *args, **kwargs):
            return SimpleNamespace(recent_messages=[])

        async def add_message(self, *args):
            pass

        async def update_profile(self, *args):
            pass

    manager = Manager()
    replies = [llm_result("请说明需要的物流信息。")]
    if outcome != "skipped":
        replies.insert(0, llm_result(
            tool_calls=[ToolCall("search-1", "search_knowledge_base", {"query": "订单物流政策"})],
            assistant_message={"role": "assistant", "content": ""},
        ))
    gateway = SequenceGateway(replies)
    orchestrator = AgentOrchestrator(api_key="test", gateway=gateway, rag_tool_manager=manager)

    async def recognize(message, history=None):
        # Greeting intent does not prevent an Agent from searching for a business question.
        return IntentResult(IntentCategory.GREETING, 0.9, UrgencyLevel.LOW, "greeting", {}, "", 1.0)

    async def recognize_without_search(message, history=None):
        # Refund intent does not force a search when the Agent asks for more information.
        return IntentResult(IntentCategory.REFUND, 0.9, UrgencyLevel.LOW, "billing", {}, "", 1.0)

    orchestrator.recognize_intent = recognize_without_search if outcome == "skipped" else recognize
    monkeypatch.setattr(api, "_orchestrator", orchestrator)
    monkeypatch.setattr(api, "_memory", Memory())
    monkeypatch.setattr(api, "_tool_manager", manager)
    contexts = []

    def build(memory, knowledge, message):
        contexts.append(knowledge)
        return SimpleNamespace(text="记忆上下文", metadata={})

    monkeypatch.setattr(api, "_context_builder", SimpleNamespace(build=build))
    response = TestClient(api.app).post("/chat", json={"message": "请说明政策", "user_id": "test"})
    assert response.status_code == 200
    payload = response.json()
    assert contexts == [""]
    assert manager.calls == int(outcome != "skipped")
    assert payload["knowledge_used"] is (outcome != "skipped")
    assert payload["knowledge_status"] == ("error" if outcome == "raised" else outcome)
    assert "knowledge_already_loaded" not in vars(technical_request())
