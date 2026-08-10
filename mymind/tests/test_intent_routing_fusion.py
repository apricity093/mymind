import asyncio
from types import SimpleNamespace

from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


class NoCallGateway:
    provider = "test"

    async def complete(self, request):
        raise AssertionError("routing-only tests must not call the LLM gateway")


def test_fine_grained_intent_refines_generic_vote_and_extracts_entities():
    async def run():
        recognizer = IntentRecognizer(
            api_key="test",
            base_url="https://example.test",
            model="test-model",
        )
        calls = 0

        async def generic_billing(message, history):
            nonlocal calls
            calls += 1
            return {
                "intent": IntentCategory.BILLING,
                "confidence": 0.6,
                "reasoning": "generic billing result",
            }

        recognizer._llm_recognize = generic_billing
        message = "订单号 AB-1234 明天申请退款500元，页面错误 500"
        result = await recognizer.recognize(message)
        cached = await recognizer.recognize(message)

        assert result.intent == IntentCategory.REFUND
        assert result.intent_group == IntentCategory.BILLING.value
        assert result.confidence == 0.51
        assert result.source_scores == {
            "llm": 0.6,
            "embedding": 0.0,
            "pattern": 0.5,
            "refined_by_pattern": 0.5,
        }
        assert result.entities == {
            "order_id": ["AB-1234"],
            "product": [],
            "date": ["明天"],
            "amount": ["500元"],
            "error_code": ["500"],
        }
        assert cached is result
        assert calls == 1
        assert recognizer.cache_stats["hits"] == 1

    asyncio.run(run())


def test_structured_route_selects_primary_and_supporting_agents():
    orchestrator = AgentOrchestrator(api_key="test", model="test-model", gateway=NoCallGateway())
    request = Request(
        message="登录后发现重复扣款，还要退款和发票，金额99元",
        user_id="user",
        conv_id="conv",
        intent=IntentCategory.TECHNICAL_LOGIN,
        intent_group=IntentCategory.TECHNICAL.value,
        urgency=UrgencyLevel.LOW,
        intent_confidence=0.9,
        entities={"amount": ["99元"]},
    )

    decision = orchestrator._route_decision(request)

    assert decision.primary_agent == AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
    assert decision.agent_types == [AgentType.TECHNICAL, AgentType.BILLING]
    assert decision.multi_agent is True
    assert decision.confidence == 0.75
    assert "group=technical" in decision.reason
    assert "primary=technical" in decision.reason
    assert "supporting=billing" in decision.reason


def test_composite_route_keeps_clear_secondary_domain_when_primary_score_is_high():
    orchestrator = AgentOrchestrator(api_key="test", model="test-model", gateway=NoCallGateway())
    decision = orchestrator._route_decision(Request(
        message="登录一直报错 401，同时银行卡被重复扣款 99 元，请处理技术故障和退款",
        user_id="user",
        conv_id="conv",
        intent=IntentCategory.TECHNICAL_LOGIN,
        intent_group=IntentCategory.TECHNICAL.value,
        urgency=UrgencyLevel.LOW,
        intent_confidence=0.9,
        entities={"amount": ["99 元"], "error_code": ["401"]},
    ))

    assert decision.primary_agent == AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]


def test_low_confidence_other_returns_clarification_without_llm_call():
    async def run():
        orchestrator = AgentOrchestrator(api_key="test", model="test-model", gateway=NoCallGateway())
        request = Request(
            message="这个要怎么办",
            user_id="user",
            conv_id="conv",
            intent=IntentCategory.OTHER,
            intent_group=IntentCategory.OTHER.value,
            urgency=UrgencyLevel.LOW,
            intent_confidence=0.2,
        )

        result = await orchestrator.run(request)

        assert "请补充" in result.response
        assert result.agent_type == AgentType.GENERAL
        assert result.agent_types == [AgentType.GENERAL]
        assert result.primary_agent == AgentType.GENERAL
        assert result.supporting_agents == []
        assert result.routing_confidence == 0.2
        assert result.routing_reason == "低置信度 OTHER 意图，先澄清用户需求"
        assert result.escalated is False

    asyncio.run(run())


def test_human_handoff_reports_executing_agent_and_marks_escalation():
    class ReplyGateway:
        provider = "test"

        async def complete(self, request):
            return SimpleNamespace(text="已记录人工请求", usage=None, metadata={})

    async def run():
        orchestrator = AgentOrchestrator(api_key="test", model="test-model", gateway=ReplyGateway())
        result = await orchestrator.run(Request(
            message="我要转人工客服",
            user_id="user",
            conv_id="conv",
            intent=IntentCategory.HUMAN_HANDOFF,
            intent_group=IntentCategory.ESCALATION.value,
            urgency=UrgencyLevel.HIGH,
            intent_confidence=0.95,
        ))

        assert result.agent_type == AgentType.GENERAL
        assert result.agent_types == [AgentType.GENERAL]
        assert result.primary_agent == AgentType.GENERAL
        assert result.escalated is True
        assert "标记人工升级" in result.routing_reason

    asyncio.run(run())
