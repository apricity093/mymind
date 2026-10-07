import asyncio

import pytest
from types import SimpleNamespace

from api.main import ChatResponse
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from core.knowledge_policy import KnowledgePolicy
from evaluation.evaluator import EndToEndEvaluator, QualityScores
from mcp.knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool


def test_knowledge_policy_uses_intent_then_business_fallback():
    policy = KnowledgePolicy()
    assert policy.decide("你好", IntentCategory.GREETING).should_search is False
    assert policy.decide("退款多久到账", IntentCategory.REFUND).reason == "intent:refund"
    assert policy.decide("帮我处理退款", IntentCategory.OTHER).should_search is True
    assert policy.decide("讲个笑话", IntentCategory.OTHER).should_search is False


def test_tool_fallback_is_degraded_and_never_enters_rerank():
    async def run():
        async def broken(params, context):
            raise ConnectionError("knowledge unavailable")

        manager = MCPToolManager(api_key="test")
        manager.register(Tool(
            name="knowledge_search",
            description="test",
            handler=broken,
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
            fallback=lambda params, context, error: [{"content": "fallback"}],
        ))

        async def no_rewrite(query, n=3):
            return [query]

        manager.rewrite_query = no_rewrite
        result = await manager.search_with_rewrite("knowledge_search", "退款")
        assert result.success is False
        assert result.degraded is True
        assert result.data == []

    asyncio.run(run())


def test_knowledge_import_is_idempotent_upsert():
    class UpsertCollection:
        def __init__(self):
            self.items = {}
            self.calls = []

        def upsert(self, ids, documents, metadatas):
            self.calls.append((ids, documents, metadatas))
            self.items.update(zip(ids, documents))

        def count(self):
            return len(self.items)

    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge._collection = UpsertCollection()
    documents = [{"title": "退款政策", "content": "七天内可以退款"}]
    assert knowledge.add_documents(documents) == 1
    assert knowledge.add_documents(documents) == 1
    assert knowledge.doc_count == 1
    first, second = knowledge._collection.calls
    assert first[0] == second[0]
    assert first[1] == second[1] == ["七天内可以退款"]


def test_chat_response_keeps_legacy_contract_and_adds_optional_diagnostics():
    response = ChatResponse(
        conv_id="c1", response="ok", intent="refund", agent_type="billing",
        escalated=False, latency_ms=12.0, knowledge_used=True,
    )
    payload = response.model_dump()
    assert payload["conv_id"] == "c1"
    assert payload["agent_type"] == "billing"
    assert payload["intent_group"] == "other"
    assert payload["knowledge_status"] == "skipped"


def test_old_baseline_without_new_metrics_remains_readable():
    report = EndToEndEvaluator._report_from_dict({
        "timestamp": "2026-01-01T00:00:00",
        "total": 1,
        "passed": 1,
        "pass_rate": 1.0,
        "avg_scores": {"relevance": 0.9},
        "results": [],
    })
    assert report.avg_scores == {"relevance": 0.9}
    assert report.regressions == []


@pytest.mark.parametrize("uses_tool", [True, False])
def test_evaluator_records_routing_and_knowledge_expectations(uses_tool):
    class Orchestrator:
        async def recognize_intent(self, message, history=None):
            return IntentResult(
                intent=IntentCategory.REFUND,
                confidence=0.9,
                urgency=UrgencyLevel.LOW,
                intent_group="billing",
                entities={"order_id": []},
                reasoning="refund",
                latency_ms=1.0,
                source_scores={"pattern": 0.75},
            )

        async def run(self, request):
            from agents.agent_orchestrator import AgentType
            return SimpleNamespace(
                response="可以申请退款",
                agent_type=AgentType.BILLING,
                agent_types=[AgentType.BILLING],
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                routing_reason="intent=refund",
                routing_confidence=0.9,
                tools_used=["search_knowledge_base"] if uses_tool else [],
            )

    class Judge:
        async def judge(self, question, response, context=None):
            return QualityScores(1.0, 1.0, 1.0, 1.0)

    async def run():
        evaluator = EndToEndEvaluator(
            orchestrator=Orchestrator(),
            recognizer=SimpleNamespace(),
            api_key="test",
        )
        evaluator._judge = Judge()
        report = await evaluator.run(dialog_cases=[{
            "question": "我要退款",
            "expected_intent": "refund",
            "expected_primary_agent": "billing",
            "expect_knowledge_search": True,
        }])
        assert report.avg_scores["routing_accuracy"] == 1.0
        assert report.avg_scores["knowledge_gate_accuracy"] == float(uses_tool)
        assert report.results[0].metadata["knowledge_search"] is uses_tool
        assert report.results[0].passed is uses_tool
        assert report.results[0].metadata["intent_group"] == "billing"

    asyncio.run(run())
