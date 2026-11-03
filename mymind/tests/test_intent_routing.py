"""意图融合、业务路由与评估隔离。"""

import asyncio
import pytest
import threading
from agents.agent_orchestrator import AgentFeatureConfig, AgentOrchestrator, AgentType, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer, IntentResult, UrgencyLevel
from core.knowledge_policy import KnowledgePolicy
from dataclasses import replace
from evaluation.evaluator import EndToEndEvaluator, QualityScores
from experiments.intent_routing import RecordedIntentRecognizer, evaluate_cases
from tests.fakes import StubEmbedding
from types import SimpleNamespace


ANNOTATIONS = {
    "escalation_disabled_primary": "general",
    "primary_agent_by_intent": {"technical": "technical", "invoice": "billing", "greeting": "general", "escalation": "escalation", "account_security": "billing"},
    "case_overrides": {"urgent-account": {"primary_agent": "escalation", "reason": "凭据泄露，紧急升级"}},
}


def fixture_record(message, gold, predicted, confidence=0.95):
    case = {"id": "case", "family_id": "family", "split": "test", "message": message,
            "intent": gold, "synthetic": True}
    pattern = RecordedIntentRecognizer({})._pattern_recognize(message)
    record = {"case": case, "status": "measured",
              "llm": {"intent": predicted, "confidence": confidence},
              "embedding": {"intent": predicted, "confidence": confidence},
              "pattern": {"intent": pattern["intent"].value, "confidence": pattern["confidence"]}}
    return case, {"case": record}


def evaluate(case, records, features=None, mode="three", saved_predictions=None):
    return asyncio.run(evaluate_cases(
        [case], records, ANNOTATIONS, features or AgentFeatureConfig(),
        mode=mode, saved_predictions=saved_predictions))


def test_pipeline_uses_model_prediction_and_gold_only_for_scoring():
    case, records = fixture_record("数据同步延迟很大", "technical", "invoice")
    module = evaluate(case, records, mode="gold")
    pipeline = evaluate(case, records)
    assert module["correct"] == 1
    assert pipeline["correct"] == 0
    assert pipeline["rows"][0]["predicted_intent"] == "invoice"
    assert pipeline["rows"][0]["primary_agent"] == "billing"


def test_fine_label_error_can_keep_correct_business_agent():
    case, records = fixture_record("需要重新出具税务凭证", "invoice", "billing")
    result = evaluate(case, records)
    assert result["intent_correct"] == 0
    assert result["correct"] == 1
    assert result["intent_errors_with_correct_route"] == ["case"]


def test_actual_run_applies_other_clarification_before_keyword_route():
    case, records = fixture_record("500", "technical", "other", confidence=0)
    result = evaluate(case, records)
    assert result["rows"][0]["predicted_intent"] == "other"
    assert result["rows"][0]["primary_agent"] == "general"
    assert "澄清" in result["rows"][0]["routing_reason"]


def test_escalation_annotation_tracks_enabled_runtime():
    case, records = fixture_record("请安排上级经理接待", "escalation", "escalation")
    normal = evaluate(case, records)
    enabled = evaluate(case, records, replace(AgentFeatureConfig(), escalation_enabled=True))
    assert normal["rows"][0]["primary_agent"] == "general"
    assert enabled["rows"][0]["primary_agent"] == "escalation"
    assert normal["correct"] == enabled["correct"] == 1
    assert normal["rows"][0]["escalated"] is True


def test_replay_rejects_different_saved_predictions():
    case, records = fixture_record("你好", "greeting", "greeting")
    with pytest.raises(ValueError, match="生产融合与已保存预测不一致"):
        evaluate(case, records, saved_predictions={"case": {"predicted": "technical", "confidence": 0.9}})


def test_replay_rejects_changed_keyword_rules():
    case, records = fixture_record("你好", "greeting", "greeting")
    records["case"]["pattern"]["intent"] = IntentCategory.OTHER.value
    with pytest.raises(ValueError, match="关键词规则与原分类运行不一致"):
        evaluate(case, records)


def test_urgent_account_case_uses_annotated_escalation_priority():
    case, records = fixture_record("凭据落到别人手里怎样紧急处理", "account_security", "account_security")
    case["id"] = "urgent-account"
    records = {case["id"]: records["case"]}
    for features in (AgentFeatureConfig(), replace(AgentFeatureConfig(), escalation_enabled=True)):
        result = evaluate(case, records, features)
        assert result["correct"] == 1
        assert result["rows"][0]["urgency"] == "CRITICAL"
        assert result["rows"][0]["escalated"] is True
        assert result["rows"][0]["expected_primary_agent"] == ("escalation" if features.escalation_enabled else "general")


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
        assert result.confidence == pytest.approx(0.585)
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
        orchestrator = AgentOrchestrator(
            api_key="test", model="test-model", gateway=ReplyGateway(),
            features=AgentFeatureConfig.for_variant("E5"),
        )
        result = await orchestrator.run(Request(
            message="我要转人工客服",
            user_id="user",
            conv_id="conv",
            intent=IntentCategory.HUMAN_HANDOFF,
            intent_group=IntentCategory.ESCALATION.value,
            urgency=UrgencyLevel.HIGH,
            intent_confidence=0.95,
        ))

        assert result.agent_type == AgentType.ESCALATION
        assert result.agent_types == [AgentType.ESCALATION]
        assert result.primary_agent == AgentType.ESCALATION
        assert result.escalated is True
        assert "人工升级节点" in result.routing_reason

    asyncio.run(run())


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.BILLING, 0.3),       # weighted score 0.21
        (IntentCategory.TECHNICAL_LOGIN, 0.9),  # weighted score 0.18
        (IntentCategory.TECHNICAL, 0.4),     # weighted score 0.04
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)


def test_knowledge_policy_uses_intent_then_business_fallback():
    policy = KnowledgePolicy()
    assert policy.decide("你好", IntentCategory.GREETING).should_search is False
    assert policy.decide("退款多久到账", IntentCategory.REFUND).reason == "intent:refund"
    assert policy.decide("帮我处理退款", IntentCategory.OTHER).should_search is True
    assert policy.decide("讲个笑话", IntentCategory.OTHER).should_search is False


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




def test_intent_weights_failure_recovery_parallel_and_template_cache():
    async def run():
        embedding = StubEmbedding()
        recognizer = IntentRecognizer("test", base_url="https://chat-only.test", embedding=embedding)
        seen = []

        async def llm(message, history):
            seen.append("llm")
            await asyncio.sleep(0)
            return {"intent": IntentCategory.GREETING, "confidence": 0.8}

        async def emb(message):
            seen.append("embedding")
            await asyncio.sleep(0)
            return {"intent": IntentCategory.GREETING, "confidence": 0.9}

        recognizer._llm_recognize = llm
        recognizer._embedding_recognize = emb
        recognizer._pattern_recognize = lambda _: {"intent": IntentCategory.GREETING, "confidence": 0.5}
        result = await recognizer.recognize("test")
        assert seen == ["llm", "embedding"]
        assert result.confidence == pytest.approx(0.7 * .8 + .2 * .9 + .1 * .5)

        async def failed(message):
            return {"intent": IntentCategory.OTHER, "confidence": 0, "failed": True}

        recognizer._embedding_recognize = failed
        result = await recognizer.recognize("failure")
        assert result.confidence == pytest.approx(.85 * .8 + .15 * .5)
        recognizer._embedding_recognize = emb
        assert (await recognizer.recognize("failure")).confidence == pytest.approx(.79)
        await recognizer._load_template_embeddings()
        calls = len(embedding.calls)
        await recognizer._load_template_embeddings()
        assert len(embedding.calls) == calls
        assert all(call[1] == "CLASSIFICATION" for call in embedding.calls)
        recognizer._embedding_enabled = False
        assert (await recognizer.recognize("no-vector")).confidence == pytest.approx(.755)

    asyncio.run(run())


def test_actual_intent_failure_does_not_call_old_hash_and_recovers(monkeypatch):
    async def run():
        embedding = StubEmbedding()
        recognizer = IntentRecognizer("test", embedding=embedding)
        monkeypatch.setattr(recognizer, "_local_embedding", lambda _: pytest.fail("production used old hash"))
        embedding.fail = True
        assert (await recognizer._embedding_recognize("hello"))["failed"]
        embedding.fail = False
        assert not (await recognizer._embedding_recognize("hello")).get("failed")
        assert embedding.telemetry.snapshot()["counts"]["intent:recovered"] == 1

    asyncio.run(run())


def test_template_cache_restart_invalidation_and_failure_recovery(tmp_path):
    path = str(tmp_path / "templates.json")
    async def run():
        first = StubEmbedding()
        recognizer = IntentRecognizer("test", embedding=first, template_cache_path=path)
        await asyncio.gather(*[recognizer._load_template_embeddings() for _ in range(3)])
        assert len(first.calls) == 1
        second = StubEmbedding()
        restarted = IntentRecognizer("test", embedding=second, template_cache_path=path)
        await restarted._load_template_embeddings()
        assert second.calls == []
        restarted.learn("新增开发表达", IntentCategory.REFUND)
        await restarted._load_template_embeddings()
        assert len(second.calls) == 1
        second.fail = True
        assert (await restarted._embedding_recognize("测试"))["failed"]
        second.fail = False
        assert not (await restarted._embedding_recognize("测试")).get("failed")
        Path = __import__("pathlib").Path
        Path(path).write_text("invalid", encoding="utf-8")
        third = StubEmbedding()
        rebuilt = IntentRecognizer("test", embedding=third, template_cache_path=path)
        await rebuilt._load_template_embeddings()
        assert len(third.calls) == 1
        for instance in [recognizer, restarted, rebuilt]:
            await instance.close()
    asyncio.run(run())


def test_template_preparation_has_bounded_wait_and_shared_task(tmp_path):
    embedding = StubEmbedding()
    release = threading.Event()
    original = embedding.embed
    def blocked(*args, **kwargs):
        assert release.wait(3)
        return original(*args, **kwargs)
    embedding.embed = blocked
    async def run():
        recognizer = IntentRecognizer("test", embedding=embedding, template_cache_path=str(tmp_path / "cache"), template_wait_s=.03)
        results = await asyncio.gather(*[recognizer._embedding_recognize("测试") for _ in range(3)])
        assert all(r["failed"] for r in results)
        release.set()
        await recognizer._template_task
        assert len(embedding.calls) == 1
        assert not (await recognizer._embedding_recognize("测试")).get("failed")
        await recognizer.close()
    asyncio.run(run())


def test_intent_cache_key_isolates_history_model_and_normalizes_message():
    recognizer = IntentRecognizer(api_key="test", model="model-a")
    first = recognizer._cache_key("  可以  ", [{"role": "assistant", "content": "是否退款？"}])
    same = recognizer._cache_key("可以", [{"role": "assistant", "content": "是否退款？"}])
    other_history = recognizer._cache_key("可以", [{"role": "assistant", "content": "是否取消？"}])
    assert first == same
    assert first != other_history

    long_message = "查询" * 600
    assert recognizer._cache_key(long_message + "退款") != recognizer._cache_key(long_message + "发票")
    long_history = [{"role": "assistant", "content": "内容" * 300}]
    assert recognizer._cache_key("可以", long_history) != recognizer._cache_key(
        "可以", [{"role": "assistant", "content": long_history[0]["content"] + "退款"}]
    )
