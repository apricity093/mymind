import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest

from agents.agent_orchestrator import AgentFeatureConfig
from core.intent_recognizer import IntentCategory
from experiments.intent_routing import RecordedIntentRecognizer, evaluate_cases


ANNOTATIONS = json.loads((Path(__file__).parents[1] / "data/eval/intent_routing_v2.json").read_text(encoding="utf-8"))


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
    case["id"] = "intent-v2-account_security-4-1"
    records = {case["id"]: records["case"]}
    for features in (AgentFeatureConfig(), replace(AgentFeatureConfig(), escalation_enabled=True)):
        result = evaluate(case, records, features)
        assert result["correct"] == 1
        assert result["rows"][0]["urgency"] == "CRITICAL"
        assert result["rows"][0]["escalated"] is True
        assert result["rows"][0]["expected_primary_agent"] == ("escalation" if features.escalation_enabled else "general")
