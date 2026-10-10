"""重放真实分类输出，比较标准意图与预测意图经过生产路由的结果。"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from agents.agent_orchestrator import (
    AgentFeatureConfig, AgentOrchestrator, AgentResponse, Request,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


class NoNetworkGateway:
    provider = "recorded-real-classification"

    async def complete(self, request):
        raise AssertionError("路由统计不允许新增模型调用")


class RecordedIntentRecognizer(IntentRecognizer):
    """仅重放保存的模型输出；融合、实体和紧急度仍由生产识别器计算。"""

    def __init__(self, records, threshold=0.5, embedding_enabled=True):
        super().__init__(
            api_key="routing-replay", model="deepseek-flash",
            confidence_threshold=threshold, gateway=NoNetworkGateway(),
            embedding=SimpleNamespace(
                enabled=True, identity={"model": "qwen3.7-text-embedding-flash", "dimensions": 768},
            ),
        )
        self._embedding_enabled = embedding_enabled
        self.records = {record["case"]["message"]: record for record in records.values()}

    def _record(self, message):
        record = self.records[message]
        if record["status"] != "measured":
            raise ValueError("分类记录未完成真实测量")
        pattern = self._pattern_recognize(message)
        saved_pattern = record["pattern"]
        if (pattern["intent"].value != saved_pattern["intent"]
                or pattern["confidence"] != saved_pattern["confidence"]):
            raise ValueError("关键词规则与原分类运行不一致，不能复用")
        return record

    async def _llm_recognize(self, message, history):
        if history:
            raise ValueError("本批真实分类记录仅覆盖单轮问题")
        value = self._record(message)["llm"]
        return {**value, "intent": IntentCategory(value["intent"])}

    async def _embedding_recognize(self, message):
        value = self._record(message)["embedding"]
        return {**value, "intent": IntentCategory(value["intent"])}


class RoutingCapture(AgentOrchestrator):
    """执行生产路由流程；决策后的回答生成不参与路由正确性计分。"""

    def __init__(self, recognizer, features):
        super().__init__(
            api_key="routing-replay", model="routing-only",
            gateway=NoNetworkGateway(), intent_recognizer=recognizer, features=features,
        )
        self._composer = SimpleNamespace(compose=self._capture_composition)

    async def _execute(self, request, agent_type):
        return AgentResponse(agent_type, "", True)

    async def _capture_composition(self, request, responses):
        return ""


def expected_primary(role, features, annotations):
    if role == "escalation" and not features.escalation_enabled:
        return annotations["escalation_disabled_primary"]
    return role


def routing_summary(rows):
    correct = sum(row["route_match"] for row in rows)
    groups = {}
    for role in sorted({row["expected_primary_agent"] for row in rows}):
        group = [row for row in rows if row["expected_primary_agent"] == role]
        hits = sum(row["route_match"] for row in group)
        groups[role] = {"n": len(group), "correct": hits, "accuracy": hits / len(group)}
    return {
        "n": len(rows), "correct": correct,
        "routing_accuracy": correct / len(rows) if rows else None,
        "by_expected_agent": groups,
        "confusion": dict(Counter(
            f"{row['expected_primary_agent']}->{row['primary_agent']}" for row in rows)),
        "errors": [row["id"] for row in rows if not row["route_match"]],
    }


async def evaluate_cases(cases, records, annotations, features, threshold=0.5,
                         mode="three", saved_predictions=None):
    recognizer = RecordedIntentRecognizer(records, threshold, embedding_enabled=mode != "two")
    orchestrator = RoutingCapture(recognizer, features)
    rows = []
    for case in cases:
        recorded = records.get(case["id"])
        if mode != "gold" and (recorded is None or recorded["case"] != case):
            raise ValueError(f"冻结题目与真实分类记录不一致：{case['id']}")
        request = Request(case["message"], "routing-eval", case["id"])
        if mode == "gold":
            request.intent = IntentCategory(case["intent"])
            request.intent_group = recognizer._intent_group(request.intent)
            request.intent_confidence = 1.0
            request.urgency = recognizer._urgency(request.message, request.intent)
            request.entities = recognizer._extract_entities(request.message)
        result = await orchestrator.run(request)
        if saved_predictions is not None:
            saved = saved_predictions[case["id"]]
            if (request.intent.value != saved["predicted"]
                    or abs(request.intent_confidence - saved["confidence"]) > 1e-9):
                raise ValueError(f"生产融合与已保存预测不一致：{case['id']}")
        override = annotations["case_overrides"].get(case["id"])
        expected_role = (override["primary_agent"] if override
                         else annotations["primary_agent_by_intent"][case["intent"]])
        expected = expected_primary(expected_role, features, annotations)
        rows.append({
            "id": case["id"], "family_id": case["family_id"], "message": case["message"],
            "expected_intent": case["intent"], "predicted_intent": request.intent.value,
            "intent_match": request.intent.value == case["intent"],
            "intent_confidence": request.intent_confidence,
            "urgency": request.urgency.name, "entities": request.entities,
            "expected_primary_agent": expected, "primary_agent": result.primary_agent.value,
            "annotation_reason": override["reason"] if override else "标准意图对应业务角色",
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "escalated": result.escalated, "routing_reason": result.routing_reason,
            "route_match": result.primary_agent.value == expected,
        })
    return {
        **routing_summary(rows),
        "intent_correct": sum(row["intent_match"] for row in rows),
        "intent_accuracy": sum(row["intent_match"] for row in rows) / len(rows) if rows else None,
        "intent_errors_with_correct_route": [row["id"] for row in rows
                                             if not row["intent_match"] and row["route_match"]],
        "rows": rows,
    }


async def evaluate_migration(cases, annotations, features):
    recognizer = RecordedIntentRecognizer({})
    orchestrator = RoutingCapture(recognizer, features)
    rows = []
    for case in cases:
        intent = IntentCategory(case["expected_intent"])
        result = await orchestrator.run(Request(
            case["question"], "routing-eval", case["id"], intent=intent,
            intent_group=recognizer._intent_group(intent), intent_confidence=1.0,
            urgency=UrgencyLevel[case["urgency"]],
            entities=recognizer._extract_entities(case["question"]),
        ))
        expected = expected_primary(case["expected_primary_agent"], features, annotations)
        rows.append({
            "id": case["id"], "category": case["category"], "message": case["question"],
            "expected_primary_agent": expected, "primary_agent": result.primary_agent.value,
            "route_match": result.primary_agent.value == expected,
            "expected_supporting_agents": case["expected_supporting_agents"],
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "support_match": sorted(agent.value for agent in result.supporting_agents)
                             == sorted(case["expected_supporting_agents"]),
            "escalation_match": result.escalated == case["expect_escalation"],
            "routing_reason": result.routing_reason,
        })
    return {
        **routing_summary(rows),
        "supporting_correct": sum(row["support_match"] for row in rows),
        "supporting_accuracy": sum(row["support_match"] for row in rows) / len(rows),
        "escalation_accuracy": sum(row["escalation_match"] for row in rows) / len(rows),
        "rows": rows,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


async def run(run_dir, output, features):
    backend = Path(__file__).resolve().parents[1]
    data_dir = backend / "data/eval"
    data = read_json(data_dir / "intent_optimization_v2.json")
    annotations = read_json(data_dir / "intent_routing_v2.json")
    cases = [case for case in data["cases"] if case["split"] == "test"]
    records = read_json(run_dir / "intent_raw_test.json")
    selection = read_json(run_dir / "selection.json")
    measured = read_json(run_dir / "intent_test.json")
    threshold = selection["intent_threshold"]
    if measured["status"] != "measured" or selection["status"] != "frozen_before_test":
        raise ValueError("正式分类或开发选择未完成")
    results = {"module": await evaluate_cases(
        cases, records, annotations, features, threshold, mode="gold")}
    for mode in ("three", "two"):
        predictions = {row["id"]: row for row in measured["thresholds"][str(threshold)][mode]["rows"]}
        results[f"pipeline_{mode}"] = await evaluate_cases(
            cases, records, annotations, features, threshold, mode, predictions)
    migration = await evaluate_migration(
        read_json(data_dir / "agent_migration_dataset.json"), annotations, features)
    report = {
        "status": "measured", "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "真实历史分类输出重放到生产recognize/run；决策后的回答、工具和Composer不计分",
        "annotation_method": annotations["annotation_method"],
        "annotation_overrides": annotations["case_overrides"],
        "source_run": str(run_dir.resolve()), "split": "test", "family_count": len({case["family_id"] for case in cases}),
        "features": asdict(features), "intent_threshold": threshold,
        "selection_frozen_at": selection["frozen_at"], "api_usage_new": {"llm": 0, "embedding_http": 0, "encoded_texts": 0},
        "paired_test": results, "migration_module": migration,
        "limitations": ["合成数据与规则派生路由标注，非人工审核或真实客服分布",
                        "152题为单域单轮，240题按原单条question输入，未执行其多轮turns",
                        "健康Agent池，未统计监控降权、Agent执行失败和线上时变模型",
                        "复用原真实分类结果；此次没有重新请求模型或测量线上延迟"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    backend = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=backend / "artifacts/qwen-optimization-v2/20261010-qwen-01")
    parser.add_argument("--output", type=Path, default=backend / "artifacts/qwen-optimization-v2/20261010-qwen-01/routing/current_test.json")
    args = parser.parse_args()
    load_dotenv(backend / ".env")
    report = asyncio.run(run(args.run_dir, args.output, AgentFeatureConfig.from_env()))
    print(json.dumps({name: {key: value for key, value in group.items()
                             if key in ("n", "correct", "routing_accuracy", "intent_correct")}
                      for name, group in report["paired_test"].items()}, ensure_ascii=True))
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
