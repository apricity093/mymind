"""Reproducible E0-E5 Agent/tool migration ablation experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agents.agent_orchestrator import AgentFeatureConfig, AgentOrchestrator, Request
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.llm_gateway import CacheUsage, LLMResult, ToolCall
from experiments.reporting import metadata, write_report


ALL_VARIANTS = ("E0", "E1", "E2", "E3", "E4", "E5")


class PolicyGateway:
    """Deterministic tool policy: validates the runtime without claiming model quality."""

    provider = "fake-policy"
    tool_protocol = "openai"

    async def complete(self, request):
        if "Response Composer" in request.stable_prompt:
            return self._result("已合并技术排障与账单核验建议，并保留人工审核边界。")
        already_used = any(message.get("role") == "tool" for message in request.messages)
        available = {item["name"] for item in (request.tools or [])}
        message = " ".join(str(item.get("content", "")) for item in request.messages if item.get("role") == "user")
        if available and not already_used:
            selected = self._select_tool(available, message)
            if selected is not None:
                name, arguments = selected
                call = ToolCall(f"offline-{name}", name, arguments)
                return LLMResult(
                    "", CacheUsage(self.provider, input_tokens=20), {}, [call],
                    {"role": "assistant", "content": "", "tool_calls": []},
                )
        return self._result("已根据当前信息给出可验证的处理建议；需要后台权限的操作应由人工核验。")

    @staticmethod
    def _select_tool(available: set[str], message: str):
        code = re.search(r"\b(401|403|404|500)\b", message)
        if "lookup_error_code" in available and code:
            return "lookup_error_code", {"error_code": code.group(1)}
        if "compare_amounts" in available and ("相差" in message or "不一致" in message):
            numbers = [float(item) for item in re.findall(r"(?<![A-Za-z])\b\d+(?:\.\d+)?\b", message)]
            amounts = [item for item in numbers if item not in {401, 403, 404, 500}]
            if len(amounts) >= 2:
                return "compare_amounts", {"amount_a": amounts[-2], "amount_b": amounts[-1]}
        if "build_diagnostic_plan" in available and any(term in message for term in ("崩溃", "超时", "无法登录", "重启")):
            return "build_diagnostic_plan", {"environment": "评测环境", "reproduced": "稳定" in message}
        if "check_billing_fields" in available and any(term in message for term in ("扣款", "扣了", "退款", "账单", "支付", "发票", "订阅")):
            return "check_billing_fields", {"payment_channel": "未提供"}
        if "suggest_required_fields" in available and any(term in message for term in ("物流", "取消", "提供什么信息")):
            return "suggest_required_fields", {}
        if "inspect_request_context" in available and "不确定" in message:
            return "inspect_request_context", {"focus": "triage"}
        return None

    def _result(self, text: str):
        return LLMResult(text, CacheUsage(self.provider, input_tokens=20), {})


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 240:
        raise ValueError("Agent migration dataset must contain exactly 240 rows")
    required = {
        "id", "category", "question", "expected_intent", "expected_primary_agent",
        "expected_supporting_agents", "required_tools", "allowed_tools", "forbidden_tools",
        "expect_escalation", "expect_knowledge_search", "forbidden_claims",
    }
    ids = set()
    for row in data:
        missing = required - set(row)
        if missing:
            raise ValueError(f"{row.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        if row["id"] in ids:
            raise ValueError(f"duplicate id: {row['id']}")
        ids.add(row["id"])
    return data


def _entities(question: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    codes = re.findall(r"\b(?:401|403|404|500)\b", question)
    if codes:
        result["error_code"] = codes
    if any(term in question for term in ("扣款", "退款", "金额", "账单", "支付")):
        result["amount"] = re.findall(r"\b\d+(?:\.\d+)?\b", question) or ["未提供"]
    return result


async def run_variant(name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gateway = PolicyGateway()
    orchestrator = AgentOrchestrator(
        api_key="offline", model="fake-policy", gateway=gateway,
        features=AgentFeatureConfig.for_variant(name),
    )
    route_matches = 0
    support_matches = 0
    escalation_matches = 0
    required_total = 0
    required_hits = 0
    executed_total = 0
    allowed_executed = 0
    unauthorized_executed = 0
    forbidden_claims = 0
    latencies: List[float] = []
    details: List[Dict[str, Any]] = []
    for row in rows:
        urgency = UrgencyLevel[row.get("urgency", "MEDIUM")]
        request = Request(
            message=row["question"], user_id="eval", conv_id=f"eval-{row['id']}",
            intent=IntentCategory(row["expected_intent"]),
            intent_group=row["category"], urgency=urgency, intent_confidence=0.95,
            entities=_entities(row["question"]), request_id=f"{name}-{row['id']}",
        )
        started = time.perf_counter()
        result = await orchestrator.run(request)
        latencies.append((time.perf_counter() - started) * 1000)
        actual_primary = (result.primary_agent or result.agent_type).value
        actual_support = sorted(item.value for item in result.supporting_agents)
        expected_support = sorted(row["expected_supporting_agents"])
        route_match = actual_primary == row["expected_primary_agent"]
        support_match = actual_support == expected_support
        escalation_match = bool(result.escalated) == bool(row["expect_escalation"])
        route_matches += route_match
        support_matches += support_match
        escalation_matches += escalation_match
        required = set(row["required_tools"])
        executed = set(result.tools_used)
        allowed = set(row["allowed_tools"])
        forbidden = set(row["forbidden_tools"])
        required_total += len(required)
        required_hits += len(required & executed)
        executed_total += len(executed)
        allowed_executed += len(executed & allowed)
        unauthorized_executed += len(executed & forbidden)
        forbidden_claims += sum(claim in result.response for claim in row["forbidden_claims"])
        details.append({
            "id": row["id"], "route_match": route_match, "support_match": support_match,
            "escalation_match": escalation_match, "expected_tools": sorted(required),
            "executed_tools": sorted(executed), "unauthorized": sorted(executed & forbidden),
        })
    count = len(rows)
    recall = required_hits / required_total if required_total else 1.0
    precision = allowed_executed / executed_total if executed_total else (1.0 if required_total == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "name": name,
        "cases": count,
        "routing_accuracy": round(route_matches / count, 4),
        "supporting_accuracy": round(support_matches / count, 4),
        "escalation_accuracy": round(escalation_matches / count, 4),
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "tool_f1": round(f1, 4),
        "unauthorized_tool_executions": unauthorized_executed,
        "forbidden_claims": forbidden_claims,
        "p50_latency_ms": round(percentile(latencies, 0.5), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "details": details,
    }


def run_agent_migration(
    output_dir: Path,
    dataset_path: Path,
    variants: Iterable[str] = ALL_VARIANTS,
) -> Dict[str, Any]:
    rows = load_dataset(dataset_path)
    selected = [item.upper() for item in (variants or ALL_VARIANTS)]
    variant_reports = {name: asyncio.run(run_variant(name, rows)) for name in selected}
    candidate = variant_reports.get("E5") or variant_reports[selected[-1]]
    checks = {
        "dataset_has_240_cases": len(rows) == 240,
        "multi_turn_gte_25_percent": sum(bool(row.get("turns")) for row in rows) >= 60,
        "adversarial_gte_20_percent": sum(bool(row.get("adversarial")) for row in rows) >= 48,
        "routing_accuracy_gte_90": candidate["routing_accuracy"] >= 0.90,
        "supporting_accuracy_gte_90": candidate["supporting_accuracy"] >= 0.90,
        "escalation_accuracy_gte_95": candidate["escalation_accuracy"] >= 0.95,
        "tool_f1_gte_90": candidate["tool_f1"] >= 0.90,
        "unauthorized_tool_execution_zero": candidate["unauthorized_tool_executions"] == 0,
        "forbidden_claims_zero": candidate["forbidden_claims"] == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    config = {"dataset": str(dataset_path), "variants": selected, "cases": len(rows), "model": "fake-policy"}
    report = {
        "title": "Python Agent Tool Migration E0-E5 Experiment",
        "artifact_type": "agent-migration-offline-v1",
        "metadata": metadata(config, "fake-policy"),
        "config": config,
        "variants": variant_reports,
        "acceptance": {"checks": checks, "passed": not failures},
        "failures": failures,
        "overall_passed": not failures,
    }
    report["artifacts"] = write_report(report, output_dir, "agent-migration-offline")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/agent_migration_dataset.json"))
    parser.add_argument("--variants", nargs="*", default=list(ALL_VARIANTS))
    args = parser.parse_args()
    report = run_agent_migration(args.output_dir, args.dataset, args.variants)
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
