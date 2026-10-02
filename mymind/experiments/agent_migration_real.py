"""Opt-in DeepSeek E0/E5 paired migration experiment with a hard call budget."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from agents.agent_orchestrator import AgentFeatureConfig, AgentOrchestrator, Request
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.llm_gateway import LLMRequest, build_gateway
from experiments.agent_migration import load_dataset, percentile
from experiments.reporting import metadata, write_report


def paired_bootstrap(values_a: Sequence[float], values_b: Sequence[float], samples: int = 2000) -> Dict[str, float]:
    if len(values_a) != len(values_b) or not values_a:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    deltas = [b - a for a, b in zip(values_a, values_b)]
    rng = random.Random(20260831)
    means = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(statistics.mean(draw))
    means.sort()
    return {
        "delta": round(statistics.mean(deltas), 4),
        "ci_low": round(means[int(samples * 0.025)], 4),
        "ci_high": round(means[min(samples - 1, int(samples * 0.975))], 4),
    }


class BudgetedGateway:
    """Count every model call and prohibit hidden incomplete-response retries."""

    def __init__(self, delegate: Any, max_calls: int = 600):
        self.delegate = delegate
        self.max_calls = max(1, int(max_calls))
        self.calls = 0
        self.provider = getattr(delegate, "provider", "unknown")
        self.tool_protocol = getattr(delegate, "tool_protocol", "unknown")

    async def complete(self, request: LLMRequest):
        if self.calls >= self.max_calls:
            raise RuntimeError(f"model_call_budget_exhausted:{self.max_calls}")
        self.calls += 1
        request.max_tokens = max(2048, request.max_tokens)
        request.retry_incomplete = False
        return await self.delegate.complete(request)


def _settings(provider: str) -> Dict[str, str]:
    load_dotenv()
    api_key = (
        os.getenv("LLM_API_KEY", "")
        or os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        raise RuntimeError("未设置 LLM_API_KEY、DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY")
    model = os.getenv("LLM_MODEL", "") or os.getenv("ANTHROPIC_MODEL", "deepseek-chat")
    base_url = os.getenv("LLM_BASE_URL", "") or os.getenv("ANTHROPIC_BASE_URL", "")
    return {"provider": provider, "api_key": api_key, "model": model, "base_url": base_url}


def _sample(rows: Sequence[Dict[str, Any]], per_category: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    categories = sorted({row["category"] for row in rows})
    for category in categories:
        selected.extend([row for row in rows if row["category"] == category][:per_category])
    return selected


def _overall_score(data: Dict[str, Any]) -> float:
    fields = ("relevance", "accuracy", "completeness", "helpfulness")
    values = [max(0.0, min(float(data.get(field, 0.5)), 1.0)) for field in fields]
    return statistics.mean(values)


def _token_summary(snapshot: Dict[str, Any], top_level_requests: int) -> Dict[str, Any]:
    counters = snapshot.get("counters", {})
    input_tokens = sum(int(value) for key, value in counters.items() if key.endswith(".input_tokens"))
    output_tokens = sum(int(value) for key, value in counters.items() if key.endswith(".output_tokens"))
    total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "average_tokens_per_top_level_request": round(total_tokens / max(1, top_level_requests), 3),
    }


async def _paired_judge(
    gateway: BudgetedGateway,
    model: str,
    question: str,
    e0_response: str,
    e5_response: str,
) -> tuple[float, float, bool]:
    prompt = f"""用户问题：{question}

回答 E0：
{e0_response}

回答 E5：
{e5_response}

请独立评价两份回答的 relevance、accuracy、completeness、helpfulness，范围均为 0.0-1.0。
不得因为回答标签或篇幅给予偏好。只返回 JSON：
{{"e0":{{"relevance":0.0,"accuracy":0.0,"completeness":0.0,"helpfulness":0.0}},
 "e5":{{"relevance":0.0,"accuracy":0.0,"completeness":0.0,"helpfulness":0.0}}}}"""
    try:
        result = await gateway.complete(LLMRequest(
            model=model,
            stable_prompt="你是严格、中立的客服回答配对评审，只返回要求的 JSON。",
            messages=[{"role": "user", "content": prompt}],
            cache_identity=f"evaluation:{model}:paired-rubric-v1",
            max_tokens=512,
            temperature=0.0,
            retry_incomplete=False,
        ))
        start, end = result.text.find("{"), result.text.rfind("}") + 1
        payload = json.loads(result.text[start:end])
        return _overall_score(payload["e0"]), _overall_score(payload["e5"]), False
    except Exception:
        return 0.5, 0.5, True


async def _run(
    rows: Sequence[Dict[str, Any]],
    settings: Dict[str, str],
    repeat: int,
    max_model_calls: int,
) -> Dict[str, Any]:
    raw_gateway = build_gateway(
        settings["provider"], settings["api_key"], settings["model"],
        settings["base_url"] or None, cache_enabled=True,
    )
    gateway = BudgetedGateway(raw_gateway, max_model_calls)
    variant_rows: Dict[str, List[Dict[str, Any]]] = {"E0": [], "E5": []}
    first_responses: Dict[str, Dict[str, str]] = {"E0": {}, "E5": {}}
    variant_metrics: Dict[str, Dict[str, Any]] = {}
    variant_model_calls: Dict[str, int] = {}

    for variant in ("E0", "E5"):
        calls_before = gateway.calls
        orchestrator = AgentOrchestrator(
            api_key=settings["api_key"], base_url=settings["base_url"] or None,
            model=settings["model"], provider=settings["provider"], gateway=gateway,
            features=AgentFeatureConfig.for_variant(variant),
        )
        for repeat_index in range(repeat):
            for row_index, row in enumerate(rows, start=1):
                started = time.perf_counter()
                result = await orchestrator.run(Request(
                    message=row["question"], user_id="real-eval", conv_id=f"{variant}-{repeat_index}-{row['id']}",
                    intent=IntentCategory(row["expected_intent"]), intent_group=row["category"],
                    urgency=UrgencyLevel[row.get("urgency", "MEDIUM")], intent_confidence=0.95,
                ))
                elapsed = (time.perf_counter() - started) * 1000
                if repeat_index == 0:
                    first_responses[variant][row["id"]] = result.response
                variant_rows[variant].append({
                    "id": row["id"], "repeat": repeat_index, "latency_ms": elapsed,
                    "primary_agent": (result.primary_agent or result.agent_type).value,
                    "tools_used": result.tools_used, "escalated": result.escalated,
                    "quality": None, "judge_failed": None,
                    "forbidden_claim": any(claim in result.response for claim in row["forbidden_claims"]),
                })
                processed = repeat_index * len(rows) + row_index
                if processed % 20 == 0:
                    print(f"[{variant}] top-level {processed}/{repeat * len(rows)}, model calls={gateway.calls}", flush=True)
        variant_metrics[variant] = orchestrator.metrics.snapshot()
        variant_model_calls[variant] = gateway.calls - calls_before

    judge_scores: Dict[str, List[float]] = {"E0": [], "E5": []}
    judge_failures = 0
    first_by_key = {
        (variant, item["id"]): item
        for variant, items in variant_rows.items()
        for item in items if item["repeat"] == 0
    }
    for judge_index, row in enumerate(rows, start=1):
        e0, e5, failed = await _paired_judge(
            gateway, settings["model"], row["question"],
            first_responses["E0"].get(row["id"], ""), first_responses["E5"].get(row["id"], ""),
        )
        judge_scores["E0"].append(e0)
        judge_scores["E5"].append(e5)
        judge_failures += int(failed)
        for variant, score in (("E0", e0), ("E5", e5)):
            first_by_key[(variant, row["id"])]["quality"] = score
            first_by_key[(variant, row["id"])]["judge_failed"] = failed
        if judge_index % 20 == 0:
            print(f"[judge] paired {judge_index}/{len(rows)}, model calls={gateway.calls}", flush=True)

    summaries: Dict[str, Dict[str, Any]] = {}
    for variant in ("E0", "E5"):
        items = variant_rows[variant]
        latencies = [item["latency_ms"] for item in items]
        summaries[variant] = {
            "requests": len(items),
            "judge_requests": len(rows),
            "quality_mean": round(statistics.mean(judge_scores[variant]), 4),
            "p50_latency_ms": round(percentile(latencies, 0.5), 3),
            "p95_latency_ms": round(percentile(latencies, 0.95), 3),
            "tool_request_rate": round(sum(bool(item["tools_used"]) for item in items) / len(items), 4),
            "forbidden_claim_rate": round(sum(item["forbidden_claim"] for item in items) / len(items), 4),
            "judge_failures": judge_failures,
            "model_calls": variant_model_calls[variant],
            "token_usage": _token_summary(variant_metrics[variant], len(items)),
            "cache_metrics": variant_metrics[variant],
        }

    comparison = paired_bootstrap(judge_scores["E0"], judge_scores["E5"])
    e0, e5 = summaries["E0"], summaries["E5"]
    e0_tokens = e0["token_usage"]["average_tokens_per_top_level_request"]
    e5_tokens = e5["token_usage"]["average_tokens_per_top_level_request"]
    checks = {
        "quality_gain_gte_5pp": comparison["delta"] >= 0.05,
        "quality_ci_above_zero": comparison["ci_low"] > 0,
        "p95_latency_within_1_8x": e5["p95_latency_ms"] <= e0["p95_latency_ms"] * 1.8,
        "forbidden_claim_rate_lte_2_percent": e5["forbidden_claim_rate"] <= 0.02,
        "judge_failures_zero": judge_failures == 0,
        "model_calls_within_budget": gateway.calls <= max_model_calls,
        "average_token_cost_proxy_within_1_6x": e5_tokens <= e0_tokens * 1.6,
    }
    return {
        "variants": summaries,
        "rows": variant_rows,
        "comparison": comparison,
        "model_calls": gateway.calls,
        "model_call_budget": max_model_calls,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def run_real_agent_migration(
    output_dir: Path,
    dataset_path: Path,
    confirm_cost: bool,
    provider: str = "deepseek",
    repeat: int = 3,
    per_category: int = 10,
    max_model_calls: int = 600,
) -> Dict[str, Any]:
    if not confirm_cost:
        raise RuntimeError("真实模型实验会产生费用；必须显式传入 --confirm-cost")
    if repeat != 3 or per_category != 10:
        raise ValueError("正式迁移实验固定为六类各 10 条、每个变体重复 3 次")
    if not 300 <= max_model_calls <= 600:
        raise ValueError("正式迁移实验模型调用预算必须在 300 到 600 次之间")
    settings = _settings(provider)
    rows = _sample(load_dataset(dataset_path), per_category)
    outcome = asyncio.run(_run(rows, settings, repeat, max_model_calls))
    config = {
        "provider": provider, "model": settings["model"], "base_url": settings["base_url"] or "official",
        "cases": len(rows), "repeat": repeat, "variants": ["E0", "E5"],
        "top_level_requests": len(rows) * repeat * 2,
        "judge_policy": "one paired judge call per first-repeat case",
        "max_model_calls": max_model_calls,
    }
    report = {
        "title": "Python Agent Migration DeepSeek Paired Experiment",
        "artifact_type": "agent-migration-real-v1",
        "metadata": metadata(config, settings["model"]),
        "config": config,
        "variants": outcome["variants"],
        "comparison": outcome["comparison"],
        "model_calls": outcome["model_calls"],
        "model_call_budget": outcome["model_call_budget"],
        "rows": outcome["rows"],
        "acceptance": {"checks": outcome["checks"], "passed": not outcome["failures"]},
        "failures": outcome["failures"],
        "overall_passed": not outcome["failures"],
    }
    report["artifacts"] = write_report(report, output_dir, "agent-migration-real")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/agent_migration_dataset.json"))
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--provider", choices=("deepseek", "openai", "anthropic"), default="deepseek")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--max-model-calls", type=int, default=600)
    args = parser.parse_args()
    report = run_real_agent_migration(
        args.output_dir, args.dataset, args.confirm_cost, args.provider,
        args.repeat, args.per_category, args.max_model_calls,
    )
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
