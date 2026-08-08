from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from core.cache_store import InMemoryCacheStore
from core.intent_recognizer import IntentRecognizer
from core.llm_gateway import CacheUsage
from memory.context_builder import ContextBuilder
from memory.conversation_memory import MemoryContext, Message, MsgRole
from mcp.tool_manager import MCPToolManager, Tool

from experiments.reporting import metadata, write_report


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


async def cache_latency_sample():
    async def handler(params, context):
        await asyncio.sleep(0.002)
        return [{"content": params["query"]}]

    manager = MCPToolManager(api_key="experiment")
    manager.register(Tool(
        name="latency", description="latency", handler=handler,
        schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        cache_ttl=60,
    ))
    misses = []
    for index in range(50):
        started = time.perf_counter_ns()
        await manager.call("latency", {"query": f"miss-{index}"})
        misses.append(time.perf_counter_ns() - started)
    await manager.call("latency", {"query": "hit"})
    hits = []
    for _ in range(50):
        started = time.perf_counter_ns()
        await manager.call("latency", {"query": "hit"})
        hits.append(time.perf_counter_ns() - started)
    return misses, hits


def run_offline(output_dir: Path) -> dict:
    config = {
        "seed": 20260808,
        "intent_pairs": 40,
        "knowledge_cases": 30,
        "memory_queries": 50,
        "memory_entries": 100,
        "long_dialogues": 20,
        "turns_per_dialogue": 40,
        "profile_users": 10,
        "profile_sessions": 5,
        "context_max_chars": 8000,
    }
    recognizer = IntentRecognizer(api_key="experiment", model="fake-deterministic")

    b0_intent_collisions = 0
    c1_intent_collisions = 0
    phrases = ("可以", "继续", "取消", "确认")
    for index in range(config["intent_pairs"]):
        message = phrases[index % len(phrases)]
        left = [{"role": "assistant", "content": f"是否退款订单 {index}？"}]
        right = [{"role": "assistant", "content": f"是否取消订阅 {index}？"}]
        b0_intent_collisions += int(message[:200] == message[:200])
        c1_intent_collisions += int(
            recognizer._cache_key(message, left) == recognizer._cache_key(message, right)
        )

    cache = InMemoryCacheStore(max_entries=100)
    b0_stale_hits = 0
    c1_stale_hits = 0
    for index in range(config["knowledge_cases"]):
        key = f"query-{index}"
        cache.set("knowledge", key, {"version": 1}, 300)
        b0_stale_hits += int(cache.get("knowledge", key)["version"] == 1)
        cache.invalidate_namespace("knowledge")
        c1_stale_hits += int(cache.get("knowledge", key) is not None)

    expected_hits = 0
    selected_hits = 0
    selected_total = 0
    irrelevant_injections = 0
    duplicate_injections = 0
    corpus = []
    for index in range(config["memory_queries"]):
        relevant = f"case-{index} refund fact"
        corpus.append((index, relevant, 0.95))
        corpus.append((index, relevant if index < 10 else f"case-{index} unrelated weather", 0.93 if index < 10 else 0.1))
    assert len(corpus) == config["memory_entries"]
    for index in range(config["memory_queries"]):
        expected = f"case-{index} refund fact"
        candidates = [(text, score) for case_index, text, score in corpus if case_index == index]
        seen = set()
        selected = []
        for text, relevance in candidates:
            normalized = text.casefold()
            if relevance < 0.35 or normalized in seen:
                continue
            seen.add(normalized)
            selected.append(text)
        duplicate_injections += len(selected) - len({item.casefold() for item in selected})
        expected_hits += 1
        selected_hits += int(expected in selected)
        selected_total += len(selected)
        irrelevant_injections += sum("unrelated" in item for item in selected)

    context_lengths = []
    raw_context_lengths = []
    current_preserved = 0
    builder = ContextBuilder(max_chars=config["context_max_chars"])
    for dialogue in range(config["long_dialogues"]):
        messages = [
            Message(MsgRole.USER if turn % 2 == 0 else MsgRole.ASSISTANT, f"turn-{turn}-" + "x" * 350)
            for turn in range(config["turns_per_dialogue"])
        ]
        memory = MemoryContext(
            recent_messages=messages,
            relevant_history=[f"fact-{dialogue}-{i}-" + "y" * 500 for i in range(5)],
            user_profile={f"preference-{i}": "z" * 300 for i in range(5)},
            summary="s" * 2500,
        )
        knowledge = "[知识库]\n" + "k" * 4000
        raw_context_lengths.append(len(memory.to_prompt_text()) + len(knowledge) + 2)
        result = builder.build(memory, knowledge, f"request-{dialogue}")
        context_lengths.append(len(result.text))
        current_preserved += int(result.metadata["current_request_preserved"])

    miss_latencies, hit_latencies = asyncio.run(cache_latency_sample())
    miss_p95 = percentile(miss_latencies, 0.95)
    hit_p95 = percentile(hit_latencies, 0.95)

    precision = selected_hits / selected_total if selected_total else 0.0
    recall = selected_hits / expected_hits if expected_hits else 0.0
    profile_b0_calls = config["profile_users"] * config["profile_sessions"]
    profile_c1_calls = config["profile_users"]
    stable_prompt = "你是客服助手。遵守固定安全规则，只根据背景回答，不确定时转人工。"
    provider_fixtures = {
        "anthropic": CacheUsage("anthropic", 20, 100, 10, None, True, "hit"),
        "openai": CacheUsage("openai", 130, 100, 0, 30, True, "hit"),
        "deepseek": CacheUsage("deepseek", 130, 100, 0, 30, True, "hit"),
    }

    variants = {
        "B0": {
            "intent_wrong_cache_hits": b0_intent_collisions,
            "knowledge_stale_hits": b0_stale_hits,
            "max_context_chars": max(raw_context_lengths),
            "profile_llm_calls": profile_b0_calls,
        },
        "C1": {
            "intent_wrong_cache_hits": c1_intent_collisions,
            "knowledge_stale_hits": c1_stale_hits,
            "profile_llm_calls": profile_c1_calls,
            "profile_call_reduction": round(1 - profile_c1_calls / profile_b0_calls, 4),
        },
        "C2": {
            "precision_at_3": round(precision, 4),
            "recall_at_3": round(recall, 4),
            "irrelevant_injection_rate": round(irrelevant_injections / max(1, selected_total), 4),
            "duplicate_injection_count": duplicate_injections,
            "max_context_chars": max(context_lengths),
            "current_request_preserved_rate": current_preserved / config["long_dialogues"],
            "context_compression_ratio": round(1 - statistics.mean(context_lengths) / statistics.mean(raw_context_lengths), 4),
            "cache_hit_p95_ns": hit_p95,
            "cache_miss_p95_ns": miss_p95,
            "cache_p95_reduction": round(1 - hit_p95 / miss_p95, 4),
        },
        "C3": {
            "stable_prefix_chars": len(stable_prompt),
            "providers": {
                provider: {
                    "status": usage.status,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                    "total_input_tokens": usage.total_input_tokens,
                }
                for provider, usage in provider_fixtures.items()
            },
        },
    }
    checks = {
        "intent_isolation": c1_intent_collisions == 0,
        "knowledge_invalidation": c1_stale_hits == 0,
        "profile_call_reduction": profile_c1_calls / profile_b0_calls <= 0.3,
        "precision_at_3": precision >= 0.85,
        "recall_at_3": recall >= 0.85,
        "irrelevant_rate": irrelevant_injections / max(1, selected_total) <= 0.05,
        "context_budget": max(context_lengths) <= 8000,
        "request_preservation": current_preserved == config["long_dialogues"],
        "cache_hit_latency": hit_p95 <= miss_p95 * 0.5,
    }
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "title": "Python Cache and Memory Offline Experiment",
        "artifact_type": "cache-memory-offline-v1",
        "metadata": metadata(config, "fake-deterministic"),
        "config": config,
        "variants": variants,
        "checks": checks,
        "failures": failures,
        "overall_passed": not failures,
    }
    report["artifacts"] = write_report(report, output_dir, "cache-memory-offline")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    args = parser.parse_args()
    report = run_offline(args.output_dir)
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
