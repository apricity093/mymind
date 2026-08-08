from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import chromadb
import redis

from core.cache_store import RedisCacheStore
from core.cache_metrics import RedisCacheMetricsCollector
from core.llm_gateway import CacheUsage
from mcp.tool_manager import MCPToolManager, Tool
from memory.conversation_memory import MemoryManager, MsgRole
from experiments.reporting import metadata, write_report


class DeterministicMessages:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=[{"type": "text", "text": f"summary-{self.calls}"}])


async def run_integration(redis_url: str, chroma_host: str, chroma_port: int, output_dir: Path) -> dict:
    parsed = urlparse(redis_url)
    database = int((parsed.path or "/0").lstrip("/") or 0)
    if database == 0:
        raise ValueError("Integration experiments require a non-production Redis DB (for example /15).")

    redis_client = redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    chroma_client.heartbeat()

    run_id = uuid.uuid4().hex[:10]
    episodic_name = f"mymind_experiment_episodic_{run_id}"
    profile_name = f"mymind_experiment_profile_{run_id}"
    cache_prefix = f"mymind:experiment:{run_id}"
    metrics_key = f"{cache_prefix}:metrics"
    llm = SimpleNamespace(messages=DeterministicMessages())
    config = {
        "redis_db": database,
        "chroma_host": chroma_host,
        "chroma_port": chroma_port,
        "concurrency": 20,
        "rounds": 20,
        "worker_adapters": 4,
    }
    failures = []
    try:
        memory = MemoryManager(
            api_key="experiment", redis_client=redis_client, chroma_client=chroma_client,
            llm_client=llm, episodic_collection=episodic_name, profile_collection=profile_name,
        )
        for round_index in range(config["rounds"]):
            await asyncio.gather(*[
                memory.add_message(
                    "experiment-user", "experiment-conversation", MsgRole.USER,
                    f"round-{round_index:02d}-message-{message_index:02d}",
                )
                for message_index in range(config["concurrency"])
            ])
        recent = await memory._get_working_memory("experiment-user", "experiment-conversation")
        episodic_count = chroma_client.get_collection(episodic_name).count()
        represented_messages = episodic_count * 10 + len(recent)
        expected_messages = config["rounds"] * config["concurrency"]
        if represented_messages != expected_messages:
            failures.append(f"message_accounting:{represented_messages}!={expected_messages}")

        stores = [RedisCacheStore(redis_client, cache_prefix) for _ in range(4)]
        metric_collectors = [RedisCacheMetricsCollector(redis_client, metrics_key) for _ in range(4)]
        for collector in metric_collectors:
            collector.record_provider(
                "deepseek", "experiment",
                CacheUsage("deepseek", 100, 70, 0, 30, True, "hit"),
            )
        shared_metrics = metric_collectors[0].snapshot()["counters"]
        if shared_metrics.get("provider.deepseek.experiment.requests") != 4:
            failures.append("cross_worker_metric_aggregation")
        stores[0].set("knowledge", "shared", {"value": 1}, 300)
        cross_worker_visible = all(store.get("knowledge", "shared") == {"value": 1} for store in stores)
        stores[-1].invalidate_namespace("knowledge")
        cross_worker_invalidated = all(store.get("knowledge", "shared") is None for store in stores)
        if not cross_worker_visible:
            failures.append("cross_worker_cache_visibility")
        if not cross_worker_invalidated:
            failures.append("cross_worker_cache_invalidation")

        handler_calls = 0

        async def handler(params, context):
            nonlocal handler_calls
            handler_calls += 1
            await asyncio.sleep(0.02)
            return [{"title": "result", "content": params["query"]}]

        managers = []
        for store in stores:
            manager = MCPToolManager(api_key="experiment", cache_store=store)
            manager.register(Tool(
                name="knowledge_search", description="integration", handler=handler,
                schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
                cache_ttl=300,
            ))
            managers.append(manager)
        results = await asyncio.gather(*[
            managers[index % len(managers)].call("knowledge_search", {"query": "refund"})
            for index in range(50)
        ])
        if handler_calls > len(managers):
            failures.append(f"expensive_call_count:{handler_calls}")
        if any(not result.success for result in results):
            failures.append("concurrent_cache_result_failure")

        variants = {
            "B0": {"cache_scope": "process", "memory_lock": False},
            "C1": {
                "expected_messages": expected_messages,
                "represented_messages": represented_messages,
                "episodic_segments": episodic_count,
                "recent_messages": len(recent),
                "cross_worker_cache_visible": cross_worker_visible,
                "cross_worker_invalidation": cross_worker_invalidated,
                "fifty_request_handler_calls": handler_calls,
                "cross_worker_provider_requests": shared_metrics.get("provider.deepseek.experiment.requests", 0),
            },
            "C2": {"retrieval_collection": episodic_name, "context_budget_chars": 8000},
            "C3": {"status": "not_run_in_integration_layer"},
        }
    finally:
        for name in (episodic_name, profile_name):
            try:
                chroma_client.delete_collection(name)
            except Exception:
                pass
        for key in redis_client.scan_iter(match=f"{cache_prefix}:*"):
            redis_client.delete(key)
        for key in redis_client.scan_iter(match="*:experiment-user:experiment-conversation"):
            redis_client.delete(key)
        redis_client.delete(
            "profile:fingerprint:experiment-user",
            "profile:updated_at:experiment-user",
            "lock:memory:experiment-user:experiment-conversation",
            metrics_key,
        )

    report = {
        "title": "Python Cache and Memory Docker Integration Experiment",
        "artifact_type": "cache-memory-integration-v1",
        "metadata": metadata(config, "fake-deterministic"),
        "config": config,
        "variants": variants,
        "failures": failures,
        "overall_passed": not failures,
    }
    report["artifacts"] = write_report(report, output_dir, "cache-memory-integration")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://:mymind123@localhost:6379/15")
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    args = parser.parse_args()
    report = asyncio.run(run_integration(args.redis_url, args.chroma_host, args.chroma_port, args.output_dir))
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
