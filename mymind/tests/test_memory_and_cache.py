"""多源缓存计量、会话记忆与压缩。"""

import asyncio
import json
import memory.conversation_memory as memory_module
import pytest
from agents.agent_orchestrator import GeneralAgent, Request
from core.cache_metrics import CacheMetricsCollector
from core.cache_store import InMemoryCacheStore
from core.llm_gateway import (
    AnthropicGateway,
    CacheUsage,
    DeepSeekAnthropicGateway,
    LLMRequest,
    LLMResult,
    OpenAIGateway,
)
from core.prompt_cache import PromptCachePolicy
from memory.context_builder import ContextBuilder
from memory.conversation_memory import MemoryContext, MemoryManager, Message, MsgRole
from monitor.performance_monitor import PerformanceMonitor
from tests.fakes import FakeChroma, FakeRedis, fake_llm
from types import SimpleNamespace


def test_cache_store_ttl_lru_and_namespace_invalidation():
    now = [10.0]
    cache = InMemoryCacheStore(max_entries=2, clock=lambda: now[0])
    cache.set("knowledge", "a", 1, 5)
    cache.set("knowledge", "b", 2, 5)
    assert cache.get("knowledge", "a") == 1
    cache.set("knowledge", "c", 3, 5)
    assert cache.get("knowledge", "b") is None
    now[0] = 16.0
    assert cache.get("knowledge", "a") is None
    cache.set("knowledge", "d", 4, 5)
    cache.invalidate_namespace("knowledge")
    assert cache.get("knowledge", "d") is None


def test_context_builder_enforces_budget_and_keeps_current_request_external():
    memory = MemoryContext(
        recent_messages=[Message(MsgRole.USER, "recent-" + "x" * 900) for _ in range(10)],
        relevant_history=["history-" + "y" * 700 for _ in range(5)],
        user_profile={f"key-{i}": "z" * 400 for i in range(5)},
        summary="s" * 2500,
    )
    result = ContextBuilder(max_chars=8000).build(memory, "[知识库]\n" + "k" * 4000, "当前请求")
    assert len(result.text) <= 8000
    assert result.metadata["current_request_preserved"] is True
    assert result.metadata["current_request_chars"] == len("当前请求")
    assert result.metadata["reductions"]


def test_prompt_cache_requires_explicit_enablement_and_minimum_prefix():
    policy = PromptCachePolicy(enabled=True, min_stable_chars=10)
    plain, metadata = policy.build_system("short", "dynamic")
    assert isinstance(plain, str)
    assert metadata["prompt_cache_eligible"] is False
    blocks, metadata = policy.build_system("x" * 10, "dynamic")
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert metadata["prompt_cache_eligible"] is True


def test_memory_concurrent_append_compresses_once_without_overwriting_latest_messages():
    async def run():
        redis = FakeRedis()
        chroma = FakeChroma()
        llm = fake_llm(["messages 0 through 9 summarized"])
        manager = MemoryManager(
            api_key="test",
            redis_client=redis,
            chroma_client=chroma,
            llm_client=llm,
        )
        await asyncio.gather(*[
            manager.add_message("user", "conv", MsgRole.USER, f"message-{index:02d}")
            for index in range(20)
        ])
        recent = await manager._get_working_memory("user", "conv")
        assert sorted(message.content for message in recent) == [f"message-{index:02d}" for index in range(10, 20)]
        assert len(chroma.collections["episodic"].items) == 1
        assert llm.messages.create.call_count == 1

    asyncio.run(run())


def test_memory_close_only_closes_owned_clients(monkeypatch):
    async def run():
        owned_redis = FakeRedis()
        monkeypatch.setattr(memory_module.redis, "from_url", lambda *args, **kwargs: owned_redis)

        owned_manager = MemoryManager(
            api_key="test",
            chroma_client=FakeChroma(),
            llm_client=fake_llm([]),
        )
        await owned_manager.close()
        await owned_manager.close()
        assert owned_redis.closed is True

        injected_redis = FakeRedis()
        injected_manager = MemoryManager(
            api_key="test",
            redis_client=injected_redis,
            chroma_client=FakeChroma(),
            llm_client=fake_llm([]),
        )
        await injected_manager.close()
        assert injected_redis.closed is False

    asyncio.run(run())


def test_profile_upsert_is_user_scoped_and_throttled():
    async def run():
        clock = [1000.0]
        redis = FakeRedis()
        chroma = FakeChroma()
        llm = fake_llm([
            json.dumps({"preferences": ["email"], "entities": {}}),
            json.dumps({"preferences": ["sms"], "entities": {}}),
        ])
        manager = MemoryManager(
            api_key="test", redis_client=redis, chroma_client=chroma,
            llm_client=llm, clock=lambda: clock[0],
        )
        await manager.add_message("u1", "c1", MsgRole.USER, "请用邮件联系")
        await manager.update_profile("u1", "c1")
        await manager.update_profile("u1", "c1")
        assert llm.messages.create.call_count == 1
        clock[0] += 601
        await manager.add_message("u1", "c1", MsgRole.USER, "改用短信联系")
        await manager.update_profile("u1", "c1")
        await manager.add_message("u2", "c2", MsgRole.USER, "另一个用户")
        assert llm.messages.create.call_count == 2
        assert (await manager._get_profile("u1"))["preferences"] == ["sms"]
        assert await manager._get_profile("u2") == {}

    asyncio.run(run())


def test_redis_memory_keys_cannot_collide_across_user_and_conversation_parts():
    assert MemoryManager._wm_key("a:b", "c") != MemoryManager._wm_key("a", "b:c")
    assert MemoryManager._summary_key("a:b", "c") != MemoryManager._summary_key("a", "b:c")


def test_episodic_retrieval_filters_low_relevance_and_duplicates():
    async def run():
        chroma = FakeChroma()
        episodic = chroma.get_or_create_collection("episodic")
        manager = MemoryManager(
            api_key="test", redis_client=FakeRedis(), chroma_client=chroma,
            llm_client=fake_llm([]),
        )
        episodic.query_result = {
            "documents": [["退款审核需要三天", "退款审核需要三天", "无关天气信息"]],
            "distances": [[0.1, 0.12, 0.9]],
            "metadatas": [[{"ts": "2026-01-01T00:00:00"}, {"ts": "2025-01-01T00:00:00"}, {}]],
        }
        assert await manager._search_episodic("u1", "退款多久") == ["退款审核需要三天"]

    asyncio.run(run())


def test_provider_usage_formulas_are_normalized():
    anthropic = CacheUsage("anthropic", input_tokens=20, cache_read_tokens=100, cache_write_tokens=10)
    assert anthropic.total_input_tokens == 130
    openai = CacheUsage("openai", input_tokens=130, cache_read_tokens=100)
    assert openai.total_input_tokens == 130
    deepseek = CacheUsage("deepseek", input_tokens=None, cache_read_tokens=100, cache_miss_tokens=30)
    assert deepseek.total_input_tokens == 130


def test_gateway_preserves_api_and_usage_serialization_errors():
    import httpx
    from anthropic import APITimeoutError

    class Messages:
        async def create(self, **kwargs):
            raise APITimeoutError(request=httpx.Request("POST", "https://example.test"))

    class BrokenUsage:
        def model_dump(self):
            raise ValueError("usage serialization failed")

    async def run():
        gateway = AnthropicGateway("test", "model")
        gateway.client = SimpleNamespace(messages=Messages())
        with pytest.raises(APITimeoutError):
            await gateway.complete(LLMRequest(model="model", stable_prompt="stable"))

        async def create(**kwargs):
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}], usage=BrokenUsage())

        gateway.client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with pytest.raises(ValueError, match="usage serialization failed"):
            await gateway.complete(LLMRequest(model="model", stable_prompt="stable"))

    asyncio.run(run())


def test_metrics_separate_application_and_provider_rates():
    metrics = CacheMetricsCollector()
    metrics.record_application("knowledge", "hit")
    metrics.record_application("knowledge", "miss")
    metrics.record_provider("deepseek", "model", CacheUsage(
        "deepseek", 100, 70, 0, 30, True, "hit", output_tokens=25,
    ))
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["application.knowledge.hit"] == 1
    assert snapshot["counters"]["provider.deepseek.model.read_tokens"] == 70
    assert snapshot["counters"]["provider.deepseek.model.output_tokens"] == 25


def test_agent_response_keeps_request_scoped_cache_metadata():
    class Gateway:
        provider = "deepseek"

        async def complete(self, request: LLMRequest):
            await asyncio.sleep(0)
            usage = CacheUsage("deepseek", 10, 5, 0, 5, True, "hit")
            return LLMResult("已处理", usage, {"request": request.messages[-1]["content"]})

    async def run():
        agent = GeneralAgent(None, "model", gateway=Gateway())
        first, second = await asyncio.gather(
            agent.handle(Request("甲", "u1", "c1")),
            agent.handle(Request("乙", "u2", "c2")),
        )
        assert first.cache_metadata["request"] == "甲"
        assert second.cache_metadata["request"] == "乙"

    asyncio.run(run())


def test_openai_adapter_reads_cached_token_details_and_sends_cache_key():
    class Completions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content="ok")
            usage = {
                "prompt_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 1024, "cache_write_tokens": 0},
            }
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage,
                                   model_dump=lambda: {"usage": usage})

    async def run():
        gateway = OpenAIGateway("test", "gpt-test")
        completions = Completions()
        gateway.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        result = await gateway.complete(LLMRequest(
            model="gpt-test", stable_prompt="stable", messages=[{"role": "user", "content": "q"}],
            cache_identity="agent:v1",
        ))
        assert completions.kwargs["prompt_cache_key"] == "agent:v1"
        assert result.usage.cache_read_tokens == 1024
        assert result.usage.total_input_tokens == 1200

    asyncio.run(run())


def test_deepseek_anthropic_adapter_uses_automatic_cache_and_parses_usage():
    class Messages:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                content=[{"type": "text", "text": "ok"}],
                usage={"input_tokens": 200, "prompt_cache_hit_tokens": 150,
                       "prompt_cache_miss_tokens": 50},
            )

    async def run():
        gateway = DeepSeekAnthropicGateway("test", "deepseek", "https://api.deepseek.com/anthropic")
        messages = Messages()
        gateway.client = SimpleNamespace(messages=messages)
        result = await gateway.complete(LLMRequest(
            model="deepseek", stable_prompt="stable", dynamic_prompt="dynamic",
            messages=[{"role": "user", "content": "q"}],
        ))
        assert isinstance(messages.kwargs["system"], str)
        assert result.usage.cache_read_tokens == 150
        assert result.usage.cache_miss_tokens == 50
        assert result.usage.status == "hit"

    asyncio.run(run())


def test_deepseek_anthropic_adapter_retries_truncated_text_with_more_tokens():
    class Messages:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    content=[{"type": "text", "text": "partial"}],
                    stop_reason="max_tokens",
                    usage={"input_tokens": 100, "output_tokens": 20},
                )
            return SimpleNamespace(
                content=[{"type": "text", "text": "ok"}],
                stop_reason="end_turn",
                usage={"input_tokens": 200, "output_tokens": 30},
            )

    async def run():
        gateway = DeepSeekAnthropicGateway("test", "deepseek", "https://api.deepseek.com/anthropic")
        messages = Messages()
        gateway.client = SimpleNamespace(messages=messages)
        result = await gateway.complete(LLMRequest(
            model="deepseek", stable_prompt="stable",
            messages=[{"role": "user", "content": "q"}], max_tokens=256,
        ))
        assert result.text == "ok"
        assert [call["max_tokens"] for call in messages.calls] == [256, 2048]
        assert result.metadata["response_retry"] is True
        assert result.metadata["empty_response_retry"] is False
        assert result.usage.total_input_tokens == 300
        assert result.usage.output_tokens == 50
        assert result.metadata["total_input_tokens"] == 300
        assert result.metadata["output_tokens"] == 50

        metrics = CacheMetricsCollector()
        metrics.record_provider("deepseek", "model", result.usage)
        counters = metrics.snapshot()["counters"]
        assert counters["provider.deepseek.model.input_tokens"] == 300
        assert counters["provider.deepseek.model.output_tokens"] == 50

    asyncio.run(run())


@pytest.mark.parametrize("gateway_type,first_usage,second_usage,total,read,write,miss", [
    (AnthropicGateway,
     {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10},
     {"input_tokens": 200, "output_tokens": 30, "cache_read_input_tokens": 70, "cache_creation_input_tokens": 20},
     450, 120, 30, None),
    (DeepSeekAnthropicGateway,
     {"input_tokens": 100, "output_tokens": 20, "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40},
     {"input_tokens": 200, "output_tokens": 30, "prompt_cache_hit_tokens": 120, "prompt_cache_miss_tokens": 80},
     300, 180, 0, 120),
    (DeepSeekAnthropicGateway,
     {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 5},
     {"input_tokens": 20, "output_tokens": 30, "cache_read_input_tokens": 70, "cache_creation_input_tokens": 10},
     165, 120, 0, 45),
])
def test_retry_usage_aggregates_provider_cache_fields(
    gateway_type, first_usage, second_usage, total, read, write, miss,
):
    class Messages:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                content=[{"type": "text", "text": "" if self.calls == 1 else "ok"}],
                stop_reason="end_turn",
                usage=first_usage if self.calls == 1 else second_usage,
            )

    async def run():
        gateway = gateway_type("test", "model")
        messages = Messages()
        gateway.client = SimpleNamespace(messages=messages)
        result = await gateway.complete(LLMRequest(model="model", stable_prompt="stable"))
        assert messages.calls == 2
        assert result.text == "ok"
        assert result.metadata["empty_response_retry"] is True
        assert result.usage.total_input_tokens == total
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_tokens == read
        assert result.usage.cache_write_tokens == write
        assert result.usage.cache_miss_tokens == miss
        assert result.metadata["total_input_tokens"] == total
        assert result.metadata["input_tokens"] == result.usage.input_tokens
        assert result.metadata["cache_read_tokens"] == read
        assert result.metadata["cache_write_tokens"] == write
        assert result.metadata["cache_miss_tokens"] == miss
        assert result.metadata["cache_status"] == "hit"

    asyncio.run(run())


def test_monitor_ignores_orchestrator_metadata_entries():
    class Orchestrator:
        def __init__(self):
            self.penalties = None

        def get_stats(self):
            return {
                "general_0": {"success_rate": 1.0, "avg_ms": 10.0, "total": 0},
                "cache_metrics": {"counters": {}, "providers": {}},
            }

        def update_routing_penalties(self, penalties):
            self.penalties = penalties

    class ToolManager:
        @staticmethod
        def get_stats():
            return {}

    async def run():
        orchestrator = Orchestrator()
        monitor = PerformanceMonitor(orchestrator, ToolManager())
        await monitor._collect()
        assert orchestrator.penalties == {"general_0": 0.0}

    asyncio.run(run())
