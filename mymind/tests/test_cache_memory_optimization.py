import asyncio
import json
from pathlib import Path

from core.cache_store import InMemoryCacheStore
from core.intent_recognizer import IntentRecognizer
from core.prompt_cache import PromptCachePolicy
from mcp.tool_manager import MCPToolManager, Tool
from memory.context_builder import ContextBuilder
from memory.conversation_memory import MemoryContext, MemoryManager, Message, MsgRole
from tests.fakes import FakeChroma, FakeLlm, FakeRedis
from experiments.offline import run_offline


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


def test_intent_cache_key_isolates_history_model_and_normalizes_message():
    recognizer = IntentRecognizer(api_key="test", model="model-a")
    first = recognizer._cache_key("  可以  ", [{"role": "assistant", "content": "是否退款？"}])
    same = recognizer._cache_key("可以", [{"role": "assistant", "content": "是否退款？"}])
    other_history = recognizer._cache_key("可以", [{"role": "assistant", "content": "是否取消？"}])
    assert first == same
    assert first != other_history


def test_knowledge_cache_collapses_fifty_concurrent_misses_and_invalidates():
    async def run():
        calls = 0

        async def handler(params, context):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return [{"title": "refund", "content": params["query"]}]

        manager = MCPToolManager(api_key="test")
        manager.register(Tool(
            name="knowledge_search",
            description="test",
            handler=handler,
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
            cache_ttl=300,
        ))
        results = await asyncio.gather(*[
            manager.call("knowledge_search", {"query": "退款"}) for _ in range(50)
        ])
        assert calls == 1
        assert sum(result.cached for result in results) == 49
        manager.invalidate_cache()
        await manager.call("knowledge_search", {"query": "退款"})
        assert calls == 2

    asyncio.run(run())


def test_cache_outage_fails_open_and_does_not_cache_fallbacks():
    class BrokenCache:
        def get(self, namespace, key):
            raise ConnectionError("redis unavailable")

        def set(self, namespace, key, value, ttl):
            raise ConnectionError("redis unavailable")

        def invalidate_namespace(self, namespace):
            raise ConnectionError("redis unavailable")

    async def run():
        calls = 0

        async def handler(params, context):
            nonlocal calls
            calls += 1
            return [{"content": "live result"}]

        manager = MCPToolManager(api_key="test", cache_store=BrokenCache())
        manager.register(Tool(
            name="knowledge_search", description="test", handler=handler,
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
            cache_ttl=300,
        ))
        first = await manager.call("knowledge_search", {"query": "退款"})
        second = await manager.call("knowledge_search", {"query": "退款"})
        assert first.success and second.success
        assert not first.cached and not second.cached
        assert calls == 2
        assert manager.invalidate_cache() == -1

    asyncio.run(run())


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
        llm = FakeLlm(["messages 0 through 9 summarized"])
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
        assert [message.content for message in recent] == [f"message-{index:02d}" for index in range(10, 20)]
        assert len(chroma.collections["episodic"].items) == 1
        assert len(llm.messages.calls) == 1

    asyncio.run(run())


def test_profile_upsert_is_user_scoped_and_throttled():
    async def run():
        clock = [1000.0]
        redis = FakeRedis()
        chroma = FakeChroma()
        llm = FakeLlm([
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
        assert len(llm.messages.calls) == 1
        clock[0] += 601
        await manager.add_message("u1", "c1", MsgRole.USER, "改用短信联系")
        await manager.update_profile("u1", "c1")
        await manager.add_message("u2", "c2", MsgRole.USER, "另一个用户")
        assert len(llm.messages.calls) == 2
        assert (await manager._get_profile("u1"))["preferences"] == ["sms"]
        assert await manager._get_profile("u2") == {}

    asyncio.run(run())


def test_redis_memory_keys_cannot_collide_across_user_and_conversation_parts():
    assert MemoryManager._wm_key("a:b", "c") != MemoryManager._wm_key("a", "b:c")
    assert MemoryManager._summary_key("a:b", "c") != MemoryManager._summary_key("a", "b:c")


def test_episodic_retrieval_filters_low_relevance_and_duplicates():
    async def run():
        chroma = FakeChroma()
        manager = MemoryManager(
            api_key="test", redis_client=FakeRedis(), chroma_client=chroma,
            llm_client=FakeLlm([]),
        )
        chroma.collections["episodic"].query_result = {
            "documents": [["退款审核需要三天", "退款审核需要三天", "无关天气信息"]],
            "distances": [[0.1, 0.12, 0.9]],
            "metadatas": [[{"ts": "2026-01-01T00:00:00"}, {"ts": "2025-01-01T00:00:00"}, {}]],
        }
        assert await manager._search_episodic("u1", "退款多久") == ["退款审核需要三天"]

    asyncio.run(run())


def test_offline_experiment_emits_all_variants(tmp_path):
    report = run_offline(tmp_path)
    assert report["overall_passed"] is True
    assert set(report["variants"]) == {"B0", "C1", "C2", "C3"}
    assert Path(report["artifacts"]["json"]).exists()
    assert Path(report["artifacts"]["markdown"]).exists()
