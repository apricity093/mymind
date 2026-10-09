import asyncio
import fnmatch

import httpx
import pytest

import api.main as api
from core.cache_metrics import CacheMetricsCollector, ObservedCacheStore
from core.cache_store import InMemoryCacheStore, RedisCacheStore
from mcp.tool_manager import MCPToolManager, Tool


class CacheRedis:
    """Shared Redis test backend for versioned cache operations."""

    def __init__(self):
        self.values = {}
        self.on_scan = None

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value

    def eval(self, script, numkeys, generation_key, key, generation, payload, ttl):
        assert numkeys == 2
        if int(self.values.get(generation_key, 0)) != generation:
            return 0
        self.values[key] = payload
        return 1

    def scan_iter(self, match, count):
        if self.on_scan:
            callback, self.on_scan = self.on_scan, None
            callback()
        yield from [key for key in self.values if fnmatch.fnmatchcase(key, match)]

    def delete(self, *keys):
        return sum(self.values.pop(key, None) is not None for key in keys)


@pytest.fixture(params=["memory", "observed", "redis"])
def cache_store(request):
    if request.param == "redis":
        return RedisCacheStore(CacheRedis())
    store = InMemoryCacheStore(clock=lambda: 10.0)
    if request.param == "observed":
        return ObservedCacheStore(store, CacheMetricsCollector())
    return store


def make_manager(cache_store, handler):
    manager = MCPToolManager(api_key="test", cache_store=cache_store)
    manager.register(Tool(
        name="knowledge_search", description="test", handler=handler,
        schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        cache_ttl=300,
    ))
    return manager


def test_invalidation_deletes_old_data_and_rejects_stale_writes(cache_store):
    generation = cache_store.get_generation("knowledge")
    cache_store.set("knowledge", "query-a", "old-a", 300)
    cache_store.set("knowledge", "query-b", "old-b", 300)
    cache_store.set("intent", "query-a", "intent", 600)

    cache_store.invalidate_namespace("knowledge")
    assert cache_store.get("knowledge", "query-a") is None
    assert cache_store.get("knowledge", "query-b") is None
    assert cache_store.get("intent", "query-a") == "intent"
    if isinstance(cache_store, RedisCacheStore):
        assert not list(cache_store.client.scan_iter("mymind:cache:knowledge:*", 500))
    else:
        store = cache_store.store if isinstance(cache_store, ObservedCacheStore) else cache_store
        assert store.size == 1

    cache_store.set("knowledge", "query-a", "fresh", 300)
    cache_store.set("knowledge", "query-a", "stale", 300, generation=generation)
    assert cache_store.get("knowledge", "query-a") == "fresh"


def test_redis_cleanup_preserves_new_results_and_other_prefixes():
    backend = CacheRedis()
    writer = RedisCacheStore(backend)
    reader = RedisCacheStore(backend)
    other = RedisCacheStore(backend, prefix="other:cache")
    writer.set("knowledge", "query", "old", 300)
    backend.values["mymind:cache:knowledge:1:previous"] = '"previous"'
    backend.values["mymind:cache:generation:knowledge"] = "1"
    other.set("knowledge", "query", "unrelated", 300)
    backend.on_scan = lambda: reader.set("knowledge", "query", "fresh", 300)

    writer.invalidate_namespace("knowledge")
    assert reader.get("knowledge", "query") == "fresh"
    assert other.get("knowledge", "query") == "unrelated"
    assert "mymind:cache:knowledge:0:query" not in backend.values
    assert "mymind:cache:knowledge:1:previous" not in backend.values


def test_query_started_before_update_cannot_repopulate_cache(cache_store):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        content = "old"
        calls = 0

        async def handler(params, context):
            nonlocal calls
            calls += 1
            result = [{"content": content}]
            if calls == 1:
                started.set()
                await release.wait()
            return result

        manager = make_manager(cache_store, handler)
        updater = (
            make_manager(RedisCacheStore(cache_store.client), handler)
            if isinstance(cache_store, RedisCacheStore) else manager
        )
        old_query = asyncio.create_task(manager.call("knowledge_search", {"query": "退款"}))
        await asyncio.wait_for(started.wait(), timeout=2)
        content = "fresh"
        updater.invalidate_cache()
        release.set()
        assert (await old_query).data == [{"content": "old"}]

        fresh = await manager.call("knowledge_search", {"query": "退款"})
        assert fresh.success and not fresh.cached
        assert fresh.data == [{"content": "fresh"}]
        assert (await manager.call("knowledge_search", {"query": "退款"})).cached
        assert calls == 2

    asyncio.run(run())


@pytest.mark.parametrize("upload", [None, "policy.md", "policy.txt", "policy.json"])
def test_knowledge_import_clears_live_cache_before_returning(monkeypatch, cache_store, upload):
    async def run():
        class Knowledge:
            content = "旧政策"
            doc_count = 1

            def add_documents(self, documents):
                self.content = documents[0]["content"]
                return len(documents)

        knowledge = Knowledge()
        calls = 0

        async def handler(params, context):
            nonlocal calls
            calls += 1
            return [{"content": knowledge.content}]

        manager = make_manager(cache_store, handler)
        monkeypatch.setattr(api, "_knowledge_base", knowledge)
        monkeypatch.setattr(api, "_tool_manager", manager)
        params = {"query": "退款"}
        assert (await manager.call("knowledge_search", params)).data == [{"content": "旧政策"}]
        assert (await manager.call("knowledge_search", params)).cached

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            if upload is None:
                response = await client.post("/knowledge/add", json={"documents": [
                    {"title": "政策", "content": "新政策"},
                ]})
            else:
                content = '[{"title": "政策", "content": "新政策"}]' if upload.endswith(".json") else "新政策"
                response = await client.post("/knowledge/upload", files={"file": (upload, content.encode())})
        assert response.status_code == 200
        assert response.json()["processed_chunks"] == 1
        fresh = await manager.call("knowledge_search", params)
        assert fresh.success and not fresh.cached
        assert fresh.data == [{"content": "新政策"}]
        assert (await manager.call("knowledge_search", params)).cached
        assert calls == 2

    asyncio.run(run())


@pytest.mark.parametrize("upload", [False, True])
def test_import_reports_cache_cleanup_failure(monkeypatch, upload):
    class BrokenCache(InMemoryCacheStore):
        def invalidate_namespace(self, namespace):
            raise ConnectionError("redis unavailable")

    class Knowledge:
        content = "旧政策"

        def add_documents(self, documents):
            self.content = documents[0]["content"]
            return 1

    async def run():
        knowledge = Knowledge()
        manager = MCPToolManager(api_key="test", cache_store=BrokenCache())
        monkeypatch.setattr(api, "_knowledge_base", knowledge)
        monkeypatch.setattr(api, "_tool_manager", manager)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            if upload:
                response = await client.post("/knowledge/upload", files={"file": ("policy.md", "新政策".encode())})
            else:
                response = await client.post("/knowledge/add", json={"documents": [
                    {"title": "政策", "content": "新政策"},
                ]})
        assert knowledge.content == "新政策"
        assert response.status_code == 503
        assert "知识库已更新" in response.json()["detail"]

    asyncio.run(run())
