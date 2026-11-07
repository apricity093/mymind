"""检索、重排、索引恢复与缓存失效。"""

import api.main as api
import asyncio
import chromadb
import fnmatch
import httpx
import json
import pytest
import threading
import time
from core.cache_metrics import CacheMetricsCollector, ObservedCacheStore
from core.cache_store import InMemoryCacheStore, RedisCacheStore, freeze_cache_key
from core.document_chunking import BudgetChunker, parse_sections
from core.embedding import (
    EmbeddingError,
    EmbeddingSettings,
    EmbeddingTelemetry,
    GeminiEmbedding,
    QwenEmbedding,
    build_embedding,
)
from core.retrieval import (
    BM25Index,
    Chunker,
    RetrievalPipeline,
    dedupe_items,
    deterministic_rank,
    parse_rerank_indices,
    source_id_for,
    stable_chunk_id,
    variant_config,
    weighted_rrf,
)
from evaluation.evaluator import EndToEndEvaluator
from experiments.retrieval_scoring import (
    evaluate_retrieval,
    rescore_retrieval_group,
    summarize_retrieval,
)
from fastapi.testclient import TestClient
from mcp.indexed_knowledge_base import KnowledgeBase
from mcp.knowledge_base import KnowledgeBase as LegacyKnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool
from tests.fakes import FakeCollection, StubEmbedding
from types import SimpleNamespace


def test_markdown_chunker_tracks_sections_and_overlap():
    content = """# 文档标题

## 第一节
这是第一节的第一段文字，包含退款审核需要 1-3 个工作日的信息。

## 第二节
这是第二节的第一段文字，包含退款到账需要 5-7 个工作日的信息。
"""
    chunks = Chunker(variant_config("r2")).chunk_document("退款指南", content)
    assert len(chunks) == 2
    assert chunks[0].section_path == "文档标题/第一节"
    assert chunks[1].section_path == "文档标题/第二节"
    assert chunks[0].chunk_id == stable_chunk_id(chunks[0].source_id, "文档标题/第一节", 0)

    long_content = "# A\n\n## 第一节\n" + ("内容单元。" * 200) + "\n\n## 第二节\n" + ("另一个单元。" * 200)
    records = Chunker(variant_config("r4")).chunk_document("长文档", long_content)
    assert len(records) > 2
    ids = [record.chunk_id for record in records]
    assert len(ids) == len(set(ids))


def test_chunk_id_stable_across_rebuild_and_dedup_uses_it():
    first = Chunker(variant_config("r1")).chunk_document("标题", "第一段。第二段。第三段。")
    second = Chunker(variant_config("r1")).chunk_document("标题", "第一段。第二段。第三段。")
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
    items = [
        {"chunk_id": "chk-1", "content": "重复内容", "score": 0.9},
        {"chunk_id": "chk-1", "content": "重复内容", "score": 0.8},
        {"chunk_id": "chk-2", "content": "另一段", "score": 0.7},
    ]
    assert len(dedupe_items(items)) == 2


def test_deterministic_rank_and_rrf_are_stable():
    items = [
        {"title": "b", "content": "x", "score": 0.9, "chunk": 1},
        {"title": "a", "content": "y", "score": 0.9, "chunk": 0},
        {"title": "c", "content": "z", "score": 0.5, "chunk": 0},
    ]
    assert [item["title"] for item in deterministic_rank(items, 3)] == ["a", "b", "c"]
    vector = [{"chunk_id": "a", "content": "a"}, {"chunk_id": "b", "content": "b"}]
    bm25 = [{"chunk_id": "b", "content": "b"}, {"chunk_id": "c", "content": "c"}]
    fused = weighted_rrf(vector, bm25)
    assert fused[0]["chunk_id"] == "b"
    assert set(fused[0]["retrieval_sources"]) == {"vector", "bm25"}
    assert all("fusion_score" in item for item in fused)


def test_bm25_prefers_exact_terms_over_noise():
    index = BM25Index()
    index.add("relevant", "接口返回 401 表示认证失败，请重新获取 token", {"chunk_id": "relevant"})
    index.add("noise", "接口调用与认证请求的常规说明", {"chunk_id": "noise"})
    assert index.search("401 认证失败", 2)[0][0] == "relevant"


def test_pipeline_falls_back_deterministically_on_rerank_failure():
    async def run():
        async def vector(query, top_k):
            return [{"chunk_id": f"c{i}", "content": f"内容{i}", "score": 1.0 - i * 0.01} for i in range(12)]

        async def rewrite(query, n=3):
            return [query, f"{query} 扩展"]

        async def broken_rerank(query, items, top_k):
            raise ValueError("simulated parse failure")

        config = variant_config("r1")
        pipeline = RetrievalPipeline(config, vector, rewriter=rewrite, reranker=broken_rerank)
        outcome = await pipeline.search("查询", top_k=5)
        assert outcome.reranked is False
        assert outcome.stats.rerank_failures == 1
        assert outcome.stats.rerank_fallbacks == 1
        assert len(outcome.items) == 5
        scores = [item["score"] for item in outcome.items]
        assert scores == sorted(scores, reverse=True)

    asyncio.run(run())


def test_pipeline_hybrid_returns_fusion_sources_and_top_k():
    async def run():
        async def vector(query, top_k):
            return [{"chunk_id": f"c{i}", "content": f"内容{i}", "score": 0.8} for i in range(top_k)]

        index = BM25Index()
        for i in range(20):
            index.add(f"c{i}", f"内容{i}", {"chunk_id": f"c{i}", "content": f"内容{i}"})
        config = variant_config("r3")
        pipeline = RetrievalPipeline(config, vector, bm25_index=index)
        outcome = await pipeline.search("查询", top_k=3)
        assert len(outcome.items) == 3
        assert all("fusion_score" in item for item in outcome.items)

    asyncio.run(run())


def test_parse_rerank_indices_rejects_bad_outputs():
    assert parse_rerank_indices('[2, 0, 1]', 3, 3) == [2, 0, 1]
    for bad in ("[0, 3]", "[0, 0]", "[true]", "没有数组"):
        try:
            parse_rerank_indices(bad, 3, 3)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject {bad}")


def test_knowledge_base_variant_search_contract_and_stable_ids():
    knowledge = LegacyKnowledgeBase.__new__(LegacyKnowledgeBase)
    knowledge._config = variant_config("r2")
    knowledge._variant_name = "r2"
    knowledge._collection_name = "mymind_rag_r2_v1"
    knowledge._chunker = Chunker(knowledge._config)
    knowledge._bm25_index = None
    knowledge._collection = FakeCollection(metric="cosine")
    knowledge.add_documents([{
        "title": "退款政策",
        "content": "# 退款政策\n\n## 时效\n退款审核需要 1-3 个工作日。\n\n## 到账\n款项将在 5-7 个工作日退回。",
    }])
    assert knowledge.doc_count == 2
    knowledge._collection.query_result = {
        "documents": [["退款审核需要 1-3 个工作日。"]],
        "metadatas": [[{"title": "退款政策", "chunk_index": 0, "chunk_id": "chk-a", "source_id": "src-a", "section_path": "退款政策/时效", "index_version": "rag-index-v1"}]],
        "distances": [[0.2]],
        "ids": [["chk-a"]],
    }
    results = knowledge.vector_search("退款审核多久", 5)
    assert results[0]["title"] == "退款政策"
    assert results[0]["content"]
    assert "score" in results[0] and "chunk" in results[0]
    assert results[0]["chunk_id"] == "chk-a"
    assert results[0]["source_id"] == "src-a"
    assert results[0]["section_path"] == "退款政策/时效"


def test_search_endpoint_keeps_contract_and_reports_top_k(monkeypatch):
    import api.main as api

    class Manager:
        def __init__(self):
            self.calls = []

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            self.calls.append((tool_name, query, top_k))
            return SimpleNamespace(
                data=[{"title": "t", "content": "c", "score": 0.9, "chunk": 0, "chunk_id": "chk-1", "source_id": "src-1"}],
                reranked=True,
            )

    manager = Manager()
    monkeypatch.setattr(api, "_tool_manager", manager)
    monkeypatch.setattr(api, "_knowledge_base", SimpleNamespace(index_version="rag-index-v1", chunk_config={"chunk_size": 500}))
    response = TestClient(api.app).post("/search", params={"query": "退款", "top_k": 7})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "退款"
    assert payload["reranked"] is True
    assert payload["requested_top_k"] == 7
    assert payload["returned"] == 1
    assert payload["results"][0]["chunk_id"] == "chk-1"
    assert manager.calls[-1][2] == 7


def test_knowledge_add_and_upload_keep_legacy_contract(monkeypatch):
    import api.main as api

    class Knowledge:
        def __init__(self):
            self.count = 10

        def import_documents(self, documents):
            added = sum(1 for doc in documents if doc.get("content"))
            self.count += added
            return {"processed_chunks": added, "status": "success"}

        @property
        def doc_count(self):
            return self.count

    class Manager:
        def invalidate_cache(self):
            return 1

    monkeypatch.setattr(api, "_knowledge_base", Knowledge())
    monkeypatch.setattr(api, "_tool_manager", Manager())
    client = TestClient(api.app)
    added = client.post("/knowledge/add", json={"documents": [
        {"title": "退款政策", "content": "七天内可以无理由退款。"},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日。"},
    ]})
    assert added.status_code == 200
    added_payload = added.json()
    assert added_payload["added_chunks"] == 2
    assert added_payload["processed_chunks"] == 2
    assert added_payload["total_chunks"] == 12

    uploaded = client.post(
        "/knowledge/upload",
        files={"file": ("退款补充.md", "# 退款补充\n\n大促审核延长到 3-5 个工作日。", "text/markdown")},
    )
    assert uploaded.status_code == 200
    upload_payload = uploaded.json()
    assert upload_payload["added_chunks"] == 1
    assert upload_payload["total_chunks"] == 13
    assert "message" in upload_payload


def test_stats_endpoint_total_chunks_is_numeric(monkeypatch):
    import api.main as api

    monkeypatch.setattr(api, "_knowledge_base", SimpleNamespace(
        doc_count=12,
        index_version="rag-index-v1",
        chunk_config={"chunk_size": 500},
    ))
    response = TestClient(api.app).get("/knowledge/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_chunks"] == 12
    assert isinstance(payload["total_chunks"], int)


def question(answerable=True):
    return {
        "id": "q", "family_id": "family", "question": "退款条件", "tags": ["condition"],
        "answerable": answerable,
        "evidence_groups": [{"alternatives": [{"source_id": "policy"}]}] if answerable else [],
    }


def test_original_precision_counts_one_source_for_three_chunks():
    items = [{"source_id": "policy", "chunk_id": str(i), "content": "不同段落"} for i in range(3)]
    row = evaluate_retrieval(question(), items)
    assert row["recall_at_5"] == row["recall_at_10"] == 1
    assert row["precision_at_3"] == pytest.approx(1 / 3)
    assert row["mrr_at_10"] == 1


def test_original_no_answer_contributes_recall_one_and_precision_zero():
    positive = evaluate_retrieval(question(), [{"source_id": "policy", "content": "退款"}])
    absent = evaluate_retrieval(question(False), [{"source_id": "policy", "content": "退款"}])
    summary = summarize_retrieval([positive, absent])
    assert summary["recall_at_5"] == 1
    assert summary["precision_at_3"] == pytest.approx(1 / 6)
    assert summary["mrr_at_10"] == 0.5
    assert summary["no_answer_false_positive"] == 1
    assert evaluate_retrieval(question(False), [])["no_answer_false_positive"] == 0


def test_rescore_preserves_ranking_and_removes_withdrawn_metrics():
    items = [{"source_id": "other", "content": "背景"}, {"source_id": "policy", "content": "退款"}]
    original = {"status": "measured", "coverage@5": 0.5, "all@5": 0, "mrr@10": 0,
                "rows": [{"id": "q", "retrieved": items, "coverage@5": 0.5, "all@5": False,
                          "mrr@10": 0, "wall_ms": 123}]}
    result = rescore_retrieval_group(original, {"q": question()})
    assert result["rows"][0]["retrieved"] == items
    assert result["rows"][0]["wall_ms"] == 123
    assert result["mrr_at_10"] == 0.5
    assert not {"coverage@5", "all@5", "mrr@10"} & result.keys()
    assert not {"coverage@5", "all@5", "mrr@10"} & result["rows"][0].keys()
    assert original["coverage@5"] == 0.5




def test_tool_fallback_is_degraded_and_never_enters_rerank():
    async def run():
        async def broken(params, context):
            raise ConnectionError("knowledge unavailable")

        manager = MCPToolManager(api_key="test")
        manager.register(Tool(
            name="knowledge_search",
            description="test",
            handler=broken,
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
            fallback=lambda params, context, error: [{"content": "fallback"}],
        ))

        async def no_rewrite(query, n=3):
            return [query]

        manager.rewrite_query = no_rewrite
        result = await manager.search_with_rewrite("knowledge_search", "退款")
        assert result.success is False
        assert result.degraded is True
        assert result.data == []

    asyncio.run(run())


def test_knowledge_import_is_idempotent_upsert():
    class UpsertCollection:
        def __init__(self):
            self.items = {}
            self.calls = []

        def upsert(self, ids, documents, metadatas):
            self.calls.append((ids, documents, metadatas))
            self.items.update(zip(ids, documents))

        def count(self):
            return len(self.items)

    knowledge = LegacyKnowledgeBase.__new__(LegacyKnowledgeBase)
    knowledge._collection = UpsertCollection()
    documents = [{"title": "退款政策", "content": "七天内可以退款"}]
    assert knowledge.add_documents(documents) == 1
    assert knowledge.add_documents(documents) == 1
    assert knowledge.doc_count == 1
    first, second = knowledge._collection.calls
    assert first[0] == second[0]
    assert first[1] == second[1] == ["七天内可以退款"]


def test_old_baseline_without_new_metrics_remains_readable():
    report = EndToEndEvaluator._report_from_dict({
        "timestamp": "2026-01-01T00:00:00",
        "total": 1,
        "passed": 1,
        "pass_rate": 1.0,
        "avg_scores": {"relevance": 0.9},
        "results": [],
    })
    assert report.avg_scores == {"relevance": 0.9}
    assert report.regressions == []


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


def test_structured_keys_preserve_parameter_types_and_mapping_order(cache_store):
    first = freeze_cache_key({"query": "退款", "filters": {"tags": []}})
    same = freeze_cache_key({"filters": {"tags": []}, "query": "退款"})
    other = freeze_cache_key({"query": "退款", "filters": {"tags": {}}})
    cache_store.set("knowledge", first, {"data": "matched"}, 60)
    assert cache_store.get("knowledge", same) == {"data": "matched"}
    assert cache_store.get("knowledge", other) is None
    assert freeze_cache_key({"value": True}) != freeze_cache_key({"value": 1})


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
def test_knowledge_import_clears_live_cache_before_returning(monkeypatch, upload):
    cache_store = InMemoryCacheStore()
    async def run():
        class Knowledge:
            content = "旧政策"
            doc_count = 1

            def import_documents(self, documents):
                self.content = documents[0]["content"]
                return {"processed_chunks": len(documents), "status": "success"}

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

        def import_documents(self, documents):
            self.content = documents[0]["content"]
            return {"processed_chunks": 1, "status": "success"}

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




class StubMiniLM:
    def __init__(self):
        self.fail = False
        self.inputs = []

    def __call__(self, input):
        self.inputs.extend(input)
        if self.fail:
            raise RuntimeError("backup unavailable")
        return [[0.0, 1.0, 0.0] for _ in input]


@pytest.fixture
def knowledge(tmp_path):
    embedding, backup = StubEmbedding(), StubMiniLM()
    changed = []
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    kb = KnowledgeBase(chroma_path=str(tmp_path), embedding=embedding, client=client,
                       backup_function=backup, backup_counter=len,
                       on_change=lambda: changed.append(True) or 0, load_defaults=False,
                       collection_prefix="test_" + tmp_path.name.replace("-", "_")[-18:])
    yield kb, embedding, backup, changed
    kb.close()


def test_embedding_configuration_is_independent(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("LLM_API_KEY", "chat-only")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        EmbeddingSettings.from_env()
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-credential")
    settings = EmbeddingSettings.from_env()
    assert settings.dimensions == 768
    assert "test-credential" not in repr(settings)
    assert settings.model == "gemini-embedding-001"
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    assert not GeminiEmbedding(EmbeddingSettings.from_env()).enabled


def test_official_api_tasks_titles_normalization_and_counting():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"supportedGenerationMethods": ["embedContent", "countTokens"]})
        if request.url.path.endswith(":countTokens"):
            return httpx.Response(200, json={"totalTokens": 10})
        return httpx.Response(200, json={"embedding": {"values": [3.0, 4.0]}})

    gateway = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 2, "unit-key"),
                             transport=httpx.MockTransport(respond))
    assert gateway.embed(["body"], "RETRIEVAL_DOCUMENT", ["path/title"]) == [[0.6, 0.8]]
    gateway.embed(["question"], "RETRIEVAL_QUERY")
    gateway.embed(["intent"], "CLASSIFICATION")
    payloads = [json.loads(r.content) for r in requests if r.url.path.endswith(":embedContent")]
    assert [p["taskType"] for p in payloads] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"]
    assert payloads[0]["title"] == "path/title"
    assert "title" not in payloads[1]
    assert payloads[0]["content"]["parts"][0]["text"] == "body"
    assert all("unit-key" not in str(r.url) for r in requests)
    assert gateway.counting_mode == "google_countTokens"


def test_unsupported_counting_and_safe_errors(caplog):
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"supportedGenerationMethods": ["embedContent"]})
        return httpx.Response(403, json={"error": "unit-secret-key body"})

    gateway = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 768, "unit-secret-key"),
                             transport=httpx.MockTransport(respond))
    assert gateway.count_tokens("中文") == 6
    assert gateway.counting_mode == "utf8_byte_upper_bound"
    with pytest.raises(EmbeddingError, match="HTTP_403") as exc:
        gateway.embed(["body"], "CLASSIFICATION")
    assert "unit-secret-key" not in str(exc.value) + caplog.text


def test_google_zero_token_response_and_empty_overlap_tail():
    calls = []

    def respond(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"supportedGenerationMethods": ["countTokens"]})
        return httpx.Response(200, json={})

    gateway = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 768, "unit-key"),
                             transport=httpx.MockTransport(respond))
    assert gateway.count_tokens("") == 0 and calls == []
    assert gateway.count_tokens("\n") == 0
    assert BudgetChunker(gateway.count_tokens, 512, 32)._tail("第一句。第二句。") == "第一句。第二句。"


def test_official_batch_keeps_document_titles_and_normalizes_each_vector():
    payloads = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"supportedGenerationMethods": ["embedContent"]})
        payload = json.loads(request.content)
        payloads.append(payload)
        assert request.url.path.endswith(":batchEmbedContents")
        return httpx.Response(200, json={"embeddings": [{"values": [3, 4]} for _ in payload["requests"]]})

    embedding = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 2, "unit-key"),
                               transport=httpx.MockTransport(respond))
    assert embedding.embed(["one", "two"], "RETRIEVAL_DOCUMENT", ["first", "second"]) == [[.6, .8], [.6, .8]]
    assert [r["title"] for r in payloads[0]["requests"]] == ["first", "second"]


def test_quota_failure_records_only_quota_and_retry_details():
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"supportedGenerationMethods": []})
        return httpx.Response(429, json={"error": {"message": "sensitive text and unit-key",
            "details": [{"violations": [{"quotaId": "EmbeddingRequestsPerMinute"}]}, {"retryDelay": "30s"}]}})

    embedding = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 2, "unit-key"),
                               transport=httpx.MockTransport(respond))
    with pytest.raises(EmbeddingError) as exc:
        embedding.embed(["body"], "CLASSIFICATION")
    assert "EmbeddingRequestsPerMinute" in str(exc.value) and "30s" in str(exc.value)
    assert "unit-key" not in str(exc.value) and "sensitive" not in str(exc.value)


def test_structure_titles_limits_overlap_and_long_unbroken_text():
    text = "# 手册\n## 退款\n" + "条件说明。" * 70 + "\n## 配送\n配送单独规则。"
    chunker = BudgetChunker(len, 100, 15, reserve=4)
    records = chunker.chunk_document("政策", text)
    assert len(records) > 3
    assert all(len(f"{chunker.embedding_title(r.title, r.section_path)}\n{r.text}") + 4 <= 100 for r in records)
    assert records[-1].section_path == "手册/配送"
    assert "条件说明" not in records[-1].text
    assert records[1].text.startswith("条件说明。条件说明。")
    uninterrupted = chunker.chunk_document("纯文本", "x" * 500, markdown=False)
    assert len(uninterrupted) > 5
    assert sum(len(r.text) for r in uninterrupted) > 500
    assert "##" in parse_sections("# 外部\n```\n## 代码\n```\n正文")[0].text
    setext = parse_sections("手册\n===\n\n政策\n---\n七天期限。\n\n## 到账\n五天到账。")
    assert [s.path for s in setext] == ["手册/政策", "手册/到账"]


def test_complete_qa_and_steps_and_independent_json_entries():
    chunker = BudgetChunker(len, 100, 0, reserve=4)
    body = "问：期限？\n答：七天。\n\n問"  # 保留问答正文
    records = chunker.chunk_document("FAQ", body)
    assert "问：期限？\n答：七天。" in records[0].text
    steps = chunker.chunk_document("操作", "1. 打开页面\n2. 选择订单\n3. 确认申请")
    assert len(steps) == 1
    assert chunker.chunk_document("一", "内容")[0].source_id != chunker.chunk_document("二", "内容")[0].source_id
    qa = "问：" + "问" * 30 + "\n答：" + "答" * 30
    with_overlap = BudgetChunker(len, 90, 20, reserve=4).chunk_document("FAQ", qa + "\n\n" + qa)
    assert len(with_overlap) == 2
    assert all(r.text == qa for r in with_overlap)


def test_duplicate_json_titles_remain_independent_documents(knowledge):
    kb, *_ = knowledge
    result = kb.import_documents([{"title": "政策", "content": "第一条"}, {"title": "政策", "content": "第二条"}])
    assert result["status"] == "success" and kb.stats()["documents"] == 2
    assert {item["content"] for item in kb.search("政策")} == {"第一条", "第二条"}


def test_bm25_uses_term_counts_not_vocabulary_size():
    index = BM25Index()
    index.add("a", "refund refund refund")
    index.add("b", "refund other")
    assert index.average_document_length == 2.5
    index.remove("a")
    assert index.average_document_length == 2


def test_dual_spaces_and_query_failover_then_recovery(knowledge):
    kb, embedding, backup, changed = knowledge
    assert kb.import_documents([{"title": "退款", "content": "七天内退款，超过七天需要质量证明。"}])["status"] == "success"
    normal = kb.search("退款")
    assert normal[0]["index_route"] == "main"
    assert all(r["chunk_id"].startswith("gemini:") for r in normal)
    assert {c[1] for c in embedding.calls} == {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}
    assert backup.inputs[0].startswith("退款\n")
    assert kb._main.get(include=["embeddings"])["embeddings"].shape[1] == 4
    assert kb._backup.get(include=["embeddings"])["embeddings"].shape[1] == 3
    embedding.fail = True
    fallback = kb.search("退款")
    assert fallback[0]["degraded"] and fallback[0]["index_route"] == "backup"
    assert all(r["chunk_id"].startswith("minilm:") for r in fallback)
    before = len(changed)
    embedding.fail = False
    assert kb.search("退款")[0]["index_route"] == "main"
    assert len(changed) > before
    assert kb.telemetry.snapshot()["counts"]["query:google_failed"] == 1


def test_import_update_stays_backup_until_all_repaired_and_deletes_stale_chunks(knowledge):
    kb, embedding, backup, changed = knowledge
    kb.import_documents([{"title": "退款", "content": "旧政策。" * 150}])
    old_count = kb._main.count()
    embedding.fail = True
    result = kb.import_documents([{"title": "退款", "content": "最新退款期限九天。"},
                                  {"title": "新物流", "content": "配送截止时间下午六点。"}])
    assert result["status"] == "degraded" and result["pending_main"] == 2
    embedding.fail = False
    result = kb.search("最新退款期限")
    assert result[0]["index_route"] == "backup"
    assert any("九天" in item["content"] for item in result)
    assert all("旧政策" not in item["content"] for item in result)
    assert kb.search("配送截止")[0]["index_route"] == "backup"
    before = len(changed)
    repair = kb.repair_main()
    assert repair["main_complete"] and repair["pending_main"] == 0
    assert len(changed) > before
    assert kb._main.count() == 2
    assert len(kb._main.get(where={"source_id": source_id_for("退款", "")}, include=[])["ids"]) == 1
    assert kb.search("配送截止")[0]["index_route"] == "main"
    source = next(row["source_id"] for row in kb._rows() if row["title"] == "退款")
    kb.delete_document(source)
    assert all(r["source_id"] != source for r in kb.search("退款"))
    assert kb._backup.count() == 1 and kb._main.count() == 1


def test_backup_failure_partial_and_original_retention(knowledge):
    kb, embedding, backup, _ = knowledge
    backup.fail = True
    result = kb.import_documents([{"title": "失败文档", "content": "完整原文保留"}])
    assert result["status"] == "failed" and result["processed_chunks"] == 0
    assert kb._rows()[0]["content"] == "完整原文保留"
    assert not kb.stats()["main_complete"]
    backup.fail = False
    assert kb.repair_main()["main_complete"]


def test_pending_state_survives_restart_and_gemini_tuning_keeps_backup_fixed(knowledge, tmp_path, monkeypatch):
    kb, embedding, backup, changed = knowledge
    embedding.fail = True
    kb.import_documents([{"title": "持久化政策", "content": "只在质量问题确认后退运费。" * 30}])
    backup_ids = kb._backup.get(include=[])["ids"]
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "80")
    restarted = KnowledgeBase(chroma_path=str(tmp_path), embedding=embedding, client=kb._client,
                              backup_function=backup, backup_counter=len, load_defaults=False,
                              collection_prefix=kb._backup.name.removesuffix("_minilm_v1"),
                              on_change=lambda: changed.append(True) or 0)
    assert restarted.stats()["pending_main"] == 1
    assert restarted.search("退运费")[0]["index_route"] == "backup"
    assert restarted._backup.get(include=[])["ids"] == backup_ids
    embedding.fail = False
    assert restarted.repair_main()["main_complete"]
    assert restarted._backup.get(include=[])["ids"] == backup_ids
    restarted.close()


def test_missing_collections_rebuild_backup_from_saved_originals(knowledge, tmp_path):
    kb, embedding, backup, _ = knowledge
    kb.import_documents([{"title": "保留原文", "content": "服务切换后仍可检索的规则"}])
    prefix = kb._backup.name.removesuffix("_minilm_v1")
    kb._client.delete_collection(kb._main.name)
    kb._client.delete_collection(kb._backup.name)
    restarted = KnowledgeBase(chroma_path=str(tmp_path), embedding=embedding, client=kb._client,
                              backup_function=backup, backup_counter=len, load_defaults=False,
                              collection_prefix=prefix)
    assert restarted.stats()["pending_main"] == 1
    assert restarted.stats()["pending_backup"] == 0
    assert restarted.search("服务切换")[0]["index_route"] == "backup"
    assert restarted.repair_main()["main_complete"]
    restarted.close()


def test_live_api_import_reports_degraded_and_repair_refreshes_cache(knowledge, monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as api
    kb, embedding, backup, changed = knowledge
    manager = MCPToolManager("test")
    monkeypatch.setattr(api, "_knowledge_base", kb)
    monkeypatch.setattr(api, "_tool_manager", manager)
    embedding.fail = True
    client = TestClient(api.app)
    response = client.post("/knowledge/add", json={"documents": [{"title": "API政策", "content": "新规则立即可查询"}]})
    assert response.status_code == 200 and response.json()["status"] == "degraded"
    assert "备用入库成功" in response.json()["message"]
    assert client.get("/knowledge/stats").json()["pending_main"] == 1
    embedding.fail = False
    before = len(changed)
    assert client.post("/knowledge/repair").json()["main_complete"]
    assert len(changed) > before
    source = kb._rows()[0]["source_id"]
    assert client.delete(f"/knowledge/documents/{source}").json()["documents"] == 0


def test_rerank_sees_tail_and_omitted_duplicate_indices_are_repaired(monkeypatch):
    async def run():
        manager = MCPToolManager("test")
        items = [{"chunk_id": str(i), "title": "政策", "content": "正文" * 180 + f"关键条件{i}"} for i in range(4)]
        prompts = []

        async def complete(prompt, temperature, identity, diagnostics=None):
            prompts.append(prompt)
            return "[2]"

        manager._complete_text = complete
        results = await manager._rerank("条件", items, 3)
        assert "关键条件3" in prompts[0]
        assert results[0]["chunk_id"] == "2"
        assert len(results) == 3 and len({r["chunk_id"] for r in results}) == 3
        monkeypatch.setenv("RAG_RERANK_MAX_CHARS", "1200")
        await manager._rerank("条件", items, 3)
        assert len(prompts[-1]) <= 1200

    asyncio.run(run())


def test_degraded_retrieval_is_real_data_and_is_not_cached():
    async def run():
        calls = []

        async def handler(params, context):
            calls.append(True)
            return [{"content": "真实备用知识", "degraded": len(calls) == 1}]

        manager = MCPToolManager("test")
        manager.register(Tool("knowledge_search", "test", handler, {}, cache_ttl=300))
        first = await manager.call("knowledge_search", {"query": "x"})
        second = await manager.call("knowledge_search", {"query": "x"})
        third = await manager.call("knowledge_search", {"query": "x"})
        assert first.data[0]["degraded"] and not second.cached and third.cached
        assert len(calls) == 2

    asyncio.run(run())


def test_qwen_configuration_factory_and_legacy_experiment_guard(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
    monkeypatch.setenv("EMBEDDING_API_KEY", "unit-qwen-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://provider.test/compatible-mode/v1/")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    settings = EmbeddingSettings.from_env()
    embedding = build_embedding(settings,transport=httpx.MockTransport(lambda _:httpx.Response(200,json={
        "data":[{"index":0,"embedding":[1.]+[0.]*767}],"usage":{"total_tokens":2}})))
    assert isinstance(embedding, QwenEmbedding)
    assert embedding.base_url == "https://provider.test/compatible-mode/v1"
    assert "unit-qwen-key" not in repr(settings) + embedding.identity
    assert embedding.count_tokens("中文") == 2
    assert embedding.count_tokens("中文") == 2
    assert embedding.usage["requests"] == 1 and embedding.usage["TOKEN_COUNT:texts"] == 1
    assert embedding.counting_mode == "qwen_single_input_usage_tokens"
    embedding.close()
    with pytest.raises(ValueError, match="Gemini 实验"):
        GeminiEmbedding(settings)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    with pytest.raises(ValueError, match="Qwen Flash"):
        EmbeddingSettings.from_env()


def test_qwen_documents_titles_index_order_and_roles():
    requests = []

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index":i,"embedding":[3.0,4.0]} for i in reversed(range(len(payload["input"])))],
                                        "usage":{"total_tokens":12}})

    embedding = QwenEmbedding(EmbeddingSettings("qwen3.7-text-embedding-flash", 2, "unit-key"),
                              transport=httpx.MockTransport(respond))
    assert embedding.embed(["正文一", "正文二"], "RETRIEVAL_DOCUMENT", ["标题一", "标题二"]) == [[.6,.8],[.6,.8]]
    embedding.embed(["查询"], "RETRIEVAL_QUERY")
    embedding.embed(["分类"], "CLASSIFICATION")
    payloads = [json.loads(request.content) for request in requests]
    assert payloads[0]["input"] == ["标题一\n正文一", "标题二\n正文二"]
    assert payloads[1]["input"] == ["查询"] and payloads[2]["input"] == ["分类"]
    assert all(request.url.path == "/compatible-mode/v1/embeddings" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer unit-key" for request in requests)
    assert all("taskType" not in payload and "encoding_format" in payload for payload in payloads)
    assert embedding.usage["reported_tokens"] == 36
    embedding.close()


def test_qwen_batch_size_and_total_input_budget():
    batches = []

    def respond(request):
        batch = json.loads(request.content)["input"]
        batches.append(batch)
        return httpx.Response(200, json={"data":[{"index":i,"embedding":[1.,0.]} for i in range(len(batch))]})

    embedding = QwenEmbedding(EmbeddingSettings("qwen3.7-text-embedding-flash", 2, "unit-key"),
                              transport=httpx.MockTransport(respond))
    assert len(embedding.embed(["短句"]*21, "CLASSIFICATION")) == 21
    assert list(map(len, batches)) == [20,1]
    batches.clear()
    assert len(embedding.embed(["a"*70000, "b"*70000], "CLASSIFICATION")) == 2
    assert list(map(len, batches)) == [1,1]
    assert all(sum(len(text.encode('utf-8'))+16 for text in batch)<=embedding.INPUT_LIMIT for batch in batches)
    embedding.close()


def test_qwen_exact_count_reuses_vector_and_requires_usage():
    calls=[]
    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json={"data":[{"index":0,"embedding":[3.,4.]}],"usage":{"total_tokens":7}})
    embedding=QwenEmbedding(EmbeddingSettings("qwen3.7-text-embedding-flash",2,"unit"),transport=httpx.MockTransport(respond))
    assert embedding.count_tokens("标题\n正文") == 7
    assert embedding.embed(["正文"],"RETRIEVAL_DOCUMENT",["标题"]) == [[.6,.8]]
    assert len(calls)==1 and embedding.usage["document_count_vector_reuse"]==1
    embedding.embed(["查询"],"RETRIEVAL_QUERY")
    assert len(calls)==2
    embedding.close()
    embedding=QwenEmbedding(EmbeddingSettings("qwen3.7-text-embedding-flash",2,"unit"),transport=httpx.MockTransport(
        lambda _:httpx.Response(200,json={"data":[{"index":0,"embedding":[1.,0.]}]})))
    with pytest.raises(EmbeddingError,match="token_usage_missing"):
        embedding.count_tokens("正文")
    embedding.close()


@pytest.mark.parametrize("data,reason", [
    ([{"index":0,"embedding":[1.,0.]}]*2,"unexpected_embedding_count"),
    ([{"index":0,"embedding":[1.]}],"unexpected_dimensions"),
    ([{"index":0,"embedding":[0.,0.]}],"invalid_vector"),
])
def test_qwen_rejects_unusable_embedding_responses(data, reason):
    embedding = QwenEmbedding(EmbeddingSettings("qwen3.7-text-embedding-flash",2,"unit-key"),
                              transport=httpx.MockTransport(lambda _:httpx.Response(200,json={"data":data})))
    with pytest.raises(EmbeddingError, match=reason):
        embedding.embed(["正文"],"CLASSIFICATION")
    embedding.close()


def test_model_switch_uses_new_collection_and_repairs_without_changing_backup(tmp_path):
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    old = StubEmbedding()
    backup = StubMiniLM()
    options = dict(chroma_path=str(tmp_path),client=client,backup_function=backup,backup_counter=len,
                   load_defaults=False,collection_prefix="qwen_migration")
    kb = KnowledgeBase(embedding=old,**options)
    kb.import_documents([{"title":"公开政策","source_id":"public-policy","content":"公开规则：七天内受理退款，需要订单号。"}])
    old_collection = kb._main
    old_name = old_collection.name
    other_dimension = client.create_collection(name="qwen_migration_gemini_001_256_v1")
    unrelated = client.create_collection(name="historical_eval_gemini_001_4_v1")
    unrelated.add(ids=["history"],documents=["历史实验文本"],embeddings=[[1.,0.,0.,0.]])
    unrelated_rows = unrelated.get(include=["documents"])
    backup_rows = kb._backup.get(include=["documents"])
    source_rows = [dict(row) for row in kb._rows()]
    kb.close()

    new = StubEmbedding()
    new.settings = SimpleNamespace(model="qwen3.7-text-embedding-flash",dimensions=4)
    new.identity = "qwen:test:4:new-provider"
    new.telemetry = EmbeddingTelemetry(new.settings.model)
    kb = KnowledgeBase(embedding=new,**options)
    assert kb._main.name != old_name and kb._main.count() == 0
    names = {collection.name for collection in client.list_collections()}
    assert old_name not in names and other_dimension.name not in names
    assert unrelated.get(include=["documents"]) == unrelated_rows
    assert [(row["source_id"],row["title"],row["content"]) for row in kb._rows()] == [
        (row["source_id"],row["title"],row["content"]) for row in source_rows]
    assert kb.telemetry.snapshot()["counts"]["cleanup:removed"] == 2
    assert kb.stats()["pending_main"] == 1
    assert kb.search("退款")[0]["index_route"] == "backup"
    assert kb._backup.get(include=["documents"]) == backup_rows
    assert kb.repair_main()["status"] == "success"
    result = kb.search("退款")
    assert result[0]["index_route"] == "main" and result[0]["chunk_id"].startswith("qwen:")
    assert all(item["embedding_model"] == new.settings.model for item in result)
    assert kb._backup.get(include=["documents"]) == backup_rows
    new.fail = True
    assert kb.search("退款")[0]["index_route"] == "backup"
    assert kb.telemetry.snapshot()["counts"]["query:provider_failed"] == 1
    kb.close()
    new.fail = False
    kb = KnowledgeBase(embedding=new,**options)
    assert kb.stats()["pending_main"] == 0 and kb.search("退款")[0]["index_route"] == "main"
    assert kb.telemetry.snapshot()["counts"]["cleanup:removed"] == 2
    assert kb._backup.get(include=["documents"]) == backup_rows
    kb.close()


def test_qwen_query_budget_returns_backup_body_through_tool(knowledge):
    kb, old, *_ = knowledge
    kb.import_documents([{"title":"公开超时政策","content":"退款期限九天，必须提供订单号。"}])

    async def delayed(request):
        await asyncio.sleep(.8)
        return httpx.Response(200,json={"data":[{"index":0,"embedding":[1.,0.,0.,0.]}]})

    qwen = QwenEmbedding(EmbeddingSettings("qwen3.7-text-embedding-flash",4,"unit-key"),
                         transport=httpx.MockTransport(delayed))
    kb.embedding, kb.telemetry, kb._failure_outcome = qwen,qwen.telemetry,"provider_failed"
    kb.query_budget_s = .4
    manager = MCPToolManager("unit")
    manager.register(Tool("knowledge_search","knowledge",kb.search_handler,{},timeout_s=2))
    result = asyncio.run(manager.call("knowledge_search",{"query":"退款期限"}))
    assert result.success and result.degraded and not result.error
    assert any("九天" in item["content"] for item in result.data)
    qwen.close()


def manager_for(kb):
    manager = MCPToolManager("test")
    manager.register(Tool("knowledge_search", "test", kb.search_handler, {}, timeout_s=1, cache_ttl=300))
    kb.on_change = manager.invalidate_cache
    return manager


def test_google_count_and_encode_share_budget_through_tool_and_agent(knowledge):
    kb, _, _, _ = knowledge
    kb.import_documents([{"title": "退款", "content": "设备质量退款必须提供照片，审核通过后原路退回。"}])
    calls = []
    async def respond(request):
        calls.append(request.url.path)
        if request.url.path.endswith(":countTokens"):
            return httpx.Response(200, json={"totalTokens": 5})
        await asyncio.Event().wait()

    google = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 4, "unit"),
                             transport=httpx.MockTransport(respond))
    try:
        google._count_supported = True
        kb.embedding = google
        kb.query_budget_s = .05
        async def run():
            from agents.tools import build_shared_rag_tools
            manager = manager_for(kb)
            started = time.monotonic()
            result = await manager.call("knowledge_search", {"query": "质量退款"})
            assert time.monotonic() - started < .9
            assert result.success and result.degraded
            assert "提供照片" in result.data[0]["content"]
            assert result.data[0]["index_route"] == "backup"
            async def rewrite(query, n=3):
                return [query]
            async def rerank(query, items, top_k):
                return items[:top_k]
            manager.rewrite_query, manager._rerank = rewrite, rerank
            req = SimpleNamespace(message="设备质量退款", tool_result_cache={})
            payload = await build_shared_rag_tools(manager)["search_knowledge_base"].handler(req, {"query": req.message, "top_k": 3})
            assert payload["success"] and payload["degraded"]
            assert "提供照片" in payload["results"][0]["content"]
            assert manager._tools["knowledge_search"].stats.failed == 0
        asyncio.run(run())
        assert any(path.endswith(":countTokens") for path in calls)
        assert any(path.endswith(":embedContent") for path in calls)
    finally:
        google.close()


@pytest.mark.parametrize("operation", ["import", "query", "repair"])
def test_stats_and_health_return_before_remote_release(knowledge, monkeypatch, operation):
    from api import main
    kb, embedding, _, _ = knowledge
    kb.import_documents([{"title": "政策", "content": "已提交快照"}])
    entered, release = threading.Event(), threading.Event()
    original = embedding.embed
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    embedding.embed = blocked
    monkeypatch.setattr(main, "_orchestrator", SimpleNamespace(get_stats=lambda: {}))
    monkeypatch.setattr(main, "_knowledge_base", kb)
    monkeypatch.setattr(main, "_monitor", SimpleNamespace(summary=lambda: {"knowledge": kb.stats()}))
    if operation == "repair":
        kb._db.execute("UPDATE sources SET main_ready=0")
        kb._db.commit()
    async def run():
        fn = (lambda: kb.import_documents([{"title": "新政策", "content": "新快照"}])) if operation == "import" else (lambda: kb.search("政策")) if operation == "query" else kb.repair_main
        pending = asyncio.create_task(asyncio.to_thread(fn))
        assert await asyncio.to_thread(entered.wait, 1)
        ticks = []
        async def heartbeat():
            for _ in range(5):
                await asyncio.sleep(.005)
                ticks.append(True)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                responses = await asyncio.wait_for(asyncio.gather(client.get("/monitor"), client.get("/knowledge/stats"), client.get("/health"), heartbeat()), .7)
                assert all(r.status_code == 200 for r in responses[:3])
                assert responses[1].json()["documents"] == 1
                assert len(ticks) == 5 and not release.is_set()
        finally:
            release.set()
            await pending
    asyncio.run(run())


def test_parallel_queries_and_update_do_not_publish_stale_cache(knowledge):
    kb, embedding, _, _ = knowledge
    kb.import_documents([{"title": "政策", "source_id": "policy", "content": "旧政策七天退款"}])
    entered, release = threading.Event(), threading.Event()
    count, lock = 0, threading.Lock()
    original = embedding.embed
    def blocked(texts, task, titles=None):
        nonlocal count
        if task == "RETRIEVAL_QUERY":
            with lock:
                count += 1
                if count >= 2:
                    entered.set()
            assert release.wait(3)
        return original(texts, task, titles)
    embedding.embed = blocked
    async def run():
        manager = manager_for(kb)
        pending = [asyncio.create_task(manager.call("knowledge_search", {"query": f"政策{i}"})) for i in range(3)]
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.to_thread(kb.import_documents, [{"title": "政策", "source_id": "policy", "content": "新政策三天退款"}])
        finally:
            release.set()
        await asyncio.gather(*pending)
        assert manager._get_cache("knowledge_search", {"query": "政策0"}) is None
        current = await manager.call("knowledge_search", {"query": "政策0"})
        assert "三天" in current.data[0]["content"]
        await asyncio.to_thread(kb.delete_document, "policy")
        assert not (await manager.call("knowledge_search", {"query": "政策0"})).data
    asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["add", "upload"])
def test_import_responses_belong_to_the_request(monkeypatch, endpoint):
    from api import main
    a_done, b_done = threading.Event(), threading.Event()
    class Imports:
        last_import = {}
        doc_count = 9
        def import_documents(self, docs):
            name = docs[0]["title"]
            outcome = {"status": "success" if name == "A" else "degraded", "degraded": name == "B",
                       "processed_chunks": 2 if name == "A" else 7, "documents": [{"source_id": name}]}
            self.last_import = outcome
            (a_done if name == "A" else b_done).set()
            return outcome
    def invalidate():
        if not b_done.is_set():
            assert b_done.wait(2)
        return 0
    monkeypatch.setattr(main, "_knowledge_base", Imports())
    monkeypatch.setattr(main, "_tool_manager", SimpleNamespace(invalidate_cache=invalidate))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            async def call(name):
                if endpoint == "add":
                    return await client.post("/knowledge/add", json={"documents": [{"title": name, "content": "正文"}]})
                return await client.post("/knowledge/upload", files={"file": (f"{name}.txt", "正文".encode())})
            a = asyncio.create_task(call("A"))
            assert await asyncio.to_thread(a_done.wait, 1)
            b = await call("B")
            a = await a
            assert a.json()["documents"] == [{"source_id": "A"}]
            assert a.json()["processed_chunks"] == 2 and a.json()["status"] == "success"
            assert b.json()["documents"] == [{"source_id": "B"}]
            assert b.json()["processed_chunks"] == 7 and b.json()["status"] == "degraded"
            assert "待补齐" in b.json()["message"] and "待补齐" not in a.json()["message"]
    asyncio.run(run())




def test_tool_immediate_failure_recovery_pending_and_double_failure(knowledge):
    kb, google, mini, _ = knowledge
    kb.import_documents([{"source_id":"policy", "title":"政策", "content":"七天受理"}])
    async def run():
        manager = manager_for(kb)
        google.fail = True
        fallback = await manager.call("knowledge_search", {"query":"受理"})
        assert fallback.success and fallback.degraded and "七天" in fallback.data[0]["content"]
        google.fail = False
        main = await manager.call("knowledge_search", {"query":"受理"})
        assert main.success and not main.cached and main.data[0]["index_route"]=="main"
        google.fail = True
        imported = await asyncio.to_thread(kb.import_documents, [{"source_id":"policy", "title":"政策", "content":"九天受理"}])
        assert imported["status"]=="degraded"
        google.fail = False
        pending = await manager.call("knowledge_search", {"query":"受理"})
        assert pending.degraded and "九天" in pending.data[0]["content"]
        await asyncio.to_thread(kb.repair_main)
        recovered = await manager.call("knowledge_search", {"query":"受理"})
        assert recovered.data[0]["index_route"]=="main" and "九天" in recovered.data[0]["content"]
        google.fail = mini.fail = True
        failed = await manager.call("knowledge_search", {"query":"另一个问题"})
        assert not failed.success and failed.data is None and "backup unavailable" in failed.error
        assert kb.telemetry.snapshot()["counts"]["query:failed"]>=1
    asyncio.run(run())


def test_query_budget_does_not_wait_for_slow_dns_executor(knowledge, monkeypatch):
    import socket
    kb, _, _, _ = knowledge
    kb.import_documents([{"title":"退款", "content":"质量退款需要提供照片"}])
    async def resolving_transport(request):
        await asyncio.get_running_loop().getaddrinfo("localhost", 1)
        return httpx.Response(200, json={"totalTokens": 5})
    google=GeminiEmbedding(EmbeddingSettings("gemini-embedding-001",4,"unit"),
                           transport=httpx.MockTransport(resolving_transport))
    google.ENDPOINT="http://localhost:1"
    google._count_supported=True
    kb.embedding=google
    kb.query_budget_s=1.5
    entered,release=threading.Event(),threading.Event()
    real_resolve=socket.getaddrinfo
    def resolve(*args,**kwargs):
        entered.set()
        assert release.wait(3)
        return real_resolve(*args,**kwargs)
    monkeypatch.setattr(socket,"getaddrinfo",resolve)
    async def run():
        manager=manager_for(kb)
        manager._tools["knowledge_search"].timeout_s=3
        try:
            result=await manager.call("knowledge_search",{"query":"质量退款"})
            assert entered.is_set() and not release.is_set()
            assert result.success and result.degraded
            assert "照片" in result.data[0]["content"]
        finally:
            release.set()
    asyncio.run(run())
    google.close()


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
        def get_generation(self, namespace):
            raise ConnectionError("redis unavailable")

        def get(self, namespace, key):
            raise ConnectionError("redis unavailable")

        def set(self, namespace, key, value, ttl, *, generation=None):
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

def test_prepared_collection_opens_without_models_or_index_writes(knowledge, tmp_path, monkeypatch):
    kb, embedding, backup, changed = knowledge
    kb.import_documents([{'source_id':'policy', 'title':'政策', 'content':'退款在审核通过后处理。', 'format':'md'}])
    prefix=kb._backup.name.removesuffix('_minilm_v1')
    db=tmp_path/f'{prefix}_sources.sqlite3'
    before=db.read_bytes()
    from mcp import indexed_knowledge_base as module
    monkeypatch.setattr(module,'prepare_minilm',lambda *a,**k: pytest.fail('must not load or download a model'))
    opened=KnowledgeBase(chroma_path=str(tmp_path),embedding=embedding,client=kb._client,
                         collection_prefix=prefix,prepared_only=True)
    assert opened.stats()['main_complete'] and opened.search('退款')
    assert db.read_bytes()==before
    embedding.fail=True
    with pytest.raises(Exception): opened.search('退款')
    opened.close()
