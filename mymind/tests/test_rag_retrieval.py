import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from core.retrieval import (
    BM25Index,
    Chunker,
    RetrievalPipeline,
    dedupe_items,
    deterministic_rank,
    parse_rerank_indices,
    stable_chunk_id,
    variant_config,
    weighted_rrf,
)
from mcp.knowledge_base import KnowledgeBase
from tests.fakes import FakeCollection


def test_variant_matrix_isolates_each_toggle():
    configs = {
        name: variant_config(name)
        for name in ("r0", "r1", "r2", "r3", "r4", "r4-no-rewrite", "r4-no-rerank", "r4-no-bm25", "r4-no-overlap")
    }
    assert configs["r0"].stable_ids is False
    assert configs["r1"].stable_ids is True and configs["r1"].cosine is True
    assert configs["r2"].overlap == 80 and configs["r2"].chunk_mode == "markdown"
    assert configs["r3"].hybrid is True and configs["r3"].llm_rerank is False
    assert configs["r4"].rewrite is True and configs["r4"].llm_rerank is True
    assert configs["r4-no-rewrite"].rewrite is False and configs["r4-no-rewrite"].hybrid is True
    assert configs["r4-no-rerank"].llm_rerank is False
    assert configs["r4-no-bm25"].hybrid is False
    assert configs["r4-no-overlap"].overlap == 0
    assert len({config.collection_name for config in configs.values()}) == len(configs)


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
    assert deterministic_rank(items, 3) == deterministic_rank(items, 3)
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
    knowledge = KnowledgeBase.__new__(KnowledgeBase)
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


def test_search_endpoint_keeps_contract_and_reports_top_k():
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
    old_manager, old_kb = api._tool_manager, api._knowledge_base
    api._tool_manager = manager
    api._knowledge_base = SimpleNamespace(index_version="rag-index-v1", chunk_config={"chunk_size": 500})
    try:
        response = TestClient(api.app).post("/search", params={"query": "退款", "top_k": 7})
    finally:
        api._tool_manager, api._knowledge_base = old_manager, old_kb
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "退款"
    assert payload["reranked"] is True
    assert payload["requested_top_k"] == 7
    assert payload["returned"] == 1
    assert payload["results"][0]["chunk_id"] == "chk-1"
    assert manager.calls[-1][2] == 7


def test_knowledge_add_and_upload_keep_legacy_contract():
    import api.main as api

    class Knowledge:
        def __init__(self):
            self.count = 10

        def add_documents(self, documents):
            added = sum(1 for doc in documents if doc.get("content"))
            self.count += added
            return added

        @property
        def doc_count(self):
            return self.count

    class Manager:
        def invalidate_cache(self):
            return 1

    old_kb, old_manager = api._knowledge_base, api._tool_manager
    api._knowledge_base = Knowledge()
    api._tool_manager = Manager()
    try:
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
    finally:
        api._knowledge_base, api._tool_manager = old_kb, old_manager


def test_chat_endpoint_keeps_legacy_and_diagnostic_fields():
    import api.main as api
    from agents.agent_orchestrator import AgentType
    from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel

    class Orchestrator:
        def get_stats(self):
            return {"agents": 3}

        async def recognize_intent(self, message, history=None):
            return IntentResult(
                intent=IntentCategory.REFUND,
                confidence=0.9,
                urgency=UrgencyLevel.LOW,
                intent_group="billing",
                entities={"order_id": ["#123"]},
                reasoning="refund",
                latency_ms=2.0,
                source_scores={"pattern": 0.9},
            )

        async def run(self, request):
            return SimpleNamespace(
                response="已为您登记退款，款项将在 5-7 个工作日退回。",
                intent=IntentCategory.REFUND,
                agent_type=AgentType.BILLING,
                agent_types=[AgentType.BILLING],
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                routing_reason="intent=refund",
                routing_confidence=0.9,
                escalated=False,
                latency_ms=3.0,
            )

    class Memory:
        async def get_context(self, user_id, conv_id, query=None):
            return SimpleNamespace(recent_messages=[])

        async def add_message(self, *args):
            return None

        async def update_profile(self, *args):
            return None

    async def build_knowledge_context(message, intent, top_k=3):
        return api.KnowledgeContextResult(
            text="[知识库检索结果]",
            used=True,
            status="used",
            reason="intent:refund",
        )

    old_orch = api._orchestrator
    old_memory = api._memory
    old_builder = api._context_builder
    old_build = api._build_knowledge_context
    api._orchestrator = Orchestrator()
    api._memory = Memory()
    api._context_builder = SimpleNamespace(build=lambda *args: SimpleNamespace(text="ctx", metadata={}))
    api._build_knowledge_context = build_knowledge_context
    try:
        response = TestClient(api.app).post("/chat", json={"message": "我要退款"})
    finally:
        api._orchestrator = old_orch
        api._memory = old_memory
        api._context_builder = old_builder
        api._build_knowledge_context = old_build
    assert response.status_code == 200
    payload = response.json()
    for field in ("conv_id", "response", "intent", "agent_type", "escalated", "latency_ms",
                  "knowledge_used", "knowledge_status", "knowledge_reason",
                  "intent_group", "entities", "agent_types", "primary_agent", "supporting_agents"):
        assert field in payload, field
    assert payload["knowledge_used"] is True
    assert payload["knowledge_status"] == "used"


def test_stats_endpoint_total_chunks_is_numeric():
    import api.main as api

    old_kb = api._knowledge_base
    api._knowledge_base = SimpleNamespace(
        doc_count=12,
        index_version="rag-index-v1",
        chunk_config={"chunk_size": 500},
    )
    try:
        response = TestClient(api.app).get("/knowledge/stats")
    finally:
        api._knowledge_base = old_kb
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_chunks"] == 12
    assert isinstance(payload["total_chunks"], int)


def test_dataset_has_120_plus_queries_and_partitions():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "data" / "eval" / "rag_dataset.json").read_text(encoding="utf-8"))
    assert len(dataset) >= 120
    categories = {row["category"] for row in dataset}
    assert {"refund", "logistics", "payment", "account", "subscription", "api", "error_codes", "no_answer"} <= categories
    partitions = {}
    for row in dataset:
        partitions.setdefault(row["partition"], []).append(row)
    assert all(len(rows) > 1 for rows in partitions.values())
    no_answer = [row for row in dataset if row["no_answer"]]
    assert all(row["relevant_source_ids"] == [] for row in no_answer)
