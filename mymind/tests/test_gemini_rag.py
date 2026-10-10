"""模型 HTTP 契约与双索引故障恢复测试。向量 stub 仅用于单元测试。"""
import asyncio
import json
from types import SimpleNamespace

import chromadb
import httpx
import pytest
from pathlib import Path

from core.document_chunking import BudgetChunker, parse_sections
from core.embedding import EmbeddingError, EmbeddingSettings, EmbeddingTelemetry, GeminiEmbedding
from core.intent_recognizer import IntentCategory, IntentRecognizer
from core.retrieval import BM25Index, source_id_for
from mcp.indexed_knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool


class StubEmbedding:
    INPUT_LIMIT = 2048

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.settings = SimpleNamespace(model="gemini-embedding-001" if enabled else "", dimensions=4)
        self.telemetry = EmbeddingTelemetry()
        self.identity = "test:4"
        self.counting_mode = "test_characters"
        self.calls = []
        self.fail = False

    def count_tokens(self, text):
        return len(text)

    def embed(self, texts, task, titles=None):
        self.calls.append((list(texts), task, titles))
        if self.fail:
            raise EmbeddingError("HTTP_503")
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


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


@pytest.mark.parametrize("filename", ["gemini_rag_eval.json", "gemini_rag_boundary_eval.json"])
def test_long_evaluation_facts_are_anchored_in_source_and_splits_are_disjoint(filename):
    path = Path(__file__).resolve().parents[1] / "data/eval" / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    assert all(len(doc["content"]) > 1000 for doc in data["corpus"])
    corpus = "\n".join(doc["content"] for doc in data["corpus"])
    assert all(fact in corpus for q in data["questions"] for fact in q["evidence"])
    dev = {q["question"] for q in data["questions"] if q["split"] == "dev"}
    test = {q["question"] for q in data["questions"] if q["split"] == "test"}
    assert not dev.intersection(test)


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


def test_intent_weights_failure_recovery_parallel_and_template_cache():
    async def run():
        embedding = StubEmbedding()
        recognizer = IntentRecognizer("test", base_url="https://chat-only.test", embedding=embedding)
        seen = []

        async def llm(message, history):
            seen.append("llm")
            await asyncio.sleep(0)
            return {"intent": IntentCategory.GREETING, "confidence": 0.8}

        async def emb(message):
            seen.append("embedding")
            await asyncio.sleep(0)
            return {"intent": IntentCategory.GREETING, "confidence": 0.9}

        recognizer._llm_recognize = llm
        recognizer._embedding_recognize = emb
        recognizer._pattern_recognize = lambda _: {"intent": IntentCategory.GREETING, "confidence": 0.5}
        result = await recognizer.recognize("test")
        assert seen == ["llm", "embedding"]
        assert result.confidence == pytest.approx(0.7 * .8 + .2 * .9 + .1 * .5)

        async def failed(message):
            return {"intent": IntentCategory.OTHER, "confidence": 0, "failed": True}

        recognizer._embedding_recognize = failed
        result = await recognizer.recognize("failure")
        assert result.confidence == pytest.approx(.85 * .8 + .15 * .5)
        recognizer._embedding_recognize = emb
        assert (await recognizer.recognize("failure")).confidence == pytest.approx(.79)
        await recognizer._load_template_embeddings()
        calls = len(embedding.calls)
        await recognizer._load_template_embeddings()
        assert len(embedding.calls) == calls
        assert all(call[1] == "CLASSIFICATION" for call in embedding.calls)
        recognizer._embedding_enabled = False
        assert (await recognizer.recognize("no-vector")).confidence == pytest.approx(.755)

    asyncio.run(run())


def test_actual_intent_failure_does_not_call_old_hash_and_recovers(monkeypatch):
    async def run():
        embedding = StubEmbedding()
        recognizer = IntentRecognizer("test", embedding=embedding)
        monkeypatch.setattr(recognizer, "_local_embedding", lambda _: pytest.fail("production used old hash"))
        embedding.fail = True
        assert (await recognizer._embedding_recognize("hello"))["failed"]
        embedding.fail = False
        assert not (await recognizer._embedding_recognize("hello")).get("failed")
        assert embedding.telemetry.snapshot()["counts"]["intent:recovered"] == 1

    asyncio.run(run())


def test_rerank_sees_tail_and_omitted_duplicate_indices_are_repaired(monkeypatch):
    async def run():
        manager = MCPToolManager("test")
        items = [{"chunk_id": str(i), "title": "政策", "content": "正文" * 180 + f"关键条件{i}"} for i in range(4)]
        prompts = []

        async def complete(prompt, temperature, identity):
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
