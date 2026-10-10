"""Qwen HTTP 契约与模型替换集成；stub 不用于质量评价。"""
import asyncio
import json
from types import SimpleNamespace

import chromadb
import httpx
import pytest

from core.embedding import EmbeddingError, EmbeddingSettings, EmbeddingTelemetry, GeminiEmbedding, QwenEmbedding, build_embedding
from mcp.indexed_knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool
from tests.test_gemini_rag import StubEmbedding, StubMiniLM, knowledge


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
