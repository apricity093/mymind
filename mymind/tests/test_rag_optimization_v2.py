"""第二轮预算、并发、缓存和单路隔离；stub 只作为工程证据。"""
import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from tests.test_gemini_rag import knowledge, StubEmbedding
from core.embedding import GeminiEmbedding, EmbeddingSettings
from core.intent_recognizer import IntentRecognizer, IntentCategory
from mcp.tool_manager import MCPToolManager, Tool


def manager_for(kb):
    manager = MCPToolManager("test")
    manager.register(Tool("knowledge_search", "test", kb.search_handler, {}, timeout_s=1, cache_ttl=300))
    kb.on_change = manager.invalidate_cache
    return manager


@contextmanager
def slow_google():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append(self.path)
            time.sleep(.13 if self.path.endswith(":countTokens") else .6)
            result = {"totalTokens": 5} if self.path.endswith(":countTokens") else {"embedding": {"values": [1, 0, 0, 0]}}
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_google_count_and_encode_share_budget_through_tool_and_agent(knowledge):
    kb, _, _, _ = knowledge
    kb.import_documents([{"title": "退款", "content": "设备质量退款必须提供照片，审核通过后原路退回。"}])
    with slow_google() as (url, calls):
        google = GeminiEmbedding(EmbeddingSettings("gemini-embedding-001", 4, "unit"))
        google.ENDPOINT = url
        google._count_supported = True
        kb.embedding = google
        kb.query_budget_s = .5
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
            payload = await build_shared_rag_tools(manager)["search_knowledge_base"].handler(req, {})
            assert payload["success"] and payload["degraded"]
            assert "提供照片" in payload["results"][0]["content"]
            assert manager._tools["knowledge_search"].stats.failed == 0
        asyncio.run(run())
        assert any(path.endswith(":countTokens") for path in calls)
        assert any(path.endswith(":embedContent") for path in calls)
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


def test_bm25_experiment_never_embeds():
    from experiments.gemini_rag import retrieval_experiments
    class NeverEmbed(StubEmbedding):
        def embed(self, *args, **kwargs):
            raise AssertionError("BM25 must not embed")
    def mini(*args):
        raise AssertionError("BM25 must not load vectors")
    args = SimpleNamespace(models="both", variants=["gemini-structure-bm25", "minilm-structure-bm25"],
                           chunk_budget=512, top_k=5, answers=False)
    result = asyncio.run(retrieval_experiments([{"title": "退款", "content": "七天退款"}],
        [{"id": "one", "question": "退款", "evidence": ["七天退款"]}], NeverEmbed(), mini, len, args, None, None))
    assert all(row["status"] == "measured" for row in result.values())


def test_template_cache_restart_invalidation_and_failure_recovery(tmp_path):
    path = str(tmp_path / "templates.json")
    async def run():
        first = StubEmbedding()
        recognizer = IntentRecognizer("test", embedding=first, template_cache_path=path)
        await asyncio.gather(*[recognizer._load_template_embeddings() for _ in range(3)])
        assert len(first.calls) == 1
        second = StubEmbedding()
        restarted = IntentRecognizer("test", embedding=second, template_cache_path=path)
        await restarted._load_template_embeddings()
        assert second.calls == []
        restarted.learn("新增开发表达", IntentCategory.REFUND)
        await restarted._load_template_embeddings()
        assert len(second.calls) == 1
        second.fail = True
        assert (await restarted._embedding_recognize("测试"))["failed"]
        second.fail = False
        assert not (await restarted._embedding_recognize("测试")).get("failed")
        Path = __import__("pathlib").Path
        Path(path).write_text("invalid", encoding="utf-8")
        third = StubEmbedding()
        rebuilt = IntentRecognizer("test", embedding=third, template_cache_path=path)
        await rebuilt._load_template_embeddings()
        assert len(third.calls) == 1
        for instance in [recognizer, restarted, rebuilt]:
            await instance.close()
    asyncio.run(run())


def test_template_preparation_has_bounded_wait_and_shared_task(tmp_path):
    embedding = StubEmbedding()
    release = threading.Event()
    original = embedding.embed
    def blocked(*args, **kwargs):
        assert release.wait(3)
        return original(*args, **kwargs)
    embedding.embed = blocked
    async def run():
        recognizer = IntentRecognizer("test", embedding=embedding, template_cache_path=str(tmp_path / "cache"), template_wait_s=.03)
        results = await asyncio.gather(*[recognizer._embedding_recognize("测试") for _ in range(3)])
        assert all(r["failed"] for r in results)
        release.set()
        await recognizer._template_task
        assert len(embedding.calls) == 1
        assert not (await recognizer._embedding_recognize("测试")).get("failed")
        await recognizer.close()
    asyncio.run(run())


def test_saved_chunks_allow_bm25_with_all_google_interfaces_unavailable(tmp_path, monkeypatch):
    from experiments import optimization_v2 as exp
    from pathlib import Path
    monkeypatch.setattr(exp, "chat_gateway", lambda: (None, None))
    args = SimpleNamespace(output=tmp_path, api_interval=0,
        rag_data=exp.ROOT/"data/eval/rag_optimization_v2.json", intent_data=exp.ROOT/"data/eval/intent_optimization_v2.json")
    runner = exp.Runner(args)
    # 用保存的同一正式分块契约；此用例仅验证隔离，不作为质量实验。
    q = next(q for q in runner.rag["questions"] if q["split"]=="dev" and q["answerable"])
    sid = q["evidence_groups"][0]["alternatives"][0]["source_id"]
    doc = runner.by_source[sid]
    items = [{"chunk_id":f"test:{i}", "source_id":sid, "title":doc["title"], "section_path":"", "content":doc["content"]} for i in range(50)]
    exp.write(tmp_path/"chunks/gemini-structure-512-0.json", {"data_version":runner.rag["version"], "corpus":runner.corpus,"items":items})
    def forbidden(*args, **kwargs):
        raise AssertionError("BM25 with saved chunks must not contact Google")
    runner.google.count_tokens = runner.google.cached_embed = runner.google._request = forbidden
    result = runner.retrieval("gemini", "structure", 512, 0, "dev", ["bm25"], 1)
    assert len(result["bm25"])==1
    assert runner.google.ledger["google_http"]==0
    assert runner.google.ledger["google_encoded_texts"]==0
    runner.google.close()


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
