"""公开合成数据的真实 Qwen/Chroma/工具与模板缓存接入验证。"""
import argparse
import asyncio
import json
import statistics
import time
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import chromadb
from dotenv import load_dotenv

from core.embedding import EmbeddingError, build_embedding
from core.cache_store import RedisCacheStore
from core.intent_recognizer import IntentRecognizer
from mcp.indexed_knowledge_base import KnowledgeBase, prepare_minilm
from mcp.tool_manager import MCPToolManager, Tool


ROOT = Path(__file__).resolve().parents[1]


async def run(output, chroma_host=None, chroma_port=8001, redis_db=None):
    load_dotenv(ROOT/".env")
    embedding = build_embedding()
    if embedding.settings.model != "qwen3.7-text-embedding-flash":
        raise RuntimeError("qwen_configuration_required")
    output.mkdir(parents=True, exist_ok=True)
    settings = chromadb.Settings(anonymized_telemetry=False)
    if chroma_host:
        client = chromadb.HttpClient(host=chroma_host,port=chroma_port,settings=settings)
        client.heartbeat()
    else:
        client = chromadb.PersistentClient(path=str(output/"isolated-index"),settings=settings)
    prefix = f"qwen_http_smoke_{uuid.uuid4().hex[:8]}" if chroma_host else "qwen_public_smoke"
    existing = {c.name:client.get_collection(c.name).count() for c in client.list_collections()} if chroma_host else {}
    legacy = [f"{prefix}_gemini_001_{dimension}_v1" for dimension in (768,256)] if chroma_host else []
    for name in legacy:
        client.create_collection(name=name,embedding_function=None)
    cache_client,cache_prefix,store = None,None,None
    if redis_db is not None:
        if redis_db <= 0:
            raise ValueError("smoke 必须使用非生产 Redis 库")
        import redis
        parts = urlsplit(os.getenv("REDIS_URL","redis://:mymind123@localhost:6379/0"))
        cache_client = redis.from_url(urlunsplit(parts._replace(path=f"/{redis_db}")),decode_responses=True)
        cache_client.ping()
        cache_prefix = f"mymind:experiment:qwen:{uuid.uuid4().hex[:8]}"
        store = RedisCacheStore(cache_client,cache_prefix)
    backup, counter = prepare_minilm()
    manager = MCPToolManager("connectivity-smoke",cache_store=store)
    options = dict(chroma_path=str(output/"isolated-index"),client=client,embedding=embedding,
                   backup_function=backup,backup_counter=counter,load_defaults=False,
                   collection_prefix=prefix,on_change=manager.invalidate_cache)
    kb = KnowledgeBase(**options)
    manager.register(Tool("knowledge_search","knowledge",kb.search_handler,{},cache_ttl=300))
    docs = [
        {"source_id":"public-refund","title":"公开样例退款政策","content":"# 退款\n仅限官网零售订单，七天内可申请退款，需要原订单申请人提供订单号和付款凭证。"},
        {"source_id":"public-logistics","title":"公开样例配送政策","content":"# 配送\n配送查询需要运单号；仓库每天下午六点停止发货。"},
        {"source_id":"public-invoice","title":"公开样例发票政策","content":"# 发票\n电子发票申请需要订单号、发票抬头和税号。"},
    ]
    result = {"model":embedding.settings.model,"dimensions":embedding.settings.dimensions,
              "base_url":embedding.base_url,"scope":"公开合成数据、真实向量和MiniLM；无聊天LLM调用，非质量对比",
              "status":"running","checks":{},"queries":[],"storage":"http" if chroma_host else "local",
              "cache":"redis" if cache_client else "memory","redis_db":redis_db}
    checks = result["checks"]
    if chroma_host:
        names = {c.name for c in client.list_collections()}
        checks["http_legacy_cleanup"] = not set(legacy)&names
        result["legacy_cleanup"] = {"scope":"controlled_empty_experiment_collections","removed":legacy}
    peer = None
    if cache_client:
        peer = MCPToolManager("connectivity-peer",cache_store=RedisCacheStore(cache_client,cache_prefix))
        peer.register(Tool("knowledge_search","knowledge",kb.search_handler,{},cache_ttl=300))
    original = embedding.embed
    recognizers = []
    try:
        imported = await asyncio.to_thread(kb.import_documents,docs)
        checks["real_qwen_import"] = imported["status"] == "success"
        for query,source in [("退款需要什么材料？","public-refund"),("仓库几点停止发货？","public-logistics"),("电子发票要提供哪些信息？","public-invoice")]:
            value = await manager.call("knowledge_search",{"query":query,"top_k":3})
            result["queries"].append({"query":query,"success":value.success,"cached":value.cached,"results":value.data})
            checks[source] = value.success and any(item["source_id"]==source and item["index_route"]=="main" for item in value.data)
        hit = await manager.call("knowledge_search",{"query":"退款需要什么材料？","top_k":3})
        checks["main_tool_cache_hit"] = hit.success and hit.cached
        if peer:
            peer_hit = await peer.call("knowledge_search",{"query":"退款需要什么材料？","top_k":3})
            checks["redis_cross_worker_cache_hit"] = peer_hit.success and peer_hit.cached

        def failure(*args,**kwargs):
            raise EmbeddingError("injected_provider_failure")
        embedding.embed = failure
        fallback = await manager.call("knowledge_search",{"query":"公开退款七天凭证"},use_cache=False)
        checks["failure_real_minilm_body"] = fallback.success and fallback.degraded and any("七天" in i["content"] for i in fallback.data)
        embedding.embed = original
        restored = await manager.call("knowledge_search",{"query":"公开退款七天凭证"},use_cache=False)
        checks["query_recovered_main"] = restored.success and not restored.degraded
        embedding.embed = failure
        changed = {**docs[0],"content":"# 退款\n公开更新规则：九天内受理退款，必须提供订单号。"}
        updated = await asyncio.to_thread(kb.import_documents,[changed])
        embedding.embed = original
        pending = await manager.call("knowledge_search",{"query":"公开更新九天退款订单号"},use_cache=False)
        checks["pending_new_content"] = updated["status"]=="degraded" and pending.degraded and any("九天" in i["content"] for i in pending.data)
        if peer:
            peer_updated = await peer.call("knowledge_search",{"query":"退款需要什么材料？","top_k":3})
            checks["redis_update_invalidates_peer"] = peer_updated.success and not peer_updated.cached and any("九天" in i["content"] for i in peer_updated.data)
        pending_count = kb.stats()["pending_main"]
        kb.close()
        kb = KnowledgeBase(**options)
        manager.register(Tool("knowledge_search","knowledge",kb.search_handler,{},cache_ttl=300))
        checks["restart_keeps_pending"] = pending_count == kb.stats()["pending_main"] == 1
        repaired = await asyncio.to_thread(kb.repair_main)
        after = await manager.call("knowledge_search",{"query":"公开更新九天退款订单号"},use_cache=False)
        checks["repair_then_main_new_body"] = repaired["status"]=="success" and after.success and not after.degraded and any("九天" in i["content"] for i in after.data)
        await asyncio.to_thread(kb.delete_document,"public-refund")
        removed = await manager.call("knowledge_search",{"query":"公开更新九天退款订单号"},use_cache=False)
        checks["deleted_current_main_and_backup"] = removed.success and not any(i["source_id"]=="public-refund" for i in removed.data) and not any(
            collection.get(where={"source_id":"public-refund"},include=[])["ids"] for collection in [kb._main,kb._backup])
        result["fault_sequence"] = {"failure":"injected_provider_failure","fallback":fallback.data,
                                     "restored":restored.data,"pending":pending.data,"after_repair":after.data}
        result["index_stats"] = kb.stats()

        cache = output/"intent_templates.json"
        recognizer = IntentRecognizer("unused-no-chat-call",embedding=embedding,template_cache_path=str(cache),template_wait_s=180)
        recognizers.append(recognizer)
        before = dict(embedding.usage)
        cache_present = cache.exists()
        started = time.perf_counter()
        await recognizer._load_template_embeddings()
        prepare_ms = (time.perf_counter()-started)*1000
        template_texts = embedding.usage["CLASSIFICATION:texts"]-before.get("CLASSIFICATION:texts",0)
        restarted = IntentRecognizer("unused-no-chat-call",embedding=embedding,template_cache_path=str(cache),template_wait_s=180)
        recognizers.append(restarted)
        before = dict(embedding.usage)
        started = time.perf_counter()
        await restarted._load_template_embeddings()
        cache_ms = (time.perf_counter()-started)*1000
        checks["real_template_cache_hit"] = embedding.usage["requests"]==before.get("requests",0)
        cache_http = embedding.usage["requests"]-before.get("requests",0)
        classifications = []
        for text in ["订单状态处理到哪了？","我要申请退款。","我的账号登录失败了。"]:
            started = time.perf_counter()
            outcome = await restarted._embedding_recognize(text)
            classifications.append({"text":text,"intent":outcome["intent"].value,"similarity":outcome["confidence"],
                                    "failed":outcome.get("failed",False),"wall_ms":(time.perf_counter()-started)*1000})
        checks["real_classification_branch"] = all(not row["failed"] for row in classifications)
        result["intent_embedding"] = {"cache_present_before":cache_present,"prepare_ms":prepare_ms,"template_texts":template_texts,
                                      "cache_hit_ms":cache_ms,"cache_hit_http":cache_http,
                                      "hot_queries":classifications,"hot_p50_ms":statistics.median(row["wall_ms"] for row in classifications),
                                      "scope":"真实embedding分支与缓存；未调用LLM，不代表三路质量或完整分类延迟"}
        result["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as ex:
        result.update(status="failed",reason=str(ex) if isinstance(ex,EmbeddingError) else type(ex).__name__)
    finally:
        embedding.embed = original
        for recognizer in recognizers:
            await recognizer.close()
        result["api_usage"] = dict(embedding.usage)
        result["counting_mode"] = embedding.counting_mode
        kb.close()
        embedding.close()
        if chroma_host:
            for collection in client.list_collections():
                if collection.name in {kb._main.name,kb._backup.name,*legacy}:
                    client.delete_collection(collection.name)
            after = {c.name:client.get_collection(c.name).count() for c in client.list_collections()}
            checks["existing_http_collections_preserved"] = after==existing
            result["server_preservation"] = {"before":existing,"after":after}
        if cache_client:
            for key in cache_client.scan_iter(match=f"{cache_prefix}:*"):
                cache_client.delete(key)
            checks["redis_experiment_keys_cleaned"] = not list(cache_client.scan_iter(match=f"{cache_prefix}:*"))
            cache_client.close()
        if not all(checks.values()):
            result["status"] = "failed"
        (output/"smoke.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k in ["status","checks","api_usage","counting_mode","reason","intent_embedding"]},ensure_ascii=False,indent=2))
    return result["status"]=="passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/qwen-embedding-migration/real-smoke")
    parser.add_argument("--chroma-host",default=None)
    parser.add_argument("--chroma-port",type=int,default=8001)
    parser.add_argument("--redis-db",type=int,default=None)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output,args.chroma_host,args.chroma_port,args.redis_db)) else 1)


if __name__ == "__main__":
    main()
