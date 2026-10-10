"""控制变量的真实 embedding / 检索 / 回答 / 意图实验；缺项显式标为 blocked。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from core.document_chunking import BudgetChunker, parse_sections
from core.embedding import EmbeddingError, EmbeddingSettings, GeminiEmbedding
from core.intent_recognizer import IntentCategory, IntentRecognizer, _TEMPLATES, _cosine
from core.llm_gateway import LLMRequest, build_gateway
from core.retrieval import BM25Index, weighted_rrf
from mcp.indexed_knowledge_base import KnowledgeBase, prepare_minilm
from mcp.knowledge_base import KnowledgeBase as BaselineKnowledgeBase
from mcp.tool_manager import MCPToolManager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/eval/gemini_rag_eval.json"


def normalize(text):
    return re.sub(r"\s+", "", text)


def duplicate_fraction(items):
    """最终上下文中重复出现的中文/英文句子字符占比。"""
    seen, duplicate, total = set(), 0, 0
    for item in items:
        for sentence in re.split(r"(?<=[。！？.!?])", normalize(item["content"])):
            if not sentence:
                continue
            total += len(sentence)
            if sentence in seen:
                duplicate += len(sentence)
            seen.add(sentence)
    return duplicate / total if total else 0


class MeasuredGateway:
    def __init__(self, gateway):
        self.gateway = gateway
        self.provider = gateway.provider
        self.tool_protocol = gateway.tool_protocol
        self.calls = 0
        self.usage = []

    async def complete(self, request):
        self.calls += 1
        result = await self.gateway.complete(request)
        usage = result.usage
        self.usage.append({"input_tokens": usage.total_input_tokens, "output_tokens": usage.output_tokens,
                           "cache_read_tokens": usage.cache_read_tokens})
        return result


def chat_gateway():
    key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None, None
    model = os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    url = os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    provider = os.getenv("LLM_PROVIDER") or ("deepseek" if url and "deepseek" in url.lower() else "anthropic")
    return MeasuredGateway(build_gateway(provider, key, model, url)), model


def model_chunks(corpus, kind, counter, budget, overlap, space):
    items = []
    chunker = BudgetChunker(counter, budget, overlap, space, reserve=0 if space == "minilm" else 16)
    for doc in corpus:
        if kind == "legacy":
            texts = BaselineKnowledgeBase._chunk_text(None, doc["content"])
            items.extend({"title": doc["title"], "content": text, "section_path": "",
                          "chunk_id": f"{space}:{doc['title']}:{i}"} for i, text in enumerate(texts))
        else:
            items.extend({**r.metadata, "content": r.text} for r in chunker.chunk_document(
                doc["title"], doc["content"], structural=kind == "structure"))
    return items


async def answer_and_judge(gateway, model, question, items):
    evidence = "\n\n".join(f"{i}. {r['title']} / {r['section_path']}\n{r['content']}" for i, r in enumerate(items))
    answer = await gateway.complete(LLMRequest(model, "仅根据知识证据回答，完整说明条件；证据不足时明确说明。",
        messages=[{"role": "user", "content": f"问题：{question['question']}\n证据：\n{evidence}"}],
        max_tokens=4096, temperature=0))
    judge = await gateway.complete(LLMRequest(model, "依据原文事实审核回答，不将改写误判为遗漏。只输出 JSON。",
        messages=[{"role": "user", "content": json.dumps({"question": question["question"], "answer": answer.text,
            "required_facts_and_conditions": question["evidence"], "retrieved_evidence": evidence,
            "output": {"covered": ["每项事实及其条件是否完整，按原顺序返回布尔值"],
                       "unsupported_claims": ["回答中无证据支撑的断言"]}}, ensure_ascii=False)}],
        max_tokens=4096, temperature=0))
    raw = judge.text
    score = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    covered = score["covered"]
    if len(covered) != len(question["evidence"]) or any(type(v) is not bool for v in covered):
        raise ValueError("invalid_judge_output")
    return {"answer": answer.text, "fact_condition_coverage": sum(covered) / len(covered),
            "all_facts_and_conditions": all(covered), "unsupported_claims": score["unsupported_claims"],
            "evaluation": "real_llm_judge_requires_human_review"}


async def retrieval_experiments(corpus, questions, gemini, mini, mini_count, args, gateway, model):
    # 每组只改变一个因素；legacy 两模型使用完全相同片段。
    specs = [("minilm-legacy-vector", "minilm", "legacy", "vector", 0),
             ("minilm-structure-vector", "minilm", "structure", "vector", 0),
             ("minilm-structure-bm25", "minilm", "structure", "bm25", 0),
             ("minilm-structure-hybrid", "minilm", "structure", "hybrid", 0)]
    if args.models == "both":
        specs += [("gemini-legacy-vector", "gemini", "legacy", "vector", 0),
                  ("gemini-recursive-vector", "gemini", "recursive", "vector", 0),
                  ("gemini-structure-vector", "gemini", "structure", "vector", 0),
                  ("gemini-structure-bm25", "gemini", "structure", "bm25", 0),
                  ("gemini-structure-hybrid", "gemini", "structure", "hybrid", 0),
                  ("gemini-overlap32-hybrid", "gemini", "structure", "hybrid", 32),
                  ("gemini-overlap64-hybrid", "gemini", "structure", "hybrid", 64)]
    if args.variants:
        unknown = set(args.variants) - {spec[0] for spec in specs}
        if unknown:
            raise ValueError(f"未知检索变体: {sorted(unknown)}")
        specs = [spec for spec in specs if spec[0] in args.variants]
    outputs, vector_cache, query_cache = {}, {}, {}
    reranker = MCPToolManager("experiment", model=model or "unconfigured", gateway=gateway)
    for name, encoder, kind, mode, overlap in specs:
        print(f"retrieval variant={name}", flush=True)
        if encoder == "gemini" and not gemini.enabled:
            outputs[name] = {"status": "blocked", "reason": "GEMINI_API_KEY 或 EMBEDDING_MODEL 未配置"}
            continue
        try:
            counter = gemini.count_tokens if encoder == "gemini" else mini_count
            items = model_chunks(corpus, kind, counter, args.chunk_budget if encoder == "gemini" else 240, overlap, encoder)
            index = BM25Index()
            truncation = sum(mini_count(f"{i['title']}\n{i['content']}") > 256 for i in items) if encoder == "minilm" else 0
            build_start = time.perf_counter()
            pending = []
            for item in items if mode != "bm25" else []:
                title = BudgetChunker.embedding_title(item["title"], item["section_path"])
                key = (encoder, title, item["content"])
                if key not in vector_cache:
                    pending.append(key)
            pending = list(dict.fromkeys(pending))
            if pending:
                values = (gemini.embed([key[2] for key in pending], "RETRIEVAL_DOCUMENT", [key[1] for key in pending])
                          if encoder == "gemini" else mini([f"{key[1]}\n{key[2]}" for key in pending]))
                vector_cache.update(zip(pending, values))
            vectors = []
            for item in items:
                title = BudgetChunker.embedding_title(item["title"], item["section_path"])
                key = (encoder, title, item["content"])
                if mode != "bm25":
                    vectors.append(vector_cache[key])
                index.add(item["chunk_id"], f"{title}\n{item['content']}", item)
            build_ms = (time.perf_counter() - build_start) * 1000
            rows = []
            for question in questions:
                started = time.perf_counter()
                query = question["question"]
                key = (encoder, query)
                embedding_ms = 0
                if mode != "bm25":
                    if key not in query_cache:
                        before = time.perf_counter()
                        query_cache[key] = ((gemini.embed([query], "RETRIEVAL_QUERY")[0]
                                             if encoder == "gemini" else mini([query])[0]),
                                            (time.perf_counter() - before) * 1000)
                    qv, embedding_ms = query_cache[key]
                    started = time.perf_counter()
                    scores = [_cosine(qv, vector) for vector in vectors]
                    order = sorted(range(len(items)), key=lambda i: -scores[i])[:max(args.top_k, 20)]
                    ranked = [{**items[i], "score": scores[i], "retrieval_sources": ["vector"]} for i in order]
                else:
                    ranked = []
                bm = [{**index.items[ident], "score": score, "retrieval_sources": ["bm25"]}
                      for ident, score in index.search(query, max(args.top_k, 20))] if mode != "vector" else []
                selected = weighted_rrf(ranked, bm) if mode == "hybrid" else ranked if mode == "vector" else bm
                retrieval_ms = (time.perf_counter() - started) * 1000 + embedding_ms
                selected = selected[:max(args.top_k, 12)] if args.answers else selected[:args.top_k]
                row = {"id": question["id"], "question": query, "latency_ms": retrieval_ms}
                if args.answers:
                    if gateway is None:
                        row["answer_status"] = "blocked: missing LLM credential"
                    else:
                        answer_started = time.perf_counter()
                        try:
                            calls_before = gateway.calls
                            success_before = reranker.rerank_stats["llm_success"]
                            failed_before = reranker.rerank_stats["failed_fallback"]
                            selected = await reranker._rerank(query, selected, args.top_k)
                            row["rerank_llm_calls"] = gateway.calls - calls_before
                            row["rerank_status"] = ("measured" if reranker.rerank_stats["llm_success"] > success_before else
                                "failed_fallback" if reranker.rerank_stats["failed_fallback"] > failed_before else "skipped")
                            row.update(await answer_and_judge(gateway, model, question, selected))
                            row["answer_status"] = "measured"
                        except Exception as ex:
                            row["answer_status"] = "failed"
                            row["answer_error_type"] = type(ex).__name__
                        row["answer_latency_ms"] = (time.perf_counter() - answer_started) * 1000
                text = normalize("\n".join(i["content"] for i in selected[:args.top_k]))
                covered = [normalize(fact) in text for fact in question["evidence"]]
                row.update(evidence_coverage=sum(covered) / len(covered), all_evidence=all(covered),
                           duplicate_fraction=duplicate_fraction(selected[:args.top_k]),
                           evidence_anchors=question["evidence"], covered=covered,
                           retrieved=selected[:args.top_k])
                rows.append(row)
            outputs[name] = {"status": "measured", "model": encoder, "chunking": kind, "overlap": overlap,
                "budget": args.chunk_budget if encoder == "gemini" else 240, "mode": mode,
                "chunks": len(items), "minilm_truncated_chunks": truncation, "build_ms": build_ms,
                "max_chunk_input_tokens": max(counter(BudgetChunker.embedding_title(i["title"], i["section_path"]) + "\n" + i["content"])
                                               for i in items),
                "max_section_input_tokens": max(counter(BudgetChunker.embedding_title(d["title"], s.path) + "\n" + s.text)
                                                 for d in corpus for s in parse_sections(d["content"])),
                "evidence_coverage": statistics.mean(r["evidence_coverage"] for r in rows),
                "all_evidence_rate": statistics.mean(r["all_evidence"] for r in rows),
                "duplicate_fraction": statistics.mean(r["duplicate_fraction"] for r in rows),
                "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
                "p95_latency_ms": sorted(r["latency_ms"] for r in rows)[max(0, math.ceil(len(rows) * .95) - 1)],
                "query_embeddings_reused": True, "rows": rows}
        except Exception as ex:
            outputs[name] = {"status": "failed", "error_type": type(ex).__name__,
                             "reason": KnowledgeBase._reason(ex)}
    return outputs


def macro_f1(rows):
    labels = sorted({r["expected"] for r in rows} | {r["predicted"] for r in rows})
    values = []
    for label in labels:
        tp = sum(r["expected"] == label and r["predicted"] == label for r in rows)
        fp = sum(r["expected"] != label and r["predicted"] == label for r in rows)
        fn = sum(r["expected"] == label and r["predicted"] != label for r in rows)
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return statistics.mean(values) if values else None


async def intent_experiments(cases, gemini, args, gateway, model):
    if gateway is None:
        return {"status": "blocked", "reason": "LLM credential missing"}
    recognizer = IntentRecognizer("experiment", model=model, gateway=gateway, embedding=gemini,
                                  confidence_threshold=args.threshold)
    modes = {"no-vector": [], "old-character": [], "gemini": []}
    threshold_rows = {mode: {level: [] for level in [.4, .5, .6]} for mode in modes}
    legacy_templates = {cat: [recognizer._local_embedding(text) for text in texts] for cat, texts in _TEMPLATES.items()}
    for case in cases:
        print(f"intent case={case['intent']}", flush=True)
        started = time.perf_counter()

        async def timed_llm():
            before = time.perf_counter()
            result = await recognizer._llm_recognize(case["message"], None)
            return result, (time.perf_counter() - before) * 1000

        async def timed_embedding():
            before = time.perf_counter()
            result = await recognizer._embedding_recognize(case["message"]) if gemini.enabled else {"failed": True}
            return result, (time.perf_counter() - before) * 1000

        (llm, llm_ms), (emb, embedding_ms) = await asyncio.gather(timed_llm(), timed_embedding())
        parallel_ms = (time.perf_counter() - started) * 1000
        if llm.get("failed"):
            return {"status": "failed", "reason": "LLM intent call failed"}
        pattern = recognizer._pattern_recognize(case["message"])
        before = time.perf_counter()
        vector = recognizer._local_embedding(case["message"])
        best_cat, best_score = max(((cat, max(_cosine(vector, v) for v in vectors))
                                   for cat, vectors in legacy_templates.items()), key=lambda row: row[1])
        local = {"intent": best_cat, "confidence": best_score}
        local_ms = (time.perf_counter() - before) * 1000
        for mode in modes:
            if mode == "gemini" and (not gemini.enabled or emb.get("failed")):
                continue
            recognizer._embedding_enabled = mode != "no-vector"
            recognizer.threshold = args.threshold
            picked, confidence, _ = recognizer._vote(llm, local if mode == "old-character" else emb,
                                                    pattern)
            modes[mode].append({"message": case["message"], "expected": case["intent"],
                               "predicted": picked.value, "confidence": confidence,
                               "latency_ms": parallel_ms if mode == "gemini" else
                                             max(llm_ms, local_ms) if mode == "old-character" else llm_ms})
            if args.split == "dev":
                for level in threshold_rows[mode]:
                    recognizer.threshold = level
                    prediction, _, _ = recognizer._vote(llm, local if mode == "old-character" else emb, pattern)
                    threshold_rows[mode][level].append({"expected": case["intent"], "predicted": prediction.value})
    outputs = {}
    for mode, rows in modes.items():
        confusion = Counter((r["expected"], r["predicted"]) for r in rows if r["expected"] != r["predicted"])
        outputs[mode] = {"status": "measured" if len(rows) == len(cases) else "blocked_or_partial",
            "macro_f1": macro_f1(rows), "rows": rows, "threshold": args.threshold,
            "confusions": [{"expected": k[0], "predicted": k[1], "count": v} for k, v in confusion.items()],
            "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows) if rows else None,
            "llm_results_reused": True}
        if args.split == "dev":
            outputs[mode]["development_thresholds"] = {str(level): macro_f1(values)
                                                        for level, values in threshold_rows[mode].items()}
    return outputs


async def run(args):
    load_dotenv(ROOT / ".env")
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    corpus = data["corpus"] + (json.loads((ROOT / "data/eval/rag_corpus.json").read_text(encoding="utf-8"))
                              if data.get("include_legacy_corpus", True) else [])
    questions = [q for q in data["questions"] if q["split"] == args.split]
    cases = [q for q in data["intents"] if q["split"] == args.split]
    if args.limit:
        questions, cases = questions[:args.limit], cases[:args.limit]
    key_present = bool(os.getenv("GEMINI_API_KEY"))
    configured = bool(os.getenv("EMBEDDING_MODEL"))
    settings = EmbeddingSettings.from_env() if not configured or key_present else EmbeddingSettings()
    gemini = GeminiEmbedding(settings, min_interval=args.api_interval)
    gateway, model = chat_gateway()
    report = {"date": "2026-10-09", "split": args.split, "suite": args.suite, "dataset": args.dataset.name,
              "configuration": {"gemini_key_present": key_present, "embedding_model_present": configured,
                  "llm_key_present": gateway is not None, "embedding_model": settings.model,
                  "dimensions": settings.dimensions, "chunk_budget": args.chunk_budget,
                  "threshold": args.threshold, "top_k": args.top_k},
              "llm_model": model, "api_interval_seconds": args.api_interval,
              "corpus_documents": len(corpus), "long_documents": [dict(title=d["title"], characters=len(d["content"]))
                  for d in data["corpus"]], "questions": len(questions), "intents": len(cases),
              "limits": ["评测语料为合成客服政策和手册，不能代表真实用户分布。",
                         "原文锚点按去空白后字面覆盖计量；回答指标由真实 LLM 审核，需要人工复核。",
                         "模型/切块/召回比较固定不使用查询改写；可选真实重排与回答设置各变体一致。",
                         "查询向量在变体间复用，延迟加回实测编码耗时；不是生产并发负载测量。",
                         "LLM 次数为网关调用数，SDK 内部重试及未完整返回的 token 用量无法完整计量。"]}
    try:
        needs_mini = args.suite in {"smoke", "recovery", "all"} or (
            args.suite == "rag" and (not args.variants or any(v.startswith("minilm") for v in args.variants)))
        bm25_only = args.suite == "rag" and args.variants and all(v.endswith("bm25") for v in args.variants)
        mini, mini_count = prepare_minilm(warm=not bm25_only) if needs_mini else (None, None)
        if args.suite in {"smoke", "all"}:
            path = args.output / "smoke-index"
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
            kb = KnowledgeBase(chroma_path=str(path), embedding=gemini, client=client,
                               backup_function=mini, backup_counter=mini_count, load_defaults=False)
            started = time.perf_counter()
            imported = kb.import_documents(data["corpus"])
            results = kb.search(questions[0]["question"], args.top_k)
            report["smoke"] = {"status": "measured", "import": imported, "stats": kb.stats(),
                               "results": results, "latency_ms": (time.perf_counter() - started) * 1000}
            kb.close()
        if args.suite == "recovery":
            if not gemini.enabled:
                report["recovery"] = {"status": "blocked", "reason": "Gemini configuration missing"}
            else:
                path = args.output / "recovery-index"
                client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
                invalidations = []
                kb = KnowledgeBase(chroma_path=str(path), embedding=gemini, client=client,
                                   backup_function=mini, backup_counter=mini_count, load_defaults=False,
                                   on_change=lambda: invalidations.append(True) or 0)
                first = kb.import_documents([{"title": "恢复验证", "content": "测试政策：退货期限七天。"}])
                real_embed = gemini.embed

                def injected_failure(*params, **kwargs):
                    raise EmbeddingError("injected_google_failure")

                gemini.embed = injected_failure
                query_degraded = kb.search("退货期限")
                gemini.embed = real_embed
                transient_recovered = kb.search("退货期限")
                gemini.embed = injected_failure
                imported = kb.import_documents([{"title": "恢复验证", "content": "测试政策更新：退货期限九天。"},
                    {"title": "恢复新增", "content": "新增规则：物流查件需提供订单号。"}])
                gemini.embed = real_embed
                pending = kb.search("退货期限九天")
                repaired = kb.repair_main()
                recovered = kb.search("物流查件")
                checks = {"initial_main_success": first["status"] == "success",
                          "query_fallback": bool(query_degraded) and all(r["index_route"] == "backup" for r in query_degraded),
                          "transient_recovery": bool(transient_recovered) and all(r["index_route"] == "main" for r in transient_recovered),
                          "import_degraded_success": imported["status"] == "degraded",
                          "pending_stays_backup": bool(pending) and all(r["index_route"] == "backup" for r in pending),
                          "update_visible": any("九天" in r["content"] for r in pending),
                          "old_content_removed": not any("七天" in r["content"] for r in pending),
                          "repair_complete": repaired["main_complete"],
                          "new_content_on_main": any("订单号" in r["content"] and r["index_route"] == "main" for r in recovered),
                          "cache_invalidated": len(invalidations) >= 4}
                report["recovery"] = {"status": "measured", "all_checks_passed": all(checks.values()),
                    "fault": "显式注入 Google 失败；入库、备用模型和恢复后 Gemini 均实际调用",
                    "checks": checks, "query_degraded": query_degraded, "pending_results": pending,
                    "recovered_results": recovered, "stats": kb.stats(), "cache_invalidations": len(invalidations)}
                kb.close()
        if args.suite in {"rag", "all"}:
            report["retrieval"] = await retrieval_experiments(corpus, questions, gemini, mini, mini_count,
                                                             args, gateway, model)
        if args.suite in {"intent", "all"}:
            report["intent"] = await intent_experiments(cases, gemini, args, gateway, model)
    except Exception as ex:
        report["execution_error"] = {"type": type(ex).__name__, "reason": KnowledgeBase._reason(ex)}
    report["gemini_counting_mode"] = gemini.counting_mode
    report["api_usage"] = {"gemini": dict(gemini.usage), "llm_calls": gateway.calls if gateway else 0,
                           "llm_usage": gateway.usage if gateway else []}
    gemini.close()
    args.output.mkdir(parents=True, exist_ok=True)
    filename = args.output / f"{args.suite}-{args.split}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    filename.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(filename.resolve())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "recovery", "rag", "intent", "all"], default="smoke")
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--models", choices=["backup", "both"], default="both")
    parser.add_argument("--variants", nargs="*", help="只运行明确列出的检索变体，避免重复整组实验")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-budget", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="调用真实聊天模型重排、回答和事实条件审核")
    parser.add_argument("--api-interval", type=float, default=1, help="Google 实验请求最小间隔秒数；生产不增加此等待")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/gemini-rag")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
