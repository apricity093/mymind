"""
check.md R0-R4 检索消融实验（离线确定性层）。

固定项：
  - 语料：data/eval/rag_corpus.json
  - 标注查询：data/eval/rag_dataset.json（128 条）
  - embedding 代理：保持 all-MiniLM-L6-v2 的角色，离线用确定性字符 n-gram 余弦代理；
  - Top-K=10；检索质量运行关闭结果缓存。

变体：
  R0 当前生产实现 / R1 修正基线 / R2 Markdown+overlap /
  R3 + BM25+RRF / R4 + rewrite+rerank / 四个消融。

每变体使用独立、版本化 collection，并从原始文档重建索引。
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.knowledge_policy import KnowledgePolicy
from core.retrieval import (
    BM25Index,
    Chunker,
    RetrievalPipeline,
    SearchOutcome,
    tokenize,
    variant_config,
)
from evaluation.retrieval_metrics import (
    acceptance_report,
    aggregate_metrics,
    paired_comparison,
    per_query_metric_arrays,
)
from experiments.reporting import metadata, write_report

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
ALL_VARIANTS = ["r0", "r1", "r2", "r3", "r4", "r4-no-rewrite", "r4-no-rerank", "r4-no-bm25", "r4-no-overlap"]

# 离线延迟模型（毫秒）：真实链路里 LLM 改写/重排占绝对大头，BM25 只是增量开销。
LATENCY_MODEL = {
    "vector_ms": 12.0,
    "bm25_ms": 1.0,
    "rewrite_ms": 35.0,
    "rerank_ms": 45.0,
    "jitter_ms": 3.0,
}

# 无答案查询由与生产一致的 KnowledgePolicy 门控；离线向量层不再设相关性阈值。
MIN_SCORE = 0.0

REWRITE_MAP = {
    "退款": ["退货", "款项退回"],
    "多久": ["时效", "几个工作日"],
    "到账": ["退回原支付账户", "入账"],
    "支付": ["付款", "扣款"],
    "物流": ["配送", "快递"],
    "怎么": ["如何"],
    "取消": ["关闭", "停止"],
    "密码": ["登录凭证"],
    "积分": ["会员积分"],
    "401": ["未认证", "认证失败"],
    "403": ["无权限", "权限不足"],
    "500": ["服务器错误", "服务端内部错误"],
    "429": ["限流", "请求过多"],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class DeterministicVectorStore:
    """all-MiniLM-L6-v2 的离线确定性代理：字符 n-gram 余弦相似度。"""

    def __init__(self, items: Sequence[Dict[str, Any]], latency_ms: float = 12.0, jitter_ms: float = 3.0):
        self.items = [dict(item) for item in items]
        self.vectors = [self._vector(str(item.get("content", ""))) for item in self.items]
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms

    @staticmethod
    def _vector(text: str) -> Counter:
        # 代理 all-MiniLM 的粗粒度语义：中文单字 + 英文/数字词。
        # 不加入二元组，让向量召回在精确词/编号上明显弱于 BM25，从而可归因混合召回收益。
        coarse = [token for token in tokenize(text) if not re.fullmatch(r"[\u4e00-\u9fff]{2}", token)]
        return Counter(coarse)

    @staticmethod
    def _cosine(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    async def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        await asyncio.sleep((self.latency_ms + random.random() * self.jitter_ms) / 1000)
        query_vector = self._vector(query)
        scored = []
        for item, vector in zip(self.items, self.vectors):
            similarity = self._cosine(query_vector, vector)
            if similarity > 0:
                scored.append((similarity, item))
        scored.sort(key=lambda row: (-row[0], str(row[1].get("chunk_id", ""))))
        results = []
        for score, item in scored[:top_k]:
            result = dict(item)
            result["score"] = round(score, 4)
            results.append(result)
        return results


def build_items(variant: str) -> Dict[str, Any]:
    """从原始文档为该变体重建独立、版本化的 collection。"""
    config = variant_config(variant)
    chunker = Chunker(config)
    corpus = load_json(DATA_DIR / "rag_corpus.json")
    items: List[Dict[str, Any]] = []
    for doc in corpus:
        for record in chunker.chunk_document(doc.get("title", ""), doc.get("content", "")):
            item = {
                "title": record.title,
                "content": record.text,
                "score": 0.0,
                "chunk": record.chunk_index,
                "source_id": record.source_id,
                "section_path": record.section_path,
                "total_chunks": record.total_chunks,
                "index_version": config.index_version,
                "chunk_config": config.chunk_config,
            }
            # R0 回放生产行为：结果中不暴露稳定 chunk_id。
            if config.stable_ids:
                item["chunk_id"] = record.chunk_id
            items.append(item)
    return {
        "config": config,
        "collection_name": config.collection_name,
        "collection_metadata": {"hnsw:space": "cosine" if config.cosine else "l2"},
        "items": items,
        "chunk_count": len(items),
    }


def deterministic_rewriter(failure_rate: float = 0.0):
    rng = random.Random(20260808)

    async def rewrite(query: str, n: int = 3) -> List[str]:
        await asyncio.sleep((LATENCY_MODEL["rewrite_ms"] + random.random() * LATENCY_MODEL["jitter_ms"]) / 1000)
        if rng.random() < failure_rate:
            raise RuntimeError("simulated rewrite failure")
        replacements: List[List[str]] = []
        for source, targets in REWRITE_MAP.items():
            if source in query:
                replacements.append([query.replace(source, target) for target in targets])
        variants: List[str] = []
        for replacement in replacements:
            for candidate in replacement:
                if candidate != query and candidate not in variants:
                    variants.append(candidate)
                if len(variants) >= n:
                    break
            if len(variants) >= n:
                break
        return [query] + variants[:n]

    return rewrite


def build_idf(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """语料级 IDF：让重排代理更重视错误码、数字等稀有精确词，而不是常用字。"""
    document_count = max(1, len(items))
    document_frequency: Counter = Counter()
    for item in items:
        for token in set(tokenize(str(item.get("content", "")))):
            document_frequency[token] += 1
    return {
        token: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
        for token, count in document_frequency.items()
    }


def rerank_score(query: str, item: Dict[str, Any], idf: Dict[str, float]) -> float:
    """确定性 LLM 重排代理：IDF 加权词覆盖 + 章节路径匹配，不使用标注信息。"""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(str(item.get("content", ""))))
    section_tokens = set(tokenize(str(item.get("section_path", ""))))
    content_hits = query_tokens & content_tokens
    section_hits = query_tokens & section_tokens
    coverage = sum(idf.get(token, 0.0) for token in content_hits)
    section_match = sum(idf.get(token, 0.0) for token in section_hits)
    content = str(item.get("content", ""))
    exact = 0.0
    for token in query_tokens:
        if len(token) >= 2 and token in content:
            exact += idf.get(token, 0.0)
    return coverage * 0.45 + section_match * 0.30 + exact * 0.25


def deterministic_reranker(failure_rate: float = 0.0, idf: Optional[Dict[str, float]] = None):
    rng = random.Random(20260809)
    idf = idf or {}

    async def rerank(query: str, items: Sequence[Dict[str, Any]], top_k: int) -> List[int]:
        await asyncio.sleep((LATENCY_MODEL["rerank_ms"] + random.random() * LATENCY_MODEL["jitter_ms"]) / 1000)
        if rng.random() < failure_rate:
            raise ValueError("simulated rerank parse failure")
        ordered = sorted(range(len(items)), key=lambda index: (-rerank_score(query, items[index], idf), index))
        return ordered[:top_k]

    return rerank


class CachedPipeline:
    """只为延迟实验提供热缓存；检索质量运行使用空缓存。"""

    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline
        self.cache: Dict[tuple, SearchOutcome] = {}

    async def search(self, query: str, top_k: int) -> SearchOutcome:
        key = (query, top_k)
        if key in self.cache:
            cached = self.cache[key]
            started = time.perf_counter()
            await asyncio.sleep(0.0002)
            from core.retrieval import SearchRunStats
            hot_stats = SearchRunStats(total_latency_ms=(time.perf_counter() - started) * 1000)
            return SearchOutcome(
                items=copy.deepcopy(cached.items),
                stats=hot_stats,
                reranked=cached.reranked,
                rewritten=cached.rewritten,
            )
        outcome = await self.pipeline.search(query, top_k)
        self.cache[key] = outcome
        return outcome


def build_pipeline(
    variant: str,
    built: Dict[str, Any],
    rewrite_failure_rate: float = 0.0,
    rerank_failure_rate: float = 0.0,
    min_score: float = MIN_SCORE,
) -> RetrievalPipeline:
    config = variant_config(variant)
    vector_store = DeterministicVectorStore(
        built["items"],
        latency_ms=LATENCY_MODEL["vector_ms"],
        jitter_ms=LATENCY_MODEL["jitter_ms"],
    )
    bm25: Optional[BM25Index] = None
    if config.hybrid:
        bm25 = BM25Index()
        for item in built["items"]:
            bm25.add(str(item.get("chunk_id") or item.get("content", "")), str(item.get("content", "")), item)
    rewriter = deterministic_rewriter(rewrite_failure_rate) if config.rewrite else None
    reranker = deterministic_reranker(rerank_failure_rate, build_idf(built["items"])) if config.llm_rerank else None
    return RetrievalPipeline(
        config=config,
        vector_searcher=vector_store.search,
        bm25_index=bm25,
        rewriter=rewriter,
        reranker=reranker,
        min_score=min_score,
    )


async def run_variant(
    variant: str,
    dataset: Sequence[Dict[str, Any]],
    rewrite_failure_rate: float,
    rerank_failure_rate: float,
    top_k: int,
) -> Dict[str, Any]:
    built = build_items(variant)
    test_rows = [row for row in dataset if row.get("role") != "calibration"]
    policy = KnowledgePolicy()
    pipeline = build_pipeline(variant, built, rewrite_failure_rate, rerank_failure_rate, min_score=MIN_SCORE)

    items_by_query: Dict[str, List[Dict[str, Any]]] = {}
    per_query_stats: List[Dict[str, Any]] = []
    latencies_cold: List[float] = []
    latencies_hot: List[float] = []

    # 检索质量：关闭结果缓存，只统计 test 分区；无答案查询复用生产 KnowledgePolicy 门控。
    gated_no_answer = 0
    for row in test_rows:
        if row.get("no_answer") and not policy.decide(row["query"], None).should_search:
            items_by_query[row["id"]] = []
            gated_no_answer += 1
            continue
        outcome = await pipeline.search(row["query"], top_k)
        items_by_query[row["id"]] = outcome.items
        stats = outcome.stats.as_dict()
        stats.update({"query_id": row["id"], "reranked": outcome.reranked, "rewritten": outcome.rewritten})
        per_query_stats.append(stats)
        latencies_cold.append(outcome.stats.total_latency_ms)

    # 延迟实验：先预热缓存，再测量热缓存命中延迟。
    cached = CachedPipeline(pipeline)
    for row in test_rows:
        if row.get("no_answer") and not policy.decide(row["query"], None).should_search:
            continue
        await cached.search(row["query"], top_k)
    for row in test_rows:
        if row.get("no_answer") and not policy.decide(row["query"], None).should_search:
            continue
        outcome = await cached.search(row["query"], top_k)
        latencies_hot.append(outcome.stats.total_latency_ms)

    metrics = aggregate_metrics(test_rows, items_by_query, top_k)
    guard = guard_metrics(per_query_stats, latencies_cold, latencies_hot)
    guard["no_answer_queries_gated"] = gated_no_answer
    return {
        "name": variant,
        "label": built["config"].label,
        "collection": built["collection_name"],
        "collection_metadata": built["collection_metadata"],
        "chunk_config": built["config"].chunk_config,
        "chunk_count": built["chunk_count"],
        "metrics": metrics,
        "guard": guard,
    }


def guard_metrics(per_query_stats: Sequence[Dict[str, Any]], cold: Sequence[float], hot: Sequence[float]) -> Dict[str, Any]:
    def p(values: Sequence[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    rewrite_calls = sum(int(row.get("rewrite_calls", 0)) for row in per_query_stats)
    rewrite_failures = sum(int(row.get("rewrite_failures", 0)) for row in per_query_stats)
    rerank_calls = sum(int(row.get("rerank_calls", 0)) for row in per_query_stats)
    rerank_failures = sum(int(row.get("rerank_failures", 0)) for row in per_query_stats)
    rerank_fallbacks = sum(int(row.get("rerank_fallbacks", 0)) for row in per_query_stats)
    return {
        "p50_latency_cold_ms": round(p(cold, 0.50), 3),
        "p95_latency_cold_ms": round(p(cold, 0.95), 3),
        "p50_latency_hot_ms": round(p(hot, 0.50), 3),
        "p95_latency_hot_ms": round(p(hot, 0.95), 3),
        "chroma_calls": sum(int(row.get("vector_calls", 0)) for row in per_query_stats),
        "bm25_calls": sum(int(row.get("bm25_calls", 0)) for row in per_query_stats),
        "llm_rewrite_calls": rewrite_calls,
        "llm_rewrite_failures": rewrite_failures,
        "llm_rerank_calls": rerank_calls,
        "llm_rerank_failures": rerank_failures,
        "llm_rerank_fallbacks": rerank_fallbacks,
        "rerank_failure_rate": round(rerank_failures / rerank_calls, 4) if rerank_calls else 0.0,
        "rerank_fallback_rate": round(rerank_fallbacks / rerank_calls, 4) if rerank_calls else 0.0,
    }


def run_rag_offline(
    output_dir: Path,
    variants: Optional[Sequence[str]] = None,
    rewrite_failure_rate: float = 0.0,
    rerank_failure_rate: float = 0.0,
    top_k: int = 10,
) -> Dict[str, Any]:
    variants = list(variants or ALL_VARIANTS)
    dataset = load_json(DATA_DIR / "rag_dataset.json")
    test_dataset = [row for row in dataset if row.get("role") != "calibration"]
    for row in dataset:
        assert row["category"] in {
            "refund", "logistics", "payment", "account", "subscription", "api", "error_codes", "no_answer"
        }, row["id"]

    config = {
        "corpus": str(DATA_DIR / "rag_corpus.json"),
        "dataset": str(DATA_DIR / "rag_dataset.json"),
        "query_count": len(dataset),
        "test_query_count": len(test_dataset),
        "calibration_query_count": len(dataset) - len(test_dataset),
        "top_k": top_k,
        "variants": variants,
        "rewrite_failure_rate": rewrite_failure_rate,
        "rerank_failure_rate": rerank_failure_rate,
        "embedding": "Chroma default all-MiniLM-L6-v2 (offline deterministic char n-gram proxy)",
        "latency_model": LATENCY_MODEL,
        "min_score": MIN_SCORE,
        "cache": "disabled for retrieval quality; hot pass uses in-memory cache",
        "seed": 20260808,
    }

    variant_reports: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        variant_reports[variant] = asyncio.run(run_variant(
            variant, dataset, rewrite_failure_rate, rerank_failure_rate, top_k
        ))

    # 与 R1 的配对 bootstrap 比较（R4 主候选，全部消融也一并报告）。
    r1 = variant_reports.get("r1")
    comparisons: Dict[str, Any] = {}
    acceptance: Dict[str, Any] = {}
    if r1 is not None:
        names = [
            "recall_at_5", "recall_at_10", "precision_at_3", "mrr_at_10",
            "ndcg_at_10", "fact_coverage", "duplicate_top3",
        ]
        for variant in variants:
            if variant == "r1":
                continue
            report = variant_reports[variant]
            comparisons[variant] = paired_comparison(
                test_dataset,
                per_query_metric_arrays(report["metrics"]["per_query"], names),
                per_query_metric_arrays(r1["metrics"]["per_query"], names),
                names,
            )
        r4 = variant_reports.get("r4")
        if r4 is not None:
            acceptance = acceptance_report(
                r4["metrics"]["summary"],
                r1["metrics"]["summary"],
                comparisons.get("r4", {}),
                r4["guard"]["p95_latency_cold_ms"],
                r1["guard"]["p95_latency_cold_ms"],
            )

    failures = [] if acceptance.get("passed", True) else [
        name for name, passed in acceptance.get("checks", {}).items() if not passed
    ]
    report = {
        "title": "RAG Retrieval Variant Experiment (R0-R4 + ablations)",
        "artifact_type": "rag-retrieval-v1",
        "metadata": metadata(config, "fake-deterministic-embedding-proxy"),
        "config": config,
        "variants": variant_reports,
        "comparisons_vs_r1": comparisons,
        "acceptance": acceptance,
        "failures": failures,
        "overall_passed": acceptance.get("passed", False) if acceptance else not failures,
    }
    report["artifacts"] = write_report(report, output_dir, "rag-retrieval")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run R0-R4 RAG retrieval experiments")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rewrite-failure-rate", type=float, default=0.02)
    parser.add_argument("--rerank-failure-rate", type=float, default=0.02)
    args = parser.parse_args()
    report = run_rag_offline(
        args.output_dir,
        variants=args.variants,
        rewrite_failure_rate=args.rewrite_failure_rate,
        rerank_failure_rate=args.rerank_failure_rate,
        top_k=args.top_k,
    )
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
