"""
真实 Chroma + 默认 all-MiniLM-L6-v2 embedding 的 RAG 变体实验。

与 rag_retrieval.py 的离线字符代理不同，本层：
  - 每个变体创建独立、版本化 collection，并从 data/eval/rag_corpus.json 重建索引；
  - 使用 Chroma 默认 ONNXMiniLM_L6_V2（模型经 RAG_ONNX_PATH 指向 workspace 内下载目录）；
  - 改写/重排仍使用确定性代理（真实 LLM 层需 --confirm-cost，另行评估）。

用途：验证真实 embedding 下切块/BM25/RRF 的行为，不作为 LLM rewrite/rerank 的最终结论。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from core.knowledge_policy import KnowledgePolicy
from core.retrieval import BM25Index, RetrievalPipeline, SearchOutcome
from evaluation.retrieval_metrics import (
    acceptance_report,
    aggregate_metrics,
    paired_comparison,
    per_query_metric_arrays,
)
from experiments.rag_retrieval import (
    ALL_VARIANTS,
    LATENCY_MODEL,
    build_idf,
    deterministic_reranker,
    deterministic_rewriter,
)
from experiments.reporting import metadata, write_report
from mcp.knowledge_base import KnowledgeBase

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
DEFAULT_CHROMA_PATH = Path("artifacts/experiments/chroma_rag")
DEFAULT_ONNX_PATH = Path("artifacts/experiments/onnx_models")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_embedding_function(onnx_path: Path) -> ONNXMiniLM_L6_V2:
    function = ONNXMiniLM_L6_V2()
    function.DOWNLOAD_PATH = Path(onnx_path)
    return function


async def run_variant(
    variant: str,
    corpus: Sequence[Dict[str, str]],
    test_rows: Sequence[Dict[str, Any]],
    embedding_function: ONNXMiniLM_L6_V2,
    top_k: int,
    rewrite_failure_rate: float,
    rerank_failure_rate: float,
    chroma_path: Path,
) -> Dict[str, Any]:
    knowledge = KnowledgeBase(
        chroma_host="localhost",
        chroma_port=8000,
        chroma_path=str(chroma_path),
        variant=variant,
        embedding_function=embedding_function,
    )
    knowledge.add_documents([dict(doc) for doc in corpus])

    # 用同一语料的切块集合构建 IDF，供确定性重排代理使用。
    idf_items: List[Dict[str, Any]] = []
    try:
        data = knowledge._collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or [{} for _ in documents]
        ids = data.get("ids") or []
        for index, text in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) else {}
            idf_items.append({
                "chunk_id": ids[index] if index < len(ids) else meta.get("chunk_id", ""),
                "content": text,
                "title": meta.get("title", ""),
                "source_id": meta.get("source_id", ""),
                "section_path": meta.get("section_path", ""),
            })
    except Exception:
        idf_items = []

    config = knowledge._config
    assert config is not None
    pipeline = RetrievalPipeline(
        config=config,
        vector_searcher=knowledge.vector_search,
        bm25_index=knowledge.bm25_index,
        rewriter=deterministic_rewriter(rewrite_failure_rate) if config.rewrite else None,
        reranker=deterministic_reranker(rerank_failure_rate, build_idf(idf_items)) if config.llm_rerank else None,
    )

    policy = KnowledgePolicy()
    items_by_query: Dict[str, List[Dict[str, Any]]] = {}
    stats_rows: List[Dict[str, Any]] = []
    cold_latencies: List[float] = []
    gated = 0
    for row in test_rows:
        if row.get("no_answer") and not policy.decide(row["query"], None).should_search:
            items_by_query[row["id"]] = []
            gated += 1
            continue
        outcome = await pipeline.search(row["query"], top_k)
        items_by_query[row["id"]] = outcome.items
        stats = outcome.stats.as_dict()
        stats.update({"query_id": row["id"], "reranked": outcome.reranked, "rewritten": outcome.rewritten})
        stats_rows.append(stats)
        cold_latencies.append(outcome.stats.total_latency_ms)

    metrics = aggregate_metrics(test_rows, items_by_query, top_k)

    def percentile(values: Sequence[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))] if ordered else 0.0

    guard = {
        "collection": knowledge.collection_name,
        "chunk_count": knowledge.doc_count,
        "p50_latency_cold_ms": round(percentile(cold_latencies, 0.50), 3),
        "p95_latency_cold_ms": round(percentile(cold_latencies, 0.95), 3),
        "vector_calls": sum(row["vector_calls"] for row in stats_rows),
        "bm25_calls": sum(row["bm25_calls"] for row in stats_rows),
        "rewrite_calls": sum(row["rewrite_calls"] for row in stats_rows),
        "rerank_calls": sum(row["rerank_calls"] for row in stats_rows),
        "rerank_failures": sum(row["rerank_failures"] for row in stats_rows),
        "rerank_fallbacks": sum(row["rerank_fallbacks"] for row in stats_rows),
        "no_answer_queries_gated": gated,
    }
    return {
        "name": variant,
        "label": config.label,
        "collection": knowledge.collection_name,
        "collection_metadata": {"hnsw:space": "cosine" if config.cosine else "l2"},
        "chunk_config": config.chunk_config,
        "metrics": metrics,
        "guard": guard,
    }


def run_rag_chroma(
    output_dir: Path,
    variants: Optional[Sequence[str]] = None,
    rewrite_failure_rate: float = 0.02,
    rerank_failure_rate: float = 0.02,
    top_k: int = 10,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    onnx_path: Path = DEFAULT_ONNX_PATH,
) -> Dict[str, Any]:
    variants = list(variants or ["r0", "r1", "r2", "r3", "r4"])
    corpus = load_json(DATA_DIR / "rag_corpus.json")
    dataset = load_json(DATA_DIR / "rag_dataset.json")
    test_rows = [row for row in dataset if row.get("role") != "calibration"]
    embedding_function = build_embedding_function(onnx_path)
    config = {
        "corpus": str(DATA_DIR / "rag_corpus.json"),
        "dataset": str(DATA_DIR / "rag_dataset.json"),
        "query_count": len(test_rows),
        "top_k": top_k,
        "variants": variants,
        "embedding": "Chroma default all-MiniLM-L6-v2 (ONNX, workspace model)",
        "rewrite": "deterministic proxy",
        "rerank": "deterministic proxy",
        "rewrite_failure_rate": rewrite_failure_rate,
        "rerank_failure_rate": rerank_failure_rate,
        "latency_model": LATENCY_MODEL,
        "chroma_path": str(chroma_path),
        "onnx_path": str(onnx_path),
        "cache": "disabled",
    }
    variant_reports: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        variant_reports[variant] = asyncio.run(run_variant(
            variant, corpus, test_rows, embedding_function, top_k,
            rewrite_failure_rate, rerank_failure_rate, chroma_path,
        ))

    comparisons: Dict[str, Any] = {}
    acceptance: Dict[str, Any] = {}
    failures: List[str] = []
    r1 = variant_reports.get("r1")
    r4 = variant_reports.get("r4")
    if r1 is not None:
        names = ["recall_at_5", "recall_at_10", "precision_at_3", "mrr_at_10",
                 "ndcg_at_10", "fact_coverage", "duplicate_top3"]
        for variant in variants:
            if variant == "r1":
                continue
            comparisons[variant] = paired_comparison(
                test_rows,
                per_query_metric_arrays(variant_reports[variant]["metrics"]["per_query"], names),
                per_query_metric_arrays(r1["metrics"]["per_query"], names),
                names,
            )
        if r4 is not None:
            acceptance = acceptance_report(
                r4["metrics"]["summary"],
                r1["metrics"]["summary"],
                comparisons.get("r4", {}),
                r4["guard"]["p95_latency_cold_ms"],
                r1["guard"]["p95_latency_cold_ms"],
            )
            failures = [name for name, passed in acceptance.get("checks", {}).items() if not passed]
    report = {
        "title": "RAG Retrieval Variant Experiment (real Chroma all-MiniLM-L6-v2)",
        "artifact_type": "rag-retrieval-chroma-v1",
        "metadata": metadata(config, "all-MiniLM-L6-v2"),
        "config": config,
        "variants": variant_reports,
        "comparisons_vs_r1": comparisons,
        "acceptance": acceptance,
        "failures": failures,
        "overall_passed": acceptance.get("passed", False) if acceptance else True,
        "note": "Embedding 为真实 Chroma 默认模型；rewrite/rerank 仍为确定性代理，最终结论需真实 LLM 层。",
    }
    report["artifacts"] = write_report(report, output_dir, "rag-retrieval-chroma")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--variants", nargs="*", default=["r0", "r1", "r2", "r3", "r4"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    args = parser.parse_args()
    report = run_rag_chroma(args.output_dir, args.variants, top_k=args.top_k,
                            chroma_path=args.chroma_path, onnx_path=args.onnx_path)
    print(report["artifacts"]["json"])


if __name__ == "__main__":
    main()
