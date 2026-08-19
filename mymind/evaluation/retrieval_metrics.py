"""
RAG 检索实验指标与统计。

指标定义（均为逐查询计算后取均值）：
  - Recall@k：Top-k 中出现至少一个相关 source_id 的查询占比。
  - Precision@3：Top-3 中首次出现的相关 source 数量 / 3。
  - MRR@10：首个相关 source 排名倒数的均值。
  - nDCG@10：source 级别 0/1/2 分级增益；同一 source 重复出现时第二次及以后增益为 0。
  - 关键事实覆盖率：每个 must_recall_fact 是否出现在 Top-k 内容中的比例。
  - 无答案误召回率：no_answer 查询返回非空结果的占比。
  - Top-3 重复 chunk 率：Top-3 内出现重复 chunk_id（无 id 时按规范化内容）的查询占比。
  - bootstrap 95% CI：按查询自助采样，配对比较两变体逐查询指标差。
"""
from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def result_source_id(item: Dict[str, Any]) -> str:
    return str(
        item.get("source_id")
        or item.get("sourceId")
        or item.get("_source_id")
        or ""
    )


def result_chunk_key(item: Dict[str, Any]) -> str:
    chunk_id = item.get("chunk_id") or item.get("chunkId") or item.get("id")
    if chunk_id:
        return f"id:{chunk_id}"
    return "content:" + _norm(item.get("content", ""))


def query_metrics(row: Dict[str, Any], items: Sequence[Dict[str, Any]], top_k: int = 10) -> Dict[str, Any]:
    """单查询指标。items 为最终返回结果（已重排、已去重）。"""
    relevant_sources = set(row.get("relevant_source_ids") or [])
    grades = {str(source): int(grade) for source, grade in (row.get("relevance") or {}).items()}
    top_items = list(items[:top_k])
    source_positions: Dict[str, int] = {}
    seen_sources: set = set()
    gains: List[int] = []
    precision_hits = 0

    for position, item in enumerate(top_items, start=1):
        source = result_source_id(item)
        if source and source in relevant_sources and source not in seen_sources:
            seen_sources.add(source)
            source_positions[source] = position
            grade = grades.get(source, 1)
            gains.append(grade)
            if position <= 3:
                precision_hits += 1
        else:
            gains.append(0)

    recall_at_k = {
        k: float(any(source in relevant_sources for source in [
            result_source_id(item) for item in top_items[:k]
        ]))
        for k in (5, 10)
    } if relevant_sources else {5: 1.0, 10: 1.0}

    first_rank = min(source_positions.values()) if source_positions else 0
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted((grades.get(source, 1) for source in relevant_sources), reverse=True)[:top_k]
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0

    facts = row.get("must_recall_facts") or []
    fact_hits = 0
    corpus_text = _norm(" ".join(str(item.get("content", "")) for item in top_items))
    for fact in facts:
        if _norm(fact) and _norm(fact) in corpus_text:
            fact_hits += 1

    top3_keys = [result_chunk_key(item) for item in top_items[:3]]
    duplicate_top3 = len(top3_keys) != len(set(top3_keys))

    return {
        "query_id": row.get("id", ""),
        "category": row.get("category", ""),
        "query_type": row.get("query_type", ""),
        "recall_at_5": recall_at_k[5],
        "recall_at_10": recall_at_k[10],
        "precision_at_3": precision_hits / 3.0,
        "mrr_at_10": mrr,
        "ndcg_at_10": ndcg,
        "fact_coverage": fact_hits / len(facts) if facts else 1.0,
        "no_answer_false_positive": float(
            bool(row.get("no_answer")) and len(top_items) > 0
        ),
        "duplicate_top3": float(duplicate_top3),
        "first_relevant_rank": first_rank,
    }


def aggregate_metrics(rows: Sequence[Dict[str, Any]], items_by_query: Dict[str, Sequence[Dict[str, Any]]], top_k: int = 10) -> Dict[str, Any]:
    """逐查询指标汇总为数据集级报告。"""
    per_query: List[Dict[str, Any]] = []
    for row in rows:
        per_query.append(query_metrics(row, items_by_query.get(row["id"], []), top_k))

    no_answer_queries = [m for m in per_query if m["no_answer_false_positive"] in (0.0, 1.0) and m["query_type"] == "no_answer"]
    fpr = (
        sum(m["no_answer_false_positive"] for m in no_answer_queries) / len(no_answer_queries)
        if no_answer_queries else 0.0
    )

    metric_names = [
        "recall_at_5", "recall_at_10", "precision_at_3",
        "mrr_at_10", "ndcg_at_10", "fact_coverage",
        "no_answer_false_positive", "duplicate_top3",
    ]
    summary: Dict[str, float] = {}
    for name in metric_names:
        values = [m[name] for m in per_query]
        summary[name] = statistics.mean(values) if values else 0.0
    summary["no_answer_false_positive"] = fpr
    summary["duplicate_top3_rate"] = summary.pop("duplicate_top3")

    by_category = group_summary(per_query, "category")
    by_query_type = group_summary(per_query, "query_type")
    return {
        "summary": {name: round(value, 4) for name, value in summary.items()},
        "by_category": by_category,
        "by_query_type": by_query_type,
        "per_query": per_query,
        "query_count": len(per_query),
    }


def group_summary(per_query: Sequence[Dict[str, Any]], group_field: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for metric in per_query:
        groups.setdefault(str(metric.get(group_field, "")), []).append(metric)
    result: Dict[str, Dict[str, float]] = {}
    names = [
        "recall_at_5", "recall_at_10", "precision_at_3", "mrr_at_10",
        "ndcg_at_10", "fact_coverage", "duplicate_top3",
    ]
    for group, metrics in sorted(groups.items()):
        row_values: Dict[str, float] = {}
        for name in names:
            values = [m[name] for m in metrics]
            row_values[name] = round(statistics.mean(values), 4) if values else 0.0
        result[group] = row_values
    return result


def paired_bootstrap_delta(
    rows: Sequence[Dict[str, Any]],
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    samples: int = 2000,
    seed: int = 20260808,
    confidence: float = 0.95,
    cluster_by_partition: bool = True,
) -> Dict[str, float]:
    """配对 bootstrap，估计 (a - b) 的 95% CI。

    同一 partition 的不同改写共享标注，按 partition 做 cluster bootstrap，
    避免把改写查询当作独立样本而低估 CI 宽度。
    """
    if len(rows) < 2:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "samples": samples, "confidence": confidence, "clusters": 0}
    rng = random.Random(seed)
    deltas = [x - y for x, y in zip(metric_a, metric_b)]
    if cluster_by_partition and any(row.get("partition") for row in rows):
        clusters: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            clusters.setdefault(str(row.get("partition", index)), []).append(index)
        cluster_ids = list(clusters)
    else:
        clusters = {str(index): [index] for index in range(len(rows))}
        cluster_ids = list(clusters)
    boot = []
    cluster_count = len(cluster_ids)
    for _ in range(max(1, samples)):
        indices: List[int] = []
        for _ in range(cluster_count):
            indices.extend(clusters[rng.choice(cluster_ids)])
        sample = [deltas[index] for index in indices]
        boot.append(statistics.mean(sample))
    tail = (1.0 - confidence) / 2.0
    ordered = sorted(boot)
    lower = ordered[max(0, int(tail * len(ordered)) - 1)]
    upper = ordered[min(len(ordered) - 1, int((1.0 - tail) * len(ordered)) - 1)]
    return {
        "mean_delta": round(statistics.mean(deltas), 4),
        "ci_low": round(lower, 4),
        "ci_high": round(upper, 4),
        "samples": samples,
        "confidence": confidence,
        "clusters": cluster_count,
    }


def paired_comparison(
    rows: Sequence[Dict[str, Any]],
    metrics_a: Dict[str, Sequence[float]],
    metrics_b: Dict[str, Sequence[float]],
    metric_names: Iterable[str],
    samples: int = 2000,
    seed: int = 20260808,
) -> Dict[str, Dict[str, float]]:
    return {
        name: paired_bootstrap_delta(rows, metrics_a[name], metrics_b[name], samples, seed)
        for name in metric_names
    }


def per_query_metric_arrays(per_query: Sequence[Dict[str, Any]], names: Sequence[str]) -> Dict[str, List[float]]:
    return {
        name: [row[name] for row in per_query]
        for name in names
    }


def acceptance_report(
    r4_summary: Dict[str, float],
    r1_summary: Dict[str, float],
    comparisons: Dict[str, Dict[str, float]],
    r4_latency_p95: float,
    r1_latency_p95: float,
) -> Dict[str, Any]:
    """check.md 第六节验收规则的机械化判定。"""
    checks = {
        "recall_at_5_gte_90": r4_summary["recall_at_5"] >= 0.90,
        "recall_at_5_vs_r1_gte_5pp": r4_summary["recall_at_5"] - r1_summary["recall_at_5"] >= 0.05,
        "recall_at_5_ci_above_zero": comparisons.get("recall_at_5", {}).get("ci_low", -1) > 0,
        "mrr_not_down_over_2pp": r4_summary["mrr_at_10"] - r1_summary["mrr_at_10"] >= -0.02,
        "ndcg_not_down_over_2pp": r4_summary["ndcg_at_10"] - r1_summary["ndcg_at_10"] >= -0.02,
        "no_answer_fpr_lte_5": r4_summary["no_answer_false_positive"] <= 0.05,
        "top3_duplicate_zero": r4_summary["duplicate_top3_rate"] == 0.0,
        "p95_latency_within_1_2x": r4_latency_p95 <= r1_latency_p95 * 1.2,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds": {
            "recall_at_5": 0.90,
            "recall_vs_r1_pp": 0.05,
            "mrr_ndcg_tolerance_pp": -0.02,
            "no_answer_fpr": 0.05,
            "top3_duplicate": 0.0,
            "latency_factor": 1.2,
        },
    }
