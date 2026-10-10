"""将冻结实验题的来源标注交给原项目检索指标函数。"""
import statistics

from evaluation.retrieval_metrics import query_metrics


RETRIEVAL_METRICS = (
    "recall_at_5", "recall_at_10", "precision_at_3", "mrr_at_10", "ndcg_at_10",
)


def evaluate_retrieval(question, ranked):
    sources = sorted({alternative["source_id"]
                      for group in question["evidence_groups"]
                      for alternative in group["alternatives"]})
    gold = {
        "id": question["id"],
        "relevant_source_ids": sources,
        "relevance": {source: 2 for source in sources},
        "no_answer": not question["answerable"],
        "query_type": "answerable" if question["answerable"] else "no_answer",
    }
    metrics = query_metrics(gold, ranked, top_k=10)
    return {
        "id": question["id"], "family_id": question["family_id"],
        "question": question["question"], "tags": question["tags"],
        "answerable": question["answerable"], "retrieved": list(ranked[:10]),
        **{name: metrics[name] for name in RETRIEVAL_METRICS},
        **{name: metrics[name] for name in (
            "no_answer_false_positive", "duplicate_top3", "first_relevant_rank",
        )},
    }


def summarize_retrieval(rows):
    """与原算法一致：所有查询参与均值，无答案非空率单独汇总。"""
    absent = [row for row in rows if not row["answerable"]]
    result = {name: statistics.mean(row[name] for row in rows) if rows else 0.0
              for name in RETRIEVAL_METRICS}
    result["no_answer_false_positive"] = (
        statistics.mean(row["no_answer_false_positive"] for row in absent) if absent else 0.0
    )
    result["duplicate_top3_rate"] = (
        statistics.mean(row["duplicate_top3"] for row in rows) if rows else 0.0
    )
    return result


def rescore_retrieval_group(value, questions):
    """仅重算已保存的排名；移除被撤销的派生指标，不调用模型。"""
    removed = {"coverage@1", "coverage@3", "coverage@5", "all@5", "mrr@10"}
    rows = [{**{key: item for key, item in row.items() if key not in removed},
             **evaluate_retrieval(questions[row["id"]], row["retrieved"])}
            for row in value["rows"]]
    return {
        **{key: item for key, item in value.items() if key not in removed},
        **summarize_retrieval(rows), "rows": rows,
        "metric_definition": "evaluation.retrieval_metrics.query_metrics; mean over all queries",
        "answerable_summary": summarize_retrieval([row for row in rows if row["answerable"]]),
    }
