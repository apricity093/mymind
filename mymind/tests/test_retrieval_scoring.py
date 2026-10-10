import json
from pathlib import Path

import pytest

from experiments.retrieval_scoring import (
    evaluate_retrieval, rescore_retrieval_group, summarize_retrieval,
)


def question(answerable=True):
    return {
        "id": "q", "family_id": "family", "question": "退款条件", "tags": ["condition"],
        "answerable": answerable,
        "evidence_groups": [{"alternatives": [{"source_id": "policy"}]}] if answerable else [],
    }


def test_original_precision_counts_one_source_for_three_chunks():
    items = [{"source_id": "policy", "chunk_id": str(i), "content": "不同段落"} for i in range(3)]
    row = evaluate_retrieval(question(), items)
    assert row["recall_at_5"] == row["recall_at_10"] == 1
    assert row["precision_at_3"] == pytest.approx(1 / 3)
    assert row["mrr_at_10"] == 1


def test_original_no_answer_contributes_recall_one_and_precision_zero():
    positive = evaluate_retrieval(question(), [{"source_id": "policy", "content": "退款"}])
    absent = evaluate_retrieval(question(False), [{"source_id": "policy", "content": "退款"}])
    summary = summarize_retrieval([positive, absent])
    assert summary["recall_at_5"] == 1
    assert summary["precision_at_3"] == pytest.approx(1 / 6)
    assert summary["mrr_at_10"] == 0.5
    assert summary["no_answer_false_positive"] == 1
    assert evaluate_retrieval(question(False), [])["no_answer_false_positive"] == 0


def test_rescore_preserves_ranking_and_removes_withdrawn_metrics():
    items = [{"source_id": "other", "content": "背景"}, {"source_id": "policy", "content": "退款"}]
    original = {"status": "measured", "coverage@5": 0.5, "all@5": 0, "mrr@10": 0,
                "rows": [{"id": "q", "retrieved": items, "coverage@5": 0.5, "all@5": False,
                          "mrr@10": 0, "wall_ms": 123}]}
    result = rescore_retrieval_group(original, {"q": question()})
    assert result["rows"][0]["retrieved"] == items
    assert result["rows"][0]["wall_ms"] == 123
    assert result["mrr_at_10"] == 0.5
    assert not {"coverage@5", "all@5", "mrr@10"} & result.keys()
    assert not {"coverage@5", "all@5", "mrr@10"} & result["rows"][0].keys()
    assert original["coverage@5"] == 0.5


def test_saved_summary_judge_uses_answerable_denominator_and_citation_errors(tmp_path):
    from experiments.summarize_optimization_v2 import summarize
    data = json.loads((Path(__file__).resolve().parents[1] / "data/eval/rag_optimization_v2.json").read_text(encoding="utf-8"))
    cases = [next(q for q in data["questions"] if q["answerable"]),
             next(q for q in data["questions"] if not q["answerable"])]
    directory = tmp_path / "answers"
    directory.mkdir()
    originals = {}
    for index, case in enumerate(cases):
        record = {"question": case, "status": "measured", "rerank_status": "success",
                  "audit": {"facts_supported": True, "conditions_complete": index == 1,
                            "unanswerable_fabricated": False, "unsupported_claims": [],
                            "citation_errors": ["错误引用"] if index == 0 else []}}
        path = directory / f"qwen-test-{case['id']}.json"
        originals[path] = json.dumps(record, ensure_ascii=False)
        path.write_text(originals[path], encoding="utf-8")
    result = summarize(tmp_path)
    stats = result["answer_summary"]["qwen-test"]
    assert stats["measured"] == 2 and stats["answerable_n"] == 1
    assert stats["rates"]["conditions_complete"] == 0.5
    assert stats["answerable_rates"]["answerable_conditions_complete"] == 0
    assert stats["citation_error_answers_n"] == stats["citation_errors_n"] == 1
    assert stats["unanswerable_fabrication_rate"] == 0
    assert stats["all_checks_passed"] == 1
    assert all(path.read_text(encoding="utf-8") == text for path, text in originals.items())
