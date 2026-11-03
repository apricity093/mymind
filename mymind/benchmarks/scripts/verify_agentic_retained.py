"""核对正式训练输入、示范来源和监督mask，不调用模型或重新判分。"""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify(root, host):
    root = Path(root).resolve()
    runs = root / ("agentic_runs" if host == "local" else "runs")

    def locate(name):
        if host == "local" and name.startswith("runs/"):
            name = name.replace("runs/", "agentic_runs/", 1)
        elif host == "server" and name.startswith("agentic_runs/"):
            name = name.replace("agentic_runs/", "runs/", 1)
        return root / name

    load = read
    active = load(runs / "active_run_baseline.json")
    baseline = load(runs / active["directory"] / active["baseline"])
    assert baseline["revision"] == "stage7-baseline-20261014-v13-v5-158"
    public = load(locate(baseline["public_tasks"]))
    standards = load(locate(baseline["standards"]))
    assert len(public) == len(standards) == 600
    assert {row["task_id"] for row in public} == {row["task_id"] for row in standards}
    assert Counter(row["split"] for row in standards) == {"train": 400, "validation": 80, "test": 120}
    standard_by_id = {row["task_id"]: row for row in standards}
    pool = load(locate(baseline["common_pool"]))["effective_sft_rl_pool"]
    selected = load(locate(baseline["demonstrations"]))
    students = load(locate(baseline["student_data"]))
    assert len(pool) == len(set(pool)) == 158
    assert [row["task_id"] for row in selected] == [row["task_id"] for row in students] == pool
    assert all(standard_by_id[tid]["split"] == "train" for tid in pool)
    for row, student in zip(selected, students):
        original = load(locate(row["source_record"]))
        for key in ("task_id", "messages", "generations", "raw_result", "trajectory", "tool_diagnostics"):
            assert row[key] == original[key], (row["task_id"], key)
        positions = {i for span in student["spans"] for i in range(span["start"], span["end"])}
        assert {i for i, value in enumerate(student["labels"]) if value != -100} == positions
        assert all(student["labels"][i] == student["input_ids"][i] for i in positions)
    review = load(runs / "stage6-20261013/independent-ai-review-20261014/task_reviews.json")
    assert len(review["rows"]) == 201 and review["verdict"]["can_freeze_stage6"]
    assert not review["verdict"]["blocking_issues"]
    frozen = load(runs / "stage6-20261013/freeze-v4-20261014/data_manifest.frozen.json")
    assert frozen["acceptance"]["human_confirmed"] is False
    assert frozen["acceptance"]["reviewed"] == {"train": 60, "validation": 21, "test": 120}
    assert locate(baseline["tokenizer_template"]).is_file()
    return dict(host=host, status="passed", baseline=baseline["revision"], original_tasks=600,
                common_pool=158, demonstration_sources=158, supervised_masks=158,
                validation=80, test=120, ai_review_items=201, model_calls=0, parameter_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", choices=("local", "server"), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.host), ensure_ascii=False, indent=2))
