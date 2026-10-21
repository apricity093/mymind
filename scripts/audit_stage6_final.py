"""独立重评已完成的真实候选，保留原标准评分并导出首条选择。"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mymind"))
from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.evaluate import process_errors
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs

WORK = Path(os.getenv("MYMIND_WORK_ROOT", str(REPO)))
RUN = WORK / ("runs" if WORK != REPO else "agentic_runs") / "stage6-20261013"
DATA = WORK / "data/agentic_rl/formal"
OUT = RUN / "post-review-v4-20261014"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    tasks = load_public_tasks(DATA / "train_public_tasks.json")
    original = load_private_standards(DATA / "train_private_standards.json")
    standards = load_private_standards(DATA / "train_private_standards.v4.json")
    clauses = load_policy_support(DATA / "policy_support.json")
    validate_task_pairs(tasks, standards, clauses)
    audits, changes, selected, gaps = [], [], [], []
    counts = defaultdict(lambda: dict(tasks=0, candidates=0, accepted=0))
    for task_id, task in tasks.items():
        standard = standards[task_id]
        group = counts[standard.group_id]
        group["tasks"] += 1
        first = None
        for candidate in (1, 2):
            path = RUN / "teacher-v3" / f"{task_id}-candidate-{candidate}.json"
            if not path.exists():
                continue
            row = read(path)
            trajectory = VisibleTrajectory.model_validate(row["trajectory"])
            assert row["task_id"] == task_id and row["candidate"] == candidate
            assert row["messages"][:2] == task.messages()
            assert row["trajectory"]["materials"] == [m.model_dump() for m in task.materials()]
            assert len(trajectory.searches) <= 3
            old_grade = grade_result(original[task_id], row["raw_result"], trajectory, clauses).to_dict()
            assert old_grade == row["grade"], (task_id, candidate, "原v3评分未能复算")
            grade = grade_result(standard, row["raw_result"], trajectory, clauses).to_dict()
            if grade != old_grade:
                assert standard.group_id in {"F01", "F02"}
                changes.append(dict(task_id=task_id, candidate=candidate, original=old_grade, current=grade))
            errors = process_errors(row)
            qualified = grade["complete_success"] and not row["infrastructure_failure"] and not row["runtime_error"] and not errors
            is_first = qualified and first is None
            if is_first:
                first = deepcopy(row)
                first.update(grade_before_task_standard_v4=old_grade, grade=grade,
                    source_record=f"runs/stage6-20261013/teacher-v3/{path.name}", standards_revision="stage6-standards-v4")
            audits.append(dict(task_id=task_id, group_id=standard.group_id, candidate=candidate,
                source_record=f"runs/stage6-20261013/teacher-v3/{path.name}", original_grade=old_grade,
                grade=grade, selected=is_first, qualified=qualified, process_errors=errors,
                runtime_error=row["runtime_error"], infrastructure_failure=row["infrastructure_failure"]))
            group["candidates"] += 1
        if first is None:
            gaps.append(dict(task_id=task_id, group_id=standard.group_id, disposition=standard.dispositions[0],
                candidates=[a["candidate"] for a in audits if a["task_id"] == task_id]))
        else:
            selected.append(first)
            group["accepted"] += 1
    assert len(audits) == 717 and len(tasks) == 400
    assert all(len(gap["candidates"]) == 2 for gap in gaps)
    assert len(selected) == len({r["task_id"] for r in selected})
    for row in selected:
        source = read(RUN / "teacher-v3" / f"{row['task_id']}-candidate-{row['candidate']}.json")
        for key in ("messages", "generations", "raw_result", "trajectory", "tool_diagnostics"):
            assert row[key] == source[key]
    config = read(RUN / "configs/teacher-v3.json")
    config.update(standards="data/agentic_rl/formal/train_private_standards.v4.json", standards_revision="stage6-standards-v4",
        task_ids=[r["task_id"] for r in selected], trajectories="runs/stage6-20261013/post-review-v4-20261014/accepted_trajectories.json",
        output_dir=f"runs/stage6-20261013/student-data-v4-{len(selected):03d}", context_tokens=16384,
        export_scope="合格示范子集，正式共同池尚未批准")
    write(RUN / "configs/student-data-v4.json", config)
    write(OUT / "grading_audit.json", audits)
    write(OUT / "grading_changes.json", changes)
    write(OUT / "accepted_trajectories.json", selected)
    write(OUT / "task_gaps.json", gaps)
    summary = dict(executed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        standards_revision="stage6-standards-v4", grading_revision="stage5-independent-review-v11", candidates=717,
        attempted_tasks=400, unstarted_tasks=0, accepted=len(selected), gaps=len(gaps),
        rewards=dict(Counter(str(a["grade"]["reward"]) for a in audits)),
        changed_grade_records=len(changes), changed_reward_records=sum(c["original"]["reward"] != c["current"]["reward"] for c in changes),
        covered_groups=sum(v["accepted"] > 0 for v in counts.values()), groups=dict(counts),
        domains=dict(Counter(tasks[r["task_id"]].role for r in selected)),
        dispositions=dict(Counter(standards[r["task_id"]].dispositions[0] for r in selected)),
        selected_search_counts=dict(Counter(len(r["trajectory"]["searches"]) for r in selected)),
        search_after_observation=sum(any(g["assistant_message"].get("tool_calls") and any(m["role"] == "tool" for m in g["input_messages"])
            for g in row["generations"]) for row in selected),
        new_model_calls=0, new_candidates=0, parameter_updates=0, final_test_policy_runs=0,
        human_test_confirmations=0, effective_sft_rl_pool=None, stage6_complete=False, data_frozen=False)
    write(OUT / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("candidates", "accepted", "gaps", "covered_groups", "domains", "dispositions", "changed_grade_records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
