"""离线重评保留的320+717条记录，在批准池内重新选择首条示范。"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mymind"))
from agents.customer_policy.protocol import PublicTask, VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_private_standards, load_policy_support, validate_task_pairs
from experiments.agentic_rl.evaluate import process_errors as formal_process_errors
from review_stage5 import BATCHES, process_errors as calibration_process_errors

REVISION = "pre-stage7-grading-v13-20261014"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def original_records(paths):
    """一次读取所需归档成员，不解包或改写原文件。"""
    refs = {(r["host"], r["path"]): r for r in read(ROOT / "agentic_runs/archive/index.json")["files"]}
    wanted, records = defaultdict(dict), {}
    for path in paths:
        if (ROOT / path).is_file():
            records[path] = read(ROOT / path)
        else:
            ref = refs[("local", path)]
            wanted[ref["archive"]][ref["member"]] = path
    for archive_name, members in wanted.items():
        with tarfile.open(ROOT / "agentic_runs/archive" / archive_name) as archive:
            for member in archive:
                if member.name in members:
                    records[members[member.name]] = json.loads(archive.extractfile(member).read())
    assert set(records) == set(paths)
    return records


def historical_grader():
    path = ROOT / "agentic_runs/stage6-20261013/freeze-v4-20261014/snapshot/mymind/experiments/agentic_rl/grading.py"
    spec = importlib.util.spec_from_file_location("grading_frozen_v11", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.grade_result


def main(output):
    output = Path(output).resolve()
    assert not (output / "regrade_summary.json").exists(), "重评产物已存在，使用新目录"
    old_grade = historical_grader()
    run5 = ROOT / "agentic_runs/stage5-20261012"
    run6 = ROOT / "agentic_runs/stage6-20261013"
    audit5 = read(run5 / "grading_audit.json")
    audit6 = read(run6 / "post-review-v4-20261014/grading_audit.json")
    paths5 = ["agentic_runs/stage5-20261012/" + r["path"] for r in audit5]
    paths6 = [r["source_record"].replace("runs/", "agentic_runs/", 1) for r in audit6]
    records = original_records(paths5 + paths6)
    deltas, summaries, selections = [], {}, {}
    pool = read(run6 / "common_pool.json")["effective_sft_rl_pool"]
    standards_deltas = []
    for stage, audit, paths, mode, filename in (
        ("stage5", audit5, paths5, "calibration", "private_standards.v12.json"),
        ("stage6", audit6, paths6, "formal", "private_standards.v5.json")):
        data = ROOT / "data/agentic_rl" / mode
        old_standards = load_private_standards(data / ("private_standards.v4.json" if stage == "stage6" else "private_standards.json"))
        standards = load_private_standards(output / "standards" / mode / filename)
        clauses = load_policy_support(data / "policy_support.json")
        tasks = {r["task_id"]: PublicTask.model_validate(r) for r in read(data / "public_tasks.json")}
        validate_task_pairs(tasks, standards, clauses)
        for task_id, standard in standards.items():
            prev = old_standards[task_id]
            assert standard.model_dump(exclude={"expectations"}) == prev.model_dump(exclude={"expectations"})
            assert [e for e in standard.expectations if e.required] == [e for e in prev.expectations if e.required]
            if standard != prev:
                standards_deltas.append(dict(stage=stage, task_id=task_id, split=standard.split,
                    added=[e.model_dump() for e in standard.expectations if e not in prev.expectations],
                    removed=[e.model_dump() for e in prev.expectations if e not in standard.expectations]))
        rows, first = [], {}
        ordered = list(zip(audit, paths))
        if stage == "stage5":
            ordered.sort(key=lambda pair: (BATCHES.index(pair[0]["batch"]) if pair[0]["batch"] in BATCHES else len(BATCHES),
                                            pair[0]["path"]))
        else:
            ordered.sort(key=lambda pair: (pair[0]["task_id"], pair[0]["candidate"]))
        for original, path in ordered:
            row = records[path]
            task_id = row["task_id"]
            assert row["candidate"] == original["candidate"]
            trajectory = VisibleTrajectory.model_validate(row["trajectory"])
            previous = original["grade"] if stage == "stage6" else original["reviewed_grade"]
            assert old_grade(old_standards[task_id], row["raw_result"], trajectory, clauses).to_dict() == previous, path
            grade = grade_result(standards[task_id], row["raw_result"], trajectory, clauses).to_dict()
            process = (formal_process_errors if stage == "stage6" else calibration_process_errors)(row)
            assert process == original["process_errors"], path
            is_teacher = stage == "stage6" or original["batch"] in BATCHES
            qualified = bool(is_teacher and grade["complete_success"] and not row["infrastructure_failure"]
                             and not row["runtime_error"] and not process)
            if stage == "stage6":
                assert row["messages"][:2] == tasks[task_id].messages()
                assert trajectory.materials == tasks[task_id].materials()
                tools = [m for m in row["messages"] if m["role"] == "tool"]
                assert [json.loads(m["content"]) for m in tools] == [t.observation.model_dump() for t in trajectory.searches]
            entry = dict(task_id=task_id, candidate=row["candidate"], source_record=path,
                group_id=standards[task_id].group_id, previous_grade=previous, grade=grade,
                qualified=qualified, selected=False, process_errors=process,
                runtime_error=row["runtime_error"], infrastructure_failure=row["infrastructure_failure"],
                grading_revision=REVISION)
            if stage == "stage5":
                entry["batch"] = original["batch"]
            rows.append(entry)
            if qualified and task_id not in first:
                selected = deepcopy(row)
                selected.update(source_record=path, previous_grade=previous, grade=grade, grading_revision=REVISION)
                if stage == "stage5":
                    selected.update(collection_batch=original["batch"], original_file=original["path"])
                first[task_id] = selected
            if previous != grade:
                delta = dict(stage=stage, **entry, reward_changed=previous["reward"] != grade["reward"],
                    severe_changed=previous["severe_error"] != grade["severe_error"],
                    success_changed=previous["complete_success"] != grade["complete_success"],
                    reasons_added=[e for e in grade["errors"] if e not in previous["errors"]],
                    reasons_removed=[e for e in previous["errors"] if e not in grade["errors"]])
                deltas.append(delta)
        approved = list(pool) if stage == "stage6" else list(tasks)
        accepted = [first[task_id] for task_id in approved if task_id in first]
        selected_paths = {r["source_record"] for r in accepted}
        for entry in rows:
            entry["selected"] = entry["source_record"] in selected_paths
        outside = [dict(task_id=task_id, candidate=r["candidate"], source_record=r["source_record"],
                        group_id=standards[task_id].group_id, grade=r["grade"], in_approved_pool=False)
                   for task_id, r in first.items() if task_id not in approved]
        old_selected = read((run6 / "post-review-v4-20261014" if stage == "stage6" else run5) / "accepted_trajectories.json")
        old_by_id = {r["task_id"]: r for r in old_selected}
        changes = []
        for task_id in sorted(set(old_by_id) | {r["task_id"] for r in accepted}):
            old, new = old_by_id.get(task_id), first.get(task_id)
            old_key = ((old.get("collection_batch"), old["candidate"]) if stage == "stage5" else old["candidate"]) if old else None
            new_key = ((new.get("collection_batch"), new["candidate"]) if stage == "stage5" else new["candidate"]) if new else None
            if old_key != new_key:
                changes.append(dict(task_id=task_id, old_selection=old_key, new_selection=new_key,
                                    source_record=new["source_record"] if new else None))
        missing = [task_id for task_id in approved if task_id not in first]
        write(output / stage / "grading_audit.json", rows)
        write(output / stage / "accepted_trajectories.json", accepted)
        write(output / stage / "selection_changes.json", changes)
        write(output / stage / "outside_pool_qualified.json", outside)
        write(output / stage / "pool_without_demonstration.json", missing if stage == "stage6" else [])
        write(output / stage / "unqualified_tasks.json", [t for t in tasks if t not in first])
        selections[stage] = accepted
        summaries[stage] = dict(records=len(rows), previous_rewards=dict(Counter(str(r["previous_grade"]["reward"]) for r in rows)),
            new_rewards=dict(Counter(str(r["grade"]["reward"]) for r in rows)),
            previous_severe=sum(r["previous_grade"]["severe_error"] for r in rows), new_severe=sum(r["grade"]["severe_error"] for r in rows),
            previous_complete=sum(r["previous_grade"]["complete_success"] for r in rows), new_complete=sum(r["grade"]["complete_success"] for r in rows),
            qualified_candidates=sum(r["qualified"] for r in rows), qualified_tasks=len(first), selected=len(accepted),
            selection_changes=changes, outside_pool_qualified=outside, pool_without_demonstration=missing if stage == "stage6" else [],
            changed_rewards=sum(d["reward_changed"] for d in deltas if d["stage"] == stage),
            changed_grade_records=sum(d["stage"] == stage for d in deltas))
        if stage == "stage6":
            for split in ("train", "validation", "test"):
                write(output / "standards/formal" / f"{split}_private_standards.v5.json",
                      [s.model_dump() for s in standards.values() if s.split == split])
            write(output / "common-pool-158/public_tasks.json", [tasks[t].model_dump() for t in pool])
            write(output / "common-pool-158/private_standards.json", [standards[t].model_dump() for t in pool])
            write(output / "common_pool.json", dict(effective_sft_rl_pool=pool, count=158,
                status="修复完成，待独立复审", frozen=False, new_pool_members=0, removed_pool_members=0,
                original_pool="agentic_runs/stage6-20261013/common_pool.json",
                qualified_outside_pool=[r["task_id"] for r in outside], pool_without_demonstration=missing,
                training_started=False))
    write(output / "score_deltas.json", deltas)
    write(output / "standards_deltas.json", standards_deltas)
    summary = dict(grading_revision=REVISION, status="修复完成，待独立复审", frozen=False,
        stages=summaries, standards_changes=dict(Counter((d["stage"] + ":" + d["split"]) for d in standards_deltas)),
        original_grade_reproduced=1037, raw_records_modified=0, public_tasks_modified=0,
        model_calls=0, parameter_updates=0, final_test_policy_runs=0)
    write(output / "regrade_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
