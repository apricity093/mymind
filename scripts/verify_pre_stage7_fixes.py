"""核验修复候选、固定池、原轨迹与学生监督，保留旧冻结核验。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mymind"))
from experiments.agentic_rl.sft_data import student_messages
from regrade_pre_stage7_fixes import original_records


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_candidate(root, manifest, host="local", records=None):
    root = Path(root).resolve()
    manifest = read(manifest)
    candidate = root / manifest["directory"]
    source_snapshot = candidate / manifest.get("source_directory", "source")
    assert manifest["status"] in {"修复完成，待独立复审", "已验收，正式运行基线已固定（本会话有限复核）"} and not manifest["frozen"]
    assert not manifest["stage7_8_authorized"]
    runs = root / ("agentic_runs" if host == "local" else "runs")
    source_root = root if host == "local" else root / "src"
    baseline_source = None
    if manifest.get("accepted_run_baseline"):
        active = read(runs / "active_run_baseline.json")
        baseline_source = runs / active["directory"] / "source"
        for name in active["source_files"]:
            assert (baseline_source / name).read_bytes() == (source_root / name).read_bytes(), name
        # 原候选快照仍代表验收时F1/F2/F3实现，训练接入的后续修改另有基线。
        assert (source_snapshot / "mymind/experiments/agentic_rl/grading.py").read_bytes() == (source_root / "mymind/experiments/agentic_rl/grading.py").read_bytes()
    else:
        for name in manifest["source_files"]:
            assert (source_snapshot / name).read_bytes() == (source_root / name).read_bytes(), name
    frozen = read(runs / "stage6-20261013/freeze-v4-20261014/data_manifest.frozen.json")
    pool = read(candidate / "common_pool.json")["effective_sft_rl_pool"]
    assert pool == read(runs / "stage6-20261013/common_pool.json")["effective_sft_rl_pool"]
    assert set(pool) == set(frozen["effective_training_pool"]["task_ids"]) and len(pool) == 158
    standards = read(candidate / "standards/formal/private_standards.v5.json")
    old_standards = read(root / "data/agentic_rl/formal/private_standards.v4.json")
    assert [r["task_id"] for r in standards] == [r["task_id"] for r in old_standards]
    standard_changes = []
    for old, new in zip(old_standards, standards):
        assert {k:v for k,v in old.items() if k != "expectations"} == {k:v for k,v in new.items() if k != "expectations"}
        assert [e for e in old["expectations"] if e["required"]] == [e for e in new["expectations"] if e["required"]]
        if old != new:
            assert new["group_id"] == "B01" and new["split"] == "train"
            assert all(e in new["expectations"] for e in old["expectations"])
            standard_changes.append(new["task_id"])
    assert len(standard_changes) == 40
    for split in ("train", "validation", "test"):
        assert read(candidate / "standards/formal" / f"{split}_private_standards.v5.json") == [r for r in standards if r["split"] == split]
    old_public = {r["task_id"]:r for r in read(root / "data/agentic_rl/formal/public_tasks.json")}
    assert read(candidate / "common-pool-158/public_tasks.json") == [old_public[t] for t in pool]
    by_id = {r["task_id"]:r for r in standards}
    assert read(candidate / "common-pool-158/private_standards.json") == [by_id[t] for t in pool]
    audit = read(candidate / "stage6/grading_audit.json")
    assert len(audit) == 717
    selected = read(candidate / "stage6/accepted_trajectories.json")
    assert [r["task_id"] for r in selected] == pool
    first = {}
    for entry in sorted(audit,key=lambda r:(r["task_id"],r["candidate"])):
        if entry["qualified"]:
            first.setdefault(entry["task_id"], entry)
    assert sum(r["selected"] for r in audit) == 158
    outside = read(candidate / "stage6/outside_pool_qualified.json")
    assert {r["task_id"] for r in outside} == set(first)-set(pool)
    assert all(not r["in_approved_pool"] for r in outside)
    assert not read(candidate / "stage6/pool_without_demonstration.json")
    paths = [r["source_record"] for r in selected]
    if records is None:
        if host != "local":
            raise ValueError("服务器修复验证须由 retained 核验提供原记录")
        records = original_records(paths)
    students = read(candidate / "stage6/student_data.json")
    old_students = {r["task_id"]:r for r in read(runs / "stage6-20261013/student-data-v4-158/student_data.json")}
    assert [r["task_id"] for r in students] == pool
    changed, supervision = [], 0
    for selected_row, student in zip(selected, students):
        task_id, path = selected_row["task_id"], selected_row["source_record"]
        assert path == first[task_id]["source_record"] and selected_row["grade"] == first[task_id]["grade"]
        raw = records[path]
        for key in ("messages","generations","raw_result","trajectory","tool_diagnostics"):
            assert selected_row[key] == raw[key], (task_id,key)
        assert student["messages"] == student_messages(raw)
        assert student["messages"][-1]["content"] == raw["generations"][-1]["raw_text"]
        assert all(set(m) <= {"role","content","tool_calls","name","tool_call_id"} for m in student["messages"])
        assert [s["message_index"] for s in student["spans"]] == [i for i,m in enumerate(student["messages"]) if m["role"] == "assistant"]
        positions = set()
        for span in student["spans"]:
            assert 0 <= span["start"] < span["end"] <= len(student["input_ids"])
            assert not (positions & set(range(span["start"],span["end"])))
            positions.update(range(span["start"],span["end"]))
            message = student["messages"][span["message_index"]]
            assert span["kind"] == ("tool_action" if message.get("tool_calls") else "final")
            if span["kind"] == "final":
                assert span["text"].startswith(message["content"])
        assert student["spans"][-1]["kind"] == "final"
        assert len(student["input_ids"]) == len(student["labels"]) == len(student["attention_mask"]) <= 16384
        assert all(v == 1 for v in student["attention_mask"])
        assert all(label == (token if i in positions else -100) for i,(token,label) in enumerate(zip(student["input_ids"],student["labels"])))
        supervision += len(positions)
        if student != old_students[task_id]:
            changed.append(task_id)
    assert changed == [r["task_id"] for r in read(candidate / "stage6/selection_changes.json")]
    assert read(candidate / "stage5/student_data.json") == read(runs / "stage5-20261012/student-data-independent-fix-20261013/student_data.json")
    assert (candidate / "chat_template.jinja").read_bytes() == (runs / "stage6-20261013/student-data-v4-158/chat_template.jinja").read_bytes()
    budget = read(candidate / "remote-budget.json")
    assert budget["metadata"]["versions"]["trl"] == "1.10.0"
    assert budget["metadata"]["installed_trl_matches_review_reference"]
    assert all(p["actual_visible_evidence_and_mask_match"] and p["dapo_effective_token_denominator"] == p["policy_tokens"] for p in budget["probes"])
    summary = read(candidate / "regrade_summary.json")
    assert summary["original_grade_reproduced"] == 1037
    current_grading_check = None
    if manifest.get("grading_verification"):
        current_grading_check = read(candidate / manifest["grading_verification"])
        assert current_grading_check["grading_revision"] == manifest["grading_revision"]
        assert current_grading_check["compared_to"] == summary["grading_revision"]
        assert current_grading_check["checked"]["stage5"]["records"] == 320
        assert current_grading_check["checked"]["stage6"]["records"] == len(audit)
        assert current_grading_check["total_records"] == 1037
        assert current_grading_check["changed_records"] == 0 and not current_grading_check["changes"]
        assert current_grading_check["existing_scores_reusable"]
    return dict(status="passed", status_of_candidate=manifest["status"], frozen=False,
        grading_revision=manifest["grading_revision"],
        retained_scores_checked_under_current_revision=current_grading_check["total_records"] if current_grading_check else None,
        original_grades_reproduced=1037, current_source_matches=len(active["source_files"]) if baseline_source else len(manifest["source_files"]),
        current_source_basis="accepted_run_baseline" if baseline_source else "repair_candidate",
        fixed_pool=158, demonstrations=158, first_candidate_rule=True, changed_exports=changed,
        unchanged_formal_exports=158-len(changed), unchanged_calibration_exports=30,
        mask_sequences=158, supervised_tokens=supervision, max_sequence_tokens=max(len(r["input_ids"]) for r in students),
        outside_pool_qualified=[r["task_id"] for r in outside], new_pool_members=0, removed_pool_members=0,
        standards_changed=40, validation_test_standard_changes=0, controlled_budget_cases=len(budget["probes"]),
        model_calls=0, parameter_updates=0, new_independent_acceptance=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_candidate(args.root, args.candidate),ensure_ascii=False,indent=2))
