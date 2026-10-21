"""正式数据归属、公开隔离与训练侧标准核验；生成待人工材料。"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mymind"))
from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs

WORK = Path(os.getenv("MYMIND_WORK_ROOT", str(REPO)))
DATA = WORK / "data/agentic_rl/formal"
RUN = WORK / ("runs" if WORK != REPO else "agentic_runs") / "stage6-20261013"


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def public_body(task):
    return task.user_text + "\n" + "\n".join(r.content for r in task.records)


def normalize(text):
    text = re.sub(r"(?:RF-\d+-\d|REQ-\d+|ORDER-\d+|CALL-\d+|ORD-RF-\d+-\d|TX-RF-\d+-\d|rid-\d+)", "对象", text)
    return "".join(text.split())


def supported_training_result(task, standard, clauses, optional=False):
    values = list(clauses.values())
    refs = {c.clause_id: f"search:1:{i+1}" for i, c in enumerate(values)}
    trajectory = VisibleTrajectory(task_id=task.task_id, materials=task.materials(), searches=[{
        "action": {"query": "训练标准单元核验人工构造正文，不能作为教师示范"},
        "observation": {"request_number": 1, "status": {"reference_id": "search:1:status", "state": "candidates"},
            "snippets": [dict(reference_id=refs[c.clause_id], source_id=c.source_id, title="核验正文",
                section_path=[], content=c.text) for c in values]}}])
    result = dict(disposition=standard.dispositions[0], facts=[], reasons=[], next_steps=[], missing_info=[])
    for e in standard.expectations:
        if not e.required and not optional:
            continue
        item = deepcopy(e.alternatives[0])
        item["evidence"] = list(dict.fromkeys(g[0] if g[0] in standard.materials else refs[g[0]] for g in e.supports))
        result[e.section].append(item)
    return result, trajectory


def card(task, standard, source, spec, clauses):
    lines = [f"## {task.task_id}", "", f"组合：{standard.group_id}；划分：{standard.split}；来源：{source['source_id']}",
        "", "人工状态：待逐题确认。AI预核查与自动字段核查不等于人工确认。", "", "### 公开题面", "",
        task.user_text]
    for r in task.records:
        lines += ["", f"附带记录 {r.reference_id}：", "", r.content]
    lines += ["", "### 独立私有标准", "", f"整体处置：`{standard.dispositions[0]}`。", "",
        "必要事实、逐对象判断、下一步及待补项：", "", "```json"]
    lines += [json.dumps({section: [dict(alternatives=e.alternatives, rule_support=e.supports) for e in standard.expectations
        if e.section == section and e.required] for section in ("facts", "reasons", "next_steps", "missing_info")}, ensure_ascii=False, indent=2), "```"]
    lines += ["", "对象与状态构造（便于对照题面，不能进入策略输入）：", "", "```json",
        json.dumps(spec["instance"], ensure_ascii=False, indent=2), "```", "", "明确未提供的状态：", "", "```json",
        json.dumps(standard.unprovided_user_facts, ensure_ascii=False, indent=2), "```", "", "规则正文依据：", ""]
    supports = sorted({s for e in standard.expectations if e.required for g in e.supports for s in g if s in clauses})
    lines += [f"- `{s}`：{clauses[s].text}" for s in supports]
    lines += ["", "来源/核查记录：全新规则构造实例，未复用开发、校准题或教师示范；组合与来源族只在本划分。",
        "AI预核查：组合模板、对象/范围、未知和否定关系复核；自动检查公开/私有材料及字段。", "",
        "待人工确认：题面是否可解；必要事实及处置是否忠于题面和规则；是否遗漏合理可接受结果；对象、未知及否定是否正确。",
        "有异议时记录本task_id、具体原句、应改标准与规则依据。未确认项保留待核查。", ""]
    return "\n".join(lines)


def main():
    tasks = load_public_tasks(DATA / "public_tasks.json")
    standards = load_private_standards(DATA / "private_standards.json")
    clauses = load_policy_support(DATA / "policy_support.json")
    sources = read("sources.json")
    specs = {s["task_id"]: s for s in read("instance_specs.json")}
    allocation = read("group_allocation.json")
    validate_task_pairs(tasks, standards, clauses)
    assert len(tasks) == len(sources) == len(specs) == 600
    expected = {g["group_id"]: (g["split"], g["count"]) for g in allocation}
    actual = Counter(s.group_id for s in standards.values())
    assert actual == {g: n for g, (_, n) in expected.items()}
    assert Counter(s.split for s in standards.values()) == {"train": 400, "validation": 80, "test": 120}
    assert Counter(tasks[t].role for t, s in standards.items() if s.split == "train") == {"billing": 200, "technical": 200}
    assert Counter(tasks[t].role for t, s in standards.items() if s.split == "validation") == {"billing": 40, "technical": 40}
    assert Counter(tasks[t].role for t, s in standards.items() if s.split == "test") == {"billing": 60, "technical": 60}
    normalized, families = {}, defaultdict(set)
    for source in sources:
        t, s = tasks[source["task_id"]], standards[source["task_id"]]
        assert s.split == expected[s.group_id][0] == source["split"]
        families[source["source_family"]].add(s.split)
        text = public_body(t)
        assert s.group_id not in str(t.messages()) and t.task_id not in str(t.messages())
        assert all(word not in text for word in ("disposition", "expectations", "grading_revision", "训练集", "测试集"))
        key = normalize(text)
        assert key not in normalized, ("仅替换对象的重复题面", t.task_id, normalized.get(key))
        normalized[key] = t.task_id
        assert not source["calibration_reuse"] and not source["teacher_reuse"]
        assert specs[t.task_id]["disposition"] == s.dispositions[0]
        for section in s.severe_items:
            assert section in {"facts", "reasons", "next_steps", "missing_info"}
    assert all(len(splits) == 1 for splits in families.values())
    training_checks = []
    for t, s in standards.items():
        if s.split != "train":
            continue
        result, visible = supported_training_result(tasks[t], s, clauses)
        grade = grade_result(s, result, visible, clauses)
        assert grade.complete_success, (t, grade.to_dict())
        optional, optional_visible = supported_training_result(tasks[t], s, clauses, optional=True)
        optional_grade = grade_result(s, optional, optional_visible, clauses)
        assert optional_grade.complete_success, (t, "允许的可选内容", optional_grade.to_dict())
        wrong = deepcopy(result)
        wrong["disposition"] = next(d for d in ("answer", "escalate", "request_info") if d not in s.dispositions)
        assert not grade_result(s, wrong, visible, clauses).complete_success
        training_checks.append(dict(task_id=t, supported_result_success=True, wrong_disposition_rejected=True))
    write(RUN / "data_verification.json", dict(date="2026-10-13", tasks=600, groups=35,
        split_counts=dict(Counter(s.split for s in standards.values())), training_checks=training_checks,
        same_id_only_duplicates=0, source_families_cross_split=0, calibration_reuse=0, teacher_reuse=0,
        policy_clauses=len(clauses), test_used_for_grading_debug=False, human_confirmations=0,
        note="确定性检查不替代语义复核；训练侧合成核验不是实际教师轨迹。最终测试未运行策略或判分调试。"))
    grouped = defaultdict(list)
    human_rows = []
    review_indices = {"R03": {2, 3, 5}, "D01": {2, 3, 4}, "D02": {2, 3, 4},
                      "B01": {1, 3, 5}, "F01": {1, 3, 12}, "F02": {1, 3, 12}, "F03": {1, 3, 9}}
    for source in sources:
        t = source["task_id"]
        s = standards[t]
        chosen = s.split == "test" or int(t.rsplit("-", 1)[1]) in review_indices.get(s.group_id, {1, 2, 3})
        source["ai_precheck"] = "AI组合模板与实例差异预核查；确定性字段检查通过"
        source["human_review"] = "待逐题人工核查" if s.split == "test" else "分层样本待人工核查" if chosen else "未逐题人工确认"
        human_rows.append(dict(task_id=t, split=s.split, group_id=s.group_id, review_required=chosen,
            ai_precheck=True, human_confirmed=False, human_decision=None, comment=None))
        if chosen:
            grouped[(s.split, s.group_id)].append(card(tasks[t], s, source, specs[t], clauses))
    write(DATA / "sources.json", sources)
    decisions = RUN / "review/human_decisions.json"
    if not decisions.exists() or not any(r["human_confirmed"] or r["human_decision"] for r in json.loads(decisions.read_text(encoding="utf-8"))):
        write(RUN / "review/human_decisions.json", human_rows)
    for (split, group), cards in grouped.items():
        dest = RUN / "review" / split / f"{group}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"# {group} {split} 人工核查材料\n\n共{len(cards)}题。全部待人工确认。\n\n" + "\n".join(cards), encoding="utf-8")
    for split in ("train", "validation", "test"):
        lines = [f"# {split} 真值人工核查入口", "", "所有人工确认栏仍为空，AI预核查不代表人工通过。", "",
            "测试120题逐题核查；训练/验证按组合、主题、处置、单/双对象与范围匹配分层选样。", ""]
        lines += [f"- [{g}：{len(cards)}题]({g}.md)" for (s, g), cards in grouped.items() if s == split]
        (RUN / "review" / split / "index.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("validated 600 tasks; training-only grading checks=400; review train=60 validation=21 test=120 pending")


if __name__ == "__main__":
    main()
