"""汇总阶段 5 原始记录，导出逐题审阅包与首条合格轨迹。"""
from collections import Counter, defaultdict
from copy import deepcopy
import html
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mymind"))
from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support

RUN = REPO / "agentic_runs/stage5-20261012"
DATA = REPO / "data/agentic_rl/calibration"
BATCHES = ["teacher-initial", "teacher-mode-probe", "teacher-recovery", "teacher-max-probe", "teacher-final",
           "teacher-remaining-20261013", "teacher-format-probe-20261013"]
STUDENT_BATCHES = ["student-sft-initial", "student-sft-calibration", "student-sft-final",
                   "student-sft-remaining-20261013", "student-sft-format-probe-20261013"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value


def file_link(path, label="原始文件"):
    return f"[{label}](<{path.resolve().as_posix()}>)"


def process_errors(row):
    queries = [s["action"].get("query", "") for s in row["trajectory"]["searches"]]
    for i, query in enumerate(queries):
        if query in queries[:i]:
            older = row["trajectory"]["searches"][queries.index(query)]["observation"]["snippets"]
            newer = row["trajectory"]["searches"][i]["observation"]["snippets"]
            if [s["content"] for s in older] == [s["content"] for s in newer]:
                return ["同一查询重复执行且实际正文没有新增，未收入示范"]
    return []


def main():
    tasks = load_public_tasks(DATA / "public_tasks.json")
    standards = load_private_standards(DATA / "private_standards.json")
    clauses = load_policy_support(DATA / "policy_support.json")
    sources = {s["task_id"]: s for s in read(DATA / "sources.json")}
    status_path = RUN / "review/review_status.json"
    previous_status = {s["task_id"]: s for s in read(status_path)} if status_path.exists() else {}
    human_status = {tid: previous_status.get(tid, sources[tid]).get("human_review", "待人工核查")
                    for tid in tasks}
    human_conclusions = {tid: previous_status.get(tid, sources[tid]).get("human_conclusion")
                         for tid in tasks}
    ai_reviews = {tid: previous_status.get(tid, sources[tid]).get("ai_review") for tid in tasks}
    confirmed = [tid for tid in tasks if human_status[tid] == "已确认审阅结论"]
    delegated = [tid for tid in tasks if ai_reviews[tid] and ai_reviews[tid]["status"] == "已完成AI代审"]
    unreviewed = [tid for tid in tasks if tid not in confirmed and tid not in delegated]
    review_summary = (f"{len(confirmed)} 题人工确认、{len(delegated)} 题完成用户授权的 AI 代审、"
                      f"{len(unreviewed)} 题尚未审阅；整体验收待独立技术审核，通过后按用户预先条件授权办理冻结")
    revisions = read(RUN / "grading-revisions.json")
    attempts, students, selected, grades = defaultdict(list), defaultdict(list), {}, []
    metrics = dict(teacher_batches={}, student_batches={}, estimated_deepseek_cny=0.0,
                   deepseek_prompt_tokens=0, deepseek_output_tokens=0, embedding_reported_tokens=0)

    def cost(row):
        usage = [g["raw_response"]["usage"] for g in row.get("generations", []) if g.get("raw_response")]
        for d in row.get("tool_diagnostics", []):
            rerank = d.get("diagnostics", {}).get("rerank_usage", {})
            if rerank.get("raw"): usage.append(rerank["raw"])
            metrics["embedding_reported_tokens"] += sum(x.get("reported_tokens") or 0 for x in d.get("embedding_usage", []))
        for u in usage:
            total = u.get("prompt_tokens") or 0
            hit = u.get("prompt_cache_hit_tokens", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
            output = u.get("completion_tokens") or 0
            metrics["deepseek_prompt_tokens"] += total
            metrics["deepseek_output_tokens"] += output
            # 本轮调用为北京时间工作日高峰时段；按 2026-10-12 官方人民币价估算。
            metrics["estimated_deepseek_cny"] += (hit * .04 + (total - hit) * 2 + output * 8) / 1_000_000

    for batch in BATCHES + STUDENT_BATCHES:
        directory = RUN / batch
        current = []
        for path in sorted(directory.glob("C*-candidate-*.json")):
            row = read(path)
            tid = row["task_id"]
            grade = grade_result(standards[tid], row["raw_result"], VisibleTrajectory.model_validate(row["trajectory"]), clauses).to_dict()
            entry = dict(batch=batch, path=path.relative_to(RUN).as_posix(), candidate=row["candidate"],
                         row=row, grade=grade, process_errors=process_errors(row))
            (attempts if batch in BATCHES else students)[tid].append(entry)
            current.append(entry)
            cost(row)
            grades.append(dict(task_id=tid, batch=batch, path=entry["path"], candidate=row["candidate"],
                original_grade=row["grade"], reviewed_grade=grade, process_errors=entry["process_errors"]))
            if batch in BATCHES and grade["complete_success"] and not row["infrastructure_failure"] and not row["runtime_error"] and not entry["process_errors"]:
                selected.setdefault(tid, entry)
        counts = dict(attempts=len(current), rewards=dict(Counter(str(e["grade"]["reward"]) for e in current)),
            search_counts=dict(Counter(str(len(e["row"]["trajectory"]["searches"])) for e in current)),
            adaptive_search_attempts=sum(any(g.get("assistant_message",{}).get("tool_calls") and
                any(m["role"] == "tool" for m in g["input_messages"]) for g in e["row"]["generations"])
                for e in current),
            repeated_search_rows=sum(bool(e["process_errors"]) for e in current),
            complete=sum(e["grade"]["complete_success"] for e in current),
            infrastructure_attempts=sum(e["row"]["infrastructure_failure"] for e in current),
            runtime_errors=dict(Counter((e["row"]["runtime_error"] or {}).get("kind", "none") for e in current)),
            elapsed_task_seconds=sum(e["row"].get("elapsed_s", 0) for e in current),
            generation_latency_seconds=sum(g.get("latency_ms", 0)/1000 for e in current for g in e["row"]["generations"]))
        if batch in STUDENT_BATCHES:
            groups = []
            for tid in tasks:
                es = [e for e in current if e["row"]["task_id"] == tid]
                if not es: continue
                rewards = [e["grade"]["reward"] for e in es]
                fault = any(e["row"]["infrastructure_failure"] for e in es)
                mean = sum(rewards)/len(rewards)
                groups.append(dict(task_id=tid,rewards=rewards,advantages=[v-mean for v in rewards],
                    infrastructure_group=fault,nonzero_advantage=not fault and len(set(rewards)) > 1,
                    samples=len(es),complete_group=len(es)==4))
            counts.update(groups=groups, nonzero_advantage_groups=sum(g["nonzero_advantage"] for g in groups if g["complete_group"]),
                zero_advantage_groups=sum(not g["nonzero_advantage"] and not g["infrastructure_group"] for g in groups if g["complete_group"]),
                truncated_generations=sum(bool(g.get("reached_token_limit")) for e in current for g in e["row"]["generations"]),
                output_token_max=max((g.get("output_tokens", 0) for e in current for g in e["row"]["generations"]), default=0),
                input_token_max=max((g.get("input_tokens", 0) for e in current for g in e["row"]["generations"]), default=0))
        metrics["teacher_batches" if batch in BATCHES else "student_batches"][batch] = counts

    metrics["accepted_search_counts"] = dict(Counter(str(len(e["row"]["trajectory"]["searches"])) for e in selected.values()))
    metrics["accepted_adaptive_search_task_ids"] = [tid for tid,e in selected.items() if any(
        g.get("assistant_message",{}).get("tool_calls") and any(m["role"] == "tool" for m in g["input_messages"])
        for g in e["row"]["generations"])]
    novelty = []
    for batch in BATCHES + STUDENT_BATCHES:
        for es in (attempts if batch in BATCHES else students).values():
            for e in es:
                if e["batch"] != batch or len(e["row"]["trajectory"]["searches"]) < 2: continue
                searches = e["row"]["trajectory"]["searches"]
                initial = {(s["source_id"],s["content"]) for s in searches[0]["observation"]["snippets"]}
                later = {(s["source_id"],s["content"]) for t in searches[1:] for s in t["observation"]["snippets"]}
                novelty.append(dict(path=e["path"],new_snippets=len(later-initial),batch=batch))
    metrics["additional_search_evidence"] = novelty

    probes = RUN / "prepare/retrieval_probes.json"
    if probes.exists():
        for row in read(probes): cost(dict(tool_diagnostics=[row["diagnostics"]]))
    accepted = []
    for tid in tasks:
        if tid in selected:
            row = deepcopy(selected[tid]["row"])
            row["grade"] = selected[tid]["grade"]
            row["collection_batch"] = selected[tid]["batch"]
            row["original_file"] = selected[tid]["path"]
            accepted.append(row)
    write(RUN / "accepted_trajectories.json", accepted)
    write(RUN / "grading_audit.json", grades)
    metrics.update(tasks=40,groups=20,accepted_task_ids=list(selected),
        accepted_by_group=dict(Counter(standards[t].group_id for t in selected)),
        accepted_by_disposition=dict(Counter(standards[t].dispositions[0] for t in selected)),
        accepted_by_domain=dict(Counter(tasks[t].role for t in selected)),
        pending_teacher_task_ids=[t for t in tasks if t not in selected],
        price_source="https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        cost_scope="已返回 DeepSeek usage + prepare 重排；包含失败尝试已返回 usage，不含无法观测的内部重试、Embedding 账单和 GPU 租金",
        human_review=review_summary, human_confirmed_task_ids=confirmed, ai_reviewed_task_ids=delegated)
    write(RUN / "calibration_metrics.json", metrics)
    audit = []
    index = ["# 阶段 5：40 题逐题审阅", "", f"执行者核查与用户人工确认分别记录；{review_summary}。", "",
             f"首条合格示范 {len(selected)}/40；未收入的题和所有失败、服务故障、截断均保留。", "",
             "| 题号 | 组合 | 主要处置 | 收录 | 教师尝试 | 学生样本 | 人工审阅 |", "| --- | --- | --- | --- | ---: | ---: | --- |"]
    articles = []
    for tid, task in tasks.items():
        s = standards[tid]
        entry = selected.get(tid)
        state = (f"首条自动合格：{entry['batch']} / 候选 {entry['candidate']}" if entry else "未收录") + f"；{human_status[tid]}"
        md = [f"# {tid} / {s.group_id}", "", f"处置：`{s.dispositions[0]}`。{state}。", "",
            f"执行者核查：{sources[tid]['executor_review']}。用户人工确认：{human_status[tid]}。", "",
            "## 公开题面", "", task.user_text, "", "## 私有标准", "",
            "| 部分 | 必需内容 | 实际支持材料／条款 |", "| --- | --- | --- |"]
        for e in s.expectations:
            if e.required:
                md.append(f"| {e.section} | `{json.dumps(e.alternatives,ensure_ascii=False)}` | `{e.supports}` |")
        md += ["", file_link(DATA / "private_standards.json", "完整私有标准与可选项"), "",
               "## 组合与来源", "", "```json", pretty(sources[tid]), "```", "", "## 教师实际尝试", ""]
        related = [v for v in revisions if tid in v["affected"]]
        if related:
            md += ["本题标准／支持修订（业务规则未改变，模型答案未改）：", "", "```json",pretty(related),"```", ""]
        for e in attempts[tid]:
            row = e["row"]
            selected_here = e is entry
            category = ("服务／检索基础设施故障" if row["infrastructure_failure"] else
                "完整合格" if e["grade"]["complete_success"] else
                "严重业务错误（兼有协议错误）" if e["grade"]["severe_error"] and e["grade"]["protocol_error"] else
                "严重业务错误" if e["grade"]["severe_error"] else
                "协议错误" if e["grade"]["protocol_error"] else
                "必要内容遗漏" if not e["grade"]["errors"] and e["grade"]["missing"] else
                "处置／事实／可见引用错误")
            md += [f"### {e['batch']} / 候选 {e['candidate']}", "",
                ("收录首条合格完整轨迹；人工确认或授权 AI 核查状态见本题记录。" if selected_here else "未收录；原因见评分、过程核查与运行错误。"), "",
                f"结果分类：{category}。", "",
                file_link(RUN / e["path"]), "", "自动评分与运行状态：", "", "```json",
                pretty(dict(original_grade=row["grade"],reviewed_grade=e["grade"],runtime_error=row["runtime_error"],
                    infrastructure_failure=row["infrastructure_failure"],process_errors=e["process_errors"])), "```", ""]
            for search in row["trajectory"]["searches"]:
                md += ["真实搜索：`" + search["action"].get("query",str(search["action"])) + "`", "", "实际展示观察：", "", "```json",pretty(search["observation"]),"```", ""]
            raw_modes = [dict(model=g["raw_response"].get("model"),finish_reason=g["raw_response"]["choices"][0]["finish_reason"],
                usage=g["raw_response"].get("usage")) for g in row["generations"] if g.get("raw_response")]
            md += ["原始响应型号、终止原因及 usage：", "", "```json",pretty(raw_modes),"```", "",
                   "最终结果（保留原始内容）：", "", "```json",pretty(row["raw_result"]),"```", "",
                   "模板呈现：", "", row.get("rendered") or "无合法终止结果。", ""]
        md += ["## 学生无更新采样", "", "策略为阶段 3 的 sft-resume/adapter；这是开发闭环产物。", ""]
        if not students[tid]: md += ["本题未安排学生诊断采样；教师采集和人工核查仍保留。", ""]
        for e in students[tid]:
            md += [f"- {e['batch']} / 样本 {e['candidate']}：奖励 {e['grade']['reward']}，搜索 {len(e['row']['trajectory']['searches'])} 次；"
                   + file_link(RUN / e["path"])]
        md += ["", "## 人工确认", "", f"状态：{human_status[tid]}。", ""]
        if human_conclusions[tid]:
            md += ["```json", pretty(human_conclusions[tid]), "```", ""]
        else:
            md += ["请确认题面／标准、引用证据、主要处置与收录是否合理；未收录项也可记录具体意见。", ""]
        if ai_reviews[tid]:
            md += ["## 用户授权的 AI 代审", "", "此处是 AI 独立审阅结论，不代表用户逐题人工通过。", "",
                   "```json", pretty(ai_reviews[tid]), "```", ""]
        path = RUN / "review" / f"{tid}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(md)
        path.write_text(content, encoding="utf-8")
        index.append(f"| {file_link(path,tid)} | {s.group_id} | {s.dispositions[0]} | {'自动合格' if entry else '未收录'} | {len(attempts[tid])} | {len(students[tid])} | {human_status[tid]} |")
        audit.append(dict(task_id=tid,group_id=s.group_id,executor_review=sources[tid]["executor_review"],human_review=human_status[tid],
            disposition=s.dispositions[0],accepted=bool(entry),selected_original_file=entry["path"] if entry else None,
            teacher_attempt_files=[e["path"] for e in attempts[tid]],student_attempt_files=[e["path"] for e in students[tid]],review_file=f"review/{tid}.md"))
        if human_conclusions[tid]:
            audit[-1]["human_conclusion"] = human_conclusions[tid]
        if ai_reviews[tid]:
            audit[-1]["ai_review"] = ai_reviews[tid]
        articles.append(f'<article id="{tid}" data-group="{s.group_id}"><h2>{tid} · {s.group_id} · {s.dispositions[0]}</h2><p>{html.escape(state)}</p><pre>{html.escape(content)}</pre></article>')
    (RUN / "review/index.md").write_text("\n".join(index), encoding="utf-8")
    write(RUN / "review/review_status.json", audit)
    nav = " ".join(f'<a href="#{t}">{t}</a>' for t in tasks)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段 5 逐题审阅</title>'
        '<style>body{margin:0;font:16px/1.65 system-ui;background:#f4f5f7;color:#17212e}header{position:sticky;top:0;background:#17212e;color:white;padding:18px 4vw}nav{display:flex;flex-wrap:wrap;gap:8px}nav a{color:#cde3ff}main{max-width:1100px;margin:30px auto;padding:0 18px}article{background:white;padding:28px;margin:24px 0;border-radius:10px;scroll-margin-top:180px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui}input{padding:8px;margin:8px 0;min-width:250px}</style>'
        f'<header><strong>阶段 5 · 40 题审阅包 · 首条合格 {len(selected)}/40 · {review_summary}</strong><br>'
        '<input placeholder="输入题号或组合，例如 C11、P05" oninput="const q=this.value.toUpperCase();document.querySelectorAll(\'article\').forEach(a=>a.hidden=q&&!a.id.includes(q)&&!a.dataset.group.includes(q))">'
        f'<nav>{nav}</nav></header><main>'+"\n".join(articles)+'</main></html>')
    (RUN / "review/review.html").write_text(page,encoding="utf-8")
    print(f"reviewed 40 tasks; first qualified trajectories {len(selected)}; teacher attempts {sum(map(len,attempts.values()))}; student samples {sum(map(len,students.values()))}")


if __name__ == "__main__":
    main()
