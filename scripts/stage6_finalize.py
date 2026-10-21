"""审计真实有限采集，导出合格示范并保留共同池缺口。"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mymind"))
from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.evaluate import process_errors
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support

WORK = Path(os.getenv("MYMIND_WORK_ROOT", str(REPO)))
RUN = WORK / ("runs" if WORK != REPO else "agentic_runs") / "stage6-20261013"
DATA = WORK / "data/agentic_rl/formal"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def usage_add(counter, usage):
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        if isinstance(usage.get(key), int):
            counter[key] += usage[key]
    detail = usage.get("completion_tokens_details") or {}
    if isinstance(detail.get("reasoning_tokens"), int):
        counter["reasoning_tokens"] += detail["reasoning_tokens"]


def main():
    executed_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    tasks = load_public_tasks(DATA / "train_public_tasks.json")
    standards = load_private_standards(DATA / "train_private_standards.json")
    clauses = load_policy_support(DATA / "policy_support.json")
    collection_config = read(RUN / "configs/teacher-current.json")
    collector = RUN / Path(collection_config["output_dir"]).name
    rows, selected, audits = [], [], []
    failures, by_group = [], defaultdict(lambda: dict(tasks=0, attempted_tasks=0, accepted=0, candidates=0, gaps=[]))
    teacher_usage, rerank_usage, embedding_usage = Counter(), Counter(), Counter()
    generation_calls = rerank_calls = search_calls = failed_generation_calls = 0
    for task_id, standard in standards.items():
        group = by_group[standard.group_id]
        group["tasks"] += 1
        found, success = [], None
        for candidate in (1, 2):
            path = collector / f"{task_id}-candidate-{candidate}.json"
            if not path.exists():
                continue
            row = read(path)
            assert row["task_id"] == task_id and row["candidate"] == candidate
            assert row["trajectory"]["materials"] == [m.model_dump() for m in tasks[task_id].materials()]
            assert row["messages"][:2] == tasks[task_id].messages()
            assert len(row["trajectory"]["searches"]) <= 3
            grade = grade_result(standard, row["raw_result"], VisibleTrajectory.model_validate(row["trajectory"]), clauses)
            assert grade.to_dict() == row["grade"], (task_id, candidate, "保存评分与当前冻结语义不一致")
            errors = process_errors(row)
            eligible = grade.complete_success and not row["infrastructure_failure"] and not row["runtime_error"] and not errors
            if success is not None and not row.get("source_record"):
                raise ValueError(f"{task_id} 已有合格首条后仍采样")
            first_eligible = eligible and success is None
            if first_eligible:
                success = row
            audit = dict(task_id=task_id, candidate=candidate, group_id=standard.group_id, file=str(path.relative_to(WORK)),
                grade=grade.to_dict(), infrastructure_failure=row["infrastructure_failure"], runtime_error=row["runtime_error"],
                process_errors=errors, accepted=bool(first_eligible), original_grade=row.get("original_grade"),
                source_record=row.get("source_record"), selection="首条完整合格" if first_eligible else "标准修订前已采后续候选，保留不收录" if eligible else "未通过完整业务/证据/过程验收")
            audits.append(audit)
            rows.append(row)
            found.append(candidate)
            group["candidates"] += 1
            observations = [m for m in row["messages"] if m["role"] == "tool"]
            assert len(observations) == len(row["trajectory"]["searches"])
            for message, turn in zip(observations, row["trajectory"]["searches"]):
                assert json.loads(message["content"]) == turn["observation"]
            for generation in row["generations"]:
                assert generation["raw_text"] == (generation["assistant_message"].get("content") or "")
                generation_calls += 1
                usage_add(teacher_usage, generation["raw_response"].get("usage") or {})
                for m in generation["input_messages"]:
                    if m["role"] == "tool":
                        assert any(m == actual for actual in observations)
            if (row["runtime_error"] or {}).get("kind") == "generation_failure":
                failed_generation_calls += 1
            search_calls += len(row["trajectory"]["searches"])
            for diagnostic in row["tool_diagnostics"]:
                detail = diagnostic.get("diagnostics") or {}
                if detail.get("rerank_usage"):
                    rerank_calls += 1
                    usage_add(rerank_usage, detail["rerank_usage"])
                for usage in diagnostic.get("embedding_usage", []):
                    embedding_usage["logical_calls_with_reported_usage"] += 1
                    embedding_usage["reported_tokens"] += usage.get("reported_tokens") or 0
        if found:
            group["attempted_tasks"] += 1
        if success is not None:
            selected.append(success)
            group["accepted"] += 1
        else:
            group["gaps"].append(task_id)
            failures.append(dict(task_id=task_id, group_id=standard.group_id, disposition=standard.dispositions[0],
                candidates=found, state="未启动" if not found else "有限候选已耗尽" if len(found) == 2 else "服务暂停或中断后剩余候选未采",
                reasons=[a for a in audits if a["task_id"] == task_id]))
    assert len(rows) <= 800 and len(selected) == len({r["task_id"] for r in selected})
    teacher_selected = read(collector / "accepted_trajectories.json") if (collector / "accepted_trajectories.json").exists() else []
    assert {r["task_id"]: r for r in teacher_selected} == {r["task_id"]: r for r in selected}
    write(RUN / "accepted_trajectories.json", selected)
    write(RUN / "collection_audit.json", audits)
    write(RUN / "task_gaps.json", failures)
    complete = len(selected) == 400
    write(RUN / "common_pool.json", dict(requested_task_ids=list(tasks), requested_count=400,
        demonstrated_task_ids=[r["task_id"] for r in selected], demonstration_count=len(selected),
        unresolved_task_ids=[f["task_id"] for f in failures], effective_sft_rl_pool=None,
        status="待人工核查后固定" if complete else "共同任务池未固定，示范缺口须用户决定",
        training_started=False, subset_export_does_not_approve_pool_change=True))
    export = dict(exported=0, sequence_max=None, validation="无合格示范，未导出")
    if selected:
        config = collection_config.copy()
        export_name = f"student-data-{len(selected):03d}"
        config.update(output_dir=f"runs/stage6-20261013/{export_name}", task_ids=[r["task_id"] for r in selected],
            trajectories="runs/stage6-20261013/accepted_trajectories.json", context_tokens=16384,
            export_scope="合格示范子集；共同池未固定前不用于正式训练")
        write(RUN / "configs/student-data.json", config)
        result = subprocess.run([sys.executable, "-B", "-m", "experiments.agentic_rl.sft_data", "--config", str(RUN / "configs/student-data.json")],
            cwd=REPO / "mymind", capture_output=True, text=True)
        (RUN / "student-data.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("SFT导出失败，详见student-data.log")
        exported = read(RUN / export_name / "student_data.json")
        for original, encoded in zip(selected, exported):
            assert original["task_id"] == encoded["task_id"]
            assert encoded["messages"][-1]["content"] == original["messages"][-1]["content"]
            assert all(set(message) <= {"role", "content", "tool_calls", "name", "tool_call_id"} for message in encoded["messages"])
            assert "reasoning_content" not in str([set(m) for m in encoded["messages"]])
            supervised = set()
            for span in encoded["spans"]:
                assert encoded["messages"][span["message_index"]]["role"] == "assistant"
                supervised.update(range(span["start"], span["end"]))
            assert all(label == token if i in supervised else label == -100
                for i, (token, label) in enumerate(zip(encoded["input_ids"], encoded["labels"])))
            assert len(encoded["input_ids"]) <= 16384
        export = dict(exported=len(exported), sequence_max=max(len(r["input_ids"]) for r in exported),
            path=f"runs/stage6-20261013/{export_name}/student_data.json",
            validation="原Qwen模板；终止文本保持；assistant-only；系统/用户/观察mask；思考字段排除；长度与实例去重通过",
            effective_common_pool_approved=False)
    write(RUN / "export_verification.json", export)
    prepare_usage = dict(rerank_calls=0, embedding_calls=0, embedding_reported_tokens=0, rerank_usage={})
    prepare_counter = Counter()
    probes = read(RUN / "prepare/retrieval_probes.json") if (RUN / "prepare/retrieval_probes.json").exists() else []
    for p in probes:
        d = p["diagnostics"]
        if (d.get("diagnostics") or {}).get("rerank_usage"):
            prepare_usage["rerank_calls"] += 1
            usage_add(prepare_counter, d["diagnostics"]["rerank_usage"])
        for u in d.get("embedding_usage", []):
            prepare_usage["embedding_calls"] += 1
            prepare_usage["embedding_reported_tokens"] += u.get("reported_tokens") or 0
    prepare_usage["rerank_usage"] = dict(prepare_counter)
    cost = dict(candidates_used=len(rows), candidates_upper_bound=800, training_tasks=400, reused_tasks=0,
        teacher_successful_response_calls=generation_calls, teacher_failed_logical_calls=failed_generation_calls,
        teacher_reported_usage=dict(teacher_usage), strategy_search_calls=search_calls,
        rerank_successful_calls_with_usage=rerank_calls, rerank_reported_usage=dict(rerank_usage),
        embedding_reported_usage=dict(embedding_usage), prepare=prepare_usage,
        actual_http_retries=None, actual_http_retries_note="SDK内部重试次数未暴露；retries=1为配置，不伪装成实测重试次数",
        billed_currency=None, actual_billed_amount=None, note="usage是服务返回的实测值；失败或未返回usage的消耗不补估，实际计费未知")
    write(RUN / "cost.json", cost)
    counts = dict(tasks=600, train=400, validation=80, test=120, candidates=len(rows), accepted=len(selected), gaps=len(failures),
        executed_at=executed_at,
        unstarted=sum(not f["candidates"] for f in failures), exhausted=sum(len(f["candidates"]) == 2 for f in failures),
        covered_training_groups=sum(bool(v["accepted"]) for v in by_group.values()), groups=dict(by_group),
        dispositions=dict(Counter(standards[r["task_id"]].dispositions[0] for r in selected)),
        domains=dict(Counter(tasks[r["task_id"]].role for r in selected)), export=export,
        stage6_complete=False, data_frozen=False, human_test_confirmations=0, training_started=False)
    write(RUN / "metrics.json", counts)
    summary = read(collector / "summary.json") if (collector / "summary.json").exists() else {}
    paused = summary.get("paused_for_service_balance", False)
    lines = ["# 阶段6当前交接", "", f"{executed_at[:10]}（北京时间）。阶段6尚未完成或冻结，未执行阶段7/8或Git提交/推送。", "",
        f"正式任务600题（训练400／验证80／测试120），35组配额与来源隔离通过。真实教师候选{len(rows)}条，上限800；首条合格{len(selected)}题，未收录{len(failures)}题。",
        f"{counts['unstarted']}题未启动，{counts['exhausted']}题两候选已耗尽；教师deepseek-flash，冻结stage5-frozen-20261013-v11，判分stage5-independent-review-v11。",
        "", f"SFT导出{export['exported']}条，最大序列{export['sequence_max']}；导出合格子集不等于批准缩小SFT/RL共同池。common_pool.json保留全部400训练任务。",
        "", "data/agentic_rl/formal/保存公开任务、独立标准、来源、实例构造及配额；teacher/保存每条原始候选/逐轮记录、响应、工具观察与usage。",
        "collection_audit.json逐条给判分/筛选理由，task_gaps.json保留全部失败或未启动题，metrics.json按组/处置/域计数，cost.json保留真实usage与未知计费/重试项。",
        "", "人工材料：review/test/index.md为测试全部120题逐题材料；review/train/index.md为60题分层样本；review/validation/index.md为21题分层样本。human_decisions.json人工确认全部留空。",
        "", "待用户处理：逐题确认测试真值及训练/验证分层样本。若存在示范缺口，task_gaps.json中的普通失败保留，选择具体有限补采预算或调整同一SFT/RL任务池；未获批准前不额外采、不缩池、不训练。"]
    if paused:
        lines += ["", "服务返回402，有限采集已停止派发新任务；保留已耗候选。余额恢复后可用原teacher配置续跑剩余额度，已完成候选不会重采。"]
    else:
        lines += ["", "本轮有限教师采集已结束；普通失败不自动追加第三候选。"]
    lines += ["", "最终测试没有送教师、没有用于判分/提示/模型/参数调试。训练标准合成检查与实际教师轨迹分别记录；没有把AI核查称为人工确认。", ""]
    (RUN / "HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN / "STATUS.md").write_text("# 阶段6执行状态\n\n" + "\n".join(lines[2:]), encoding="utf-8")
    print(f"audited candidates={len(rows)} accepted={len(selected)} gaps={len(failures)} exported={export['exported']}; stage6 pending human review")


if __name__ == "__main__":
    main()
