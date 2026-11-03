"""正式数据与 RL40/80 训练记录的只读定向审计；不加载模型或 tokenizer。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / "mymind"))
from agents.customer_policy.protocol import PublicTask


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    # 必需日志缺失时失败，不将缺失解释成零条记录。
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                fraction=numerator / denominator if denominator else None)


def lengths(values):
    values = sorted(values)
    if not values:
        return dict(n=0, total=None, min=None, median=None, p90=None, max=None, mean=None)
    return dict(n=len(values), total=sum(values), min=values[0], median=statistics.median(values),
                p90=values[math.ceil(.9 * len(values))-1], max=values[-1], mean=statistics.mean(values))


def expected_bucket(standard):
    # 以允许处置集合为互斥桶，不能把一个多允许处置任务重复计数。
    return "|".join(sorted(standard["dispositions"]))


def distribution(before, after, field):
    a, b = Counter(row[field] for row in before), Counter(row[field] for row in after)
    return [dict(bucket=key, train_count=a[key], selected_count=b[key],
                 train_share=fraction(a[key], len(before)), selected_share=fraction(b[key], len(after)),
                 selection_rate=fraction(b[key], a[key]),
                 share_difference_pp=100*(b[key]/len(after)-a[key]/len(before)))
            for key in sorted(a.keys() | b.keys())]


def tool_summary(row):
    messages = row.get("completion", row.get("messages"))
    if messages is None:
        raise ValueError("轨迹缺少实际消息")
    observed = False
    requested = after = batches = 0
    for message in messages:
        if message["role"] == "assistant":
            calls = [c for c in (message.get("tool_calls") or []) if c["function"]["name"] == "search"]
            requested += len(calls)
            batches += bool(calls)
            after += len(calls) if observed else 0
        observed |= message["role"] == "tool"
    searches = row["trajectory"]["searches"]
    tools = [m for m in messages if m["role"] == "tool"]
    observations = [json.loads(m["content"]) for m in tools]
    assert observations == [s["observation"] for s in searches]
    assert requested == row.get("search_requests", requested)
    assert len(searches) == row.get("actual_search_count", len(searches))
    statuses = Counter(o["status"]["state"] for o in observations)
    terminal = row.get("raw_result")
    if terminal is None:
        assistant = [m for m in messages if m["role"] == "assistant"]
        try:
            terminal = json.loads(assistant[-1].get("content") or "") if assistant else None
        except (ValueError, TypeError):
            terminal = None
    disposition = terminal.get("disposition") if isinstance(terminal, dict) else None
    return dict(search_requests=requested, actual_observations=len(tools), search_batches=batches,
                after_observation_requests=after, after_observation=after > 0,
                observation_statuses=dict(statuses), terminal_disposition=disposition)


def aggregate_tools(rows):
    details = [tool_summary(row) for row in rows]
    return dict(trajectories=len(rows), search_requests=sum(r["search_requests"] for r in details),
                actual_observations=sum(r["actual_observations"] for r in details),
                request_distribution=dict(Counter(r["search_requests"] for r in details)),
                observation_distribution=dict(Counter(r["actual_observations"] for r in details)),
                after_observation_trajectories=fraction(sum(r["after_observation"] for r in details), len(rows)),
                after_observation_requests=sum(r["after_observation_requests"] for r in details),
                raw_terminal_dispositions=dict(Counter(r["terminal_disposition"] or "unparseable_or_absent" for r in details)),
                observation_statuses=dict(sum((Counter(r["observation_statuses"]) for r in details), Counter())))


def deduplicate_groups(paths):
    groups = defaultdict(dict)
    for path in paths:
        row = read(path)
        rank = int(path.stem.rsplit("rank", 1)[1])
        key = row["policy_step"], row["attempt"], tuple(row["task_ids"])
        assert rank not in groups[key], (key, rank)
        groups[key][rank] = (path, row)
    result = []
    for key, ranks in sorted(groups.items()):
        assert set(ranks) == {0, 1}, (key, "缺少 rank 原件")
        a, b = ranks[0][1], ranks[1][1]
        for field in ("rows", "task_ids", "rewards", "advantages", "effective_model_tokens",
                      "optimization_model_tokens", "infrastructure_failure", "nonzero_advantage"):
            assert a[field] == b[field], (key, field)
        assert len(a["rows"]) == len(a["rewards"]) == len(a["advantages"]) == 4
        assert len(set(a["task_ids"])) == 1
        result.append(ranks)
    return result


def select_prefix(groups, stop):
    return [r for r in groups if r[0][1]["policy_step"] < stop
            and not r[0][1]["infrastructure_failure"]]


def audit_group(ranks, directory):
    g = ranks[0][1]
    rewards = [r["grade"]["reward"] for r in g["rows"]]
    assert rewards == g["rewards"]
    mean = statistics.mean(rewards)
    expected = [r-mean for r in rewards]
    assert all(abs(a-b) < 1e-5 for a, b in zip(expected, g["advantages"]))
    nonzero = any(abs(a) > 1e-8 for a in expected)
    assert nonzero == g["nonzero_advantage"]
    token_details = []
    for rank, (path, row) in sorted(ranks.items()):
        assert len(row["completion_ids"]) == len(row["completion_mask"]) == len(row["tool_mask"]) == 2
        for i, (ids, cm, tm) in enumerate(zip(row["completion_ids"], row["completion_mask"], row["tool_mask"])):
            assert len(ids) == len(cm) == len(tm)
            assert set(cm) <= {0, 1} and set(tm) <= {0, 1}
            assert sum(cm) == row["completion_lengths"][i]
            token_details.append(dict(rank=rank, local_row=i, completion_tokens=sum(cm),
                effective_policy_tokens=sum(c*t for c, t in zip(cm, tm)),
                excluded_context_tokens=sum(c*(1-t) for c, t in zip(cm, tm))))
    effective = sum(r["effective_policy_tokens"] for r in token_details)
    assert effective == g["effective_model_tokens"] == g["optimization_model_tokens"]
    update_paths = [directory / f"group-updates-step{g['policy_step']}-rank{rank}.json" for rank in (0, 1)]
    update = read(update_paths[0])
    assert update == read(update_paths[1])
    assert update["rewards"] == rewards and update["advantages"] == g["advantages"]
    assert update["completed_steps"] == [g["policy_step"]+1, g["policy_step"]+2]
    assert update["nonzero_advantage"] == nonzero and not update["paused"] and not update["missing"]
    return dict(policy_step=g["policy_step"], attempt=g["attempt"], task_id=g["task_ids"][0],
                source_records=[str(path.name) for path, _ in ranks.values()],
                rewards=rewards, reward_mean=mean, reward_range=max(rewards)-min(rewards),
                reward_population_std=statistics.pstdev(rewards), advantages=g["advantages"],
                nonzero_advantage=nonzero, effective_policy_tokens=effective,
                nonzero_advantage_policy_tokens=sum(t["effective_policy_tokens"] for t, a in zip(token_details, expected) if abs(a)>1e-8),
                token_slices=token_details, completed_steps=update["completed_steps"],
                tools=[tool_summary(row) for row in g["rows"]],
                nonzero_gradient=update["nonzero_gradient"], parameter_changed=update["parameter_changed"],
                all_gradients_finite=all(u["gradient_finite"] and math.isfinite(u["gradient_norm"])
                                         for step in update["rank_updates"] for u in step))


def candidate_coverage(train, selected, available, historical):
    selected_by_id = {r["task_id"]: r for r in selected}
    by_task = defaultdict(list)
    for r in available:
        by_task[r["task_id"]].append(r["candidate"])
    return dict(historical_manifest=historical,
        original_records_available=len(available),
        original_records_unavailable=historical["candidates"]-len(available),
        uncollected_tasks_recomputed=None, failed_candidates_recomputed=None,
        qualified_candidates_total_recomputed=None, rejection_funnel_complete=False,
        tasks=[dict(task_id=t["task_id"], available_candidates=sorted(by_task[t["task_id"]]),
                    selected_candidate=selected_by_id[t["task_id"]]["candidate"] if t["task_id"] in selected_by_id else None,
                    collection_status="original_available" if by_task[t["task_id"]] else "original_unavailable",
                    uncollected=False if by_task[t["task_id"]] else None)
               for t in train])


def run(root, output):
    runs = root / ("agentic_runs" if root == SOURCE else "runs")
    def locate(name):
        return root / (name.replace("runs/", "agentic_runs/", 1) if root == SOURCE else name.replace("agentic_runs/", "runs/", 1))
    baseline = read(runs / "stage7-preparation-20261014/baseline.json")
    public = read(locate(baseline["public_tasks"]))
    standards = read(locate(baseline["standards"]))
    selected = read(locate(baseline["demonstrations"]))
    students = read(locate(baseline["student_data"]))
    manifest = read(runs / "stage6-20261013/freeze-v4-20261014/data_manifest.frozen.json")
    public_by_id, standard_by_id = ({r["task_id"]: r for r in rows} for rows in (public, standards))
    assert len(public_by_id) == len(public) == len(standard_by_id) == len(standards) == 600
    assert public_by_id.keys() == standard_by_id.keys()
    pool_record = read(locate(baseline["common_pool"]))
    pool = pool_record["effective_sft_rl_pool"]
    decision = read(runs / "stage6-20261013/USER_DECISIONS-20261014.json")
    assert decision["sft_rl_same_task_ids"] == pool and not decision["extra_teacher_sampling_authorized"]
    assert len(set(pool)) == len(pool) == 158
    assert [r["task_id"] for r in selected] == [r["task_id"] for r in students] == pool
    data = root / "data/agentic_rl/formal"
    source_rows, specs = read(data / "sources.json"), read(data / "instance_specs.json")
    source_by_id, spec_by_id = ({r["task_id"]: r for r in rows} for rows in (source_rows, specs))
    assert len(source_rows) == len(source_by_id) == len(specs) == len(spec_by_id) == 600
    assert source_by_id.keys() == spec_by_id.keys() == standard_by_id.keys()
    allocations = read(data / "group_allocation.json")
    actual_allocation = Counter((r["group_id"], r["split"]) for r in standards)
    assert actual_allocation == {(r["group_id"], r["split"]): r["count"] for r in allocations}
    assert len({r["group_id"] for r in allocations}) == len(allocations) == 35
    task_rows = []
    for standard in standards:
        tid = standard["task_id"]
        task = public_by_id[tid]
        source, spec = source_by_id[tid], spec_by_id[tid]
        assert source["source_id"] == spec["source_id"]
        assert source["split"] == spec["split"] == standard["split"]
        assert source["group_id"] == spec["group_id"] == standard["group_id"]
        assert spec["disposition"] in standard["dispositions"]
        assert task.keys() <= {"task_id", "role", "user_text", "records"}
        targets = {a["target"] for e in standard["expectations"] for a in e["alternatives"] if a.get("target")}
        task_rows.append(dict(task_id=tid, split=standard["split"], role=task["role"], group_id=standard["group_id"],
            expected_disposition=expected_bucket(standard), selected=tid in pool,
            source_id=source["source_id"], source_origin=source["origin"],
            calibration_reuse=source["calibration_reuse"], teacher_reuse=source["teacher_reuse"],
            public_user_chars=len(task["user_text"]), expectation_target_count=len(targets),
            expectation_count=len(standard["expectations"]),
            required_expectation_count=sum(e["required"] for e in standard["expectations"])))
    train = [r for r in task_rows if r["split"] == "train"]
    retained = [r for r in task_rows if r["selected"]]
    assert all(r["split"] == "train" for r in retained)
    assert Counter(r["split"] for r in task_rows) == dict(train=400, validation=80, test=120)
    demo_rows = []
    for row, student in zip(selected, students):
        tid = row["task_id"]
        assert student["messages"][:2] == PublicTask.model_validate(public_by_id[tid]).messages()
        spans = student["spans"]
        assert {s["message_index"] for s in spans} == {i for i, m in enumerate(student["messages"]) if m["role"] == "assistant"}
        assert all(student["messages"][s["message_index"]]["role"] == "assistant" for s in spans)
        assert all(0 <= s["start"] < s["end"] <= len(student["input_ids"]) for s in spans)
        assert all(a["end"] <= b["start"] for a, b in zip(spans, spans[1:]))
        # label/id 与 span 的逐 token 一致性由现行保留核验入口执行。
        supervised = sum(label != -100 for label in student["labels"])
        action = sum(s["end"]-s["start"] for s in spans if s["kind"] == "tool_action")
        final = sum(s["end"]-s["start"] for s in spans if s["kind"] == "final")
        assert action+final == supervised
        demo_rows.append(dict(task_id=tid, candidate=row["candidate"], source_record=row["source_record"],
            expected_disposition=expected_bucket(standard_by_id[tid]),
            teacher_actual_disposition=row["raw_result"]["disposition"],
            sequence_tokens=len(student["input_ids"]), initial_input_tokens=spans[0]["start"],
            context_tokens=len(student["input_ids"])-supervised, supervised_tokens=supervised,
            action_tokens=action, final_tokens=final, supervised_ratio=supervised/len(student["input_ids"]),
            tools=tool_summary(row)))
    raw_paths = sorted((runs / "stage6-20261013/teacher-v3").glob('*-candidate-*.json'))
    available = [read(p) for p in raw_paths]
    assert len({(r["task_id"], r["candidate"]) for r in available}) == len(available)
    assert all(standard_by_id[r["task_id"]]["split"] == "train" for r in available)
    coverage = candidate_coverage(train, selected, available, manifest["teacher_collection"])
    historical_candidates = {"1": manifest["teacher_collection"]["selected_candidate_1"],
                             "2": manifest["teacher_collection"]["selected_candidate_2"]}
    current_candidates = dict(Counter(str(r["candidate"]) for r in selected))
    directory = runs / "stage7-preparation-20261014/formal/grpo"
    all_groups = deduplicate_groups(directory.glob("group-step*-rank*.json"))
    training = select_prefix(all_groups, 80)
    group_details = [audit_group(g, directory) for g in training]
    details_by_step = {g["policy_step"]: g for g in group_details}
    assert len(details_by_step) == len(training)
    updates = {rank: lines(directory / f"updates-rank{rank}.jsonl") for rank in (0, 1)}
    losses = {rank: lines(directory / f"loss-rank{rank}.jsonl") for rank in (0, 1)}
    scopes = {}
    for stop in (40, 80):
        groups = select_prefix(all_groups, stop)
        details = [details_by_step[g[0][1]["policy_step"]] for g in groups]
        rows = [r for g in groups for r in g[0][1]["rows"]]
        tids = sorted({r["task_id"] for r in rows})
        assert set(tids) <= set(pool)
        records = [next(t for t in task_rows if t["task_id"] == tid) for tid in tids]
        per_rank = {}
        for rank in (0, 1):
            us = [u for u in updates[rank] if u["step"] <= stop]
            ls = [l for l in losses[rank] if l["policy_step"] < stop]
            assert [u["step"] for u in us] == list(range(1, stop+1))
            assert [l["policy_step"] for l in ls] == list(range(stop))
            for l in ls:
                g = details_by_step[l["policy_step"]//2*2]
                assert math.isfinite(l["loss"])
                assert l["global_generation_model_tokens"] == g["effective_policy_tokens"]
                assert l["dapo_normalizer"] == g["effective_policy_tokens"]/4
            for g in groups:
                u = read(directory / f"group-updates-step{g[0][1]['policy_step']}-rank{rank}.json")
                for step in u["rank_updates"]:
                    assert step[rank] == us[step[rank]["step"]-1]
            per_rank[str(rank)] = dict(optimizer_steps=len(us), loss_records=len(ls),
                all_gradients_finite=all(u["gradient_finite"] and math.isfinite(u["gradient_norm"]) for u in us),
                nonzero_gradient_steps=sum(u["gradient_norm"]>0 for u in us),
                parameter_changed_steps=sum(u["parameter_delta_l1"]>0 for u in us),
                zero_gradient_with_parameter_change_steps=sum(u["gradient_norm"]==0 and u["parameter_delta_l1"]>0 for u in us),
                loss_effective_tokens=sum(l["local_effective_tokens"] for l in ls))
        assert read(directory / f"checkpoint-{stop}/trainer_state.json")["global_step"] == stop
        assert sorted(s for d in details for s in d["completed_steps"]) == list(range(1, stop+1))
        effective = sum(d["effective_policy_tokens"] for d in details)
        assert sum(r["loss_effective_tokens"] for r in per_rank.values()) == effective
        # 一个全局生成批次分两步消费，各 rank 一条/步；不是把同组完整重放两次。
        scopes[str(stop)] = dict(optimizer_steps=stop, groups=len(groups), trajectories=len(rows),
            unique_tasks=len(tids), task_ids=tids, scene_groups=len({r["group_id"] for r in records}),
            distributions={f: dict(Counter(r[f] for r in records)) for f in ("role", "group_id", "expected_disposition")},
            task_pool_coverage=fraction(len(tids), len(pool)), original_train_coverage=fraction(len(tids), len(train)),
            nonzero_advantage_groups=sum(d["nonzero_advantage"] for d in details),
            zero_advantage_groups=sum(not d["nonzero_advantage"] for d in details),
            distinct_reward_counts=dict(Counter(len(set(d["rewards"])) for d in details)),
            reward_distribution=dict(Counter(str(r["grade"]["reward"]) for r in rows)),
            mean_reward=statistics.mean(r["grade"]["reward"] for r in rows),
            group_reward_range=lengths([d["reward_range"] for d in details]),
            advantage_distribution=dict(Counter(str(round(a, 6)) for d in details for a in d["advantages"])),
            nonzero_advantage_trajectories=fraction(sum(abs(a)>1e-8 for d in details for a in d["advantages"]), len(rows)),
            effective_policy_tokens=effective,
            nonzero_advantage_policy_tokens=sum(d["nonzero_advantage_policy_tokens"] for d in details),
            excluded_context_tokens=sum(t["excluded_context_tokens"] for d in details for t in d["token_slices"]),
            signal_groups_with_finite_gradient_and_update=sum(d["nonzero_advantage"] and d["all_gradients_finite"] and d["nonzero_gradient"] and d["parameter_changed"] for d in details),
            updates_by_rank=per_rank, tools=aggregate_tools(rows))
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=SOURCE, text=True, encoding="utf-8").strip()
    head = git("rev-parse", "HEAD")
    stats = dict(date=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        scope="task1_only", head=head, working_tree=git("status", "--short"),
        baseline_revision=baseline["revision"], grading_revision=baseline["grading_revision"],
        source_files=["mymind/benchmarks/scripts/audit_optimization_data.py", "mymind/tests/test_optimization_data_audit.py"],
        inputs={k: baseline[k] for k in ("public_tasks", "standards", "common_pool", "demonstrations", "student_data", "tokenizer_template")},
        dataset_sources=dict(paths=["data/agentic_rl/formal/"+name for name in
            ("sources.json", "instance_specs.json", "group_allocation.json", "generation_record.json")],
            records_matched=600, source_origins=dict(Counter(r["source_origin"] for r in task_rows)),
            calibration_reuse=sum(r["calibration_reuse"] for r in task_rows),
            teacher_reuse=sum(r["teacher_reuse"] for r in task_rows)),
        pool_selection=dict(authorization="runs/stage6-20261013/USER_DECISIONS-20261014.json",
            approved_count=decision["effective_training_count"], decision=decision["pool_answer"],
            historically_qualified_outside_approved_pool=pool_record["qualified_outside_pool"],
            outside_pool_evidence="historical_common_pool_metadata_not_original_recalculation"),
        training_source=str(directory.relative_to(root)).replace("\\", "/"),
        archive_check=read(output / "archive-check.json") if (output / "archive-check.json").exists() else None,
        split_counts=dict(Counter(r["split"] for r in task_rows)),
        split_group_counts={s: len({r["group_id"] for r in task_rows if r["split"]==s}) for s in ("train", "validation", "test")},
        train_selected_distributions={f: distribution(train, retained, f) for f in ("role", "group_id", "expected_disposition")},
        candidate_coverage={k: v for k, v in coverage.items() if k != "tasks"},
        version_differences=dict(historical_manifest_grading=manifest["grading_revision"],
            current_grading=baseline["grading_revision"], historical_selected_candidates=historical_candidates,
            current_selected_candidates=current_candidates,
            selected_candidate_counts_match=historical_candidates == current_candidates,
            current_selected_with_previous_failing_grade=[dict(task_id=r["task_id"], candidate=r["candidate"],
                previous_grade=r["previous_grade"], current_grade=r["grade"], source_record=r["source_record"])
                for r in selected if not r["previous_grade"]["complete_success"]]),
        demonstrations=dict(count=len(selected), candidate_numbers=dict(Counter(r["candidate"] for r in selected)),
            expected_dispositions=dict(Counter(r["expected_disposition"] for r in demo_rows)),
            teacher_actual_dispositions=dict(Counter(r["teacher_actual_disposition"] for r in demo_rows)),
            lengths={f: lengths([r[f] for r in demo_rows]) for f in ("sequence_tokens", "initial_input_tokens", "context_tokens", "supervised_tokens", "action_tokens", "final_tokens", "supervised_ratio")},
            token_weighted_supervision=fraction(sum(r["supervised_tokens"] for r in demo_rows), sum(r["sequence_tokens"] for r in demo_rows)),
            task_complexity={scope: dict(expectation_targets=lengths([r["expectation_target_count"] for r in records]),
                multiple_targets=fraction(sum(r["expectation_target_count"]>1 for r in records), len(records)))
                for scope, records in (("train", train), ("selected", retained))},
            tools=aggregate_tools(selected), assistant_span_roles_and_public_input="passed"),
        training=scopes, deduplication=dict(rank_group_files=sum(len(g) for g in all_groups),
            unique_generation_groups=len(all_groups), excluded_infrastructure_groups=sum(g[0][1]["infrastructure_failure"] for g in all_groups),
            post_training_groups=sum(g[0][1]["policy_step"]>=80 for g in all_groups),
            post_training_trajectories=sum(len(g[0][1]["rows"]) for g in all_groups if g[0][1]["policy_step"]>=80)),
        model_calls=0, gpu_seconds=0, external_api_calls=0,
        human_confirmations=0)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("task-audit.jsonl", task_rows), ("demonstration-audit.jsonl", demo_rows),
                          ("group-audit.jsonl", group_details), ("candidate-availability.jsonl", coverage["tasks"])):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records), encoding="utf-8")
    (output / "data-audit.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", scopes={k: {f: v[f] for f in ("groups", "trajectories", "unique_tasks", "scene_groups", "nonzero_advantage_groups", "effective_policy_tokens")} for k, v in scopes.items()}, available_candidates=len(available)), indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.getenv("MYMIND_WORK_ROOT", SOURCE)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run(root, args.output.resolve() if args.output else root / ("agentic_runs" if root == SOURCE else "runs") / "optimization-20261022")
