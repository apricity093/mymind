"""正式首次 SFT120/RL40 的只读配对与失败诊断；不加载模型或改写评分。"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / "mymind"))
from pydantic import ValidationError
from agents.customer_policy.protocol import (Fact, MissingInfo, NextStep, Reason,
                                              VisibleTrajectory, parse_result)
from rl.grading import (_equivalent_fact, _payload, _same_item,
                                            _supported, grade_result)
from rl.tasks import (load_policy_support, load_private_standards,
                                          load_public_tasks, validate_task_pairs)
from benchmarks.scripts.records import read, records

SIDES = ("sft", "rl")
SECTIONS = {"reasons": Reason, "facts": Fact, "next_steps": NextStep, "missing_info": MissingInfo}
DISPOSITIONS = {"answer", "request_info", "escalate"}
SOURCE_FILES = ["mymind/benchmarks/scripts/analyze_optimization_failures.py", "mymind/benchmarks/scripts/records.py",
                "mymind/tests/test_optimization_failures.py",
                "mymind/rl/grading.py", "mymind/rl/tasks.py",
                "mymind/agents/customer_policy/protocol.py", "mymind/agents/customer_policy/vocabulary.py"]


def unique(rows, label):
    counts = Counter(r["task_id"] for r in rows)
    duplicates = [tid for tid, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"{label} 重复任务: {duplicates}")
    return {r["task_id"]: r for r in rows}


def pair_records(task_ids, left, right):
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("预定题集含重复任务")
    expected = set(task_ids)
    sides = [unique(left, "SFT首次表"), unique(right, "RL首次表")]
    for name, mapping in zip(SIDES, sides):
        if set(mapping) != expected:
            raise ValueError(f"{name} 题集不符: missing={sorted(expected-set(mapping))}, extra={sorted(set(mapping)-expected)}")
    return [(tid, sides[0][tid], sides[1][tid]) for tid in task_ids]


def quadrant(a, b):
    return {(True, True): "both_pass", (False, False): "both_fail",
            (False, True): "gained", (True, False): "lost"}[
                bool(a["complete_success"]), bool(b["complete_success"])]


def decode_terminal(raw):
    if isinstance(raw, dict):
        return raw, "object"
    if not isinstance(raw, str):
        return None, "absent_or_non_object"
    try:
        value = json.loads(raw)
        return (value, "json") if isinstance(value, dict) else (None, "non_object_json")
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data":
            value, _ = json.JSONDecoder().raw_decode(raw)
            return (value, "json_prefix_with_trailing_text") if isinstance(value, dict) else (None, "non_object_json")
        return None, "invalid_json"


def terminal_status(raw):
    decoded, json_state = decode_terminal(raw)
    try:
        parsed = parse_result(raw)
        error = None
    except (ValidationError, ValueError, TypeError) as exc:
        parsed, error = None, str(exc)
    raw_disposition = decoded.get("disposition") if decoded is not None else None
    return dict(protocol_parse="passed" if parsed is not None else "failed", json_state=json_state,
                parse_error=error, raw_disposition=raw_disposition,
                actual_disposition=parsed.disposition if parsed is not None else None,
                actual_bucket=parsed.disposition if parsed is not None else "parse_failed",
                raw_disposition_valid=raw_disposition in DISPOSITIONS if isinstance(raw_disposition, str) else False)


def historical_checks(grade, raw):
    decoded, _ = decode_terminal(raw)
    later = not grade["protocol_error"] and not grade["severe_error"]
    return dict(severe_assertion_scan="executed" if decoded is not None else "not_executable",
                protocol="failed" if grade["protocol_error"] else "passed",
                item_matching_and_evidence="executed" if later else "not_run",
                required_coverage="executed" if later else "not_run",
                disposition="executed" if later else "not_run",
                missing_interpretation="checked" if later else "unknown_not_checked")


def diagnose(standard, raw, trajectory, clauses):
    """独立检查可读取条目；无效条目不被删掉后冒充完整协议结果。"""
    decoded, json_state = decode_terminal(raw)
    parsed = terminal_status(raw)
    items, matched, supported_facts = [], set(), []
    section_states = {}
    for section, model in SECTIONS.items():
        values = decoded.get(section) if decoded is not None else None
        if not isinstance(values, list):
            section_states[section] = "unknown_unreadable_or_absent"
            continue
        section_states[section] = "checked"
        for index, value in enumerate(values):
            entry = dict(section=section, index=index, submitted=value)
            try:
                item = model.model_validate(value).model_dump(exclude_none=True)
            except (ValidationError, ValueError, TypeError) as exc:
                entry.update(status="invalid_item", error=str(exc))
                section_states[section] = "partial_invalid_items"
                items.append(entry)
                continue
            payload = _payload(item)
            candidates = [(i, e) for i, e in enumerate(standard.expectations)
                          if e.section == section and any(_same_item(payload, alt, e.required)
                                                         for alt in e.alternatives)]
            accepted = [i for i, e in candidates if _supported(e, item["evidence"], trajectory, standard, clauses)]
            matched.update(accepted)
            if accepted and section == "facts":
                supported_facts.append(payload)
            entry.update(status="supported" if accepted else "unsupported_evidence" if candidates else "not_in_standard",
                         expectation_candidates=[i for i, _ in candidates], matched_expectations=accepted)
            items.append(entry)
    for i, expected in enumerate(standard.expectations):
        if expected.section == "facts" and any(_equivalent_fact(a, supported_facts) for a in expected.alternatives):
            matched.add(i)
    visible_refs = [m.reference_id for m in trajectory.materials]
    visible_refs += [s.reference_id for t in trajectory.searches for s in t.observation.snippets]
    visible_refs += [t.observation.status.reference_id for t in trajectory.searches]
    coverage = []
    for i, e in enumerate(standard.expectations):
        if not e.required:
            continue
        state = ("supported" if i in matched else "missing_or_unsupported" if section_states[e.section] == "checked"
                 else "unknown_invalid_or_unreadable_items")
        coverage.append(dict(expectation_index=i, section=e.section, alternatives=e.alternatives, state=state,
                             expected_support_available=_supported(e, visible_refs, trajectory, standard, clauses)))
    return dict(kind="read_only_diagnostic_not_historical_grade", json_state=json_state,
                protocol=parsed, section_states=section_states, items=items, required_coverage=coverage,
                disposition_matches_standard=(parsed["actual_disposition"] in standard.dispositions
                                              if parsed["actual_disposition"] is not None else None),
                raw_disposition_matches_standard=(parsed["raw_disposition"] in standard.dispositions
                                                  if parsed["raw_disposition_valid"] else None),
                scope="有效条目独立核对标准与可见证据；无效条目/不可读section的未匹配必需项保留未知。无新reward。")


def observed_labels(grade, diagnostic):
    labels = set()
    if grade["severe_error"]:
        labels.add("historical_severe_assertion")
    if grade["protocol_error"]:
        labels.add("protocol_parse_failure")
    if diagnostic["disposition_matches_standard"] is False:
        labels.add("wrong_disposition")
    elif diagnostic["disposition_matches_standard"] is None:
        labels.add("actual_disposition_unknown")
    for item in diagnostic["items"]:
        if item["status"] == "not_in_standard":
            labels.add(f"{item['section']}_not_in_standard")
        elif item["status"] == "unsupported_evidence":
            labels.add("unsupported_evidence")
        elif item["status"] == "invalid_item":
            labels.add("invalid_item")
    for expected in diagnostic["required_coverage"]:
        if expected["state"] == "missing_or_unsupported":
            labels.add(f"required_{expected['section']}_missing_or_unsupported")
        elif expected["state"].startswith("unknown"):
            labels.add("required_coverage_unknown")
    return sorted(labels)


def brief_analysis(grade, diagnostic):
    if grade["complete_success"]:
        return "历史v13完整成功；本项只描述已记录输出与标准，不据此确认内部推理。"
    parts = []
    status = diagnostic["protocol"]
    if status["protocol_parse"] == "failed":
        parts.append("终止协议解析失败，实际处置桶未知；JSON可读字段仅作原始输出观察")
    for error in grade["errors"][:2]:
        parts.append("历史检查：" + error.split("\n")[0])
    if len(grade["errors"]) > 2:
        parts.append(f"历史共{len(grade['errors'])}项错误，余项见原grade")
    if historical_checks(grade, {})["required_coverage"] == "not_run":
        parts.append("历史条目/引用/必需覆盖/处置检查未执行，空missing不表明无遗漏")
    missing = [e for e in diagnostic["required_coverage"] if e["state"] == "missing_or_unsupported"]
    if missing:
        text = "；".join(f"{e['section']}:{json.dumps(e['alternatives'], ensure_ascii=False)}" for e in missing[:2])
        parts.append(f"独立诊断{len(missing)}项必需内容未获支持，例如：" + text)
    unexpected = [i for i in diagnostic["items"] if i["status"] != "supported"]
    if unexpected:
        parts.append(f"独立诊断{len(unexpected)}项条目问题，例如：" + "；".join(
            f"{i['section']}[{i['index']}]={json.dumps(i['submitted'], ensure_ascii=False)} ({i['status']})"
            for i in unexpected[:2]))
    parts.append("上述为输出/标准/可见证据差异；主因未知，日志不独立记录阅读或内部推理，未人工确认")
    return "。".join(parts) + "。"


def events(row):
    result = []
    for index, message in enumerate(row["messages"]):
        pointer = f"/messages/{index}"
        if message["role"] == "assistant" and message.get("tool_calls"):
            actions = []
            for call in message["tool_calls"]:
                f = call["function"]
                arguments = f["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        pass
                actions.append(dict(name=f["name"], arguments=arguments))
            result.append(dict(kind="query_or_tool_arguments", value=actions, pointer=pointer))
        elif message["role"] == "tool":
            result.append(dict(kind="displayed_observation", value=json.loads(message["content"]), pointer=pointer))
        elif message["role"] == "assistant":
            result.append(dict(kind="terminal_output", value=message.get("content"), pointer=pointer))
    return result


def first_divergence(left, right):
    a, b = events(left), events(right)
    for i in range(max(len(a), len(b))):
        x, y = a[i] if i < len(a) else None, b[i] if i < len(b) else None
        if x is None or y is None or (x["kind"], x["value"]) != (y["kind"], y["value"]):
            return dict(event_index=i, kind=x["kind"] if x and y and x["kind"] == y["kind"] else "event_sequence",
                        sft=x, rl=y, interpretation="首个可观察差异，不代表首个内部推理差异或已确认根因")
    return dict(kind="no_observed_difference", event_index=None)


def reference(path, pointer=""):
    return dict(path=path, pointer=pointer)


def analyze_side(row, standard, clauses, path, line):
    trajectory = VisibleTrajectory.model_validate(row["trajectory"])
    assert trajectory.task_id == row["task_id"] == standard.task_id
    grade = row["grade"]
    replay = grade_result(standard, row["raw_result"], trajectory, clauses).to_dict()
    if replay != grade:
        raise ValueError(f"{path} 历史v13评分与留存源码重放冲突")
    diagnostic = diagnose(standard, row["raw_result"], trajectory, clauses)
    assert (diagnostic["protocol"]["protocol_parse"] == "failed") == grade["protocol_error"]
    tool_messages = [(i, m) for i, m in enumerate(row["messages"]) if m["role"] == "tool"]
    assert [json.loads(m["content"]) for _, m in tool_messages] == [s["observation"] for s in row["trajectory"]["searches"]]
    # 观察确已进入下一轮生成输入，而不只是记录在独立trajectory字段。
    for i, generation in enumerate(row["generations"]):
        n = len(generation["input_messages"])
        assert row["messages"][:n] == generation["input_messages"], (path, i, "生成输入与实际消息不符")
    labels = observed_labels(grade, diagnostic)
    status = diagnostic["protocol"]
    diagnostic.update(task_id=row["task_id"], historical_replay=replay,
                      historical_grade_matches_replay=True, source=reference(path, "/raw_result"))
    result = dict(source=reference(path), first_record=reference(str(Path(path).parent / "first_records.jsonl").replace("\\", "/"), f"line:{line}"),
                  historical_grade=grade, historical_check_states=historical_checks(grade, row["raw_result"]),
                  terminal_status=status, raw_result=row["raw_result"],
                  terminal_generation_text=row["generations"][-1]["raw_text"],
                  actions=[e for e in events(row) if e["kind"] == "query_or_tool_arguments"],
                  observations=[dict(message_index=i, content=json.loads(m["content"])) for i, m in tool_messages],
                  runtime_error=row["runtime_error"], infrastructure_failure=row["infrastructure_failure"],
                  observable_error_labels=labels, analysis=brief_analysis(grade, diagnostic),
                  primary_cause=None, human_confirmation=False,
                  diagnostic_key=dict(file="failure-diagnostics.jsonl", task_id=row["task_id"]))
    return result, diagnostic


def counts(rows):
    result = dict(tasks=len(rows), quadrants=dict(Counter(r["quadrant"] for r in rows)), sides={})
    for side in SIDES:
        gs = [r[side]["historical_grade"] for r in rows]
        result["sides"][side] = dict(tasks=len(gs), failures=sum(not g["complete_success"] for g in gs),
            complete_success=sum(g["complete_success"] for g in gs), severe=sum(g["severe_error"] for g in gs),
            protocol=sum(g["protocol_error"] for g in gs),
            actual_dispositions=dict(Counter(r[side]["terminal_status"]["actual_bucket"] for r in rows)),
            parsed_disposition_correct=sum(r[side]["terminal_status"]["actual_disposition"] in r["expected_disposition"].split("|") for r in rows),
            parsed_disposition_wrong=sum(r[side]["terminal_status"]["actual_disposition"] is not None and
                                         r[side]["terminal_status"]["actual_disposition"] not in r["expected_disposition"].split("|") for r in rows),
            raw_disposition_correct=sum(r[side]["terminal_status"]["raw_disposition"] in r["expected_disposition"].split("|") for r in rows),
            raw_dispositions=dict(Counter(r[side]["terminal_status"]["raw_disposition"] or "unknown" for r in rows)))
    return result


def transition_ids(rows, field):
    result = {f"{int(a)}->{int(b)}": [] for a in (False, True) for b in (False, True)}
    for r in rows:
        key = f"{int(r['sft']['historical_grade'][field])}->{int(r['rl']['historical_grade'][field])}"
        result[key].append(r["task_id"])
    return result


def summarize(rows):
    summary = counts(rows)
    summary["denominators"] = dict(paired_tasks=len(rows), model_sides=2*len(rows),
        failure_sides=sum(summary["sides"][s]["failures"] for s in SIDES),
        tasks_with_any_failure=sum(r["quadrant"] != "both_pass" for r in rows),
        label_unit="每个模型侧每标签最多一次；允许多标签，标签次数不能相加当作失败侧数")
    summary["by"] = {}
    for field in ("role", "group_id", "expected_disposition"):
        summary["by"][field] = {v: counts([r for r in rows if r[field] == v]) for v in sorted({r[field] for r in rows})}
    summary["by"]["actual_disposition"] = {s: {} for s in SIDES}
    summary["label_counts"] = {}
    summary["severe_protocol_cross"] = {}
    for side in SIDES:
        for v in sorted({r[side]["terminal_status"]["actual_bucket"] for r in rows}):
            subset = [r for r in rows if r[side]["terminal_status"]["actual_bucket"] == v]
            summary["by"]["actual_disposition"][side][v] = counts(subset)["sides"][side]
        failures = [r for r in rows if not r[side]["historical_grade"]["complete_success"]]
        labels = Counter(l for r in failures for l in r[side]["observable_error_labels"])
        summary["label_counts"][side] = dict(failure_side_denominator=len(failures), counts=dict(labels))
        cross = {f"severe={s},protocol={p}": [] for s in (0, 1) for p in (0, 1)}
        for r in rows:
            g = r[side]["historical_grade"]
            cross[f"severe={int(g['severe_error'])},protocol={int(g['protocol_error'])}"].append(r["task_id"])
        summary["severe_protocol_cross"][side] = cross
    summary["transitions"] = {f: transition_ids(rows, f) for f in ("complete_success", "severe_error", "protocol_error")}
    joint, dispositions = {}, {}
    for r in rows:
        states = [f"S{int(r[s]['historical_grade']['severe_error'])}P{int(r[s]['historical_grade']['protocol_error'])}" for s in SIDES]
        joint.setdefault("->".join(states), []).append(r["task_id"])
        ds = [r[s]["terminal_status"]["actual_bucket"] for s in SIDES]
        dispositions.setdefault("->".join(ds), []).append(r["task_id"])
    summary["joint_severe_protocol_transitions"] = joint
    summary["actual_disposition_transitions"] = dispositions
    summary["severe_resolved_still_failed"] = [r["task_id"] for r in rows
        if r["sft"]["historical_grade"]["severe_error"] and not r["rl"]["historical_grade"]["severe_error"]
        and not r["rl"]["historical_grade"]["complete_success"]]
    summary["first_observable_divergence"] = dict(Counter(r["first_observable_divergence"]["kind"] for r in rows))
    return summary


def build(root):
    root = Path(root).resolve()
    test = root / "agentic_runs/stage8-20261016/test"
    baseline = read(root / "agentic_runs/stage7-preparation-20261014/baseline.json")
    def local(path):
        return root / (path.replace("runs/", "agentic_runs/", 1) if path.startswith("runs/") else path)
    tasks = load_public_tasks(local(baseline["public_tasks"]))
    standards = load_private_standards(local(baseline["standards"]))
    clauses = load_policy_support(local(baseline["policy_support"]))
    validate_task_pairs(tasks, standards, clauses)
    task_ids = [tid for tid, s in standards.items() if s.split == "test"]
    assert len(task_ids) == 120
    inputs = {"public_tasks.json": baseline["public_tasks"], "private_standards.v5.json": baseline["standards"],
              "policy_support.json": baseline["policy_support"]}
    configs, tables, metadata = {}, {}, {}
    for side, phase, step in (("sft", "sft", 120), ("rl", "grpo", 40)):
        directory = test / side
        config = read(directory / "configuration.json")
        selection = read(root / f"agentic_runs/stage7-preparation-20261014/formal/{phase}-selection.json")
        assert config["adapter_path"] == selection["selected_adapter"]
        assert config["adapter_path"].endswith(f"/{phase}/checkpoint-{step}")
        assert config["evaluation_split"] == "test" and config["run_kind"] == "formal"
        assert config["samples_per_task"] == 1 and config["run_purpose"] == "evaluate"
        assert config["tasks"] == baseline["public_tasks"] and config["policy_support"] == baseline["policy_support"]
        assert len(config["task_ids"]) == 120 and set(config["task_ids"]) == set(task_ids)
        for k in ("model_id", "model_revision", "retrieval", "budget", "grading_revision", "standards", "baseline"):
            expected = "runs/stage7-preparation-20261014/baseline.json" if k == "baseline" else baseline[k]
            assert config[k] == expected, (side, k)
        assert config["generation"] == baseline["evaluation_generation"]
        assert config["grading_revision"] == "pre-stage7-grading-v13-20261014"
        assert config["standards"].endswith("private_standards.v5.json")
        load = read(directory / "model_load.json")
        assert load["base_provenance"]["revision"] == config["model_revision"]
        assert load["base_provenance"]["repo_id"] == config["model_id"]
        assert load["adapter_path"].endswith("/" + config["adapter_path"])
        for name, path in inputs.items():
            assert read(directory / "inputs" / name) == read(local(path)), (side, name)
        for source in SOURCE_FILES[3:]:
            # 冻结源码保持原目录；仅归一化本次包迁移改变的导入名。
            archived = source.replace("mymind/rl/", "mymind/experiments/agentic_rl/")
            original = (directory / "source" / archived).read_text(encoding="utf-8-sig")
            current = (SOURCE / source).read_text(encoding="utf-8-sig")
            assert original.replace("experiments.agentic_rl", "rl") == current, (side, source)
        table = records(directory / "first_records.jsonl")
        mapping = unique(table, side)
        assert {p.stem for p in directory.glob("M6-*.json")} == set(task_ids)
        for tid, r in mapping.items():
            assert read(directory / f"{tid}.json") == r, (side, tid, "首次表与逐题原件不符")
            assert not r["infrastructure_failure"], (side, tid, "首次基础设施失败须保留而不能用诊断替代")
            assert r["trajectory"]["materials"] == [m.model_dump() for m in tasks[tid].materials()]
        metrics = read(directory / "metrics.json")
        assert metrics["table_kind"] == "first_table"
        assert read(directory / "execution.json")["exit_code"] == 0
        configs[side], tables[side] = config, table
        metadata[side] = dict(configuration=str((directory / "configuration.json").relative_to(root)).replace("\\", "/"),
            model_id=config["model_id"], model_revision=config["model_revision"], adapter=config["adapter_path"],
            selected_step=step, grading_revision=config["grading_revision"], standard_version="v5",
            source_matches_frozen=True, input_snapshots_match=True, first_records_match_individual_files=True,
            selection=reference(str(local(config["model_selection"]).relative_to(root)).replace("\\", "/")),
            generation=config["generation"], retrieval=config["retrieval"], search_limit=config["search_limit"])
    for k in configs["sft"]:
        if k not in {"adapter_path", "model_selection", "output_dir"}:
            assert configs["sft"][k] == configs["rl"][k], k
    pairs, diagnostics = [], []
    line_numbers = {s: {r["task_id"]: i+1 for i, r in enumerate(tables[s])} for s in SIDES}
    for tid, left, right in pair_records(task_ids, tables["sft"], tables["rl"]):
        standard = standards[tid]
        pair = dict(task_id=tid, role=tasks[tid].role, group_id=standard.group_id,
                    expected_disposition="|".join(sorted(standard.dispositions)),
                    standard=reference(str(local(baseline["standards"]).relative_to(root)).replace("\\", "/"),
                                       f"task_id:{tid}"),
                    expected_required=[e.model_dump() for e in standard.expectations if e.required],
                    quadrant=quadrant(left["grade"], right["grade"]),
                    first_observable_divergence=first_divergence(left, right))
        for side, row in zip(SIDES, (left, right)):
            path = f"agentic_runs/stage8-20261016/test/{side}/{tid}.json"
            pair[side], d = analyze_side(row, standard, clauses, path, line_numbers[side][tid])
            pair[side]["diagnostic_key"]["side"] = side
            pair[side]["model"] = metadata[side]
            d["side"] = side
            diagnostics.append(d)
        pairs.append(pair)
    summary = summarize(pairs)
    summary["diagnostic_summary"] = {}
    for side in SIDES:
        ds = [d for d in diagnostics if d["side"] == side]
        required = [e for d in ds for e in d["required_coverage"]]
        summary["diagnostic_summary"][side] = dict(
            side_denominator=len(ds), item_denominator=sum(len(d["items"]) for d in ds),
            item_states=dict(Counter(i["status"] for d in ds for i in d["items"])),
            required_item_denominator=len(required), required_states=dict(Counter(e["state"] for e in required)),
            required_support_available=sum(e["expected_support_available"] for e in required),
            historical_later_checks_not_run=sum(p[side]["historical_check_states"]["required_coverage"] == "not_run" for p in pairs))
    summary.update(metadata=metadata, missing_originals=[], model_calls=0, human_confirmations=0)
    historical = read(root / "agentic_runs/stage8-20261016/paired-changes.json")
    conflicts = []
    for old, current in (("both_success", "both_pass"), ("new_success", "gained"), ("lost_success", "lost")):
        if set(historical[old]) != {r["task_id"] for r in pairs if r["quadrant"] == current}:
            conflicts.append(old)
    reference_counts = dict(both_pass=40, both_fail=74, gained=3, lost=3)
    if summary["quadrants"] != reference_counts:
        conflicts.append("quadrant_counts")
    for side, severe, protocol in (("sft", 38, 24), ("rl", 27, 19)):
        actual = summary["sides"][side]
        if (actual["complete_success"], actual["severe"], actual["protocol"]) != (43, severe, protocol):
            conflicts.append(side + "_metrics")
    summary["historical_reference_conflicts"] = conflicts
    return pairs, diagnostics, summary


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(root, output):
    pairs, diagnostics, summary = build(root)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("paired-failures.jsonl", pairs), ("failure-diagnostics.jsonl", diagnostics)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write(output / "failure-summary.json", summary)
    execution_path = output / "execution.json"
    execution = read(execution_path) if execution_path.exists() else {}
    status = subprocess.run(["git", "status", "--short"], cwd=SOURCE, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SOURCE, capture_output=True, text=True, check=True)
    task_two = execution.setdefault("task_two", {})
    task_two.setdefault("initial_head", head.stdout.strip())
    task_two.setdefault("initial_working_tree", status.stdout.strip())
    task_two.update(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        command=[sys.executable, *sys.argv], source_files=SOURCE_FILES,
        inputs="正式test/sft与test/rl首次表及逐题原件；冻结v13源码、v5标准与公开题面快照逐项相等",
        output=str(output), paired_tasks=len(pairs), failure_sides=summary["denominators"]["failure_sides"],
        diagnostics=len(diagnostics), human_confirmations=0,
        consumption=dict(model_calls=0, gpu_seconds=0, external_api=0, remote_rerank=0, downloads=0, training_updates=0))
    snapshots = []
    for name in SOURCE_FILES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
        snapshots.append(str(target.relative_to(output)).replace("\\", "/"))
    task_two["source_snapshots"] = snapshots
    write(execution_path, execution)
    print(json.dumps({k: summary[k] for k in ("tasks", "quadrants", "sides", "denominators", "historical_reference_conflicts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    args = parser.parse_args()
    run(args.root.resolve(), args.root.resolve() / "agentic_runs/optimization-20261022")
