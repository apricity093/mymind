"""首次配对、提前返回、处置未知和真实原件可复算性。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_optimization_failures import (build, diagnose, first_divergence, historical_checks,
    observed_labels, pair_records, quadrant, summarize, terminal_status)
from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import PrivateStandard


def sample():
    standard = PrivateStandard.model_validate(dict(task_id="case", group_id="test", split="test",
        dispositions=["request_info"], materials={"user:1": "审核状态未知"}, expectations=[
            dict(section="facts", alternatives=[dict(target="R1", type="approval_status", value="unknown")],
                 supports=[["user:1"]]),
            dict(section="missing_info", alternatives=[dict(target="R1", field="approval_status", purpose="decision")],
                 supports=[["user:1"]])]))
    trajectory = VisibleTrajectory.model_validate(dict(task_id="case", materials=[
        dict(reference_id="user:1", content="审核状态未知")], searches=[]))
    raw = dict(disposition="request_info", reasons=[], facts=[
        dict(target="R1", type="approval_status", value="unknown", evidence=["user:1"])],
        next_steps=[], missing_info=[dict(target="R1", field="approval_status", purpose="decision", evidence=["user:1"])])
    return standard, trajectory, raw


@pytest.mark.parametrize("side", [0, 1])
def test_missing_and_extra_first_records_are_rejected(side):
    a = [[dict(task_id="a"), dict(task_id="b")], [dict(task_id="a"), dict(task_id="b")]]
    a[side] = [dict(task_id="a"), dict(task_id="other")]
    with pytest.raises(ValueError, match="missing=.*b.*extra=.*other"):
        pair_records(["a", "b"], *a)


def test_duplicate_records_and_planned_ids_are_rejected():
    a = [dict(task_id="a"), dict(task_id="a")]
    with pytest.raises(ValueError, match="重复任务"):
        pair_records(["a", "b"], a, a)
    with pytest.raises(ValueError, match="预定题集含重复"):
        pair_records(["a", "a"], a, a)


def test_pairing_uses_ids_not_row_order():
    a = [dict(task_id="b"), dict(task_id="a")]
    assert [tid for tid, _, _ in pair_records(["a", "b"], a, list(reversed(a)))] == ["a", "b"]


def test_severe_return_does_not_turn_empty_missing_into_pass():
    standard, trajectory, raw = sample()
    raw["facts"][0]["value"] = "approved"
    raw["missing_info"] = []
    grade = grade_result(standard, raw, trajectory, {}).to_dict()
    assert grade["severe_error"] and not grade["protocol_error"] and grade["missing"] == []
    assert historical_checks(grade, raw)["required_coverage"] == "not_run"
    d = diagnose(standard, raw, trajectory, {})
    assert [e["state"] for e in d["required_coverage"]] == ["missing_or_unsupported"] * 2
    assert "required_missing_info_missing_or_unsupported" in observed_labels(grade, d)


def test_severe_assertions_are_checked_before_protocol_failure():
    standard, trajectory, raw = sample()
    raw["facts"][0]["value"] = "approved"
    raw["extra"] = "invalid"
    grade = grade_result(standard, raw, trajectory, {}).to_dict()
    assert grade["severe_error"] and grade["protocol_error"]
    states = historical_checks(grade, raw)
    assert states["severe_assertion_scan"] == "executed"
    assert states["disposition"] == states["item_matching_and_evidence"] == "not_run"
    status = terminal_status(raw)
    assert status["raw_disposition"] == "request_info"
    assert status["actual_disposition"] is None and status["actual_bucket"] == "parse_failed"


@pytest.mark.parametrize("raw", ["{broken", None, "[]", '{"disposition":"answer"} trailing'])
def test_parse_failure_never_enters_an_actual_disposition_bucket(raw):
    status = terminal_status(raw)
    assert status["actual_disposition"] is None and status["actual_bucket"] == "parse_failed"


def test_partial_invalid_section_coverage_is_unknown_not_missing_or_pass():
    standard, trajectory, raw = sample()
    raw["facts"][0]["evidence"] = "user:1"
    raw["missing_info"] = None
    d = diagnose(standard, raw, trajectory, {})
    assert d["items"][0]["status"] == "invalid_item"
    assert {e["state"] for e in d["required_coverage"]} == {"unknown_invalid_or_unreadable_items"}


def test_evidence_failure_is_distinct_from_payload_mismatch():
    standard, trajectory, raw = sample()
    raw["facts"][0]["evidence"] = ["user:99"]
    d = diagnose(standard, raw, trajectory, {})
    assert d["items"][0]["status"] == "unsupported_evidence"
    assert all(e["expected_support_available"] for e in d["required_coverage"])
    raw["facts"][0]["target"] = "other"
    d = diagnose(standard, raw, trajectory, {})
    assert d["items"][0]["status"] == "not_in_standard"


def test_first_divergence_ignores_call_ids_but_preserves_query_and_observation():
    a = dict(messages=[dict(role="assistant", tool_calls=[dict(id="one", function=dict(name="search", arguments='{"query":"refund"}'))]),
                       dict(role="tool", content='{"snippets": []}')])
    b = deepcopy(a)
    b["messages"][0]["tool_calls"][0]["id"] = "different"
    assert first_divergence(a, b)["kind"] == "no_observed_difference"
    b["messages"][1]["content"] = '{"snippets": ["policy"]}'
    assert first_divergence(a, b)["kind"] == "displayed_observation"
    b["messages"][0]["tool_calls"][0]["function"]["arguments"] = '{"query":"channel"}'
    assert first_divergence(a, b)["kind"] == "query_or_tool_arguments"


@pytest.fixture(scope="module")
def actual():
    if not (ROOT / "agentic_runs/stage8-20261016/test/sft/first_records.jsonl").exists():
        pytest.skip("正式首次原件仅保留本地")
    return build(ROOT)


def test_actual_120_pairs_quadrants_denominators_and_failure_analysis(actual):
    pairs, diagnostics, summary = actual
    assert len(pairs) == len({r["task_id"] for r in pairs}) == 120
    assert len(diagnostics) == 240
    assert summary["quadrants"] == dict(both_pass=40, both_fail=74, gained=3, lost=3)
    assert sum(summary["quadrants"].values()) == 120
    assert summary["denominators"]["failure_sides"] == 154
    assert summary["denominators"]["tasks_with_any_failure"] == 80
    assert not summary["historical_reference_conflicts"]
    for side in ("sft", "rl"):
        assert summary["sides"][side]["complete_success"] == 43
        assert summary["sides"][side]["failures"] == 77
        for r in pairs:
            assert r[side]["analysis"] and r[side]["primary_cause"] is None
            if not r[side]["historical_grade"]["complete_success"]:
                assert r[side]["observable_error_labels"]
    for field in ("role", "group_id", "expected_disposition"):
        assert sum(v["tasks"] for v in summary["by"][field].values()) == 120
    for side in ("sft", "rl"):
        assert sum(v["tasks"] for v in summary["by"]["actual_disposition"][side].values()) == 120
        assert summary["by"]["actual_disposition"][side]["parse_failed"]["tasks"] == summary["sides"][side]["protocol"]
        metrics = summary["sides"][side]
        assert metrics["parsed_disposition_correct"] + metrics["parsed_disposition_wrong"] + metrics["protocol"] == 120
        labels = Counter(label for r in pairs if not r[side]["historical_grade"]["complete_success"]
                         for label in r[side]["observable_error_labels"])
        assert dict(labels) == summary["label_counts"][side]["counts"]


def test_actual_cross_and_transitions_recompute_from_individual_grades(actual):
    pairs, _, summary = actual
    for side, severe, protocol in (("sft", 38, 24), ("rl", 27, 19)):
        cross = summary["severe_protocol_cross"][side]
        assert sum(map(len, cross.values())) == 120
        assert len(cross["severe=1,protocol=0"]) + len(cross["severe=1,protocol=1"]) == severe
        assert len(cross["severe=0,protocol=1"]) + len(cross["severe=1,protocol=1"]) == protocol
    for field, mapping in summary["transitions"].items():
        assert sum(map(len, mapping.values())) == 120
        for state, tids in mapping.items():
            expected = [r["task_id"] for r in pairs
                        if f"{int(r['sft']['historical_grade'][field])}->{int(r['rl']['historical_grade'][field])}" == state]
            assert tids == expected
    assert len(summary["transitions"]["severe_error"]["1->0"]) == 13
    assert len(summary["transitions"]["severe_error"]["0->1"]) == 2
    assert len(summary["severe_resolved_still_failed"]) == 12
    assert sum(map(len, summary["joint_severe_protocol_transitions"].values())) == 120


def test_actual_original_pointers_outputs_and_diagnostics_are_reproducible(actual):
    pairs, diagnostics, summary = actual
    lookup = {(d["task_id"], d["side"]): d for d in diagnostics}
    first_tables = {side: (ROOT / f"agentic_runs/stage8-20261016/test/{side}/first_records.jsonl").read_text(encoding="utf-8").splitlines()
                    for side in ("sft", "rl")}
    for r in pairs:
        for side in ("sft", "rl"):
            s = r[side]
            original = json.loads((ROOT / s["source"]["path"]).read_text(encoding="utf-8"))
            assert original["task_id"] == r["task_id"]
            assert original["grade"] == s["historical_grade"]
            assert original["raw_result"] == s["raw_result"]
            first = s["first_record"]
            line = int(first["pointer"].split(":")[1])
            assert ROOT / first["path"] == ROOT / f"agentic_runs/stage8-20261016/test/{side}/first_records.jsonl"
            assert json.loads(first_tables[side][line-1]) == original
            for action in s["actions"]:
                assert action["pointer"].startswith("/messages/")
                assert original["messages"][int(action["pointer"].split("/")[-1])]["tool_calls"]
            for observation in s["observations"]:
                assert observation["content"] == json.loads(original["messages"][observation["message_index"]]["content"])
            d = lookup[r["task_id"], side]
            assert d["historical_replay"] == original["grade"]
            assert s["observable_error_labels"] == observed_labels(original["grade"], d)
        assert r["quadrant"] == quadrant(r["sft"]["historical_grade"], r["rl"]["historical_grade"])
    rebuilt = summarize(pairs)
    assert all(summary[k] == v for k, v in rebuilt.items())
    output = ROOT / "agentic_runs/optimization-20261022"
    if (output / "paired-failures.jsonl").exists():
        saved = [json.loads(line) for line in (output / "paired-failures.jsonl").read_text(encoding="utf-8").splitlines()]
        assert saved == pairs
        assert json.loads((output / "failure-summary.json").read_text(encoding="utf-8")) == summary
        assert [json.loads(line) for line in (output / "failure-diagnostics.jsonl").read_text(encoding="utf-8").splitlines()] == diagnostics
