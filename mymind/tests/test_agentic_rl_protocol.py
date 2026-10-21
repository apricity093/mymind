import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.customer_policy.protocol import (
    FACT_TYPES, MISSING_FIELDS, Fact, PublicTask, SearchObservation, TerminalResult,
    VisibleTrajectory, parse_result, public_value_catalog,
)
from agents.customer_policy.rendering import render_result
from experiments.agentic_rl.tasks import (
    RESOURCES, load_private_standards, load_public_tasks, validate_task_pairs,
)
from tests.agentic_rl_samples import (
    CLAUSES, KNOWLEDGE, STANDARDS, TASKS, accepted_result, s04_result, trajectory,
)


def test_development_resources_are_paired_and_only_use_training_groups():
    validate_task_pairs(TASKS, STANDARDS, CLAUSES)
    assert set(TASKS) == {f"S{i:02d}" for i in range(1, 14)} | {"D02", "P05"}
    assert {s.split for s in STANDARDS.values()} == {"dev"}
    assert {s.group_id for s in STANDARDS.values()} == {
        "R01", "R02", "R03", "R04", "R06", "B01", "P01", "P02", "P03", "P04",
        "T01", "F01", "F02", "D02", "P05",
    }
    assert all(clause.text in KNOWLEDGE for clause in CLAUSES.values())


def test_public_messages_and_observations_do_not_include_private_standard():
    messages = TASKS["P05"].messages()
    content = json.dumps(messages, ensure_ascii=False)
    for hidden in ("group_id", "expectations", "dispositions", "severe_items", "supports"):
        assert hidden not in content
    assert "生产环境的开通状态不知道" in content
    for material in TASKS["P05"].materials()[1:]:
        assert f"[{material.reference_id}]\n{material.content}" in messages[1]["content"]
    observation = trajectory("P05").searches[0].observation
    assert isinstance(observation.tool_content(), str)
    assert json.loads(observation.tool_content())["snippets"][0]["content"]
    assert "clause_id" not in observation.tool_content()


def test_public_fact_and_diagnostic_catalogs_match_accepted_fields():
    catalog = public_value_catalog()
    assert set(catalog["fact_labels"]) == set(FACT_TYPES)
    assert set(catalog["diagnostic_labels"]) == MISSING_FIELDS
    assert "response_summary" not in catalog["fact_labels"]
    assert "response_summary" in catalog["diagnostic_fields"]


def test_public_loader_rejects_a_private_task_file():
    with pytest.raises(ValidationError):
        load_public_tasks(RESOURCES / "dev_standards.json")


def test_private_loader_rejects_a_public_task_file():
    with pytest.raises(ValidationError):
        load_private_standards(RESOURCES / "dev_tasks.json")


@pytest.mark.parametrize("kind,value", [
    ("within_window", "unknown"), ("elapsed_workdays", True), ("limit_workdays", 0),
    ("http_status", True), ("elapsed_workdays", -1),
    ("feature_enabled", {"project": "Alpha", "status": True}),
    ("outbound_ip_allowed", {"project": "Alpha", "environment": "test",
                             "outbound_ip": "203.0.113.10", "status": True}),
])
def test_fact_value_types_and_scope_are_enforced(kind, value):
    with pytest.raises(ValidationError):
        Fact(target="Q1", type=kind, value=value, evidence=["user:1"])


def test_unknown_is_preserved_as_distinct_from_false():
    unknown = Fact(target="Q1", type="auth_valid", value="unknown", evidence=["user:1"])
    negative = Fact(target="Q1", type="auth_valid", value=False, evidence=["user:1"])
    assert unknown.value == "unknown"
    assert negative.value is False


@pytest.mark.parametrize("change", ["extra_text", "missing_field", "conditional_wrong_field", "missing_step_params"])
def test_terminal_result_has_no_free_assertion_or_implicit_completion(change):
    result = s04_result()
    if change == "extra_text":
        result["answer"] = "退款已执行"
    elif change == "missing_field":
        del result["facts"]
    elif change == "conditional_wrong_field":
        result["missing_info"] = [{"target": "R1", "field": "payment_channel", "purpose": "decision",
                                   "condition": "if_approved", "evidence": ["user:1"]}]
    else:
        del result["next_steps"][1]["params"]
    with pytest.raises(ValidationError):
        parse_result(result)


@pytest.mark.parametrize("state", ["empty", "unavailable", "invalid_arguments"])
def test_tool_status_is_distinct_and_does_not_become_knowledge(state):
    observation = SearchObservation(request_number=1, status={"reference_id": "search:1:status", "state": state,
                                                              "detail": "未返回知识正文"}, snippets=[])
    assert json.loads(observation.tool_content())["status"]["state"] == state
    payload = observation.model_dump()
    payload["snippets"] = trajectory("S04").searches[0].observation.model_dump()["snippets"]
    with pytest.raises(ValidationError):
        SearchObservation.model_validate(payload)


def test_trajectory_rejects_unseen_reference_namespace_and_out_of_order_search():
    payload = trajectory("S04").model_dump()
    payload["materials"][0]["reference_id"] = "private:answer"
    with pytest.raises(ValidationError):
        VisibleTrajectory.model_validate(payload)
    payload = trajectory("S04").model_dump()
    payload["searches"].append(deepcopy(payload["searches"][0]))
    with pytest.raises(ValidationError):
        VisibleTrajectory.model_validate(payload)


def test_renderer_does_not_fill_missing_threshold_or_advice():
    result = s04_result()
    result["facts"] = []
    result["next_steps"] = []
    text = render_result(result)
    assert "常规退款时效" in text
    assert "5" not in text and "4" not in text
    assert "退款流水" not in text
    assert "已执行" not in text


def test_renderer_preserves_unknown_scope_and_conditional_missing_information():
    text = render_result(accepted_result("P05"))
    assert "环境：生产" in text and "状态：未知" in text
    assert "已开通" not in text
    text = render_result(accepted_result("S01"))
    assert "若该笔已审核通过" in text
    assert "用于判断" in text


def test_public_modules_import_without_training_or_experiment_dependencies():
    backend = Path(__file__).resolve().parents[1]
    code = ("import sys; import agents.customer_policy.protocol; import agents.customer_policy.rendering; "
            "assert not any(m.startswith(('experiments.', 'trl', 'torch', 'transformers')) for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], cwd=backend, check=True)


def test_task_pair_validation_catches_changed_public_input():
    tasks = dict(TASKS)
    tasks["S04"] = PublicTask.model_validate({**TASKS["S04"].model_dump(), "user_text": "另一笔退款"})
    with pytest.raises(ValueError, match="公开材料不一致"):
        validate_task_pairs(tasks, STANDARDS, CLAUSES)
