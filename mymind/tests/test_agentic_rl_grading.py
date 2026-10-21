from copy import deepcopy

import pytest

from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from tests.agentic_rl_samples import (
    BILLING, CLAUSES, STANDARDS, TASKS, accepted_result, s04_result, trajectory,
)


def score(task_id, result, visible=None):
    return grade_result(STANDARDS[task_id], result, visible or trajectory(task_id), CLAUSES)


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_all_development_standards_have_complete_supported_results(task_id):
    grade = score(task_id, accepted_result(task_id))
    assert grade.reward == 1, grade.to_dict()
    assert grade.complete_success and not grade.severe_error


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_optional_correct_content_does_not_change_complete_success(task_id):
    grade = score(task_id, accepted_result(task_id, optional=True))
    assert grade.reward == 1, grade.to_dict()


def test_s04_four_business_reward_levels():
    complete = s04_result()
    assert score("S04", complete).reward == 1
    partial = deepcopy(complete)
    partial["next_steps"] = partial["next_steps"][:1]
    assert score("S04", partial).reward == 0.3
    repeated_input = {"disposition": "answer", "reasons": [], "facts": complete["facts"][:3],
                      "next_steps": [], "missing_info": []}
    assert score("S04", repeated_input).reward == 0
    wrong = deepcopy(complete)
    wrong["facts"][4]["value"] = 3
    wrong["facts"][5]["value"] = False
    wrong["disposition"] = "escalate"
    grade = score("S04", wrong)
    assert grade.reward == -1 and grade.severe_error


def test_supported_optional_channel_coverage_is_accepted():
    result = s04_result()
    result["facts"].append({"target": "R1", "type": "timing_policy_covered", "value": True,
                            "evidence": ["user:1", "search:1:1"]})
    assert score("S04", result).complete_success
    result["facts"][-1]["evidence"] = ["user:1"]
    assert score("S04", result).reward == 0.3


def test_interface_noun_suffix_is_not_a_business_contradiction():
    result = accepted_result("S07")
    interface = next(f for f in result["facts"] if f["type"] == "interface_name")
    interface["value"] += "接口"
    assert score("S07", result).complete_success
    interface["value"] = "基础查询接口"
    assert score("S07", result).severe_error


def test_reported_expired_credentials_support_optional_invalid_authentication():
    result = accepted_result("S10")
    result["facts"].append({"target": "Q1", "type": "auth_valid", "value": False, "evidence": ["user:1"]})
    assert score("S10", result).complete_success
    result["facts"][-1]["value"] = True
    assert score("S10", result).severe_error


def test_unknown_approval_can_explain_clock_and_limit_without_claiming_elapsed_state():
    result = accepted_result("S01")
    for kind, value in [("timer_start", "approval"), ("limit_workdays", 5)]:
        result["facts"].append({"target": "R1", "type": kind, "value": value, "evidence": ["search:1:1"]})
    assert score("S01", result).complete_success
    result["facts"].append({"target": "R1", "type": "within_window", "value": True,
                            "evidence": ["user:1", "search:1:1"]})
    assert score("S01", result).reward == -1


def test_correct_label_or_generic_escalation_without_judgment_earns_zero():
    for task_id, disposition in [("S04", "answer"), ("S12", "escalate")]:
        grade = score(task_id, {"disposition": disposition, "reasons": [], "facts": [],
                                "next_steps": [], "missing_info": []})
        assert grade.reward == 0


def test_dual_refund_object_binding_error_is_severe():
    result = accepted_result("D02")
    for fact in result["facts"]:
        if fact["type"] == "elapsed_workdays":
            fact["target"] = "R2" if fact["target"] == "R1" else "R1"
    assert score("D02", result).reward == -1


def test_dual_refund_complete_success_requires_both_records():
    result = accepted_result("D02")
    for section in ("facts", "reasons", "next_steps", "missing_info"):
        result[section] = [item for item in result[section] if item["target"] != "R2"]
    grade = score("D02", result)
    assert grade.reward == 0.3 and not grade.complete_success


@pytest.mark.parametrize("status", [True, False])
def test_other_environment_record_cannot_confirm_current_status(status):
    result = accepted_result("P05")
    for fact in result["facts"]:
        if fact["type"] == "feature_enabled":
            fact["value"]["status"] = status
    assert score("P05", result).reward == -1


@pytest.mark.parametrize("change", ["known_status", "unneeded_question", "condition_omitted"])
def test_unknown_and_genuine_missing_information_are_distinguished(change):
    result = accepted_result("S01")
    if change == "known_status":
        for fact in result["facts"]:
            if fact["type"] == "approval_status":
                fact["value"] = "approved"
        assert score("S01", result).reward == -1
    elif change == "unneeded_question":
        result["missing_info"].append({"target": "R1", "field": "payment_channel", "purpose": "decision",
                                       "evidence": ["user:1"]})
        assert score("S01", result).reward == 0.3
    else:
        result["missing_info"][1].pop("condition")
        assert score("S01", result).reward == 0.3


@pytest.mark.parametrize("evidence", [[], ["search:1:404"], ["search:1:status"], ["user:1"]])
def test_existing_correct_value_without_visible_policy_support_is_not_success(evidence):
    result = s04_result()
    result["facts"][4]["evidence"] = evidence
    grade = score("S04", result)
    assert grade.reward == 0.3 and not grade.complete_success
    assert any("正文不支持" in error for error in grade.errors)


def test_source_match_does_not_supply_unreturned_or_truncated_text():
    result = s04_result()
    wrong_body = trajectory("S04", billing_text="退款申请受理不等于审核通过。")
    assert score("S04", result, wrong_body).reward == 0
    truncated = trajectory("S04", billing_text=BILLING.split("实验银行卡渠道为", 1)[0])
    assert score("S04", result, truncated).reward == 0


def test_sufficient_relevant_excerpt_does_not_require_whole_policy_paragraph():
    clause_names = ["refund_clock", "refund_bank_limit", "refund_wait", "refund_overdue"]
    text = "\n".join(CLAUSES[key].text for key in clause_names)
    assert score("S04", s04_result(), trajectory("S04", billing_text=text)).reward == 1


def test_multiple_seen_snippets_can_support_one_judgment_with_neutral_ids():
    payload = trajectory("S04").model_dump()
    observation = payload["searches"][0]["observation"]
    original = observation["snippets"][0]
    observation["snippets"] = [
        {**original, "reference_id": "search:1:17", "content": CLAUSES["refund_clock"].text},
        {**original, "reference_id": "search:1:28", "content": "\n".join(
            CLAUSES[key].text for key in ("refund_bank_limit", "refund_wait", "refund_overdue"))},
    ]
    result = s04_result()
    for section in ("facts", "reasons", "next_steps"):
        for item in result[section]:
            if "search:1:1" in item["evidence"]:
                item["evidence"] = [ref for ref in item["evidence"] if ref != "search:1:1"] + ["search:1:17", "search:1:28"]
    assert score("S04", result, VisibleTrajectory.model_validate(payload)).reward == 1


def test_wrong_source_cannot_borrow_support_from_uncited_correct_snippet():
    payload = trajectory("S04").model_dump()
    payload["searches"][0]["observation"]["snippets"][0]["source_id"] = "unrelated-policy"
    assert score("S04", s04_result(), VisibleTrajectory.model_validate(payload)).reward == 0


def test_severe_business_error_takes_priority_even_if_protocol_is_invalid():
    result = s04_result()
    result["facts"][4]["value"] = 3
    result["unstructured_answer"] = "无关文本"
    grade = score("S04", result)
    assert grade.reward == -1 and grade.severe_error and grade.protocol_error


@pytest.mark.parametrize("raw", ["{", None, {"disposition": "answer"}])
def test_invalid_or_truncated_terminal_result_earns_no_completion_reward(raw):
    grade = score("S04", raw)
    assert grade.reward == 0 and grade.protocol_error


def test_balance_timing_is_not_inferred_from_empty_search_or_channel_limits():
    result = accepted_result("S13")
    assert score("S13", result).reward == 1
    empty = trajectory("S13").model_dump()
    empty["searches"][0]["observation"]["status"]["state"] = "empty"
    empty["searches"][0]["observation"]["snippets"] = []
    assert score("S13", result, VisibleTrajectory.model_validate(empty)).reward == 0
    result["facts"].append({"target": "R1", "type": "limit_workdays", "value": 5,
                            "evidence": ["search:1:1"]})
    assert score("S13", result).reward == -1


def test_user_material_cannot_be_replaced_by_private_parsed_facts():
    payload = trajectory("S04").model_dump()
    payload["materials"][1]["content"] = "银行卡已通过，时间未知"
    assert score("S04", s04_result(), VisibleTrajectory.model_validate(payload)).reward == 0


def test_wrong_disposition_is_an_ordinary_error_and_costs_do_not_add_reward():
    result = s04_result()
    result["disposition"] = "request_info"
    grade = score("S04", result)
    assert grade.reward == 0 and not grade.severe_error
    no_search = trajectory("S12").model_dump()
    no_search["searches"] = []
    assert score("S12", accepted_result("S12"), VisibleTrajectory.model_validate(no_search)).reward == 1


def test_task_identity_must_match_the_standard():
    with pytest.raises(ValueError, match="task_id"):
        score("S04", s04_result(), trajectory("S03"))


def test_wrong_threshold_in_advice_is_severe_without_wrong_fact():
    result = s04_result()
    result["next_steps"][1]["params"]["limit_workdays"] = 3
    grade = score("S04", result)
    assert grade.reward == -1 and grade.severe_error


def test_wrong_project_in_advice_is_severe_without_wrong_fact():
    result = accepted_result("S08")
    result["next_steps"][0]["params"]["project"] = "Beta"
    assert score("S08", result).reward == -1


def test_diagnostic_field_order_and_optional_low_risk_direction_are_flexible():
    result = accepted_result("S11")
    for step in result["next_steps"]:
        if step["code"] == "record_diagnostics":
            step["params"]["fields"].reverse()
        if step["code"] == "low_risk_checks":
            step["params"]["directions"] = ["network"]
    assert score("S11", result).reward == 1


@pytest.mark.parametrize("change", ["missing_value", "extra_fact_field", "missing_scope_param"])
def test_protocol_omission_is_not_misclassified_as_a_severe_business_error(change):
    result = s04_result() if change != "missing_scope_param" else accepted_result("S08")
    if change == "missing_value":
        result["facts"][4].pop("value")
    elif change == "extra_fact_field":
        result["facts"][4]["note"] = "多余字段"
    else:
        result["next_steps"][0]["params"].pop("environment")
    grade = score("S04" if change != "missing_scope_param" else "S08", result)
    assert grade.reward == 0 and grade.protocol_error and not grade.severe_error


def test_correct_status_without_required_scope_is_protocol_failure():
    result = accepted_result("S07")
    fact = next(item for item in result["facts"] if item["type"] == "outbound_ip_allowed")
    fact["value"] = False
    grade = score("S07", result)
    assert grade.reward == 0 and grade.protocol_error and not grade.severe_error
    fact["value"] = {"project": "Beta", "status": False}
    assert score("S07", result).severe_error


def test_known_request_environment_is_allowed_as_optional_context():
    result = accepted_result("S07")
    result["facts"].append({"target": "Q1", "type": "environment", "value": "production",
                            "evidence": ["user:2"]})
    assert score("S07", result).complete_success
    result["facts"][-1]["evidence"] = ["user:1"]
    assert score("S07", result).reward == 0.3
    result["facts"][-1]["value"] = "test"
    assert score("S07", result).severe_error


def test_unexpected_params_are_not_invented_business_thresholds():
    result = s04_result()
    result["next_steps"][0]["params"] = {"limit_workdays": 5}
    grade = score("S04", result)
    assert grade.reward == 0 and grade.protocol_error and not grade.severe_error


@pytest.mark.parametrize("kind,value", [("limit_workdays", "5"), ("received", "false")])
def test_correct_values_encoded_as_strings_are_protocol_errors(kind, value):
    result = s04_result()
    if kind == "received":
        result["facts"].append({"target": "R1", "type": "received", "value": False,
                                "evidence": ["user:1"]})
    next(item for item in result["facts"] if item["type"] == kind)["value"] = value
    grade = score("S04", result)
    assert grade.reward == 0 and grade.protocol_error and not grade.severe_error


def test_approved_stage_is_optional_for_uncovered_channel_without_timing_claim():
    result = accepted_result("S13")
    result["facts"].append({"target": "R1", "type": "refund_stage", "value": "approved",
                            "evidence": ["user:1"]})
    assert score("S13", result).complete_success
    result["facts"].append({"target": "R1", "type": "limit_workdays", "value": 5,
                            "evidence": ["search:1:1"]})
    assert score("S13", result).severe_error
