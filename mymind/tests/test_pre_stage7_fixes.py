"""训练侧原轨迹与受控生成桩验证三个修复；不调用模型。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTED_PATH = ROOT / "agentic_runs/pre-stage7-fixes-20261014/stage6/accepted_trajectories.json"
if not SELECTED_PATH.is_file():
    pytest.skip("需要本地保留的正式教师轨迹 accepted_trajectories.json", allow_module_level=True)
sys.path.insert(0, str(ROOT / "mymind"))
sys.path.insert(0, str(ROOT / "scripts"))
from agents.customer_policy.protocol import VisibleMaterial, VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import Expectation, PrivateStandard, load_private_standards, load_policy_support
from build_stage6_formal import revise_optional_recording
from tests.agentic_rl_samples import (
    CLAUSES as DEV_CLAUSES, STANDARDS as DEV_STANDARDS, accepted_result, trajectory,
)

DATA = ROOT / "data/agentic_rl/formal"
STANDARDS = load_private_standards(DATA / "train_private_standards.v4.json")
CLAUSES = load_policy_support(DATA / "policy_support.json")
SELECTED = {row["task_id"]: row for row in json.loads(SELECTED_PATH.read_text(encoding="utf-8"))}


def severe_probe(name):
    """在保留的正式示范上注入单项错误，不作为历史模型输出。"""
    task_id = {
        "wrong_refund_threshold_control": "M6-R01-001",
        "unknown_scope_as_enabled_control": "M6-P05-001",
    }.get(name, "M6-P02-003")
    row = SELECTED[task_id]
    result = deepcopy(row["raw_result"])
    if name in {"false_recovery_as_reason", "false_expired_response_as_reason"}:
        code = "limited_recovered_fault" if name == "false_recovery_as_reason" else "token_expired"
        result["reasons"].append({**result["reasons"][0], "code": code})
    elif name == "false_recovery_as_fact":
        result["facts"].append(dict(target=result["reasons"][0]["target"],
            type="recovered", value=True, evidence=["user:1"]))
    elif name == "wrong_refund_threshold_control":
        next(f for f in result["facts"] if f["type"] == "limit_workdays")["value"] = 5
    else:
        next(f for f in result["facts"] if f["type"] == "feature_enabled")["value"]["status"] = True
    return dict(task_id=task_id, result=result, evidence=row["trajectory"])


def score(row, result=None, standard=None):
    return grade_result(standard or STANDARDS[row["task_id"]], result or row["raw_result"],
                        VisibleTrajectory.model_validate(row["trajectory"]), CLAUSES)


@pytest.mark.parametrize("name", ["false_recovery_as_reason", "false_recovery_as_fact",
    "false_expired_response_as_reason", "wrong_refund_threshold_control", "unknown_scope_as_enabled_control"])
def test_report_severe_counterexamples(name):
    row = severe_probe(name)
    grade = grade_result(STANDARDS[row["task_id"]], row["result"],
                         VisibleTrajectory.model_validate(row["evidence"]), CLAUSES)
    assert grade.reward == -1 and grade.severe_error


def test_unknown_recovery_reason_is_severe_but_omitted_expectation_is_not_unknown():
    row = SELECTED["M6-P02-003"]
    result = deepcopy(row["raw_result"])
    result["reasons"].append({**result["reasons"][0], "code": "limited_recovered_fault"})
    assert score(row, result).severe_error
    standard = STANDARDS[row["task_id"]].model_copy(deep=True)
    standard.unprovided_user_facts = {}
    assert not score(row, result, standard).severe_error
    result = deepcopy(row["raw_result"])
    result["facts"].append(dict(target=result["reasons"][0]["target"], type="recovered", value="unknown", evidence=["user:1"]))
    assert not score(row, result).severe_error


@pytest.mark.parametrize("task_id", ["M6-P02-003", "M6-P05-001"])
def test_platform_reason_cannot_assert_disabled_when_enabled_or_unknown(task_id):
    row = SELECTED[task_id]
    result = deepcopy(row["raw_result"])
    result["reasons"].append({**result["reasons"][0], "code": "platform_enablement_required"})
    assert score(row, result).severe_error


@pytest.mark.parametrize("task_id,other_status,severe", [
    ("S07", False, True),   # 当前已开通，异范围未开通。
    ("S08", True, False),   # 当前未开通，异范围已开通。
    ("S09", False, True),   # 当前未知，异范围未开通。
])
@pytest.mark.parametrize("different_dimension", ["project", "environment"])
def test_platform_reason_uses_current_request_scope(task_id, other_status, severe, different_dimension):
    standard = DEV_STANDARDS[task_id].model_copy(deep=True)
    visible = trajectory(task_id)
    answer = accepted_result(task_id)
    scope_fact = next(a for e in standard.expectations if e.section == "facts"
                      for a in e.alternatives if a.get("type") == "request_scope")
    target, scope = scope_fact["target"], scope_fact["value"]
    other_scope = dict(scope)
    other_scope[different_dimension] = ("Beta" if different_dimension == "project" else
                                        "test" if scope["environment"] == "production" else "production")
    record = VisibleMaterial(reference_id="user:3", content=(
        f"另有开通记录：{other_scope['project']}项目{other_scope['environment']}环境导出功能"
        f"{'已开通' if other_status else '未开通'}；当前请求仍为{scope['project']}项目{scope['environment']}环境。"))
    visible.materials.append(record)
    standard.materials[record.reference_id] = record.content
    standard.expectations.append(Expectation(section="facts", alternatives=[dict(
        target=target, type="feature_enabled", value={**other_scope, "status": other_status})],
        supports=[[record.reference_id]], required=False))
    assert grade_result(standard, answer, visible, DEV_CLAUSES).complete_success
    if severe:
        answer["reasons"].append(dict(target=target, code="platform_enablement_required",
                                      evidence=["user:1", "user:2", "search:1:2"]))
    grade = grade_result(standard, answer, visible, DEV_CLAUSES)
    assert grade.severe_error is severe
    assert grade.reward == (-1 if severe else 1)


def test_reason_severity_precedes_invalid_terminal_protocol():
    row = severe_probe("false_recovery_as_reason")
    result = deepcopy(row["result"])
    result["extra"] = "多余字段"
    grade = grade_result(STANDARDS[row["task_id"]], result,
        VisibleTrajectory.model_validate(row["evidence"]), CLAUSES)
    assert grade.reward == -1 and grade.severe_error and grade.protocol_error


def test_known_severe_damage_contradicts_limited_reason_even_when_recovered():
    row = SELECTED["M6-F02-001"]
    standard = STANDARDS[row["task_id"]].model_copy(deep=True)
    for expected in standard.expectations:
        for alternative in expected.alternatives:
            if alternative.get("type") == "recovered":
                alternative["value"] = True
    result = deepcopy(row["raw_result"])
    result["reasons"].append({**result["reasons"][0], "code": "limited_recovered_fault"})
    assert score(row, result, standard).severe_error  # 已知支付影响仍存在


@pytest.mark.parametrize("task_id", ["M6-F01-003", "M6-T01-001", "M6-P02-003", "M6-F02-001"])
def test_correct_original_reasons_remain_complete(task_id):
    assert score(SELECTED[task_id]).complete_success


def test_ordinary_wrong_reason_and_missing_citation_are_not_severe():
    row = SELECTED["M6-F02-001"]
    result = deepcopy(row["raw_result"])
    result["reasons"].append({**result["reasons"][0], "code": "evidence_insufficient"})
    grade = score(row, result)
    assert grade.reward == 0.3 and not grade.severe_error
    result = deepcopy(row["raw_result"])
    result["reasons"][0]["evidence"] = []
    assert score(row, result).reward == 0 and not score(row, result).severe_error
    row = SELECTED["M6-T01-001"]
    result = deepcopy(row["raw_result"])
    result["reasons"].append({**result["reasons"][0], "code": "user_config_fixable"})
    assert not score(row, result).severe_error


@pytest.mark.parametrize("task_id,fields", [
    ("M6-B01-001", ["order_id"]),
    ("M6-B01-001", ["transaction_id", "payment_time"]),
    ("M6-B01-001", ["amount", "currency"]),
    ("M6-B01-001", ["payment_channel"]),
    ("M6-F02-001", None),
])
def test_correct_optional_recording_fields(task_id, fields):
    row = SELECTED[task_id]
    standard = PrivateStandard.model_validate(revise_optional_recording(STANDARDS[row["task_id"]].model_dump()))
    result = deepcopy(row["raw_result"])
    if fields is not None:
        next(step for step in result["next_steps"] if step["code"] == "record_diagnostics")["params"]["fields"] = fields
    grade = score(row, result, standard)
    assert grade.complete_success, grade.to_dict()


@pytest.mark.parametrize("change", ["unrelated", "wrong_target", "unneeded_question", "no_evidence", "unknown_claim"])
def test_recording_does_not_admit_wrong_or_unrelated_content(change):
    row = SELECTED["M6-B01-001"]
    standard = PrivateStandard.model_validate(revise_optional_recording(STANDARDS[row["task_id"]].model_dump()))
    result = deepcopy(row["raw_result"])
    step = next(step for step in result["next_steps"] if step["code"] == "record_diagnostics")
    if change == "unrelated":
        step["params"]["fields"].append("feature_enabled")
    elif change == "wrong_target":
        step["target"] = "case"
    elif change == "unneeded_question":
        result["missing_info"].append(dict(target=step["target"], field="amount", purpose="handoff", evidence=["user:1"]))
    elif change == "no_evidence":
        step["evidence"] = ["user:1"]  # 未引用政策或角色中的核验规则
    else:
        result["facts"].append(dict(target=step["target"], type="data_loss", value=False, evidence=["user:1"]))
    grade = score(row, result, standard)
    assert not grade.complete_success
    assert grade.severe_error == (change == "unknown_claim")


def test_unknown_damage_can_be_recorded_without_asserting_its_value():
    row = SELECTED["M6-F02-001"]
    result = deepcopy(row["raw_result"])
    step = next(step for step in result["next_steps"] if step["code"] == "record_diagnostics")
    step["params"]["fields"] += ["data_loss", "data_corruption"]
    assert score(row, result).complete_success
    result["facts"].append(dict(target=step["target"], type="data_loss", value=False, evidence=["user:1"]))
    assert score(row, result).severe_error
