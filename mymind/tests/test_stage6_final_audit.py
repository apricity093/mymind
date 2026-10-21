"""用保留的正式训练记录及受控变异检查故障可选项，不调用模型。"""
from copy import deepcopy
import json
import os
from pathlib import Path

from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_private_standards, load_policy_support
from tests.agentic_rl_samples import retained_path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.getenv("MYMIND_WORK_ROOT", str(REPO)))
DATA = WORK / "data/agentic_rl/formal"
RUN = WORK / ("runs" if WORK != REPO else "agentic_runs") / "stage6-20261013"


def original(task_id):
    return json.loads(retained_path(RUN / "teacher-v3" / f"{task_id}-candidate-1.json").read_text(encoding="utf-8"))


def grade(task_id, result=None):
    row = original(task_id)
    standards = load_private_standards(DATA / "train_private_standards.v4.json")
    clauses = load_policy_support(DATA / "policy_support.json")
    return grade_result(standards[task_id], result or row["raw_result"], VisibleTrajectory.model_validate(row["trajectory"]), clauses)


def test_actual_missing_fault_materials_are_optional_correct_content():
    assert grade("M6-F01-003").complete_success


def test_actual_payment_fault_diagnostic_context_is_allowed():
    assert grade("M6-F02-001").complete_success


def test_missing_conditional_escalation_remains_incomplete():
    result = deepcopy(original("M6-F01-003")["raw_result"])
    result["next_steps"] = [step for step in result["next_steps"] if step["code"] != "escalate_if_severe_impact"]
    scored = grade("M6-F01-003", result)
    assert not scored.complete_success and scored.reward == 0.3


def test_request_id_is_not_substituted_for_public_target():
    result = deepcopy(original("M6-F02-001")["raw_result"])
    for section in ("reasons", "facts", "next_steps", "missing_info"):
        for atom in result[section]:
            atom["target"] = "request-id-instead-of-call-target"
    scored = grade("M6-F02-001", result)
    assert not scored.complete_success and scored.reward == 0


def test_unneeded_payment_order_materials_are_not_added_to_fault_standard():
    result = deepcopy(original("M6-F02-001")["raw_result"])
    result["missing_info"].append(dict(target=result["reasons"][0]["target"],
        field="order_id", purpose="handoff", evidence=result["reasons"][0]["evidence"]))
    scored = grade("M6-F02-001", result)
    assert not scored.complete_success and scored.reward == 0.3
