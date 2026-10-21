"""校准数据隔离及决定性业务误判回归。"""
from copy import deepcopy
from pathlib import Path
import json
import os
import runpy

import pytest

from agents.customer_policy.protocol import KnowledgeSnippet, VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support

DATA = Path(os.getenv("MYMIND_WORK_ROOT", str(Path(__file__).resolve().parents[2]))) / "data/agentic_rl/calibration"
if not (DATA / "public_tasks.json").is_file():
    pytest.skip("需要本地保留的校准数据", allow_module_level=True)
TASKS = load_public_tasks(DATA / "public_tasks.json")
STANDARDS = load_private_standards(DATA / "private_standards.json")
CLAUSES = load_policy_support(DATA / "policy_support.json")


def supported_result(task_id):
    standard = STANDARDS[task_id]
    result = dict(disposition=standard.dispositions[0], facts=[], reasons=[], next_steps=[], missing_info=[])
    clauses = list(CLAUSES.values())
    visible = VisibleTrajectory(task_id=task_id, materials=TASKS[task_id].materials(), searches=[{
        "action": {"query": "判分测试手工观察，不能用于示范"}, "observation": {
            "request_number": 1, "status": {"reference_id": "search:1:status", "state": "candidates"},
            "snippets": [dict(reference_id=f"search:1:{i+1}",source_id=c.source_id,
                title="测试正文",section_path=[],content=c.text) for i,c in enumerate(clauses)]}}])
    for e in standard.expectations:
        if not e.required: continue
        item = deepcopy(e.alternatives[0])
        item["evidence"] = list(dict.fromkeys(group[0] if group[0] in standard.materials else
            f"search:1:{[c.clause_id for c in clauses].index(group[0])+1}" for group in e.supports))
        result[e.section].append(item)
    return result, visible


def test_all_40_tasks_are_training_combinations_and_no_private_input_reaches_policy():
    assert len(TASKS) == 40
    assert len({s.group_id for s in STANDARDS.values()}) == 20
    for task_id, task in TASKS.items():
        assert STANDARDS[task_id].split == "train"
        messages = task.messages()
        assert task_id not in messages[0]["content"]
        assert STANDARDS[task_id].group_id not in str(messages)
        result, visible = supported_result(task_id)
        grade = grade_result(STANDARDS[task_id], result, visible, CLAUSES)
        assert grade.complete_success, (task_id, grade.to_dict())


@pytest.mark.parametrize("task_id", ["C15", "C16"])
def test_escalated_double_refund_keeps_non_escalated_record_without_duplicate_reason(task_id):
    result, visible = supported_result(task_id)
    result["reasons"] = [r for r in result["reasons"] if r["code"] != "within_refund_window"]
    assert grade_result(STANDARDS[task_id], result, visible, CLAUSES).complete_success
    result["next_steps"] = [s for s in result["next_steps"]
                            if s["code"] != "verify_refund_if_overdue_unreceived"]
    incomplete = grade_result(STANDARDS[task_id], result, visible, CLAUSES)
    assert not incomplete.complete_success and incomplete.reward == 0.3
    result["reasons"].append({"target": "R2", "code": "refund_overdue",
                              "evidence": ["user:1", "search:1:1"]})
    assert grade_result(STANDARDS[task_id], result, visible, CLAUSES).severe_error


def test_dual_refund_unknown_time_and_escalation_keep_distinct_targets():
    result, visible = supported_result("C19")
    assert result["disposition"] == "escalate"
    assert any(m["target"] == "R2" and m["purpose"] == "decision" for m in result["missing_info"])
    result["missing_info"][0]["purpose"] = "handoff"
    assert not grade_result(STANDARDS["C19"], result, visible, CLAUSES).complete_success
    result, visible = supported_result("C15")
    for f in result["facts"]:
        if f["type"] == "elapsed_workdays": f["target"] = "R2" if f["target"] == "R1" else "R1"
    assert grade_result(STANDARDS["C15"], result, visible, CLAUSES).severe_error


def test_other_scope_is_unknown_not_disabled_and_whitelist_has_no_environment():
    result, visible = supported_result("C31")
    f = next(f for f in result["facts"] if f["type"] == "feature_enabled")
    assert f["value"]["status"] == "unknown"
    f["value"]["status"] = False
    assert grade_result(STANDARDS["C31"], result, visible, CLAUSES).severe_error
    result, visible = supported_result("C33")
    f = next(f for f in result["facts"] if f["type"] == "outbound_ip_allowed")
    f["value"]["environment"] = "production"
    assert grade_result(STANDARDS["C33"], result, visible, CLAUSES).protocol_error


def test_bank_five_day_boundary_and_generic_escalation():
    result, visible = supported_result("C06")
    assert grade_result(STANDARDS["C06"], result, visible, CLAUSES).complete_success
    f = next(f for f in result["facts"] if f["type"] == "limit_workdays")
    f["value"] = 3
    assert grade_result(STANDARDS["C06"], result, visible, CLAUSES).severe_error
    result, visible = supported_result("C39")
    for key in ["reasons","facts","next_steps","missing_info"]: result[key] = []
    assert grade_result(STANDARDS["C39"], result, visible, CLAUSES).reward == 0


def test_balance_cannot_borrow_either_channel_limit():
    for value in [3, 5]:
        result, visible = supported_result("C11")
        result["facts"].append(dict(target="R1",type="limit_workdays",value=value,evidence=["search:1:1"]))
        assert grade_result(STANDARDS["C11"], result, visible, CLAUSES).severe_error


def test_supplied_duplicate_channel_and_severe_handoff_context_are_optional_correct_content():
    for tid, fact_value in [("C22",dict(target="A102",type="payment_channel",value="online_payment")),
                            ("C40",dict(target="Q1",type="recovered",value=False))]:
        result, visible = supported_result(tid)
        result["facts"].append({**fact_value,"evidence":["user:1"]})
        if tid == "C40":
            result["missing_info"].append(dict(target="Q1",field="response_summary",purpose="handoff",evidence=["user:1"]))
        assert grade_result(STANDARDS[tid],result,visible,CLAUSES).complete_success


def test_aggregate_refund_reason_cannot_borrow_uncited_timing_rules():
    result, visible = supported_result("C13")
    refs = {clause.clause_id:f"search:1:{i+1}" for i,clause in enumerate(CLAUSES.values())}
    item = dict(target="case",code="within_refund_window",evidence=["user:1",refs["refund_multi"]])
    result["reasons"].append(item)
    assert grade_result(STANDARDS["C13"],result,visible,CLAUSES).reward == 0.3
    item["evidence"] += [refs[k] for k in ["refund_clock","refund_online_limit","refund_bank_limit"]]
    assert grade_result(STANDARDS["C13"],result,visible,CLAUSES).complete_success


def test_permission_reason_accepts_explicit_condition_rule_without_borrowing_other_citations():
    for tid, clause in [("C25","config_fix_allowed"),("C27","config_enablement_handoff")]:
        result, visible = supported_result(tid)
        ref = f"search:1:{len(CLAUSES)+1}"
        visible.searches[0].observation.snippets.append(KnowledgeSnippet(reference_id=ref,source_id="technical-policy",
            title="配置条件",section_path=[],content=CLAUSES["config_actors"].text+CLAUSES[clause].text))
        result["reasons"][0]["evidence"] = ["user:1",ref]
        assert grade_result(STANDARDS[tid],result,visible,CLAUSES).complete_success
        visible.searches[0].observation.snippets[-1].content = CLAUSES["config_actors"].text
        assert grade_result(STANDARDS[tid],result,visible,CLAUSES).reward == 0


def test_c01_conditional_refund_advice_is_optional_and_supported():
    result, visible = supported_result("C01")
    refs = {c.clause_id:f"search:1:{i+1}" for i,c in enumerate(CLAUSES.values())}
    result["next_steps"].append(dict(target="R1",code="verify_refund_if_overdue_unreceived",
        params={"limit_workdays":5},evidence=["user:1"]+[refs[c] for c in
            ("refund_clock","refund_bank_limit","refund_overdue")]))
    assert grade_result(STANDARDS["C01"],result,visible,CLAUSES).complete_success
    result["next_steps"][0]["params"]["limit_workdays"] = 3
    assert grade_result(STANDARDS["C01"],result,visible,CLAUSES).severe_error


def test_c01_citation_and_missing_condition_keep_supported_partial_judgment():
    _, visible = supported_result("C01")
    refs = {c.clause_id:f"search:1:{i+1}" for i,c in enumerate(CLAUSES.values())}
    result = dict(disposition="request_info", reasons=[dict(target="R1",code="refund_state_unknown",
        evidence=["user:1",refs["refund_accepted"],refs["refund_clock"]])], facts=[
        dict(target="R1",type="refund_stage",value="accepted",evidence=["user:1"]),
        dict(target="R1",type="approval_status",value="unknown",evidence=["user:1"]),
        dict(target="R1",type="timer_start",value="approval",evidence=[refs["refund_clock"]]),
        dict(target="R1",type="timing_policy_covered",value=True,evidence=[refs["refund_scope"]])],
        next_steps=[], missing_info=[
        dict(target="R1",field="approval_status",purpose="decision",evidence=["user:1"]),
        dict(target="R1",field="elapsed_workdays",purpose="decision",evidence=["user:1"])])
    grade = grade_result(STANDARDS["C01"],result,visible,CLAUSES)
    assert grade.reward == 0.3 and grade.errors and grade.missing and not grade.complete_success
    result["reasons"] = []
    result["facts"] = result["facts"][:2]
    assert grade_result(STANDARDS["C01"],result,visible,CLAUSES).reward == 0


@pytest.mark.parametrize("value", [True, False])
def test_undetermined_refund_comparison_is_severe_even_with_protocol_error(value):
    for tid in ("C01","C17","C19"):
        result, visible = supported_result(tid)
        target = "R1" if tid == "C01" else "R2"
        result["facts"].append(dict(target=target,type="within_window",value=value,evidence=["user:1"]))
        result["facts"].append(dict(target=target,type="feature_enabled",value=True,evidence=["user:1"]))
        grade = grade_result(STANDARDS[tid],result,visible,CLAUSES)
        assert grade.reward == -1 and grade.severe_error and grade.protocol_error


def test_c01_unreported_account_check_is_unknown_not_confirmed_or_denied():
    result, visible = supported_result("C01")
    item = dict(target="R1",type="original_account_checked",value="unknown",evidence=["user:1"])
    result["facts"].append(item)
    assert grade_result(STANDARDS["C01"],result,visible,CLAUSES).complete_success
    for value in (True,False):
        item["value"] = value
        assert grade_result(STANDARDS["C01"],result,visible,CLAUSES).severe_error


def test_calibration_build_and_review_preserve_explicit_human_conclusions(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    data.mkdir()
    sources = deepcopy(runpy.run_path(str(repo / "scripts/build_stage5_calibration.py"))["sources"])
    for source in sources:
        source["human_review"] = "待人工核查"
        source.pop("ai_review", None)
    sources[0].update(human_review="已确认审阅结论",human_conclusion={"teacher_qualified":False})
    sources[1]["ai_review"] = {"status":"已完成AI代审","teacher_qualified":True}
    (data / "sources.json").write_text(json.dumps(sources,ensure_ascii=False),encoding="utf-8")
    build = runpy.run_path(str(repo / "scripts/build_stage5_calibration.py"))["main"]
    build.__globals__["DEST"] = data
    build()
    assert json.loads((data / "sources.json").read_text(encoding="utf-8"))[0]["human_conclusion"] == {"teacher_qualified":False}
    run = tmp_path / "run"
    run.mkdir()
    (run / "grading-revisions.json").write_text("[]",encoding="utf-8")
    review = runpy.run_path(str(repo / "scripts/review_stage5.py"))["main"]
    review.__globals__.update(DATA=data,RUN=run)
    review()
    review()
    statuses = json.loads((run / "review/review_status.json").read_text(encoding="utf-8"))
    assert statuses[0]["human_review"] == "已确认审阅结论"
    assert statuses[0]["human_conclusion"] == {"teacher_qualified":False}
    assert all(s["human_review"] == "待人工核查" for s in statuses[1:])
    assert not statuses[0]["accepted"]
    assert statuses[1]["ai_review"]["status"] == "已完成AI代审"
    metrics = json.loads((run / "calibration_metrics.json").read_text(encoding="utf-8"))
    assert metrics["human_confirmed_task_ids"] == ["C01"]
    assert metrics["ai_reviewed_task_ids"] == ["C02"]


def test_untimed_record_can_have_a_supported_future_step_without_becoming_overdue():
    result, visible = supported_result("C17")
    refs = {c.clause_id:f"search:1:{i+1}" for i,c in enumerate(CLAUSES.values())}
    result["next_steps"].append(dict(target="R2",code="verify_refund_if_overdue_unreceived",
        params={"limit_workdays":5},evidence=["user:1"]+[refs[k] for k in
        ("refund_clock","refund_bank_limit","refund_overdue")]))
    assert grade_result(STANDARDS["C17"],result,visible,CLAUSES).complete_success
    result["next_steps"][-1]["params"]["limit_workdays"] = 3
    assert grade_result(STANDARDS["C17"],result,visible,CLAUSES).severe_error


def test_secondary_unknown_refund_does_not_need_a_duplicate_escalation_reason():
    result, visible = supported_result("C19")
    assert not any(r["target"] == "R2" for r in result["reasons"])
    assert grade_result(STANDARDS["C19"],result,visible,CLAUSES).complete_success
    result["facts"] = [f for f in result["facts"] if not (f["target"] == "R2" and f["type"] == "elapsed_workdays")]
    assert not grade_result(STANDARDS["C19"],result,visible,CLAUSES).complete_success


def test_nested_current_outbound_address_satisfies_address_information_only():
    result, visible = supported_result("C30")
    result["facts"] = [f for f in result["facts"] if f["type"] != "outbound_ip"]
    assert grade_result(STANDARDS["C30"],result,visible,CLAUSES).complete_success
    field = next(f for f in result["facts"] if f["type"] == "outbound_ip_allowed")
    field["value"]["outbound_ip"] = "203.0.113.99"
    assert grade_result(STANDARDS["C30"],result,visible,CLAUSES).severe_error


@pytest.mark.parametrize("suffix", ['"','</s>'])
def test_extra_text_never_repairs_a_terminal_but_does_not_hide_severe_claims(suffix):
    result, visible = supported_result("C05")
    raw = json.dumps(result,ensure_ascii=False)+suffix
    grade = grade_result(STANDARDS["C05"],raw,visible,CLAUSES)
    assert grade.reward == 0 and grade.protocol_error and not grade.complete_success
    next(f for f in result["facts"] if f["type"] == "within_window")["value"] = False
    grade = grade_result(STANDARDS["C05"],json.dumps(result)+suffix,visible,CLAUSES)
    assert grade.reward == -1 and grade.protocol_error and grade.severe_error


def test_reason_comparison_is_a_decisive_judgment_and_wrong_claim_is_severe():
    result, visible = supported_result("C14")
    result["facts"] = [f for f in result["facts"] if f["type"] != "within_window"]
    assert grade_result(STANDARDS["C14"],result,visible,CLAUSES).reward == 0.3
    result["reasons"][0]["code"] = "refund_overdue"
    assert grade_result(STANDARDS["C14"],result,visible,CLAUSES).severe_error


def test_known_numeric_workdays_cannot_be_changed_to_unknown():
    result, visible = supported_result("C19")
    next(f for f in result["facts"] if f["target"] == "R1" and f["type"] == "elapsed_workdays")["value"] = "unknown"
    assert grade_result(STANDARDS["C19"],result,visible,CLAUSES).severe_error


def test_user_visible_interface_identifier_is_valid_optional_context():
    result, visible = supported_result("C36")
    result["facts"].append(dict(target="Q1",type="interface_name",value="Q1",evidence=["user:1"]))
    assert grade_result(STANDARDS["C36"],result,visible,CLAUSES).complete_success


def test_timeout_does_not_supply_an_http_code_and_multiple_is_not_widespread():
    result, visible = supported_result("C37")
    result["facts"].append(dict(target="Q1",type="http_status",value=500,evidence=["user:1"]))
    assert grade_result(STANDARDS["C37"],result,visible,CLAUSES).severe_error
    result, visible = supported_result("C40")
    result["facts"].append(dict(target="Q1",type="widespread_outage",value=True,evidence=["user:1"]))
    assert grade_result(STANDARDS["C40"],result,visible,CLAUSES).severe_error


def test_diagnostic_context_can_extend_the_required_fields_without_unrelated_billing_fields():
    result, visible = supported_result("C40")
    ref = f"search:1:{list(CLAUSES).index('fault_handling')+1}"
    result["next_steps"].append(dict(target="Q1",code="record_diagnostics",
        params={"fields":["request_id","interface_name","event_time","occurrence","response_summary",
                          "environment","http_status","recovered"]},evidence=[ref]))
    assert grade_result(STANDARDS["C40"],result,visible,CLAUSES).complete_success
    result["next_steps"][-1]["params"]["fields"].append("payment_channel")
    assert not grade_result(STANDARDS["C40"],result,visible,CLAUSES).complete_success


@pytest.mark.parametrize("kind,value", [
    ("environment", "production"), ("request_read_only", False),
    ("request_scope", {"project": "Q1", "environment": "production"}),
    ("feature_enabled", {"project": "Q1", "environment": "production", "status": "unknown"}),
])
def test_401_does_not_confirm_unsupplied_context_even_with_a_correct_answer(kind, value):
    result, visible = supported_result("C35")
    result["facts"].append(dict(target="Q1", type=kind, value=value, evidence=[]))
    grade = grade_result(STANDARDS["C35"], result, visible, CLAUSES)
    assert grade.reward == -1 and grade.severe_error
    assert not grade.complete_success


def test_unknown_context_is_not_confirmed_or_added_as_a_required_output():
    result, visible = supported_result("C35")
    assert grade_result(STANDARDS["C35"], result, visible, CLAUSES).complete_success
    result["facts"].append(dict(target="Q1", type="request_read_only", value="unknown", evidence=["user:1"]))
    grade = grade_result(STANDARDS["C35"], result, visible, CLAUSES)
    assert not grade.severe_error and not grade.complete_success


def test_refund_record_does_not_confirm_payment_impact_or_charge_count():
    for kind, value in [("payment_affected", False), ("reported_charge_count", 1)]:
        result, visible = supported_result("C09")
        result["facts"].append(dict(target="R1", type=kind, value=value, evidence=["user:1"]))
        assert grade_result(STANDARDS["C09"], result, visible, CLAUSES).severe_error


@pytest.mark.parametrize("task_id", ["C25", "C27", "C33"])
def test_supported_current_enablement_scope_does_not_need_a_duplicate_request_scope(task_id):
    result, visible = supported_result(task_id)
    result["facts"] = [f for f in result["facts"] if f["type"] != "request_scope"]
    assert grade_result(STANDARDS[task_id], result, visible, CLAUSES).complete_success
    scoped = next(f for f in result["facts"] if f["type"] == "feature_enabled")
    scoped["evidence"] = []
    grade = grade_result(STANDARDS[task_id], result, visible, CLAUSES)
    assert not grade.complete_success and any("request_scope" in m for m in grade.missing)
    scoped["value"]["environment"] = "test"
    assert grade_result(STANDARDS[task_id], result, visible, CLAUSES).severe_error


def test_refund_approval_and_clock_have_supported_equivalent_expressions():
    result, visible = supported_result("C10")
    result["facts"] = [f for f in result["facts"] if f["type"] != "approval_status"]
    assert grade_result(STANDARDS["C10"], result, visible, CLAUSES).complete_success
    next(f for f in result["facts"] if f["type"] == "elapsed_workdays")["evidence"] = []
    assert not grade_result(STANDARDS["C10"], result, visible, CLAUSES).complete_success
    result, visible = supported_result("C03")
    result["facts"] = [f for f in result["facts"] if f["type"] != "timer_start"]
    assert grade_result(STANDARDS["C03"], result, visible, CLAUSES).complete_success
    next(f for f in result["facts"] if f["type"] == "within_window")["evidence"] = ["user:1"]
    assert not grade_result(STANDARDS["C03"], result, visible, CLAUSES).complete_success


def test_equivalence_does_not_repair_an_explicit_wrong_or_unsupported_claim():
    result, visible = supported_result("C27")
    next(f for f in result["facts"] if f["type"] == "request_scope")["evidence"] = []
    assert not grade_result(STANDARDS["C27"], result, visible, CLAUSES).complete_success
    next(f for f in result["facts"] if f["type"] == "request_scope")["value"]["project"] = "Birch"
    assert grade_result(STANDARDS["C27"], result, visible, CLAUSES).severe_error


@pytest.mark.parametrize("task_id", [f"C{i:02d}" for i in range(23, 35)])
@pytest.mark.parametrize("kind,value", [("widespread_outage", False), ("affected_users", "one")])
def test_403_reporter_and_error_code_do_not_confirm_impact(task_id, kind, value):
    result, visible = supported_result(task_id)
    assert grade_result(STANDARDS[task_id], result, visible, CLAUSES).complete_success
    result["facts"].append(dict(target="Q1", type=kind, value=value, evidence=["user:1"]))
    grade = grade_result(STANDARDS[task_id], result, visible, CLAUSES)
    assert grade.reward == -1 and grade.severe_error and not grade.protocol_error


@pytest.mark.parametrize("kind,value", [
    ("payment_affected", False), ("write_affected", False), ("data_loss", False),
    ("data_corruption", False), ("occurrence", "single"), ("recovered", False),
    ("reported_charge_count", 0), ("charges_same_order", False), ("request_read_only", False),
])
def test_403_configuration_does_not_confirm_other_unsupplied_states(kind, value):
    result, visible = supported_result("C30")
    result["facts"].append(dict(target="Q1", type=kind, value=value, evidence=["user:1"]))
    grade = grade_result(STANDARDS["C30"], result, visible, CLAUSES)
    assert grade.reward == -1 and grade.severe_error


@pytest.mark.parametrize("target", ["Q1", "user:1"])
@pytest.mark.parametrize("corruption", [None, "tail", "field"])
def test_403_severe_claim_precedes_tail_or_illegal_field_without_repair(target, corruption):
    result, visible = supported_result("C30")
    result["facts"].extend([
        dict(target=target, type="widespread_outage", value=False, evidence=["user:1"]),
        dict(target=target, type="affected_users", value="one", evidence=["user:1"]),
    ])
    if corruption == "field":
        result["facts"].append(dict(target="Q1", type="project", value="Cedar", evidence=["user:1"]))
    raw = json.dumps(result) + ('</s>' if corruption == "tail" else '')
    grade = grade_result(STANDARDS["C30"], raw, visible, CLAUSES)
    assert grade.reward == -1 and grade.severe_error and not grade.complete_success
    assert grade.protocol_error == (corruption is not None)
    assert any(":widespread_outage:" in error for error in grade.errors)
    assert any(":affected_users:" in error for error in grade.errors)
    assert raw == json.dumps(result) + ('</s>' if corruption == "tail" else '')


def test_403_unrelated_unknown_is_not_an_allowed_extra_fact():
    result, visible = supported_result("C30")
    result["facts"].append(dict(target="Q1", type="widespread_outage", value="unknown", evidence=["user:1"]))
    grade = grade_result(STANDARDS["C30"], result, visible, CLAUSES)
    assert not grade.severe_error and not grade.complete_success and grade.errors


@pytest.mark.parametrize("kind,value", [("auth_valid", True), ("token_expired", False)])
def test_recovered_timeout_does_not_confirm_authentication_by_misbinding_a_reference(kind, value):
    result, visible = supported_result("C37")
    result["facts"].append(dict(target="user:1", type=kind, value=value, evidence=["user:1"]))
    grade = grade_result(STANDARDS["C37"], result, visible, CLAUSES)
    assert grade.reward == -1 and grade.severe_error


def test_403_unprovided_annotation_preserves_supplied_and_inferable_context():
    for tid in ("C24", "C30"):
        result, visible = supported_result(tid)
        fields = STANDARDS[tid].unprovided_user_facts["Q1"]
        assert not set(fields).intersection({"environment", "request_scope", "auth_valid",
            "feature_enabled", "outbound_ip_allowed", "token_expired", "timeout_observed"})
        result["facts"].extend([
            dict(target="Q1", type="token_expired", value=False, evidence=["user:1"]),
            dict(target="Q1", type="timeout_observed", value=False, evidence=["user:1"]),
        ])
        if tid == "C24":
            assert "request_read_only" not in fields
            result["facts"].append(dict(target="Q1", type="request_read_only", value=True, evidence=["user:1"]))
        assert not grade_result(STANDARDS[tid], result, visible, CLAUSES).severe_error
