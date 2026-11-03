"""协议、证据判分、SFT 监督与 GRPO 归一化。"""

import asyncio
import json
import pytest
import subprocess
import sys
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import parse_qwen
from agents.customer_policy.protocol import (
    FACT_TYPES,
    Fact,
    MISSING_FIELDS,
    PublicTask,
    SearchObservation,
    VisibleTrajectory,
    parse_result,
    public_value_catalog,
)
from agents.customer_policy.rendering import render_result
from agents.customer_policy.retrieval import PolicyRetrieval
from agents.customer_policy.runner import run_task
from copy import deepcopy
from core.llm_gateway import CacheUsage, DeepSeekGateway, LLMRequest, LLMResult
from mcp.tool_manager import MCPToolManager, Tool, ToolResult
from pathlib import Path
from pydantic import ValidationError
from rl.grading import grade_result
from rl.grpo import BatchPaused, ToolEnvironment, complete_group
from rl.grpo_diagnostics import group_statistics, normalization_evidence
from rl.sft_data import encode_trajectory, student_messages
from rl.tasks import RESOURCES, load_private_standards, load_public_tasks, validate_task_pairs
from tests.agentic_rl_samples import (
    BILLING,
    CLAUSES,
    STANDARDS,
    TASKS,
    accepted_result,
    s04_result,
    trajectory,
)
from types import SimpleNamespace


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


def score(task_id, result, visible=None):
    return grade_result(STANDARDS[task_id], result, visible or trajectory(task_id), CLAUSES)


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


def test_student_conversion_excludes_supplier_reasoning_and_parses_calls():
    row = {"messages": [{"role": "assistant", "content": None, "reasoning_content": "供应商思考",
        "tool_calls": [{"type": "function", "id": "a", "function": {"name": "search", "arguments": '{"query":"政策"}'}}]},
        {"role": "tool", "name": "search", "tool_call_id": "a", "content": '{"正文":"实际观察"}'}]}
    converted = student_messages(row)
    assert "reasoning_content" not in converted[0]
    assert converted[0]["tool_calls"][0]["function"]["arguments"] == {"query": "政策"}
    assert converted[1]["content"] == row["messages"][1]["content"]
    assert isinstance(row["messages"][0]["tool_calls"][0]["function"]["arguments"], str)


def test_supervision_contains_assistant_actions_and_final_only():
    class Tokenizer:
        def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, return_dict):
            value = "".join(m["role"] + ":" + m["content"] + "|" for m in messages)
            if add_generation_prompt:
                value += "assistant:"
            return [ord(c) for c in value]
        def decode(self, ids):
            return "".join(map(chr, ids))
    encoded = encode_trajectory(Tokenizer(), [
        {"role": "system", "content": "规范"}, {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "查询", "tool_calls": [{}]},
        {"role": "tool", "content": "真实观察"}, {"role": "assistant", "content": "结论"}])
    supervised = "".join(chr(i) for i in encoded["labels"] if i != -100)
    assert supervised == "查询|结论|"
    assert [s["kind"] for s in encoded["spans"]] == ["tool_action", "final"]


def test_fault_group_is_excluded_before_bounded_same_policy_resampling():
    seen, accepted = [], []
    def sample(attempt):
        # 正常低分同样属于组；只因基础设施故障排除首次整组。
        return {"policy_step": 7, "rewards": [-1, 0, 0.3, 1]}, attempt == 0
    result = complete_group(sample, lambda attempt, result, failed: seen.append((attempt, result, failed)))
    accepted.append(result)
    assert [x[2] for x in seen] == [True, False]
    assert [x[1]["policy_step"] for x in seen] == [7, 7]
    assert accepted[0]["rewards"] == [-1, 0, 0.3, 1]


def test_second_infrastructure_fault_pauses_without_optimization():
    attempts = []
    with pytest.raises(BatchPaused):
        complete_group(lambda attempt: ("fault group", True),
                       lambda attempt, result, failed: attempts.append(attempt))
    assert attempts == [0, 1]


def test_tool_environment_reset_and_parallel_request_budget():
    class Retrieval:
        async def search(self, query, number):
            await asyncio.sleep(0)
            return SearchObservation(request_number=number,
                status={"reference_id": f"search:{number}:status", "state": "empty"}, snippets=[]), {}
    task = PublicTask(task_id="S", role="billing", user_text="公开问题")
    env = ToolEnvironment(Retrieval(), {"S": task})
    async def run():
        env.reset("S")
        await asyncio.gather(*(env.search("查询") for _ in range(3)))
        assert [s.observation.request_number for s in env._environment.searches] == [1, 2, 3]
        await env.search("第四次")
        assert env._protocol_error and len(env._environment.searches) == 3
        env.reset("S")
        assert not env._environment.searches and env._protocol_error is None
    asyncio.run(run())


def test_complete_cross_rank_group_uses_sample_standard_deviation():
    rewards = [0, -1, -1, -1]
    group = group_statistics(rewards, "group")
    assert group["reward_std"] == .5
    assert group["expected_advantages"] == pytest.approx([.75/.5001] + [-.25/.5001]*3)
    assert group_statistics(rewards, "none")["expected_advantages"] == [.75, -.25, -.25, -.25]
    for reward in [-1, 0, .3, 1]:
        assert group_statistics([reward]*4, "group")["expected_advantages"] == [0]*4


def test_grpo_and_dapo_denominators_exclude_tool_observations_and_padding():
    grpo = normalization_evidence("grpo", [2, 5], 20, 2, 1, 2)
    assert grpo["trajectory_denominators"] == [2, 5]
    assert grpo["sample_denominator"] == 2
    assert "dapo_normalizer" not in grpo
    dapo = normalization_evidence("dapo", [2, 5], 20, 2, 1, 2)
    assert dapo["dapo_normalizer"] == 5


TASK = PublicTask(task_id="dev", role="billing", user_text="银行卡退款咨询")


FINAL = json.dumps(dict(disposition="request_info", reasons=[], facts=[], next_steps=[], missing_info=[]))


class Retriever:
    async def search(self, query, number):
        return SearchObservation(request_number=number,
            status={"reference_id": f"search:{number}:status", "state": "candidates"},
            snippets=[dict(reference_id=f"search:{number}:1", content="实际返回的银行卡政策正文",
                           source_id="billing-policy", title="退款", section_path=[])]), {"query": query}


def test_last_observation_has_final_generation_and_strings_reach_context():
    class Generator:
        def __init__(self):
            self.inputs = []

        async def generate(self, messages, tools_enabled=True):
            self.inputs.append((deepcopy(messages), tools_enabled))
            if len(self.inputs) <= 3:
                return parse_qwen('<tool_call>{"name":"search","arguments":{"query":"退款"}}</tool_call>')
            return parse_qwen(FINAL)

    generator = Generator()
    row = asyncio.run(run_task(TASK, generator, PolicyEnvironment(Retriever())))
    assert len(row["trajectory"]["searches"]) == 3 and len(generator.inputs) == 4
    assert generator.inputs[-1][1] is False
    assert len([m for m in generator.inputs[-1][0] if m["role"] == "tool"]) == 3
    assert all(isinstance(m["content"], str) and "实际返回" in m["content"]
               for m in generator.inputs[-1][0] if m["role"] == "tool")
    assert row["runtime_error"] is None
    assert row["messages"][0] == TASK.messages()[0]
    assert row["messages"][-2]["role"] == "system"
    assert "不再执行检索" in generator.inputs[-1][0][-1]["content"]


def test_invalid_arguments_consume_budget_without_fabricating_query():
    env = PolicyEnvironment(Retriever())
    env.reset(TASK)
    async def run():
        for _ in range(3):
            result = await env.step("search", {})
            assert result.status.state == "invalid_arguments" and not result.snippets
        try:
            await env.step("search", {"query": "第四次"})
        except ValueError:
            return
        assert False
    asyncio.run(run())
    assert env.trajectory().searches[0].action == {}


def test_deepseek_mode_is_explicit_and_actual_response_model_is_recorded():
    requests = []
    async def create(**kwargs):
        requests.append(kwargs)
        raw = {"model": "actual-service-model", "choices": [{"finish_reason": "stop"}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
        return SimpleNamespace(model_dump=lambda: raw,
            choices=[SimpleNamespace(message=SimpleNamespace(content="[0]", tool_calls=[]))])
    gateway = DeepSeekGateway("test", "deepseek-flash", thinking="disabled")
    gateway.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = asyncio.run(gateway.complete(LLMRequest(model="deepseek-flash", stable_prompt="排序")))
    assert requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result.metadata["response_model"] == "actual-service-model"


def test_policy_can_stop_without_forced_search():
    class Generator:
        async def generate(self, messages, tools_enabled=True):
            return parse_qwen(FINAL)
    row = asyncio.run(run_task(TASK, Generator(), PolicyEnvironment(Retriever())))
    assert row["trajectory"]["searches"] == [] and row["runtime_error"] is None


def test_parallel_search_calls_count_requests_and_reject_over_budget():
    class Generator:
        async def generate(self, messages, tools_enabled=True):
            return parse_qwen('\n'.join(['<tool_call>{"name":"search","arguments":{"query":"退款"}}</tool_call>']*4))
    row = asyncio.run(run_task(TASK, Generator(), PolicyEnvironment(Retriever())))
    assert row["runtime_error"]["kind"] == "protocol_error"
    assert row["trajectory"]["searches"] == []


def test_fallback_content_is_excluded_and_empty_keeps_this_call_route():
    class Manager:
        async def call(self, *args, **kwargs):
            args[2]["retrieval_diagnostics"]["index_route"] = "main"
            return ToolResult(True, [{"source_id": "fake", "content": "故障提示"}], "x", error="timeout", degraded=True)
    retrieval = PolicyRetrieval(None, Manager())
    observation, log = asyncio.run(retrieval.search("退款", 1))
    assert observation.status.state == "unavailable" and observation.snippets == []
    assert log["index_route"] == "main"
    class EmptyManager:
        async def call(self, *args, **kwargs):
            args[2]["retrieval_diagnostics"].update(index_route="backup", degraded=True)
            return ToolResult(True, [], "x")
    observation, log = asyncio.run(PolicyRetrieval(None, EmptyManager()).search("退款", 1))
    assert observation.status.state == "empty" and observation.status.degraded
    assert log["index_route"] == "backup"


def test_observation_omits_whole_over_budget_chunks_and_internal_scores():
    class Manager:
        async def call(self, *args, **kwargs):
            return ToolResult(True, [{"source_id": "policy", "content": "规则"*1000, "score": 0.9},
                                     {"source_id": "policy", "content": "短正文", "score": 0.2}], "x")
    observation, log = asyncio.run(PolicyRetrieval(None, Manager(), observation_chars=300).search("规则", 1))
    assert [s.content for s in observation.snippets] == ["短正文"]
    assert "score" not in observation.tool_content() and len(log["data"]) == 2


def test_rerank_reports_success_skip_fallback_and_usage_per_call():
    class Gateway:
        async def complete(self, request):
            return LLMResult("[2, 1, 0]", CacheUsage("deepseek", input_tokens=42))
    async def run():
        manager = MCPToolManager("unused", gateway=Gateway())
        items = [{"content": str(i), "chunk_id": str(i)} for i in range(3)]
        async def handler(params, context):
            return items[:params["count"]]
        manager.register(Tool("x", "x", handler, {"required": [], "properties": {}}, supports_rerank=True))
        result = await manager.call("x", {"count": 3}, rerank_top_k=2)
        assert result.reranked and result.diagnostics["rerank_status"] == "llm_success"
        assert result.diagnostics["rerank_usage"]["input_tokens"] == 42
        result = await manager.call("x", {"count": 1}, rerank_top_k=2)
        assert not result.reranked and result.diagnostics["rerank_status"].startswith("skipped")
        class FailedGateway:
            async def complete(self, request):
                raise RuntimeError("service failure")
        manager._gateway = FailedGateway()
        result = await manager.call("x", {"count": 3}, rerank_top_k=2)
        assert result.success and not result.reranked
        assert result.diagnostics["rerank_status"] == "failed_fallback"
        assert "rerank_usage" not in result.diagnostics
    asyncio.run(run())
