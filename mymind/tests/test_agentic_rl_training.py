import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.customer_policy.protocol import PublicTask, SearchObservation
from experiments.agentic_rl.grpo import BatchPaused, ToolEnvironment, complete_group
from experiments.agentic_rl.sft_data import student_messages, encode_trajectory


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
