"""受控文本生成验证预算、请求计数和外部观察掩码。"""
import asyncio
import ast
from copy import deepcopy
import json
from types import SimpleNamespace
import inspect
import time

from agents.customer_policy.protocol import PublicTask, SearchObservation
from experiments.agentic_rl.grpo import (FINAL_INSTRUCTION, ToolEnvironment,
    policy_tool_call_loop, visible_trajectory)
import experiments.agentic_rl.grpo as grpo


def action(count=1):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "search", "arguments": {"query": f"政策{i}"}}}
        for i in range(count)]}


FINAL = {"role": "assistant", "content": '{"disposition":"answer","reasons":[],"facts":[],"next_steps":[],"missing_info":[]}'}


def controlled_loop(lengths=(100, 100, 100), *, calls=1, ignore_terminal=False, prompt_length=3601, observation_chars=700):
    class Retrieval:
        async def search(self, query, number):
            return SearchObservation(request_number=number, status={"reference_id": f"search:{number}:status", "state": "candidates"},
                snippets=[dict(reference_id=f"search:{number}:1", content="正文" * observation_chars,
                    source_id="policy", title="政策", section_path=[])]), {}
    env = ToolEnvironment(Retrieval(), {"S": PublicTask(task_id="S", role="billing", user_text="公开材料")})
    env.reset("S")
    contexts, decoded, caps = [], {}, []
    initial = action(calls)
    trainer = SimpleNamespace(environments=[env], turn_tokens=2048, context_tokens=16384,
        max_completion_length=8192, _tokenizer=None, _sync_tool_dicts=[{"search": lambda **kw: asyncio.run(env.search(**kw))}],
        _async_tool_dicts=[{}], _policy_turn_cap=None)
    def suffix(messages):
        return list(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    trainer._get_tool_suffix_ids = suffix
    def generate(ids, *args):
        contexts.append(deepcopy(ids[0]))
        caps.append(trainer._policy_turn_cap)
        terminal = FINAL_INSTRUCTION["content"].encode("utf-8") in bytes(v for v in ids[0][prompt_length:] if v < 256)
        message = action() if ignore_terminal or (env._search_requests < 3 and not terminal) else deepcopy(FINAL)
        n = lengths[min(len(contexts), len(lengths)-1)] if message.get("tool_calls") else 100
        assert n <= trainer._policy_turn_cap
        marker = 100000 + len(contexts)
        decoded[marker] = message
        return [[marker] * n], None
    trainer._generate_single_turn = generate
    result = policy_tool_call_loop(trainer, [[{"role": "user", "content": "公开"}]], [[0] * prompt_length],
        [[99999] * lengths[0]], [[initial]], None, None, {},
        lambda tokenizer, ids, prefix: deepcopy(decoded[ids[0]]))
    masks, completion, ids, *_ = result
    assert len(masks[0]) == len(ids[0]) <= 8192
    assert len(ids[0]) + prompt_length <= 16384
    # 每次生成实际接收的 token 前缀对应保留的完整 completion 前缀。
    for context in contexts:
        assert context == [0] * prompt_length + ids[0][:len(context)-prompt_length]
    shown = visible_trajectory(env._environment, completion[0])
    tools = [m for m in completion[0] if m["role"] == "tool"]
    assert [t.observation.model_dump() for t in shown.searches] == [json.loads(m["content"]) for m in tools]
    return result, env, contexts, caps


def test_regular_three_searches_have_final_and_external_tokens_masked():
    result, env, contexts, caps = controlled_loop(observation_chars=50)
    assert env._search_requests == 3 and result[1][0][-1] == FINAL
    assert sum(result[0][0]) == 400  # 三次动作与终止，观察和系统提示不计分母
    assert not env._protocol_error and not env._infrastructure_error


def test_three_parallel_requests_count_as_three_searches_not_three_assistant_turns():
    result, env, contexts, caps = controlled_loop(calls=3, observation_chars=50)
    assert env._search_requests == 3 and len(contexts) == 1
    assert result[1][0][-1] == FINAL and sum(result[0][0]) == 200


def test_near_budget_shortens_observation_and_still_generates_final():
    result, env, contexts, caps = controlled_loop(lengths=(2000, 2000, 1700))
    assert result[1][0][-1] == FINAL
    assert any(d.get("observation_budget_applied") for d in env._environment.diagnostics)
    assert not env._infrastructure_error and not env._protocol_error
    assert caps[-1] == 2048


def test_tool_request_without_budget_is_protocol_failure_not_infrastructure():
    result, env, contexts, caps = controlled_loop(lengths=(2000, 2000, 1700), ignore_terminal=True)
    assert env._protocol_error and not env._infrastructure_error
    assert result[1][0][-1].get("tool_calls") and result[1][0][-1] != FINAL
    assert env._search_requests < 3


def test_four_requests_in_one_turn_are_rejected_without_execution():
    result, env, contexts, caps = controlled_loop(calls=4)
    assert env._search_requests == 0 and not contexts
    assert env._protocol_error and not env._infrastructure_error


def test_context_budget_limits_feedback_and_preserves_final_generation():
    result, env, contexts, caps = controlled_loop(prompt_length=12000, observation_chars=100)
    assert result[1][0][-1] == FINAL
    assert all(len(c) + cap <= 16384 for c, cap in zip(contexts, caps))


def test_actual_generation_override_caps_and_restores_both_trl_configurations():
    tree = ast.parse(inspect.getsource(grpo.make_trainer_class))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PolicyGRPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_single_turn")
    generated = []
    class Base:
        def _generate_single_turn(self, *args):
            cap = self.generation_config.max_new_tokens
            assert cap == self.generation_kwargs["max_new_tokens"]
            generated.append(cap)
            return [[9] * cap], None
    test_cls = ast.ClassDef(name="Controlled", bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[])
    namespace = dict(Base=Base, time=time, torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test_cls], type_ignores=[])), "actual_generation_override", "exec"), namespace)
    trainer = namespace["Controlled"]()
    trainer.turn_tokens, trainer.context_tokens, trainer.max_completion_length = 2048, 16384, 8192
    trainer.generation_config = SimpleNamespace(max_new_tokens=8192)
    trainer.generation_kwargs = {"max_new_tokens":8192}
    trainer.state, trainer.turn_records = SimpleNamespace(global_step=0), []
    for input_length, extra_cap, expected in [(3601,None,2048),(15000,None,1384),(15000,777,777)]:
        trainer._policy_turn_cap = extra_cap
        trainer._generate_single_turn([[0]*input_length],None,{})
        assert generated[-1] == expected
        assert trainer.generation_config.max_new_tokens == trainer.generation_kwargs["max_new_tokens"] == 8192
