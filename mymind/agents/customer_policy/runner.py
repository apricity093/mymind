"""由策略决定查询和终止，最后一次观察后仍有生成机会。"""
from copy import deepcopy
import json

from agents.customer_policy.protocol import parse_result
from agents.customer_policy.rendering import render_result
from agents.customer_policy.protocol import VisibleTrajectory


def restore_progress(task, generation, environment, budget, state):
    """只接续已保存全部工具观察的消息边界；不重放工具调用。"""
    initial = environment.reset(task)
    trajectory = VisibleTrajectory.model_validate(state["trajectory"])
    generations = deepcopy(state["generations"])
    diagnostics = deepcopy(state["tool_diagnostics"])
    if (state["task_id"] != task.task_id or trajectory.task_id != task.task_id or
            trajectory.materials != environment.trajectory().materials or state["budget"] != budget or
            state["initial_tokens"] != generation.token_count(initial) or
            state.get("runtime_error") or not generations or
            not 0 < len(trajectory.searches) <= environment.max_searches or
            len(diagnostics) != len(trajectory.searches)):
        raise ValueError("partial任务、预算或观察状态不支持消息边界恢复")
    expected, index = deepcopy(initial), 0
    for generated in generations:
        assistant = generated["assistant_message"]
        calls = assistant.get("tool_calls") or []
        if generated["input_messages"] != expected or not calls:
            raise ValueError("partial不是完整工具观察后的生成边界")
        expected.append(assistant)
        for call in calls:
            if index >= len(trajectory.searches):
                raise ValueError("partial工具观察尚未全部落盘")
            turn = trajectory.searches[index]
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if call["function"]["name"] != "search" or arguments != turn.action.model_dump():
                raise ValueError("partial搜索与观察不对应")
            if turn.observation.request_number != index + 1:
                raise ValueError("partial搜索序号不连续")
            if turn.observation.status.degraded or turn.observation.status.state == "unavailable":
                raise ValueError("partial含未处理工具故障")
            expected.append({"role": "tool", "tool_call_id": call["id"], "name": "search",
                             "content": turn.observation.tool_content()})
            index += 1
    if index != len(trajectory.searches) or expected != state["messages"]:
        raise ValueError("partial消息与已执行搜索不一致")
    environment.searches = trajectory.searches
    environment.diagnostics = diagnostics
    return deepcopy(state["messages"]), generations, state["initial_tokens"]


async def run_task(task, generation, environment, budget=None, on_progress=None, resume_state=None, public_guidance=None):
    messages = environment.reset(task)
    if public_guidance:
        messages[0]['content'] += '\n' + public_guidance
    generations, result, error = [], None, None
    teacher_units = bool(budget and budget.get('length_unit') in {'utf8_token_upper_bound', 'retained-text-v1'})
    count_input = generation.input_token_upper_bound if teacher_units else getattr(generation, 'token_count', None)
    initial_tokens = count_input(messages) if budget else None
    if resume_state is not None:
        messages, generations, initial_tokens = restore_progress(task, generation, environment, budget, resume_state)

    def checkpoint():
        if on_progress:
            on_progress({"task_id": task.task_id, "trajectory": environment.trajectory().model_dump(),
                "messages": deepcopy(messages), "generations": deepcopy(generations),
                "tool_diagnostics": deepcopy(environment.diagnostics),
                "raw_result": generations[-1]["raw_text"] if generations else "",
                "rendered": None, "budget": budget, "initial_tokens": initial_tokens})

    checkpoint()

    def remaining():
        tokens = count_input(messages)
        output_used = sum(g['output_token_budget_used'] for g in generations) if teacher_units else tokens - initial_tokens
        return min(budget["completion_tokens"] - output_used,
                   budget["context_tokens"] - tokens)

    for _ in range(len(generations), environment.max_searches + 1):
        enabled = len(environment.searches) < environment.max_searches
        options = {}
        if budget:
            available = remaining()
            enabled = enabled and available >= 2 * budget["turn_tokens"]
            if available <= 0:
                error = {"kind": "budget_exhausted", "detail": "公开上下文已耗尽生成预算"}
                break
            options["max_new_tokens"] = min(budget["turn_tokens"], available)
        if not enabled:
            messages.append({"role": "system", "content": "本轮不再执行检索，请依据现有观察提交终止 JSON。"})
            if budget:
                available = remaining()
                if available <= 0:
                    error = {"kind": "budget_exhausted", "detail": "终止提示后无剩余生成预算"}
                    break
                options["max_new_tokens"] = min(budget["turn_tokens"], available)
        try:
            generated = await generation.generate(messages, tools_enabled=enabled, **options)
        except Exception as ex:
            error = {"kind": "generation_failure", "type": type(ex).__name__, "detail": str(ex)}
            if getattr(ex, "status_code", None) is not None:
                error.update(status_code=ex.status_code, response_body=getattr(ex, "body", None))
            break
        generations.append({"input_messages": deepcopy(messages), "raw_text": generated.text,
                            "assistant_message": deepcopy(generated.assistant), **generated.diagnostics})
        if teacher_units:
            usage = (generated.diagnostics.get('raw_response') or {}).get('usage') or {}
            used = usage.get('completion_tokens')
            trusted = type(used) is int and 0 <= used <= options['max_new_tokens']
            generations[-1].update(output_token_budget_used=used if trusted else options['max_new_tokens'],
                output_token_budget_basis='provider_completion_tokens' if trusted else 'requested_output_cap_usage_missing')
        messages.append(generated.assistant)
        checkpoint()
        choices = (generated.diagnostics.get('raw_response') or {}).get('choices') or []
        if choices and choices[0].get('finish_reason') == 'length':
            error = {'kind':'generation_truncated', 'detail':'供应商达到生成或上下文长度上限'}
            break
        if not generated.calls:
            try:
                result = parse_result(generated.text)
            except ValueError as ex:
                error = {"kind": "protocol_error", "detail": str(ex)}
            break
        if not enabled or len(environment.searches) + len(generated.calls) > environment.max_searches:
            error = {"kind": "protocol_error", "detail": "模型请求超过剩余检索预算"}
            break
        for call in generated.calls:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"invalid_json": arguments}
            try:
                observation = await environment.step(function["name"], arguments)
            except Exception as ex:
                error = {"kind": "tool_failure", "type": type(ex).__name__, "detail": str(ex)}
                break
            message = {"role": "tool", "tool_call_id": call["id"], "name": function["name"],
                       "content": observation.tool_content()}
            messages.append(message)
            if budget:
                original = observation.model_dump()
                # 省略完整片段以给终止结果预留一轮，不把未展示正文当作证据。
                while observation.snippets and remaining() < budget["turn_tokens"] + 64:
                    observation.snippets.pop()
                    observation.status.state = "candidates" if observation.snippets else "empty"
                    message["content"] = observation.tool_content()
                if original != observation.model_dump():
                    environment.diagnostics[-1].update(before_token_budget=original,
                        visible_observation=observation.model_dump(), observation_budget_applied=True)
            checkpoint()
        if error:
            break
    return {"task_id": task.task_id, "trajectory": environment.trajectory().model_dump(),
            "messages": messages, "generations": generations, "tool_diagnostics": environment.diagnostics,
            "raw_result": result.model_dump() if result else (generations[-1]["raw_text"] if generations else ""),
            "rendered": render_result(result) if result else None, "runtime_error": error,
            "budget": budget, "initial_tokens": initial_tokens,
            "infrastructure_failure": bool(error and error["kind"] in {"generation_failure", "tool_failure"}) or any(
                s.observation.status.state == "unavailable" or s.observation.status.degraded for s in environment.searches)}
