"""TRL 原生工具循环下的在线 GRPO 与异常整组补采。"""
import asyncio
from copy import deepcopy
import json
import time

from agents.customer_policy.configuration import rerank_gateway
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import TOOLS
from agents.customer_policy.protocol import SearchAction, SearchObservation, SearchTurn
from agents.customer_policy.retrieval import build_retrieval
from rl.grading import grade_result
from rl.runtime import write_json
from rl.grpo_diagnostics import (group_statistics, normalization_evidence,
    loss_components, reference_snapshot)
from rl.training import (ResumeEvidence, append_record, training_inputs,
    load_trainable_model, evidence_callback, training_arguments, save_training_result)


class BatchPaused(RuntimeError):
    pass


class GroupUpdateEvidence:
    """将同题组跨步骤、跨 rank 的更新记录合并，在完整组结束时判断。"""
    def __init__(self, output, rank):
        self.output, self.rank = output, rank

    def begin(self, group, optimizer_steps):
        self.group = {key: group[key] for key in
            ("policy_step", "attempt", "task_ids", "rewards", "advantages", "nonzero_advantage")}
        self.group["beta"] = group.get("beta", 0.0)
        self.optimizer_steps = optimizer_steps
        self.updates = []

    def record(self, rank_updates):
        self.updates.append(rank_updates)
        if len(self.updates) < self.optimizer_steps:
            return
        nonzero_gradient = any(r["gradient_norm"] > 0 for step in self.updates for r in step)
        parameter_changed = any(r["parameter_delta_l1"] > 0 for step in self.updates for r in step)
        missing = []
        if self.group["nonzero_advantage"]:
            if not nonzero_gradient:
                missing.append("整组无非零梯度")
            if not parameter_changed:
                missing.append("整组无实际LoRA参数变化")
        result = {**self.group, "optimizer_steps": self.optimizer_steps,
            "completed_steps": [rows[0]["step"] for rows in self.updates],
            "rank_updates": self.updates, "nonzero_gradient": nonzero_gradient,
            "parameter_changed": parameter_changed, "paused": bool(missing), "missing": missing,
            "reward_signal_present": self.group["nonzero_advantage"],
            "kl_enabled": self.group["beta"] != 0,
            "zero_advantage_meaning": "no_within_group_reward_distinction"}
        write_json(self.output / f"group-updates-step{self.group['policy_step']}-rank{self.rank}.json", result)
        if missing:
            raise BatchPaused("非零优势完整组处理后" + "、".join(missing) + "，暂停诊断；已消耗额度不返还")


def complete_group(sample, record, max_attempts=2):
    """异常组只补采一次；两次均在参数更新之前。"""
    for attempt in range(max_attempts):
        result, failed = sample(attempt)
        record(attempt, result, failed)
        if not failed:
            return result
    raise BatchPaused("同一策略版本完整组采样耗尽基础设施尝试上限，暂停批次")


class ToolEnvironment:
    """只有 search 暴露为工具；其余实验状态保留在属性中。"""
    def __init__(self, retrieval, tasks):
        self._environment = PolicyEnvironment(retrieval)
        self._tasks = tasks

    def reset(self, task_id, **kwargs):
        self._environment.reset(self._tasks[task_id])
        self._lock = asyncio.Lock()
        self._protocol_error = None
        self._infrastructure_error = None
        self._search_requests = 0

    async def search(self, query: str) -> str:
        """检索客服政策。

        Args:
            query: 自然语言政策查询。
        """
        async with self._lock:
            if self._search_requests >= self._environment.max_searches:
                self._protocol_error = "模型请求超过三次检索预算"
                return '{"state":"invalid_arguments","detail":"检索预算已用尽，请提交终止 JSON"}'
            self._search_requests += 1
            try:
                observation = await self._environment.step("search", {"query": query})
                return observation.tool_content()
            except Exception as error:
                self._infrastructure_error = {"type": type(error).__name__, "detail": str(error)}
                return '{"state":"unavailable","detail":"检索服务不可用"}'


FINAL_INSTRUCTION = {"role": "system", "content": "本轮不再执行检索，请依据现有观察提交终止 JSON。"}


def visible_trajectory(environment, completion):
    """以进入生成上下文的工具正文重建评分证据，包括预算裁剪后的片段。"""
    trajectory = environment.trajectory()
    visible = {}
    for message in completion:
        if message.get("role") == "tool" and message.get("name") == "search":
            payload = json.loads(message["content"])
            if "request_number" in payload:
                observation = SearchObservation.model_validate(payload)
                visible[observation.request_number] = observation
    trajectory.searches = [SearchTurn(action=turn.action, observation=visible[turn.observation.request_number])
        for turn in trajectory.searches if turn.observation.request_number in visible]
    return trajectory


def policy_tool_call_loop(trainer, prompts, prompt_ids, completion_ids, completions, logprobs,
                          images, multimodal_fields, parse_response):
    """固定 TRL 1.10.0 的文本 search 接入：保留终止轮并返回原生 loss mask。"""
    if images or multimodal_fields:
        raise ValueError("客服策略仅支持文本 search 轨迹")
    masks = [[1] * len(ids) for ids in completion_ids]
    counts, failures = 0, 0
    active = [i for i, messages in enumerate(completions) if messages[-1].get("tool_calls")]
    terminal_only = set()

    def available(i, length):
        return min(trainer.max_completion_length - length,
                   trainer.context_tokens - len(prompt_ids[i]) - length)

    while active:
        regenerate, contexts, caps = [], [], []
        for i in active:
            env = trainer.environments[i]
            calls = completions[i][-1]["tool_calls"]
            counts += len(calls)
            # 检查发起该助手轮时的空间，搜索次数按请求而非助手轮数计。
            last_model_length = 0
            for value in reversed(masks[i]):
                if not value:
                    break
                last_model_length += 1
            before = len(completion_ids[i]) - last_model_length
            if (i in terminal_only or available(i, before) < 2 * trainer.turn_tokens or
                    env._search_requests + len(calls) > env._environment.max_searches):
                env._protocol_error = "模型请求超过剩余检索或生成预算"
                failures += len(calls)
                continue
            try:
                for call in calls:
                    if call["type"] != "function" or call["function"]["name"] != "search":
                        raise ValueError("只支持 search 工具")
                    SearchAction.model_validate(call["function"]["arguments"])
            except (ValueError, KeyError, TypeError) as error:
                env._protocol_error = f"工具协议错误:{error}"
                failures += len(calls)
                continue

            tools = []
            for call in calls:
                arguments = call["function"]["arguments"]
                if "search" in trainer._sync_tool_dicts[i]:
                    content = trainer._sync_tool_dicts[i]["search"](**arguments)
                else:
                    content = asyncio.run_coroutine_threadsafe(
                        trainer._async_tool_dicts[i]["search"](**arguments), trainer.async_loop).result()
                tools.append({"role": "tool", "name": "search", "content": content})

            original_tools = deepcopy(tools)
            # 先为完整终止轮和结束提示留空间；只省略整条知识片段。
            while True:
                suffix = trainer._get_tool_suffix_ids(tools + [FINAL_INSTRUCTION])
                if available(i, len(completion_ids[i]) + len(suffix)) >= trainer.turn_tokens:
                    break
                shortened = False
                for message in reversed(tools):
                    payload = json.loads(message["content"])
                    if payload.get("snippets"):
                        observation = SearchObservation.model_validate(payload)
                        observation.snippets.pop()
                        observation.status.state = "candidates" if observation.snippets else "empty"
                        message["content"] = observation.tool_content()
                        shortened = True
                        break
                if not shortened:
                    break

            suffix_without_notice = trainer._get_tool_suffix_ids(tools)
            final_turn = (env._search_requests >= env._environment.max_searches or
                available(i, len(completion_ids[i]) + len(suffix_without_notice)) < 2 * trainer.turn_tokens)
            feedback = tools + ([deepcopy(FINAL_INSTRUCTION)] if final_turn else [])
            suffix = trainer._get_tool_suffix_ids(feedback)
            cap = min(trainer.turn_tokens, available(i, len(completion_ids[i]) + len(suffix)))
            for old, shown in zip(original_tools, tools):
                if old != shown:
                    number = json.loads(shown["content"])["request_number"]
                    env._environment.diagnostics[number - 1].update(
                        before_token_budget=json.loads(old["content"]),
                        visible_observation=json.loads(shown["content"]), observation_budget_applied=True)
            if cap <= 0:
                # 未进入生成上下文的观察不加入 completion，也不用于判分。
                env._protocol_error = "工具观察后无剩余终止预算"
                continue
            completions[i].extend(feedback)
            completion_ids[i].extend(suffix)
            masks[i].extend([0] * len(suffix))
            if logprobs is not None:
                logprobs[i].extend([0.0] * len(suffix))
            if final_turn:
                terminal_only.add(i)
            regenerate.append(i)
            contexts.append(prompt_ids[i] + completion_ids[i])
            caps.append(cap)
        if not regenerate:
            break
        # HF 按批次使用一个 max_new_tokens；最短剩余预算限制本次生成。
        trainer._policy_turn_cap = min(caps)
        try:
            generated, generated_logprobs = trainer._generate_single_turn(contexts, None, {})
        finally:
            trainer._policy_turn_cap = None
        active = []
        for position, i in enumerate(regenerate):
            ids = generated[position]
            if len(ids) > min(caps):
                raise RuntimeError("生成后端未遵守本轮 max_new_tokens")
            message = parse_response(trainer._tokenizer, ids, prefix=contexts[position]) if ids else {}
            completion_ids[i].extend(ids)
            masks[i].extend([1] * len(ids))
            if logprobs is not None:
                logprobs[i].extend(generated_logprobs[position])
            if message:
                completions[i].append(message)
                if message.get("tool_calls"):
                    active.append(i)
    return masks, completions, completion_ids, logprobs, counts, failures, [[] for _ in completions]


def make_trainer_class():
    import torch
    from trl import GRPOTrainer
    from trl.chat_template_utils import parse_response

    class PolicyGRPOTrainer(ResumeEvidence, GRPOTrainer):
        def _get_train_sampler(self, dataset=None):
            if self.run_config.get('dataset_mode') == 'three_branch_pilot':
                from trl.trainer.utils import RepeatSampler
                return RepeatSampler(dataset if dataset is not None else self.train_dataset,
                    mini_repeat_count=4, batch_size=1, repeat_count=2, shuffle=False, seed=42)
            return super()._get_train_sampler(dataset)

        def _tool_call_loop(self, *args):
            return policy_tool_call_loop(self, *args, parse_response=parse_response)

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
            cap = min(self.turn_tokens, self.context_tokens - max(map(len, prompt_ids)),
                      getattr(self, "_policy_turn_cap", None) or self.max_completion_length)
            if cap <= 0:
                raise ValueError("生成输入没有剩余上下文空间")
            start = time.monotonic()
            original = self.generation_config.max_new_tokens
            original_kwargs = self.generation_kwargs["max_new_tokens"]
            self.generation_config.max_new_tokens = cap
            self.generation_kwargs["max_new_tokens"] = cap
            try:
                result = super()._generate_single_turn(prompt_ids, images, multimodal_fields, has_tool_images)
            finally:
                self.generation_config.max_new_tokens = original
                self.generation_kwargs["max_new_tokens"] = original_kwargs
            torch.cuda.synchronize()
            self.turn_records.append({"policy_step": self.state.global_step,
                "input_lengths": list(map(len, prompt_ids)), "output_lengths": list(map(len, result[0])),
                "max_new_tokens": cap, "generation_s": time.monotonic()-start})
            return result

        def _generate_and_score_completions(self, inputs):
            # 单生成批次恰好一个完整同题组，组跨两个 rank 分布。
            if self.args.generation_batch_size != self.num_generations:
                raise ValueError("当前短程入口每次生成一个完整同题组")
            policy_step = self.state.global_step

            def sample(attempt):
                if self.run_config.get("run_kind"):
                    from rl.run_control import reserve_distributed
                    key = "student_trajectories"
                    reserve_distributed(self.work_root, self.run_config, key,
                        self.args.generation_batch_size, f"grpo-step-{policy_step}-attempt-{attempt}")
                self.turn_records = []
                self.reward_rows = []
                self.group_attempt = attempt
                result = super(PolicyGRPOTrainer, self)._generate_and_score_completions(inputs)
                gathered = [None] * self.accelerator.num_processes
                torch.distributed.all_gather_object(gathered, self.reward_rows)
                rows = [row for rank_rows in gathered for row in rank_rows]
                failed = any(row["infrastructure_failure"] for row in rows)
                if self.state.global_step != policy_step:
                    raise RuntimeError("补采前策略版本发生变化")
                mask = result["completion_mask"] * result.get("tool_mask", 1)
                count = self.accelerator.gather(mask.sum()).sum()
                if count != result["num_items_in_batch"]:
                    raise RuntimeError("生成批次分母未按有效模型 token 计算")
                actual_rewards = [row["grade"]["reward"] for row in rows]
                if len(rows) != self.num_generations or len({row["task_id"] for row in rows}) != 1:
                    raise RuntimeError("跨 rank 奖励必须覆盖完整同题四轨迹")
                statistics = group_statistics(actual_rewards, self.scale_rewards)
                expected = statistics["expected_advantages"]
                advantages = self.accelerator.gather(result["advantages"]).tolist()
                if len(advantages) != len(expected) or any(abs(a-b) > 1e-5 for a, b in zip(expected, advantages)):
                    raise RuntimeError("优势与配置的同题奖励变换不一致")
                initial_logp_difference = None
                if self.beta and policy_step == 0:
                    difference = (result["old_per_token_logps"] - result["ref_per_token_logps"]).abs()
                    initial_logp_difference = float(difference[mask.bool()].max())
                    if initial_logp_difference != 0:
                        raise RuntimeError("初始策略与参考策略的同条件 token 概率不相符")
                self.group_evidence = {"policy_step": policy_step, "attempt": attempt,
                    **statistics, "beta": self.beta, "initial_policy_reference_logp_max_difference": initial_logp_difference,
                    "task_ids": [row["task_id"] for row in rows], "rows": rows,
                    "rewards": actual_rewards, "advantages": advantages,
                    "nonzero_advantage": any(abs(x) > 1e-8 for x in advantages),
                    "effective_model_tokens": int(count), "infrastructure_failure": failed,
                    "optimization_model_tokens": 0 if failed else int(count),
                    "optimization_contribution": "excluded_before_backward" if failed else "accepted",
                    "generation": self.turn_records,
                    "completion_mask": result["completion_mask"].cpu().tolist(),
                    "tool_mask": result.get("tool_mask", torch.ones_like(mask)).cpu().tolist(),
                    "completion_ids": result["completion_ids"].cpu().tolist(),
                    "prompt_lengths": result["prompt_mask"].sum(1).cpu().tolist(),
                    "completion_lengths": result["completion_mask"].sum(1).cpu().tolist()}
                return result, failed

            def record(attempt, result, failed):
                task_id = self.group_evidence["task_ids"][0]
                path = self.evidence_output / f"group-step{policy_step}-{task_id}-attempt{attempt}-rank{self.accelerator.process_index}.json"
                write_json(path, self.group_evidence)
                if failed:
                    for task_id in set(self.group_evidence["task_ids"]):
                        counts = self.fault_counts.setdefault(task_id, {"abnormal_groups": 0, "resampled_groups": 0})
                        counts["abnormal_groups"] += 1
                        if attempt == 0:
                            counts["resampled_groups"] += 1
                    write_json(self.evidence_output / f"fault-counts-rank{self.accelerator.process_index}.json", self.fault_counts)

            result = complete_group(sample, record, max_attempts=self.run_config.get('group_attempts', 2))
            if self.state.global_step < self.state.max_steps:
                self.zero_signal_groups = 0 if self.group_evidence["nonzero_advantage"] else self.zero_signal_groups + 1
                if (self.run_config.get("run_kind") == "formal"
                        and self.zero_signal_groups >= self.run_config["stop_after_zero_signal_groups"]):
                    raise BatchPaused("连续预定组数没有非零优势，暂停诊断；不自动扩预算")
                self.group_updates.begin(self.group_evidence,
                    self.args.steps_per_generation * self.num_iterations // self.args.gradient_accumulation_steps)
            return result

        def record_group_update(self, update):
            gathered = [None] * self.accelerator.num_processes
            torch.distributed.all_gather_object(gathered, update)
            self.group_updates.record(gathered)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self.loss_started = time.monotonic()
            self.record_loss_forward = True
            try:
                loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            finally:
                self.record_loss_forward = False
            mask = inputs["completion_mask"] * inputs.get("tool_mask", 1)
            self.loss_evidence = normalization_evidence(self.loss_type, mask.sum(1).tolist(),
                int(inputs["num_items_in_batch"]), self.accelerator.num_processes,
                self.current_gradient_accumulation_steps, self.args.steps_per_generation)
            logps, entropies = self.recorded_loss_forward
            # 在小型概率张量上求两项导数，不增加模型前向或策略反向。
            with torch.enable_grad():
                diagnostic_logps = logps.detach().requires_grad_()
                reward, kl, fields = loss_components(self, inputs, diagnostic_logps, entropies.detach())
                reward_gradient = torch.autograd.grad(reward, diagnostic_logps, retain_graph=True)[0]
                kl_gradient = (torch.autograd.grad(kl, diagnostic_logps)[0] if self.beta else torch.zeros_like(logps))
            if not torch.allclose(loss.detach(), (reward+kl).detach(), atol=1e-6, rtol=1e-5):
                raise RuntimeError("损失分解与实际 TRL 目标不一致")
            self.loss_evidence.update(fields,
                reward_logp_gradient_l1=float(reward_gradient.abs().sum()),
                kl_logp_gradient_l1=float(kl_gradient.abs().sum()),
                excluded_token_gradient_l1=float((reward_gradient+kl_gradient)[~mask.bool()].abs().sum()))
            del self.recorded_loss_forward
            return loss

        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            reference = (self.beta != 0 and
                self.accelerator.unwrap_model(self.model).active_adapter == "ref")
            if reference:
                torch.cuda.synchronize()
                started = time.monotonic()
            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            if reference:
                torch.cuda.synchronize()
                append_record(self.evidence_output / f"reference-forward-rank{self.accelerator.process_index}.jsonl",
                    {"policy_step": self.state.global_step, "reference_forward_s": time.monotonic()-started,
                     "origin": self.run_config["sft_adapter"]})
            if getattr(self, "record_loss_forward", False):
                self.recorded_loss_forward = (result[0].detach(), result[1].detach())
            return result

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            torch.cuda.synchronize()
            append_record(self.evidence_output / f"loss-rank{self.accelerator.process_index}.jsonl",
                {"policy_step": self.state.global_step, "loss": float(loss),
                 "update_microbatch_s": time.monotonic()-self.loss_started, **self.loss_evidence})
            return loss

    return PolicyGRPOTrainer


def main():
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig
    root, config, output, tasks, standards, clauses = training_inputs()
    set_seed(config["training"]["seed"])
    if config.get("run_purpose"):
        from rl.run_control import check_inputs
        check_inputs(root, config, tasks, standards, "grpo")
    elif any(standards[t].split != ('train' if config.get('dataset_mode') == 'three_branch_pilot' else 'dev') for t in config["task_ids"]):
        raise ValueError("开发闭环只使用开发任务")
    gateway = rerank_gateway(config["retrieval"])
    rerank_http_requests = [0]
    async def on_rerank_request(request):
        rerank_http_requests[0] += 1
    gateway.client._client.event_hooks["request"].append(on_rerank_request)
    retrieval = build_retrieval(root, config["retrieval"], gateway,
        prepared_only=config.get('dataset_mode') == 'three_branch_pilot')
    if config.get('dataset_mode') == 'three_branch_pilot':
        from rl.three_branch_pilot import attach_budget
        attach_budget(root, config, output, gateway, retrieval)
    if retrieval.knowledge.stats()["pending_main"] or not retrieval.knowledge._main.count():
        raise RuntimeError("须复用完整 Qwen 实验索引")
    model, tokenizer = load_trainable_model(root, config)
    trainer = None

    def business_reward(completions, environments, **kwargs):
        rewards = []
        for completion, env in zip(completions, environments):
            environment = env._environment
            final = completion[-1].get("content", "") if completion[-1].get("role") == "assistant" else ""
            if env._protocol_error or completion[-1].get("tool_calls"):
                final = ""
            trajectory = visible_trajectory(environment, completion)
            grade = grade_result(standards[environment.task.task_id], final, trajectory, clauses)
            failure = bool(env._infrastructure_error) or any(
                turn.observation.status.state == "unavailable" or turn.observation.status.degraded
                for turn in environment.searches)
            trainer.reward_rows.append({"task_id": environment.task.task_id, "grade": grade.to_dict(),
                "trajectory": trajectory.model_dump(), "completion": deepcopy(completion),
                "actual_search_count": len(environment.searches),
                "search_requests": env._search_requests,
                "tool_diagnostics": environment.diagnostics, "infrastructure_failure": failure,
                "infrastructure_error": env._infrastructure_error, "protocol_error": env._protocol_error})
            rewards.append(grade.reward)
        return rewards

    grpo_options = dict(config["grpo"])
    expected_steps = grpo_options.get("steps_per_generation")
    if grpo_options.get("generation_batch_size") is not None:
        # TRL 1.10.0 不允许两者同时传入；保留显式生成批量并核对推导步数。
        grpo_options.pop("steps_per_generation", None)
    args = GRPOConfig(**training_arguments(config, output), **grpo_options)
    if expected_steps is not None and args.steps_per_generation != expected_steps:
        raise ValueError("实际进程/批量与预定 steps_per_generation 不符")
    trainer = make_trainer_class()(model=model, processing_class=tokenizer,
        args=args,
        train_dataset=Dataset.from_list([{"task_id": tid, "prompt": tasks[tid].messages()} for tid in config["task_ids"]]),
        reward_funcs=business_reward, environment_factory=lambda: ToolEnvironment(retrieval, tasks),
        callbacks=[evidence_callback(output, root, config,
            on_update=lambda update: trainer.record_group_update(update))])
    # 提示使用与教师、独立 runner 完全相同的公开工具 schema。
    trainer._env_tools[None] = TOOLS
    trainer.evidence_output = output
    trainer.turn_tokens = config["turn_tokens"]
    trainer.context_tokens = config["context_tokens"]
    trainer.fault_counts = {}
    trainer.turn_records = []
    trainer.reward_rows = []
    trainer.run_config = config
    trainer.work_root = root
    trainer.zero_signal_groups = 0
    trainer.group_updates = GroupUpdateEvidence(output, trainer.accelerator.process_index)
    if trainer.accelerator.num_processes != 2:
        raise ValueError("当前训练配置要求双进程双卡")
    try:
        reference_snapshot(trainer, output, initial=True)
        trainer.train(resume_from_checkpoint=str(root / config["resume_from_checkpoint"])
                      if config.get("resume_from_checkpoint") else None)
        reference_snapshot(trainer, output)
        save_training_result(trainer, tokenizer, output)
        # 使用刚更新的同一内存策略再次完整采样，不加载旧 checkpoint。
        trainer.model.eval()
        for task_id in config.get("post_update_task_ids", config["task_ids"]):
            batch = [{"task_id": task_id, "prompt": tasks[task_id].messages()}
                     for _ in range(trainer.args.generation_batch_size // trainer.accelerator.num_processes)]
            trainer._generate_and_score_completions(batch)
        if trainer.is_world_process_zero():
            updates = [json.loads(line) for line in (output / "updates-rank0.jsonl").read_text(encoding="utf-8").splitlines()]
            actual_update = any(row["parameter_delta_l1"] > 0 for row in updates)
            write_json(output / "post_update_sampling.json", {"policy_step": trainer.state.global_step,
                "task_ids": config.get("post_update_task_ids", config["task_ids"]),
                "actual_parameter_update": actual_update,
                "model_source": "current_in_memory_updated_policy" if actual_update else "current_in_memory_policy_without_parameter_change"})
    except BatchPaused as error:
        write_json(output / f"paused-rank{trainer.accelerator.process_index}.json",
            {"reason": str(error), "policy_step": trainer.state.global_step, "fault_counts": trainer.fault_counts})
        raise
    finally:
        write_json(output / f"api-cost-rank{trainer.accelerator.process_index}.json", {
            "embedding_requests": retrieval.knowledge.embedding.usage["requests"],
            "embedding_reported_tokens": retrieval.knowledge.embedding.usage["reported_tokens"],
            "rerank_http_requests": rerank_http_requests[0], "monetary_cost": None})
        retrieval.knowledge.close()
        retrieval.knowledge.embedding.close()
        if trainer.async_loop is not None:
            asyncio.run_coroutine_threadsafe(gateway.client.close(), trainer.async_loop).result()


if __name__ == "__main__":
    main()
