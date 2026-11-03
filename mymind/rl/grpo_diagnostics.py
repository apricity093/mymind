"""GRPO 配置对照的组统计、真实损失分解和固定参考策略证据。"""
import math


def group_statistics(rewards, scale_rewards):
    if scale_rewards not in {"none", "group"}:
        raise ValueError("本入口仅验证 none/group 奖励缩放")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1))
    advantages = [value - mean for value in rewards]
    if scale_rewards == "group":
        advantages = [value / (std + 1e-4) for value in advantages]
    return {"reward_mean": mean, "reward_std": std, "std_correction": 1,
            "advantage_epsilon": 1e-4, "scale_rewards": scale_rewards,
            "expected_advantages": advantages}


def normalization_evidence(loss_type, lengths, global_tokens, world_size, accumulation, generation_steps):
    result = {"loss_type": loss_type, "local_effective_tokens": sum(lengths),
              "trajectory_effective_model_tokens": lengths,
              "global_generation_model_tokens": global_tokens}
    if loss_type == "grpo":
        result.update(normalization="mean_of_per_trajectory_masked_token_means",
                      trajectory_denominators=[max(1, length) for length in lengths],
                      sample_denominator=len(lengths), accumulation_denominator=accumulation)
    elif loss_type == "dapo":
        result.update(normalization="generation_model_token_sum_rescaled_to_update",
                      dapo_normalizer=max(1, global_tokens) / world_size * accumulation / generation_steps)
    else:
        raise ValueError("本入口仅验证 grpo/dapo loss")
    return result


def loss_components(trainer, inputs, logps, entropies):
    """对同一次前向的 token 概率分解；不增加模型前向或改变训练目标。"""
    import torch
    mask = inputs["completion_mask"] * inputs.get("tool_mask", 1)
    old = inputs.get("old_per_token_logps")
    old = logps.detach() if old is None else old
    ratio = (logps - old).exp()
    advantages = inputs["advantages"].unsqueeze(1)
    reward = -torch.minimum(ratio * advantages,
                            ratio.clamp(1 - trainer.epsilon_low, 1 + trainer.epsilon_high) * advantages)
    kl = torch.zeros_like(logps)
    if trainer.beta:
        difference = inputs["ref_per_token_logps"] - logps
        kl = difference.exp() - difference - 1
        if trainer.args.use_bias_correction_kl:
            kl = kl * ratio
    def reduce(value):
        if trainer.loss_type == "grpo":
            return ((value * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean() / trainer.current_gradient_accumulation_steps
        denominator = max(1, int(inputs["num_items_in_batch"])) / trainer.accelerator.num_processes
        denominator *= trainer.current_gradient_accumulation_steps / trainer.args.steps_per_generation
        return (value * mask).sum() / denominator
    reward_loss, kl_loss = reduce(reward), reduce(trainer.beta * kl)
    return reward_loss, kl_loss, {
        "reward_loss": float(reward_loss.detach()), "kl_loss": float(kl_loss.detach()),
        "masked_kl": float((kl * mask).sum().detach() / mask.sum().clamp(min=1)),
        "masked_entropy": float((entropies * mask).sum().detach() / mask.sum().clamp(min=1)),
        "clip_fraction": float((((ratio < 1-trainer.epsilon_low) & (advantages < 0) |
                                  (ratio > 1+trainer.epsilon_high) & (advantages > 0)) * mask).sum().detach() / mask.sum().clamp(min=1))}


def reference_snapshot(trainer, output, *, initial=False):
    import torch
    from rl.runtime import write_json
    model = trainer.accelerator.unwrap_model(trainer.model)
    if not trainer.beta:
        return
    references = {name: parameter for name, parameter in model.named_parameters() if ".ref." in name}
    if not references or trainer.ref_model is not None:
        raise RuntimeError("KL 必须使用同一 PEFT 基座内的 SFT120 ref adapter")
    if any(p.requires_grad for p in references.values()):
        raise RuntimeError("参考 adapter 必须冻结")
    if initial:
        trainer.reference_initial = {name: parameter.detach().cpu().clone() for name, parameter in references.items()}
        pairs_equal = all(torch.equal(parameter, model.get_parameter(name.replace(".ref.", ".default.")))
                          for name, parameter in references.items())
        if not pairs_equal:
            raise RuntimeError("参考 adapter 与初始 SFT120 策略不一致")
        record = {"origin": trainer.run_config["sft_adapter"], "reference_tensors": len(references),
                  "initial_policy_reference_equal": pairs_equal, "separate_base_model": False}
    else:
        optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        unchanged = all(torch.equal(parameter.detach().cpu(), trainer.reference_initial[name])
                        for name, parameter in references.items())
        excluded = all(id(parameter) not in optimizer_ids for parameter in references.values())
        record = {"reference_tensors": len(references), "reference_unchanged": unchanged,
                  "reference_excluded_from_optimizer": excluded,
                  "reference_gradients_absent": all(p.grad is None for p in references.values())}
        if not unchanged or not excluded or not record["reference_gradients_absent"]:
            raise RuntimeError("固定参考策略或优化器隔离验证失败")
    write_json(output / f"reference-{'initial' if initial else 'final'}-rank{trainer.accelerator.process_index}.json", record)
