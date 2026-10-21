"""在安装版 TRL 上检验 GRPO/DAPO 的 mask、归一化和 KL 导数，无轨迹采样。"""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "mymind"))


def main():
    import torch
    from trl import GRPOTrainer
    from trl.trainer.grpo_trainer import nanstd
    from experiments.agentic_rl.grpo_diagnostics import loss_components, group_statistics
    from experiments.agentic_rl.runtime import write_json
    device = "cpu"
    results = []
    for loss_type, beta, equal in [("grpo", .04, False), ("grpo", .04, True), ("dapo", 0., False)]:
        logps = torch.tensor([[-.3, -.2, -.7, -.1], [-.1, -.4, -.8, -.9]], device=device, requires_grad=True)
        entropy = torch.ones_like(logps)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], device=device)
        tool = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]], device=device)
        inputs = dict(prompt_ids=torch.ones((2,2), dtype=torch.long, device=device),
            prompt_mask=torch.ones((2,2), device=device), completion_ids=torch.ones((2,4), dtype=torch.long, device=device),
            completion_mask=mask, tool_mask=tool, advantages=torch.tensor([0,0] if equal else [.5,-.5], device=device),
            old_per_token_logps=logps.detach() + .02,
            ref_per_token_logps=logps.detach() + .15,
            num_items_in_batch=torch.tensor(10,device=device))
        trainer = SimpleNamespace(loss_type=loss_type, beta=beta, top_entropy_quantile=1.,
            off_policy_mask_threshold=None, importance_sampling_level="token", epsilon_low=.2, epsilon_high=.2 if beta else .28,
            use_vllm=False, _entropy_bonus_enabled=False, aux_loss_enabled=False, current_gradient_accumulation_steps=1,
            model=SimpleNamespace(training=True), args=SimpleNamespace(use_bias_correction_kl=True, delta=None, steps_per_generation=2),
            accelerator=SimpleNamespace(num_processes=2, reduce=lambda x,**kw:x, gather=lambda x:x),
            _metrics=defaultdict(lambda:defaultdict(list)),
            _get_per_token_logps_and_entropies=lambda *a,**kw:(logps,entropy,None))
        actual = GRPOTrainer._compute_loss(trainer, inputs=inputs, model=trainer.model)
        reward, kl, diagnostics = loss_components(trainer, inputs, logps, entropy)
        assert torch.allclose(actual, reward+kl, atol=1e-6)
        gradient = torch.autograd.grad(actual, logps, retain_graph=True)[0]
        effective_mask = mask * tool
        assert float(gradient[~effective_mask.bool()].abs().sum()) == 0
        # 手算与两种目标不同的长度权重；tool 和 padding 大值不得进入结果。
        ratio = (logps-inputs["old_per_token_logps"]).exp()
        raw = -torch.minimum(ratio*inputs["advantages"][:,None],
            ratio.clamp(.8,1+trainer.epsilon_high)*inputs["advantages"][:,None])
        if beta:
            diff=inputs["ref_per_token_logps"]-logps
            raw=raw+beta*(diff.exp()-diff-1)*ratio
        expected = ((raw*effective_mask).sum(1)/torch.tensor([2.,3.],device=device)).mean() if loss_type=="grpo" else (raw*effective_mask).sum()/2.5
        assert torch.allclose(actual,expected,atol=1e-6)
        if equal:
            assert float(reward)==0 and float(gradient.abs().sum())>0
        results.append(dict(loss_type=loss_type,beta=beta,equal_reward=equal,loss=float(actual),
            excluded_token_gradient_l1=0.,gradient_l1=float(gradient.abs().sum()),**diagnostics))
    for rewards in ([0.,-1.,-1.,-1.], [1.,1.,1.,1.]):
        tensor=torch.tensor(rewards,device=device)
        actual=(tensor-tensor.mean())/(nanstd(tensor)+1e-4)
        assert actual.tolist()==__import__('pytest').approx(group_statistics(rewards,"group")["expected_advantages"],abs=1e-6)
    path=Path(os.environ["MYMIND_WORK_ROOT"])/"runs/grpo-standard-20261021/kernel-verification.json"
    write_json(path,dict(passed=True,device="server CPU; installed torch/TRL",cases=results,
        group_std_correction=1,group_epsilon=1e-4,trl_loss_implementation="installed GRPOTrainer._compute_loss"))
    print(json.dumps(read_json(path),ensure_ascii=False))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__=="__main__":
    main()
