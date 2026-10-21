"""在八条已采预检轨迹上验证受控偏移后的真实 KL 更新，不新增轨迹或奖励。"""
import json
import os
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/"mymind"))


def main():
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.grpo_trainer import use_adapter
    from agents.customer_policy.generation import TOOLS
    from experiments.agentic_rl.training import load_trainable_model, training_arguments
    from experiments.agentic_rl.runtime import write_json
    from experiments.agentic_rl.stage7 import read, RunBudget
    from experiments.agentic_rl.tasks import load_public_tasks
    root=Path(os.environ["MYMIND_WORK_ROOT"])
    run=root/"runs/grpo-standard-20261021"
    output=run/"kl-replay"
    output.mkdir(exist_ok=False)
    config=read(run/"configs/preflight.json")
    config={**config,"output_dir":"runs/grpo-standard-20261021/kl-replay"}
    set_seed(42)
    started=time.monotonic()
    os.environ["RANK"]="0"
    os.environ["LOCAL_RANK"]="0"
    model,tokenizer=load_trainable_model(root,config)
    del os.environ["RANK"],os.environ["LOCAL_RANK"]
    write_json(output/"configuration.json",config)
    import shutil
    for relative in (SOURCE/config["source_manifest"]).read_text(encoding="utf-8").splitlines():
        if relative and not relative.startswith("#"):
            target=output/"source"/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(SOURCE/relative,target)
    options=dict(config["grpo"])
    options.pop("steps_per_generation")
    args=GRPOConfig(**training_arguments(config,output),**options)
    tasks=load_public_tasks(root/config["tasks"])
    group=read(sorted((run/"preflight").glob("group-step0-*-rank0.json"))[0])
    tid=group["task_ids"][0]
    prompt=tokenizer.apply_chat_template(tasks[tid].messages(),tools=TOOLS,tokenize=True,add_generation_prompt=True,return_dict=False)
    if len(prompt)!=group["prompt_lengths"][0]:
        raise RuntimeError("回放公开prompt与原轨迹token长度不一致")
    trainer=GRPOTrainer(model=model,processing_class=tokenizer,args=args,
        train_dataset=Dataset.from_list([dict(prompt=tasks[tid].messages())]),reward_funcs=lambda completions,**kw:[1.]*len(completions))
    trainer.current_gradient_accumulation_steps=1
    policy=trainer.accelerator.unwrap_model(trainer.model)
    policy.eval()
    parameters={n:p for n,p in policy.named_parameters() if p.requires_grad}
    references={n:p for n,p in policy.named_parameters() if ".ref." in n}
    ref_initial={n:p.detach().cpu().clone() for n,p in references.items()}
    assert references and all(not p.requires_grad for p in references.values())
    assert all(torch.equal(p,policy.get_parameter(n.replace(".ref.",".default."))) for n,p in references.items())
    device=next(policy.parameters()).device
    completion=torch.tensor(group["completion_ids"][:1],device=device)
    mask=torch.tensor(group["completion_mask"][:1],device=device)
    tool=torch.tensor(group["tool_mask"][:1],device=device)
    prompt_ids=torch.tensor([prompt],device=device)
    inputs=dict(prompt_ids=prompt_ids,prompt_mask=torch.ones_like(prompt_ids),completion_ids=completion,
        completion_mask=mask,tool_mask=tool,advantages=torch.zeros(1,device=device),num_items_in_batch=(mask*tool).sum())
    ids=torch.cat([prompt_ids,completion],dim=1)
    attention=torch.cat([torch.ones_like(prompt_ids),mask],dim=1)
    def probabilities():
        return trainer._get_per_token_logps_and_entropies(trainer.model,ids,attention,completion.shape[1],compute_entropy=False)[0]
    with torch.no_grad():
        before=probabilities()
        with use_adapter(policy,"ref"):
            ref=probabilities()
    equal=torch.equal(before,ref)
    assert equal
    inputs.update(old_per_token_logps=before,ref_per_token_logps=ref)
    # 只修改本次回放副本的一张LoRA B矩阵；不称自然奖励学习。
    name=next(n for n in parameters if ".lora_B.default." in n)
    with torch.no_grad():
        parameters[name].add_(.01)
    previous={n:p.detach().cpu().clone() for n,p in parameters.items()}
    trainer.create_optimizer()
    optimizer_ids={id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert all(id(p) not in optimizer_ids for p in references.values())
    assert len(trainer.optimizer.state)==0
    RunBudget(root,config).reserve("diagnostic_updates",1,"controlled-kl-replay-no-new-trajectories")
    trainer.model.train()
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    loss=trainer.compute_loss(trainer.model,inputs)
    loss.backward()
    gradient=sum(float(p.grad.float().square().sum()) for p in parameters.values() if p.grad is not None)**.5
    torch.nn.utils.clip_grad_norm_(parameters.values(),1.)
    trainer.optimizer.step()
    delta=sum(float((p.detach().cpu().float()-previous[n].float()).abs().sum()) for n,p in parameters.items())
    unchanged=all(torch.equal(p.detach().cpu(),ref_initial[n]) for n,p in references.items())
    assert float(loss)>0 and gradient>0 and delta>0 and unchanged
    record=dict(passed=True,kind="controlled_kl_replay",new_trajectories=0,new_api_requests=0,
        source_group=str(sorted((run/"preflight").glob("group-step0-*-rank0.json"))[0].relative_to(root)),
        task_id=tid,reward=group["rewards"][0],advantage=0.,initial_policy_reference_outputs_equal=equal,
        perturbation=dict(parameter=name,additive_value=.01),loss=float(loss),reward_loss=0.,
        kl_gradient_norm=gradient,optimizer_parameter_delta_l1=delta,reference_unchanged=unchanged,
        reference_excluded_from_optimizer=True,reference_gradients_absent=all(p.grad is None for p in references.values()),
        optimizer_initial_state_entries=0,diagnostic_optimizer_steps=1,elapsed_s=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        formal_start="original SFT120; discard diagnostic perturbation and optimizer")
    write_json(output/"verification.json",record)
    print(json.dumps(record,ensure_ascii=False))


if __name__=="__main__":
    main()
