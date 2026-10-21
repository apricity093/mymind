"""汇总本轮真实预检；缺少学习信号时明确保留未通过项。"""
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE / "mymind"))
from experiments.agentic_rl.stage7 import read
from experiments.agentic_rl.runtime import write_json


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    root=Path(os.environ["MYMIND_WORK_ROOT"])
    run=root / "runs/stage7-preparation-20261014"
    tokenizer=AutoTokenizer.from_pretrained(root / "models/Qwen3-4B-Instruct-2507",local_files_only=True)
    phases={}
    for phase in ("sft","grpo"):
        folder=run / "preflight" / phase
        result=read(folder / "training_result.json")
        updates=jsonl(folder / "updates-rank0.jsonl")
        all_updates=updates+jsonl(folder / "updates-rank1.jsonl")
        phases[phase]={"optimizer_steps":result["global_step"],"world_size":result["world_size"],
            "finite_gradients":all(r["gradient_finite"] and math.isfinite(r["gradient_norm"]) for r in all_updates),
            "nonzero_gradient_steps":sum(r["gradient_norm"]>0 for r in updates),
            "nonzero_parameter_steps":sum(r["parameter_delta_l1"]>0 for r in updates),
            "parameter_delta_l1":sum(r["parameter_delta_l1"] for r in updates),
            "elapsed_update_steps_s":sum(r["elapsed_step_s"] for r in updates),
            "peak_allocated_gib":max(r["peak_allocated_bytes"] for r in all_updates)/2**30,
            "peak_reserved_gib":max(r["peak_reserved_bytes"] for r in all_updates)/2**30,
            "effective_args":{k:result["effective_args"].get(k) for k in ("num_generations","generation_batch_size","steps_per_generation","gradient_accumulation_steps","loss_type","scale_rewards")}}
    groups=[]; counts=Counter(); failures=[]; lengths=[]; api_usage=[]; tool_messages=0
    for file in sorted((run / "preflight/grpo").glob("group-*-rank0.json")):
        left=read(file); right=read(file.with_name(file.name.replace("rank0","rank1")))
        assert len(set(left["task_ids"]))==1 and len(left["rows"])==4
        total=0
        for rank,record in enumerate((left,right)):
            for i,(ids,mask,tools) in enumerate(zip(record["completion_ids"],record["completion_mask"],record["tool_mask"])):
                total+=sum(m*t for m,t in zip(mask,tools))
                external=tokenizer.decode([value for value,m,t in zip(ids,mask,tools) if m and not t])
                row=record["rows"][rank*len(record["completion_ids"])+i]
                messages=row["completion"]
                for message in messages:
                    if message.get("role")=="tool":
                        assert message["content"] in external
                        tool_messages+=1
                if any(m.get("role")=="tool" for m in messages):
                    assert messages[-1]["role"]=="assistant"
                lengths.append({"prompt":record["prompt_lengths"][i],"completion":record["completion_lengths"][i]})
                assert lengths[-1]["completion"]<=8192 and sum(lengths[-1].values())<=16384
        assert total==left["effective_model_tokens"]
        mean=sum(left["rewards"])/4
        assert all(abs(a-(r-mean))<1e-5 for a,r in zip(left["advantages"],left["rewards"]))
        for row in left["rows"]:
            counts[str(row["grade"]["reward"])]+=1
            assert row["search_requests"]<=3
            for diag in row["tool_diagnostics"]:
                api_usage.append({"embedding_usage":diag.get("embedding_usage"),"rerank":diag.get("diagnostics",{}).get("rerank_usage"),
                    "rerank_status":diag.get("diagnostics",{}).get("rerank_status")})
        groups.append({"file":file.name,"policy_step":left["policy_step"],"task_id":left["task_ids"][0],
            "rewards":left["rewards"],"advantages":left["advantages"],"nonzero_advantage":left["nonzero_advantage"],
            "effective_tokens":total,"infrastructure_failure":left["infrastructure_failure"],
            "search_requests":sum(r["search_requests"] for r in left["rows"]),
            "grade_errors":[r["grade"]["errors"] for r in left["rows"]],
            "generation_s_max_rank":max(sum(t["generation_s"] for t in record["generation"]) for record in (left,right))})
    saved_sft=load_file(str(run / "preflight/sft/adapter/adapter_model.safetensors"))
    saved_rl=load_file(str(run / "preflight/grpo/adapter/adapter_model.safetensors"))
    delta=sum(float((saved_rl[n].float()-value.float()).abs().sum()) for n,value in saved_sft.items())
    if not any(g["nonzero_advantage"] for g in groups if g["policy_step"]<4): failures.append("训练rollout没有非零组相对优势")
    if phases["grpo"]["nonzero_gradient_steps"]==0: failures.append("GRPO没有非零梯度")
    if delta==0: failures.append("GRPO adapter 相对 SFT 没有实际参数变化")
    evaluations={name:read(run / f"preflight/{name}/metrics.json") for name in ("evaluate","sft-evaluate","one-search")}
    one_rows=[read(run / "preflight/one-search" / f"{tid}.json") for tid in read(run / "configs/preflight-one-search.json")["task_ids"]]
    one_observed_final=any(row["trajectory"]["searches"] and not row["runtime_error"] for row in one_rows)
    if not one_observed_final:
        failures.append("真实一次检索轨迹未出现观察后终止：本轮同消息两搜被正确拒绝；CPU终止机会对照已通过")
    result={"run_id":"stage7-preparation-20261014", "preflight_passed":not failures, "missing_evidence":failures,
        "phases":phases,"rl_adapter_delta_from_sft":delta,"groups":groups,"reward_distribution":dict(counts),
        "tool_messages_verified_in_masked_context":tool_messages,"dapo_denominator_and_advantages_verified":True,
        "max_prompt_tokens":max(r["prompt"] for r in lengths),"max_completion_tokens":max(r["completion"] for r in lengths),
        "policy_resampled_after_optimizer_steps":read(run / "preflight/grpo/post_update_sampling.json"),
        "actual_updated_policy_resampled":delta>0,
        "one_search_observation_then_final_observed":one_observed_final,
        "evaluations":evaluations,"api_usage":api_usage,"limits":read(run / "preflight/limits.json"),
        "no_teacher_calls":True,"formal_training_executed":False,"final_test_policy_runs":0}
    write_json(run / "preflight-result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"evaluations","api_usage","groups"}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
