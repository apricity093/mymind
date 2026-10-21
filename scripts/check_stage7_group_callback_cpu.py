"""固定 TRL 与真实回调的 CPU 接线检查；只使用受控张量和伪造采样记录。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mymind"))

import torch
from trl import GRPOTrainer

from experiments.agentic_rl.grpo import BatchPaused, GroupUpdateEvidence, make_trainer_class
from experiments.agentic_rl.runtime import write_json
from experiments.agentic_rl.training import evidence_callback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    output=parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    cls=make_trainer_class()
    trainer=object.__new__(cls)
    trainer.state=SimpleNamespace(global_step=0,max_steps=80)
    trainer.args=SimpleNamespace(generation_batch_size=4,steps_per_generation=2,gradient_accumulation_steps=1)
    trainer.num_generations,trainer.num_iterations=4,1
    trainer.accelerator=SimpleNamespace(num_processes=2,process_index=0,
        gather=lambda tensor: torch.cat([tensor.reshape(-1)]*2))
    trainer.run_config=dict(run_kind="formal",stop_after_zero_signal_groups=8)
    trainer.work_root=output
    trainer.evidence_output=output
    trainer.fault_injector=None
    trainer.zero_signal_groups=0
    trainer.group_updates=GroupUpdateEvidence(output,0)
    nonzero=[True]

    def sampled(self, inputs):
        rewards=[-1,0] if nonzero[0] else [0,0]
        self.reward_rows=[dict(task_id="controlled",grade=dict(reward=r),infrastructure_failure=False) for r in rewards]
        return dict(completion_mask=torch.ones(2,1,dtype=torch.long),num_items_in_batch=torch.tensor(4),
            advantages=torch.tensor([-.5,.5] if nonzero[0] else [0.,0.]),
            completion_ids=torch.ones(2,1,dtype=torch.long),prompt_mask=torch.ones(2,1,dtype=torch.long))

    def gather(rows, value):
        rows[:]=[value,value]

    with ExitStack() as stack:
        stack.enter_context(patch.object(GRPOTrainer,"_generate_and_score_completions",sampled))
        stack.enter_context(patch("torch.distributed.all_gather_object",gather))
        stack.enter_context(patch("experiments.agentic_rl.stage7.reserve_distributed"))
        for name in ("reset_peak_memory_stats","synchronize","max_memory_allocated","max_memory_reserved"):
            stack.enter_context(patch.object(torch.cuda,name,return_value=0))
        for scenario,gradient,change,paused in (("updated",1.,.25,False),("unchanged",1.,0.,True),("no-gradient",0.,0.,True)):
            directory=output / scenario
            directory.mkdir(exist_ok=True)
            trainer.group_updates=GroupUpdateEvidence(directory,0)
            trainer.state.global_step=0
            trainer._generate_and_score_completions([{}])
            parameter=torch.nn.Parameter(torch.tensor([1.]))
            model=SimpleNamespace(named_parameters=lambda: [("lora_A",parameter)])
            callback=evidence_callback(directory,on_update=trainer.record_group_update)
            callback.on_train_begin(SimpleNamespace(process_index=0),trainer.state,None,model)
            for step in (1,2):
                parameter.grad=torch.tensor([0. if step==1 else gradient])
                callback.on_pre_optimizer_step(None,trainer.state,None)
                with torch.no_grad():
                    parameter.add_(0. if step==1 else change)
                trainer.state.global_step=step
                try:
                    callback.on_step_end(None,trainer.state,None,SimpleNamespace(state={}))
                except BatchPaused:
                    assert step==2 and paused
                else:
                    assert step==1 or not paused
            result=json.loads((directory / "group-updates-step0-rank0.json").read_text(encoding="utf-8"))
            assert result["paused"]==paused and result["completed_steps"]==[1,2]
        nonzero[0]=False
        trainer.zero_signal_groups=0
        for count in range(1,9):
            try:
                trainer._generate_and_score_completions([{}])
            except BatchPaused:
                assert count==8
                break
            assert count<8
        else:
            raise AssertionError("连续8组零优势未暂停")
        assert trainer.zero_signal_groups==8
        trainer.state.global_step=80
        trainer.zero_signal_groups=7
        trainer._generate_and_score_completions([{}])
        assert trainer.zero_signal_groups==7
    result=dict(status="passed",trainer_class=cls.__name__,device="cpu",
        callback_scenarios=["zero-first-slice-then-update","gradient-without-parameter-change","no-gradient-or-change"],
        zero_advantage_pause_at=8,post_training_sampling_excluded=True,
        controlled_records=True,model_calls=0,api_requests=0,gpu_operations=0,
        real_rl_update_accepted=False)
    write_json(output / "result.json",result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
