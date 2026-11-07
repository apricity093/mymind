"""Fixed small-pilot execution, evaluation and validation-only model selection."""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from rl.runtime import write_json

REL = Path('runs/success-pool-v1/three-branch-pilot-v1')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def attach_budget(root, config, output, gateway, retrieval):
    from rl.teacher_budget import CollectionBudget
    rank=int(os.environ.get('RANK','0'))
    directory=output/f'http-rank{rank}'
    directory.mkdir()
    budget=CollectionBudget(directory,config['collection_limits'],'main')
    hooks=budget.hooks('rerank')
    for key, values in hooks.items(): gateway.client._client.event_hooks[key].extend(values)
    retrieval.knowledge.embedding.collection_budget=budget
    return budget

async def evaluate(root, cfg):
    from agents.customer_policy.configuration import load_configuration, rerank_gateway
    from agents.customer_policy.environment import PolicyEnvironment
    from agents.customer_policy.generation import QwenGeneration
    from agents.customer_policy.protocol import VisibleTrajectory
    from agents.customer_policy.retrieval import build_retrieval
    from agents.customer_policy.runner import run_task
    from rl.grading import grade_result, GRADING_REVISION
    from rl.metrics import summarize
    from rl.tasks import load_public_tasks, load_private_standards, load_policy_support
    output=root/cfg['output_dir']; output.mkdir(parents=True)
    write_json(output/'configuration.json',cfg)
    # Configure retrieval environment before opening the existing index.
    load_configuration(output/'configuration.json')
    tasks=load_public_tasks(root/cfg['tasks'])
    gateway=rerank_gateway(cfg['retrieval'])
    retrieval=build_retrieval(root,cfg['retrieval'],gateway,prepared_only=True)
    budget=attach_budget(root,cfg,output,gateway,retrieval)
    rows=[]
    try:
        assert retrieval.knowledge._main_complete and retrieval.knowledge._main.count()
        generation=QwenGeneration(root/cfg['model_dir'],cfg['generation'],root/cfg['adapter_path'] if cfg.get('adapter_path') else None)
        generation.torch.manual_seed(42)
        write_json(output/'model_load.json',generation.load_info)
        for tid in cfg['task_ids']:
            budget.check(); start=time.monotonic()
            budget.request_context={'task_id':tid}
            row=await run_task(tasks[tid],generation,PolicyEnvironment(retrieval),budget=cfg['budget'],
                on_progress=lambda state:write_json(output/f'{tid}.partial.json',state))
            row['elapsed_s']=time.monotonic()-start
            # Private scoring data is loaded only after student generation.
            standards=load_private_standards(root/cfg['standards'])
            clauses=load_policy_support(root/cfg['policy_support'])
            row['grade']=grade_result(standards[tid],row['raw_result'],VisibleTrajectory.model_validate(row['trajectory']),clauses).to_dict()
            row['grading_revision']=GRADING_REVISION
            rows.append(row); write_json(output/f'{tid}.json',row)
            write_json(output/'records.json',rows)
            if row['infrastructure_failure']: raise RuntimeError('evaluation infrastructure failure; no retry')
            budget.check()
        result=summarize(cfg['task_ids'],rows,tasks,standards)
        result['branch_equal_protocol_error_rate']=sum(sum(r['protocol_failure'] for r in result['rows'] if standards[r['task_id']].group_id==b)/2 for b in ('A01','B01','R01'))/3
        write_json(output/'metrics.json',result)
    finally:
        budget.mark_inflight_unknown('process_end')
        retrieval.knowledge.close(); retrieval.knowledge.embedding.close()
        await gateway.client.close()

def select(root, phase, candidates):
    records=[]
    for step in candidates:
        metric=read(root/REL/f'output/{phase}-val-{step}/metrics.json')
        assert metric['all_tasks_finished'] and not metric['infrastructure_failures']
        records.append(dict(step=step,success=metric['combination_equal_success_rate'],
            severe=metric['combination_equal_severe_error_rate'],protocol=metric['branch_equal_protocol_error_rate']))
    selected=min(records,key=lambda r:(-r['success'],r['severe'],r['protocol'],r['step']))
    result=dict(phase=phase,candidates=records,selected_adapter=str(REL/f'output/{phase}/checkpoint-{selected["step"]}'))
    write_json(root/REL/f'output/{phase}_selection.json',result)
    return result['selected_adapter']

def run(root):
    prep=root/REL
    configs=prep/'configs'
    def launch(module,name,gpus='2',nproc=None):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus,'PYTHONUNBUFFERED':'1'}
        if nproc:
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={nproc}','-m',module]
        else: cmd=[sys.executable,'-m',module]
        cmd+=['--config',str(configs/f'{name}.json')]
        print(json.dumps(dict(command=cmd,cuda=gpus)),flush=True)
        identity=dict(command=cmd,cuda=gpus,started_at=datetime.now(timezone.utc).isoformat())
        with subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL) as child:
            identity.update(pid=child.pid,proc_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
            write_json(prep/f'output/{name}-process.json',identity)
            code=child.wait()
        identity.update(exit_code=code,ended_at=datetime.now(timezone.utc).isoformat())
        write_json(prep/f'output/{name}-process.json',identity)
        if code: raise RuntimeError(f'{name} exited {code}; no retry')
    def evaluation(name,adapter=None):
        cfg=read(configs/f'{name}.json')
        if adapter: cfg['adapter_path']=adapter
        write_json(configs/f'{name}-resolved.json',cfg)
        launch('rl.three_branch_pilot',name+'-resolved')
    (prep/'output').mkdir()
    write_json(prep/'output/status.json',dict(training='pending',evaluation='pending',content_acceptance='pending'))
    launch('rl.sft','gpu-preflight','2,3',2)
    assert read(prep/'output/gpu-preflight/training_result.json')['global_step']==1
    evaluation('base-val')
    launch('rl.sft','sft','2',1)
    assert read(prep/'output/sft/training_result.json')['global_step']==9
    for step in [3,6,9]: evaluation(f'sft-val-{step}',str(REL/f'output/sft/checkpoint-{step}'))
    sft=select(root,'sft',[3,6,9])
    cfg=read(configs/'rl.json'); cfg['sft_adapter']=sft
    write_json(configs/'rl-resolved.json',cfg)
    launch('rl.grpo','rl-resolved','2,3',2)
    assert read(prep/'output/rl/training_result.json')['global_step']==22
    groups=[read(p) for p in (prep/'output/rl').glob('group-step*-rank0.json')]
    assert len(groups)==11 and {g['task_ids'][0] for g in groups}==set(cfg['task_ids'])
    assert all(not g['infrastructure_failure'] for g in groups)
    write_json(prep/'output/rl_group_summary.json',dict(attempted_groups=11,accepted_groups=11,
        signal_groups=sum(g['nonzero_advantage'] for g in groups),optimizer_steps=22,
        effective_model_tokens=sum(g['effective_model_tokens'] for g in groups)))
    for step in [8,16,22]: evaluation(f'rl-val-{step}',str(REL/f'output/rl/checkpoint-{step}'))
    rl=select(root,'rl',[8,16,22])
    write_json(prep/'output/status.json',dict(training='complete',evaluation='pending',content_acceptance='pending'))
    write_json(prep/'output/model_lock.json',dict(base='original revision',sft=sft,rl=rl))
    for name,adapter in [('base',None),('sft',sft),('rl',rl)]: evaluation(name+'-test',adapter)
    tables={name:{r['task_id']:r for r in read(prep/f'output/{name}-test/records.json')} for name in ['base','sft','rl']}
    paired={}
    for before,after in [('base','sft'),('sft','rl'),('base','rl')]:
        changes={}
        for metric in ['complete_success','severe_error','protocol_error']:
            changes[metric]={
                'added':[tid for tid in ids_for_test(tables) if not tables[before][tid]['grade'][metric] and tables[after][tid]['grade'][metric]],
                'lost':[tid for tid in ids_for_test(tables) if tables[before][tid]['grade'][metric] and not tables[after][tid]['grade'][metric]]}
        paired[before+'-to-'+after]=changes
    write_json(prep/'output/paired_test.json',paired)
    # Content review is a separate human/AI activity after raw machine execution.
    write_json(prep/'output/status.json',dict(training='complete',evaluation='complete',content_acceptance='pending'))

def ids_for_test(tables):
    return sorted(tables['base'])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config'); parser.add_argument('--run',action='store_true')
    args=parser.parse_args(); root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    if args.run: run(root)
    else: asyncio.run(evaluate(root,read(args.config)))

if __name__=='__main__': main()
