"""第二候选的固定题单、独立输入与停止行为；无真实调用。"""
import asyncio
from pathlib import Path
import pytest
from rl import teacher_mainline as module
from rl.teacher_budget import CollectionBudget, CollectionStopped

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/success-pool-v1/second-candidates-v1'
if not PREP.exists():
    pytest.skip('需要单独保留的 second-candidates-v1 实验原件', allow_module_level=True)


@pytest.mark.parametrize('stop',[None,'failure','timeout','budget'])
def test_second_round(tmp_path,monkeypatch,stop):
    cfg=module.read(PREP/'configuration.json')
    # 调度测试读取固定输入；部署预检的源码快照比较不适用于当前源码。
    tasks=module.load_public_tasks(ROOT/cfg['tasks'])
    standards=module.load_private_standards(ROOT/cfg['standards'])
    tasks={tid:tasks[tid] for tid in cfg['task_ids']}
    standards={tid:standards[tid] for tid in cfg['task_ids']}
    clauses=module.load_policy_support(ROOT/cfg['policy_support'])
    module.validate_task_pairs(tasks,standards,clauses)
    meta=module.read(PREP/'metadata/task_manifest.json')
    prior=module.read(PREP/'prior_acceptance.json')
    old=module.read(ROOT/cfg['prior_output_dir']/'configuration.json')
    assert cfg['task_ids']==[t for t in prior['original_order'] if t not in prior['qualified_task_ids']]
    assert len(tasks)==26 and len(set(tasks))==26
    assert cfg['output_dir']!=cfg['prior_output_dir']
    assert cfg['collection_limits']['total']['usd']==1
    for key in ['tasks','standards','policy_support']:
        assert (ROOT/cfg[key]).read_bytes()==(ROOT/old[key]).read_bytes()
    calls=[]
    budget=CollectionBudget(tmp_path,cfg['collection_limits'],'main')
    async def fake(task,generation,environment,**kwargs):
        calls.append(task.task_id)
        assert budget.request_context['candidate']==2
        assert budget.request_context['prior_original'].endswith('-candidate-1.json')
        original=module.load_public_tasks(ROOT/old['tasks'])[task.task_id]
        assert task.messages()==original.messages()
        assert kwargs['public_guidance']==old['public_teacher_guidance']
        if stop=='timeout':
            kwargs['on_progress'](dict(partial=True))
            raise asyncio.TimeoutError
        if stop=='budget':
            budget.stop('global_limit:total')
            raise CollectionStopped('global_limit:total')
        return dict(task_id=task.task_id,raw_result=dict(disposition='answer',facts=[],reasons=[],next_steps=[],missing_info=[]),
            trajectory=dict(task_id=task.task_id,materials=[m.model_dump() for m in task.materials()],searches=[]),
            messages=task.messages(),generations=[dict(output_token_budget_basis='provider_completion_tokens')],
            tool_diagnostics=[],runtime_error=dict(kind='generation_failure') if stop=='failure' else None,
            infrastructure_failure=stop=='failure')
    monkeypatch.setattr(module,'run_task',fake)
    rows=asyncio.run(module.collect_candidates(cfg,tmp_path,tasks,standards,clauses,None,None,budget,meta))
    summary=module.read(tmp_path/'summary.json')
    assert calls==cfg['task_ids'][:1 if stop else 26]
    assert all(r['candidate']==2 and 'candidate-2.json' in r['file'] for r in rows)
    assert not list(tmp_path.glob('*candidate-1*'))
    assert summary['prior_qualified_successes']==10 and summary['prior_failed_denominator']==26
    assert summary['second_candidate_added_successes'] is None
    assert summary['cumulative_qualified_successes'] is None
    assert summary['round_attempts']==len(calls)
    assert summary['stopped']=={None:'fixed_26_completed','failure':'generation_failure',
        'timeout':'trajectory_timeout','budget':'global_limit:total'}[stop]
    if stop in ('timeout','budget'):
        assert summary['incomplete_task_ids']==cfg['task_ids'][:1]
