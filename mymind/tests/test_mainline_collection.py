"""新主采集调度和故障停止；只使用模拟轨迹。"""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from rl import teacher_mainline as module
from rl.teacher_budget import CollectionBudget

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/success-pool-v1/mainline-candidates-v1'
if not PREP.exists():
    pytest.skip('需要单独保留的 mainline-candidates-v1 实验原件', allow_module_level=True)


@pytest.mark.parametrize('fail_at',[None,2])
def test_fixed_one_attempt_schedule_and_failure_stop(tmp_path,monkeypatch,fail_at):
    cfg=module.read(PREP/'configuration.json'); # 调度测试读取固定输入；部署预检的源码快照比较不适用于当前源码。
    tasks=module.load_public_tasks(ROOT/cfg['tasks'])
    standards=module.load_private_standards(ROOT/cfg['standards'])
    tasks={tid:tasks[tid] for tid in cfg['task_ids']}
    standards={tid:standards[tid] for tid in cfg['task_ids']}
    clauses=module.load_policy_support(ROOT/cfg['policy_support'])
    module.validate_task_pairs(tasks,standards,clauses)
    calls=[]
    async def fake(task,generation,environment,**kwargs):
        calls.append(task.task_id); fail=len(calls)==fail_at
        return dict(task_id=task.task_id,raw_result=dict(disposition='answer',facts=[],reasons=[],next_steps=[],missing_info=[]),
            trajectory=dict(task_id=task.task_id,materials=[m.model_dump() for m in task.materials()],searches=[]),
            messages=task.messages(),generations=[dict(output_token_budget_basis='provider_completion_tokens')],
            tool_diagnostics=[],runtime_error=dict(kind='generation_failure') if fail else None,infrastructure_failure=fail)
    monkeypatch.setattr(module,'run_task',fake)
    budget=CollectionBudget(tmp_path,cfg['collection_limits'],'main')
    rows=asyncio.run(module.collect_candidates(cfg,tmp_path,tasks,standards,clauses,SimpleNamespace(),None,budget,[]))
    count=fail_at or 36
    assert calls==cfg['task_ids'][:count] and len(rows)==count
    summary=module.read(tmp_path/'summary.json')
    assert len(summary['unstarted'])==36-count and summary['qualified_successes'] is None
    assert summary['formal_pool_successes']==0
    assert summary['stopped']==('generation_failure' if fail_at else 'fixed_36_completed')
    assert not (tmp_path/'accepted_trajectories.json').exists()


def test_timeout_retains_attempt_and_partial(tmp_path,monkeypatch):
    cfg=module.read(PREP/'configuration.json'); # 调度测试读取固定输入；部署预检的源码快照比较不适用于当前源码。
    tasks=module.load_public_tasks(ROOT/cfg['tasks'])
    standards=module.load_private_standards(ROOT/cfg['standards'])
    tasks={tid:tasks[tid] for tid in cfg['task_ids']}
    standards={tid:standards[tid] for tid in cfg['task_ids']}
    clauses=module.load_policy_support(ROOT/cfg['policy_support'])
    module.validate_task_pairs(tasks,standards,clauses)
    async def fake(task,generation,environment,**kwargs):
        kwargs['on_progress'](dict(task_id=task.task_id,partial=True))
        raise asyncio.TimeoutError
    monkeypatch.setattr(module,'run_task',fake)
    budget=CollectionBudget(tmp_path,cfg['collection_limits'],'main')
    asyncio.run(module.collect_candidates(cfg,tmp_path,tasks,standards,clauses,None,None,budget,[]))
    summary=module.read(tmp_path/'summary.json')
    assert summary['attempted_task_ids']==cfg['task_ids'][:1]
    assert summary['incomplete_task_ids']==cfg['task_ids'][:1]
    assert summary['stopped']=='trajectory_timeout' and list(tmp_path.glob('*.partial.json'))
