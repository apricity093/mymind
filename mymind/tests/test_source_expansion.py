"""仅验证本轮新增的两遍调度，不调用服务。"""
import asyncio
from pathlib import Path
import pytest
from rl import teacher_source_expansion as module
from rl.teacher_budget import CollectionBudget

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/success-pool-v1/source-expansion-v1'
if not PREP.exists():
    pytest.skip('需要单独保留的 source-expansion-v1 实验原件', allow_module_level=True)


@pytest.mark.parametrize('stop', [None, 'second_pass_failure', 'timeout'])
def test_two_pass_filter_and_stop(tmp_path, monkeypatch, stop):
    cfg=module.read(PREP/'configuration.json');# 调度测试读取固定输入；部署预检的源码快照比较不适用于当前源码。
    tasks=module.load_public_tasks(ROOT/cfg['tasks'])
    standards=module.load_private_standards(ROOT/cfg['standards'])
    tasks={tid:tasks[tid] for tid in cfg['task_ids']}
    standards={tid:standards[tid] for tid in cfg['task_ids']}
    clauses=module.load_policy_support(ROOT/cfg['policy_support'])
    module.validate_task_pairs(tasks,standards,clauses)
    metadata=module.read(PREP/'metadata/task_manifest.json')
    references={r['task_id']:r['result'] for r in module.read(PREP/'checks/offline_references.json')}
    from benchmarks.scripts.balanced_policy_design import fixture
    calls=[];counts={};failed=cfg['task_ids'][::3]
    async def fake(task,generation,environment,**kwargs):
        tid=task.task_id;counts[tid]=counts.get(tid,0)+1;calls.append(tid)
        assert kwargs['public_guidance']==cfg['public_teacher_guidance']
        assert task.messages()==tasks[tid].messages()
        if stop=='timeout':
            kwargs['on_progress'](dict(task_id=tid,partial=True));raise asyncio.TimeoutError
        infrastructure=stop=='second_pass_failure' and counts[tid]==2
        raw=references[tid] if counts[tid]==2 or tid not in failed else dict(disposition='answer',facts=[],reasons=[],next_steps=[],missing_info=[])
        return dict(task_id=tid,raw_result=raw,trajectory=fixture(task).model_dump(),messages=task.messages(),
            generations=[dict(output_token_budget_basis='provider_completion_tokens')],tool_diagnostics=[],
            runtime_error=dict(kind='generation_failure') if infrastructure else None,infrastructure_failure=infrastructure)
    monkeypatch.setattr(module,'run_task',fake)
    budget=CollectionBudget(tmp_path,cfg['collection_limits'],'main')
    rows=asyncio.run(module.collect_candidates(cfg,tmp_path,tasks,standards,clauses,None,None,budget,metadata))
    summary=module.read(tmp_path/'summary.json')
    if stop=='timeout':
        assert calls==cfg['task_ids'][:1] and not rows
        assert summary['incomplete'][0]['candidate']==1
        assert summary['stopped']=='trajectory_timeout' and list(tmp_path.glob('*.partial.json'))
    else:
        expected=cfg['task_ids']+(failed[:1] if stop else failed)
        assert calls==expected
        assert all(count<=2 for count in counts.values())
        assert [r['task_id'] for r in rows if r['candidate']==2]==(failed[:1] if stop else failed)
        assert summary['stopped']==('generation_failure' if stop else 'fixed_two_passes_completed')
        if not stop:assert summary['score_passes']==38
    assert summary['qualified_successes'] is None and summary['formal_pool_successes']==0
    assert not list(tmp_path.glob('*accepted*'))
