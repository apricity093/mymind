"""固定清单的有限主采集；默认零调用预检，执行需下一次明确授权。"""
import argparse
import asyncio
import os
from pathlib import Path
import shutil
import sys
import time

from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.protocol import VisibleTrajectory
from agents.customer_policy.runner import run_task
from rl.evaluate import process_errors
from rl.grading import GRADING_REVISION, grade_result
from rl.runtime import write_json
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs
from rl.teacher_collection import collect_real, read
from rl.teacher_budget import CollectionStopped


def preflight(root, cfg):
    tasks=load_public_tasks(root/cfg['tasks']); standards=load_private_standards(root/cfg['standards'])
    clauses=load_policy_support(root/cfg['policy_support']); validate_task_pairs(tasks,standards,clauses)
    meta=read(root/cfg['frozen_dir']/'metadata/task_manifest.json')
    candidate=cfg.get('candidate',1)
    assert candidate in (1,2)
    if candidate==2:
        tasks={tid:tasks[tid] for tid in cfg['task_ids']}
        standards={tid:standards[tid] for tid in cfg['task_ids']}
        links=read(root/cfg['candidate_manifest'])
        assert len(tasks)==26 and [m['task_id'] for m in links]==list(tasks)
        assert all(m['candidate']==2 and m['prior_candidate']==1 for m in links)
        prior=read(root/cfg['frozen_dir']/'prior_acceptance.json')
        assert cfg['task_ids']==[t for t in prior['original_order'] if t not in prior['qualified_task_ids']]
        assert cfg['prior_qualified_successes']==10
        assert cfg['output_dir']!=cfg['prior_output_dir']
    else:
        assert len(tasks)==36
    assert list(tasks)==cfg['task_ids']==[m['task_id'] for m in meta]
    assert cfg['schedule']==[dict(sequence=i+1,task_id=t) for i,t in enumerate(tasks)]
    assert cfg['candidates_per_task']==cfg['concurrency']==1 and not cfg.get('resume_from')
    assert cfg['grading_revision']==GRADING_REVISION=='v17-policy-constant-equivalence'
    assert cfg['standards_revision']=='success-pool-evidence-v4'
    assert all(s.split=='candidate' for s in standards.values())
    assert all(m['split'] is None and m['formal_pool_eligible'] for m in meta)
    assert len({m['family_id'] for m in meta})==(11 if candidate==2 else 12)
    baseline=read(root/cfg['baseline_configuration'])
    for key in ('generation','retrieval','embedding','budget','public_teacher_guidance','trajectory_wall_seconds'):
        assert cfg[key]==baseline[key],key
    assert cfg['collection_limits']['request']==baseline['collection_limits']['request']
    source_root=Path(__file__).resolve().parents[2]
    for name in (root/cfg['source_manifest']).read_text(encoding='utf-8').splitlines():
        assert (source_root/name).read_bytes()==(root/cfg['frozen_dir']/'source_snapshot'/name).read_bytes(),name
    return tasks,standards,clauses


async def collect_candidates(cfg, output, tasks, standards, clauses, generation, retrieval, budget, metadata):
    rows=[]; attempted=[]; candidate=cfg.get('candidate',1)
    links={m['task_id']:m for m in metadata} if candidate==2 else {}
    try:
        for step in cfg['schedule']:
            budget.check(); tid=step['task_id']; attempted.append(tid)
            budget.request_context={**step,'candidate':candidate}
            if candidate==2:
                budget.request_context.update(family_id=links[tid]['family_id'],prior_original=links[tid]['prior_original'])
            partial=output/f'{tid}-candidate-{candidate}.partial.json'; started=time.monotonic()
            try:
                row=await asyncio.wait_for(run_task(tasks[tid],generation,PolicyEnvironment(retrieval),
                    budget=cfg['budget'],public_guidance=cfg['public_teacher_guidance'],
                    on_progress=lambda state:write_json(partial,state)),
                    timeout=min(cfg['trajectory_wall_seconds'],budget.remaining_seconds()))
            except asyncio.TimeoutError:
                budget.mark_inflight_unknown('trajectory_timeout'); budget.stop('trajectory_timeout')
                break  # 已产生partial与HTTP账本保留；无完整输出不评分。
            budget.mark_inflight_unknown('trajectory_finished_without_response_usage')
            grade=grade_result(standards[tid],row['raw_result'],VisibleTrajectory.model_validate(row['trajectory']),clauses)
            row.update(**step,candidate=candidate,elapsed_s=time.monotonic()-started,grade=grade.to_dict(),
                process_errors=process_errors(row),grading_revision=cfg['grading_revision'],standards_revision=cfg['standards_revision'])
            row['score_pass']=grade.complete_success and not row.get('runtime_error') and not row['infrastructure_failure'] and not row['process_errors']
            row['content_review']='pending; score_pass is not qualified success'
            if candidate==2:
                row.update(family_id=links[tid]['family_id'],prior_original=links[tid]['prior_original'])
            path=output/f'{tid}-candidate-{candidate}.json'; write_json(path,row)
            rows.append(dict(task_id=tid,candidate=candidate,file=path.name,grade=row['grade'],score_pass=row['score_pass'],
                             process_errors=row['process_errors'],runtime_error=row.get('runtime_error')))
            kind=(row.get('runtime_error') or {}).get('kind')
            if row['infrastructure_failure'] or kind in {'generation_truncated','budget_exhausted','generation_failure'}:
                budget.stop(kind or 'infrastructure_failure')
            elif any(g.get('output_token_budget_basis')!='provider_completion_tokens' for g in row['generations']):
                budget.stop('teacher_completion_usage_missing_or_invalid')
            elif any(e['accounting_state']!='settled' for e in budget.state['requests']):
                budget.stop('unsettled_request_requires_review')
            budget.check()
    except CollectionStopped:
        pass
    except Exception:
        budget.mark_inflight_unknown('unexpected_failure'); budget.stop('unexpected_failure_requires_review')
        raise
    finally:
        write_json(output/'summary.json',dict(attempts=rows,attempted_task_ids=attempted,
            unstarted=[t for t in cfg['task_ids'] if t not in attempted],
            incomplete_task_ids=[t for t in attempted if t not in {r['task_id'] for r in rows}],
            score_passes=sum(r['score_pass'] for r in rows),qualified_successes=None,formal_pool_successes=0,
            stopped=budget.state['stop_reason'] or f'fixed_{len(cfg["task_ids"])}_completed',budget=budget.totals(),
            candidate=candidate,round_attempts=len(attempted),
            prior_failed_denominator=26 if candidate==2 else None,
            prior_qualified_successes=cfg.get('prior_qualified_successes',0),
            second_candidate_added_successes=None,cumulative_qualified_successes=None,
            content_review='pending; no teacher or student dataset exported'))
    return rows


async def check_index(root, cfg):
    from agents.customer_policy.configuration import rerank_gateway
    from agents.customer_policy.retrieval import build_retrieval
    from rl.teacher_collection import DeepSeekGeneration
    gateway = rerank_gateway(cfg['retrieval'])
    generation = DeepSeekGeneration(os.environ['DEEPSEEK_API_KEY'], cfg['generation'])
    retrieval = None
    try:
        retrieval = build_retrieval(root, cfg['retrieval'], gateway, prepared_only=True)
        embedding = retrieval.knowledge.embedding
        expected_embedding = cfg['embedding']
        assert (embedding.settings.model, embedding.settings.dimensions, embedding.base_url) == (
            expected_embedding['model'], expected_embedding['dimensions'], expected_embedding['base_url'])
        assert retrieval.knowledge._main_complete and retrieval.knowledge._main.count()
        text = (root/cfg['frozen_dir']/'policy/knowledge.md').read_text(encoding='utf-8')
        split = text.index('### 接口认证')
        assert {r['source_id']: r['content'] for r in retrieval.knowledge._rows()} == {
            'billing-policy': text[:split], 'technical-policy': text[split:]}
        print('Existing index/configuration/policy and clients ready; no HTTP dispatched')
    finally:
        await generation.close()
        await gateway.client.close()
        if retrieval:
            retrieval.knowledge.close()
            retrieval.knowledge.embedding.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check-index',action='store_true'); args=parser.parse_args()
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve(); cfg=read(root/args.config)
    tasks,standards,clauses=preflight(root,cfg)
    if not args.execute:
        if args.check_index:
            asyncio.run(check_index(root, cfg))
        print(f'{len(tasks)} candidates / candidate {cfg.get("candidate",1)}; v4/v17 preflight passed; 0 calls'); return
    output=root/cfg['output_dir']
    if output.exists(): raise ValueError('仅使用新目录；不自动恢复或重跑')
    from agents.customer_policy.configuration import load_configuration
    _,_,output=load_configuration(root/args.config)
    write_json(output/'configuration.json',cfg); write_json(output/'invocation.json',dict(argv=sys.argv))
    for key in ('tasks','standards','policy_support'):
        dest=output/'inputs'/key/Path(cfg[key]).name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/cfg[key],dest)
    if cfg.get('candidate')==2:
        shutil.copy2(root/cfg['candidate_manifest'],output/'inputs'/'candidate_manifest.json')
        shutil.copy2(root/cfg['frozen_dir']/'prior_acceptance.json',output/'inputs'/'prior_acceptance.json')
    for name in (root/cfg['source_manifest']).read_text(encoding='utf-8').splitlines():
        dest=output/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/cfg['frozen_dir']/'source_snapshot'/name,dest)
    asyncio.run(collect_real(root,cfg,output,tasks,standards,clauses,collector=collect_candidates))


if __name__=='__main__': main()
