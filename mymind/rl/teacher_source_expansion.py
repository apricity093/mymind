"""固定新来源清单的两遍采集；默认只做本地预检。"""
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
from rl.teacher_budget import CollectionStopped
from rl.teacher_collection import collect_real, read
from rl.teacher_mainline import check_index


def preflight(root, cfg):
    tasks=load_public_tasks(root/cfg['tasks']);standards=load_private_standards(root/cfg['standards'])
    clauses=load_policy_support(root/cfg['policy_support']);validate_task_pairs(tasks,standards,clauses)
    meta=read(root/cfg['frozen_dir']/'metadata/task_manifest.json')
    assert len(tasks)==38 and len({m['family_id'] for m in meta})==16
    assert list(tasks)==cfg['task_ids']==[m['task_id'] for m in meta]
    assert cfg['schedule']==[dict(sequence=i+1,task_id=t) for i,t in enumerate(tasks)]
    assert cfg['second_pass_order']==cfg['task_ids']
    assert cfg['candidates_per_task']==2 and cfg['concurrency']==1 and not cfg.get('resume_from')
    assert cfg['grading_revision']==GRADING_REVISION=='v17-policy-constant-equivalence'
    assert cfg['standards_revision']=='success-pool-evidence-v5'
    assert all(s.split=='candidate' for s in standards.values())
    assert all(m['split'] is None and m['formal_pool_eligible'] for m in meta)
    eligibility=read(root/cfg['frozen_dir']/'metadata/source_eligibility.json')
    excluded=set(eligibility['excluded_dev_families'])|{m['family_id'] for m in eligibility['sealed_families']}
    assert not excluded & {m['family_id'] for m in meta}
    baseline=read(root/cfg['baseline_configuration'])
    for key in ('generation','retrieval','embedding','budget','public_teacher_guidance','trajectory_wall_seconds'):
        assert cfg[key]==baseline[key],key
    assert cfg['collection_limits']['request']==baseline['collection_limits']['request']
    source_root=Path(__file__).resolve().parents[2]
    for name in (root/cfg['source_manifest']).read_text(encoding='utf-8').splitlines():
        assert (source_root/name).read_bytes()==(root/cfg['frozen_dir']/'source_snapshot'/name).read_bytes(),name
    return tasks,standards,clauses


async def collect_candidates(cfg, output, tasks, standards, clauses, generation, retrieval, budget, metadata):
    rows=[];attempted=[];passed=set();meta={m['task_id']:m for m in metadata}
    try:
        for candidate in (1,2):
            for step in cfg['schedule']:
                tid=step['task_id']
                if candidate==2 and tid in passed:continue
                budget.check()
                context=dict(**step,candidate=candidate,family_id=meta[tid]['family_id'])
                attempted.append(context);budget.request_context=context
                partial=output/f'{tid}-candidate-{candidate}.partial.json';started=time.monotonic()
                try:
                    # 每次独立run_task，不向第二遍传第一遍消息、回复、分数或私有答案。
                    row=await asyncio.wait_for(run_task(tasks[tid],generation,PolicyEnvironment(retrieval),
                        budget=cfg['budget'],public_guidance=cfg['public_teacher_guidance'],
                        on_progress=lambda state:write_json(partial,state)),
                        timeout=min(cfg['trajectory_wall_seconds'],budget.remaining_seconds()))
                except asyncio.TimeoutError:
                    budget.mark_inflight_unknown('trajectory_timeout');budget.stop('trajectory_timeout')
                    raise CollectionStopped('trajectory_timeout')
                budget.mark_inflight_unknown('trajectory_finished_without_response_usage')
                grade=grade_result(standards[tid],row['raw_result'],VisibleTrajectory.model_validate(row['trajectory']),clauses)
                row.update(**context,elapsed_s=time.monotonic()-started,grade=grade.to_dict(),
                    process_errors=process_errors(row),grading_revision=cfg['grading_revision'],standards_revision=cfg['standards_revision'])
                row['score_pass']=grade.complete_success and not row.get('runtime_error') and not row['infrastructure_failure'] and not row['process_errors']
                row['content_review']='pending; score_pass is not qualified success'
                path=output/f'{tid}-candidate-{candidate}.json';write_json(path,row)
                rows.append(dict(task_id=tid,candidate=candidate,file=path.name,score_pass=row['score_pass'],grade=row['grade'],process_errors=row['process_errors']))
                if row['score_pass']:passed.add(tid)
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
        budget.mark_inflight_unknown('unexpected_failure');budget.stop('unexpected_failure_requires_review')
        raise
    finally:
        completed={(r['task_id'],r['candidate']) for r in rows}
        write_json(output/'summary.json',dict(attempts=rows,attempted=attempted,
            unstarted=[t for t in cfg['task_ids'] if not any(a['task_id']==t for a in attempted)],
            incomplete=[a for a in attempted if (a['task_id'],a['candidate']) not in completed],
            score_passes=len(passed),qualified_successes=None,formal_pool_successes=0,
            stopped=budget.state['stop_reason'] or 'fixed_two_passes_completed',budget=budget.totals(),
            content_review='pending; first score/process pass stops attempts, later content rejection triggers no extra collection'))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--check-index',action='store_true')
    args=parser.parse_args();root=Path(os.environ['MYMIND_WORK_ROOT']).resolve();cfg=read(root/args.config)
    tasks,standards,clauses=preflight(root,cfg)
    if not args.execute:
        if args.check_index:asyncio.run(check_index(root,cfg))
        print('38 candidates, maximum two passes; v5/v17 preflight passed; 0 calls');return
    output=root/cfg['output_dir']
    if output.exists():raise ValueError('仅使用新目录；不恢复或重跑')
    from agents.customer_policy.configuration import load_configuration
    _,_,output=load_configuration(root/args.config)
    write_json(output/'configuration.json',cfg);write_json(output/'invocation.json',dict(argv=sys.argv))
    for key in ('tasks','standards','policy_support'):
        dest=output/'inputs'/key/Path(cfg[key]).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/cfg[key],dest)
    for key in ('task_manifest.json','source_families.json','source_eligibility.json'):
        dest=output/'inputs/metadata'/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/cfg['frozen_dir']/'metadata'/key,dest)
    for name in (root/cfg['source_manifest']).read_text(encoding='utf-8').splitlines():
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/cfg['frozen_dir']/'source_snapshot'/name,dest)
    asyncio.run(collect_real(root,cfg,output,tasks,standards,clauses,collector=collect_candidates))


if __name__=='__main__':main()
