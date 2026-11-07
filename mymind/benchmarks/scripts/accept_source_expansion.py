"""合并已经逐条阅读并核对原件的新来源成功项；不修改采集原件。"""
from collections import Counter
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    base=root/'runs/success-pool-v1'
    prep=base/'source-expansion-v1'
    run=base/'teacher-source-expansion-v1'
    summary=read(run/'summary.json')
    audit=read(run/'evidence_audit.json')
    reviews=read(prep/'checks/content_reviews.json')
    meta={r['task_id']:r for r in read(run/'inputs/metadata/task_manifest.json')}
    passed={(r['task_id'],r['candidate']) for r in summary['attempts'] if r['score_pass']}
    assert passed=={(r['task_id'],r['candidate']) for r in reviews}
    assert len(passed)==len(reviews)
    assert audit['fixed_two_pass_schedule'] and len(audit['attempt_checks'])==len(summary['attempts'])
    review_map={(r['task_id'],r['candidate']):r for r in reviews}
    rows=[]
    for attempt in summary['attempts']:
        row=read(run/attempt['file']); tid=row['task_id']; m=meta[tid]
        review=review_map.get((tid,row['candidate']))
        accepted=bool(row['score_pass'] and m['formal_pool_eligible'] and review and review['qualified'])
        rows.append(dict(task_id=tid,family_id=m['family_id'],branch=m['stratum_id'],
            candidate=row['candidate'],original=(run/attempt['file']).relative_to(root).as_posix(),
            original_grade=row['grade'],score_pass=row['score_pass'],process_ok=not row['process_errors'] and not row['runtime_error'] and not row['infrastructure_failure'],
            source_kind=m['source_kind'],source_use='fixed_collection_and_content_acceptance',source_eligible=m['formal_pool_eligible'],
            content_acceptance=review if review else dict(qualified=False,reason='机器筛选未通过；未开展开发性失败分析；早返回未检查项保持未知'),
            content_accepted=accepted,in_pool=accepted,standards_revision=row['standards_revision'],grading_revision=row['grading_revision']))
    added=[r for r in rows if r['in_pool']]
    old=read(base/'unified-pool-v5-v1/success_pool.json')['tasks']
    assert len(old)==15 and len({r['family_id'] for r in old})==8
    pool=old+added
    assert len({r['task_id'] for r in pool})==len(pool)
    assert not {r['family_id'] for r in old}&{r['family_id'] for r in added}
    branches={}
    for b in ('A01','B01','F02','M02','P04','R01'):
        selected=[r for r in pool if r['branch']==b]
        families=sorted({r['family_id'] for r in selected})
        branches[b]=dict(tasks=len(selected),families=families,family_count=len(families),
            added_tasks=sum(r['branch']==b for r in added),minimum_family_gap=max(0,3-len(families)))
    gaps={b:v['minimum_family_gap'] for b,v in branches.items() if v['minimum_family_gap']}
    assert gaps, '实际分支已达三族，需要进行联合划分求解'
    counts=dict(first_pass_attempts=sum(r['candidate']==1 for r in rows),second_pass_attempts=sum(r['candidate']==2 for r in rows),
        first_pass_score_passes=sum(r['candidate']==1 and r['score_pass'] for r in rows),second_pass_score_passes=sum(r['candidate']==2 and r['score_pass'] for r in rows),
        first_pass_qualified=sum(r['candidate']==1 for r in added),second_pass_added_qualified=sum(r['candidate']==2 for r in added),
        qualified=len(added),content_rejected=[r['original'] for r in rows if r['score_pass'] and not r['content_accepted']],
        failed_attempts=sum(not r['score_pass'] for r in rows),failed_tasks=38-len(added),
        partial_only=len(summary['incomplete']),unstarted=len(summary['unstarted']),
        retained_progress_partial_files=len(list(run.glob('*.partial.json'))))
    acceptance=dict(machine_collection='complete',content_acceptance='complete',**counts,
        content_reviews=reviews,evidence_audit='evidence_audit.json',stopped=summary['stopped'],
        totals=audit['totals'],early_return_unchecked='unknown; original grades retained',
        source_use='fixed collection and content acceptance; no development failure analysis',
        reviewer='AI数据构造审阅，非人工金标或独立盲审')
    version='unified-pool-v5-expansion-v1'
    out=base/version
    capacity=dict(eligible_successes=len(pool),successful_families=len({r['family_id'] for r in pool}),branches=branches,
        joint_six_branch_partition_feasible=False,proof='分支独立成功族不足3，不能分配到三个不重叠整族集合；此必要条件已否定联合可行性，无需枚举',
        family_gaps=gaps,joint_six_branch_minimum_denominator_across_three_sets=0,
        formal_split_created=False,training_readiness=False,training_started=False,
        recommendation='不启动六分支联合训练或选模，不自动追加采集；F02/M02/P04来源缺口见分支统计。若另行批准收窄范围，仅定位小规模先导，须另定数据、配置与GPU/选模预算。')
    write(run/'content_acceptance.json',acceptance)
    write(out/'attempts.json',rows)
    write(out/'success_pool.json',dict(version=version,status='content accepted; no formal splits',
        prior_pool='runs/success-pool-v1/unified-pool-v5-v1/success_pool.json',prior_tasks=15,added_tasks=len(added),tasks=pool))
    write(out/'capacity.json',capacity)
    write(out/'validation.json',dict(status='passed',real_calls=0,originals=len(rows),
        earliest_qualified_per_task=True,raw_replies_unchanged=True,prior_pool_preserved=True,
        content_reviews=len(reviews),generation_provenance='各批生成配置、原分、源码和原件分别保留；统一池不表示同配置生成',
        new_batch_evidence_audit='runs/success-pool-v1/teacher-source-expansion-v1/evidence_audit.json'))
    print(json.dumps(dict(counts=counts,capacity=capacity),ensure_ascii=False))


if __name__=='__main__':
    main()
