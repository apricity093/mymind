"""用已验收的 v5/v17 重验两轮原件；仅本地评分和元数据容量计算。"""
from collections import Counter
from itertools import product
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.customer_policy.protocol import VisibleTrajectory
from rl.grading import grade_result, GRADING_REVISION
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    root = Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    base = root/'runs/success-pool-v1'
    out = base/'unified-pool-v5-v1'
    if (out/'success_pool.json').exists():
        raise ValueError('新版成功池已存在，不覆盖')
    prior = read(base/'fact-sources-v5/original_reply_comparison.json')
    for file in ['mymind/rl/grading.py', 'mymind/agents/customer_policy/protocol.py']:
        assert (root/file).read_bytes() == (base/'fact-sources-v5/source_snapshot'/file).read_bytes()
    cached = {(r['group'], r['file']): r for r in prior}
    scope = {r['task_id']: r for r in read(base/'expression-pair-v1/metadata/source_use.json')['source_scope']}
    meta = {r['task_id']: r for r in read(base/'mainline-candidates-v1/metadata/task_manifest.json')}
    rows, provenance = [], []
    for name, count in [('teacher-mainline-v1', 36), ('teacher-second-v1', 26)]:
        run = base/name
        cfg = read(run/'configuration.json')
        tasks = load_public_tasks(run/'inputs/tasks/candidates.json')
        standards_path = base/'fact-sources-v5'/name/'private/dev.json'
        standards = load_private_standards(standards_path)
        clauses = load_policy_support(run/'inputs/policy_support/policy_support.json')
        assert read(base/'fact-sources-v5'/name/'policy/policy_support.json') == read(run/'inputs/policy_support/policy_support.json')
        validate_task_pairs(tasks, standards, clauses)
        audit = {r['task_id']: r for r in read(run/'evidence_audit.json')['attempt_checks']}
        reviews = {r['task_id']: r for r in read(run/'content_acceptance.json')['content_reviews']}
        attempts = read(run/'summary.json')['attempts']
        assert len(attempts) == count
        provenance.append(dict(run=name, configuration=str((run/'configuration.json').relative_to(root)),
            original_standard=cfg['standards'], original_source=str((run/'source').relative_to(root)),
            original_score_audit=str((run/'evidence_audit.json').relative_to(root)),
            generation_standards_revision=read(run/attempts[0]['file'])['standards_revision']))
        for attempt in attempts:
            row = read(run/attempt['file']); tid = row['task_id']
            tr = VisibleTrajectory.model_validate(row['trajectory'])
            assert [m.model_dump() for m in tasks[tid].materials()] == row['trajectory']['materials']
            assert standards[tid].materials == {m.reference_id:m.content for m in tasks[tid].materials() if m.reference_id.startswith('user:')}
            check = audit[tid]
            assert all(check[k] for k in ['original_score_reproduced', 'raw_reply_unchanged', 'exact_message_chain'])
            assert row['grade'] == attempt['grade']
            cache = cached.get((name, attempt['file']))
            if cache:
                assert cache['original'] == row['grade']
                unified = cache['candidate']
            else:
                unified = grade_result(standards[tid], row['raw_result'], tr, clauses).to_dict()
            process_ok = not row['process_errors'] and not row['runtime_error'] and not row['infrastructure_failure']
            review = reviews.get(tid)
            # 新通过项依据实际原件深读，其他通过项复用已完成的逐项内容审阅。
            if unified['complete_success'] and not row['grade']['complete_success']:
                assert tid == 'mainline-balanced-P04-18' and row['candidate'] == 1
                review = dict(qualified=True, review='Q1通过业务路径关联换版组调用；user:3明确403、凭据有效、批量报表生成、分组发布/production、出口198.51.100.108、配置权与出口登记true、功能状态明确未知。12项事实逐项与材料一致。search:1:2支持export、白名单及平台开通要求；search:1:1支持范围、平台主体及缺状态待补。permission_state_unknown理由同时引用两段实际政策和user:3；当前request_info、仅补feature_enabled、无当前开通/配置动作正确。旧误判仅为两项直接状态被强加config_scope，无新增例外或盲点。',
                    reviewer='本次AI读取原公开材料、终止回复及全部实际引用；非人工金标或独立盲审')
            accepted = bool(unified['complete_success'] and process_ok and review and review['qualified'])
            s = scope[tid]
            rows.append(dict(task_id=tid, family_id=s['family_id'], branch=meta[tid]['stratum_id'],
                run=name, candidate=row['candidate'], original=str((run/attempt['file']).relative_to(root)),
                original_grade=row['grade'], unified_grade=unified,
                scoring_evidence='reused_fact_sources_v5' if cache else 'completed_v5_scoring',
                source_use=s['source_use'], source_eligible=s['formal_pool_eligible'], process_ok=process_ok,
                content_acceptance=review if accepted else dict(qualified=False, reason='统一评分未通过；未开展提示/教师开发性失败分析' if not unified['complete_success'] else '过程或内容验收未通过'),
                content_accepted=accepted, standards_revision='success-pool-evidence-v5', grading_revision=GRADING_REVISION,
                standard=str(standards_path.relative_to(root)), original_standards_revision=row['standards_revision'],
                original_grading_revision=row['grading_revision'], in_pool=False))
    assert len(rows) == 62
    grouped = {}
    for r in rows:
        grouped.setdefault(r['task_id'], []).append(r)
    assert len(grouped) == 36 and all(len(v) in [1, 2] for v in grouped.values())
    tasks_out = []
    for tid, attempts in grouped.items():
        attempts.sort(key=lambda r:r['candidate'])
        good = next((r for r in attempts if r['source_eligible'] and r['content_accepted']), None)
        if good:
            good['in_pool'] = True
        tasks_out.append(dict(task_id=tid, family_id=attempts[0]['family_id'],
            attempts=[dict(original=r['original'], candidate=r['candidate'], original_score=r['original_grade']['reward'], unified_score=r['unified_grade']['reward'], in_pool=r['in_pool']) for r in attempts],
            selected_original=good['original'] if good else None))
    pool = [r for r in rows if r['in_pool']]
    families = sorted({r['family_id'] for r in pool})
    branches = {}
    for b in ['A01','B01','F02','M02','P04','R01']:
        p = [r for r in pool if r['branch']==b]
        fam = sorted({r['family_id'] for r in p})
        branches[b] = dict(qualified_tasks=len(p), independent_success_families=fam,
            minimum_family_gap=max(0,3-len(fam)), selected_by_candidate=dict(Counter(str(r['candidate']) for r in p)),
            attempts_by_round=dict(Counter(r['run'] for r in rows if r['branch']==b)))
    # 枚举整族着色仅计算覆盖，不输出或启用正式划分。
    masks = {f:{r['branch'] for r in pool if r['family_id']==f} for f in families}
    target = set(branches)
    feasible, best = 0, 0
    for assignment in product(range(3), repeat=len(families)):
        coverage = [set() for _ in range(3)]
        for f, part in zip(families, assignment):
            coverage[part].update(masks[f])
        best = max(best, len(set.intersection(*coverage)))
        feasible += int(all(c==target for c in coverage))
    capacity = dict(branches=branches, eligible_successes=len(pool), successful_families=len(families),
        joint_partitions_checked=3**len(families), joint_six_branch_partitions=feasible,
        maximum_branches_common_to_three_sets=best, formal_split_created=False,
        training_readiness=False, excluded_attempts=sum(not r['source_eligible'] for r in rows),
        historical_successes_lost=[r['original'] for r in rows if r['original_grade']['complete_success'] and not r['content_accepted']],
        added=[r['original'] for r in pool if not r['original_grade']['complete_success']])
    eligibility = read(base/'mainline-candidates-v1/metadata/source_eligibility.json')
    available = sorted(set(eligibility['eligible_old_train']['families'])-set(eligibility['selected_old_train']))
    oldmeta = read(root/'data/agentic_rl/balanced-policy-v1/metadata/task_manifest.json')
    inventory = [r for r in oldmeta if r['split']=='train' and r['family_id'] in available and r['stratum_id'] in branches]
    write(out/'available_sources.json', dict(families=available, tasks=inventory,
        provenance='source_eligibility.json + remaining_capacity.json + old train manifest only; no old test public/private materials',
        historical_access=read(base/'source-dev-v1/source_selection.json')['access_limit'],
        excluded_dev=eligibility['excluded_dev_families']+read(base/'expression-pair-v1/metadata/source_use.json')['dev_families'],
        sealed=eligibility['sealed_families']))
    write(out/'attempts.json', rows)
    write(out/'task_decisions.json', tasks_out)
    write(out/'success_pool.json', dict(version='unified-pool-v5-v1', status='content accepted; no formal splits',
        historical_successes=16, previous_eligible_successes=14, tasks=pool))
    write(out/'capacity.json', capacity)
    write(out/'validation.json', dict(status='passed', calls=0, originals=62, tasks=36,
        v5_reused=sum(r['scoring_evidence']=='reused_fact_sources_v5' for r in rows),
        v5_completed=sum(r['scoring_evidence']=='completed_v5_scoring' for r in rows),
        original_audit_reused=62, contract_checks_reused=str((base/'fact-sources-v5/validation.json').relative_to(root)),
        contract_passed=29, earliest_qualified_per_task=True, raw_reply_and_input_unchanged=True,
        generation_provenance=provenance, generation_configuration_claim='各批原协议、v4标准、源码、预算与调度分别保留；统一重评不等于同配置生成'))
    print(json.dumps(capacity, ensure_ascii=False))


if __name__ == '__main__':
    main()
