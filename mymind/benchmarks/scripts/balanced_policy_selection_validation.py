"""Focused regression evidence for the two concrete job-selection forms."""
from copy import deepcopy

from agents.customer_policy.protocol import PublicTask
from rl.tasks import PrivateStandard
from benchmarks.scripts.balanced_policy_data_validation import public_selection_audit
from benchmarks.scripts.balanced_policy_events import materialize_event
from benchmarks.scripts.balanced_policy_content import load, dump, SPLITS


def check(data, logs):
    before=logs/'before-selection-r4'
    planned=load(data/'design/candidate_manifest.json')
    old={t['task_id']:PublicTask.model_validate(t) for s in SPLITS for t in load(before/f'public/{s}.json')}
    standards={t['task_id']:PrivateStandard.model_validate(t) for s in SPLITS for t in load(before/f'private/{s}.json')}
    conflicts=[dict(task_id=tid,issues=public_selection_audit(t,standards[tid])) for tid,t in old.items()]
    conflicts=[r for r in conflicts if r['issues']]
    assert {r['task_id'] for r in conflicts}=={'balanced-P01-08','balanced-P01-16'},conflicts
    current={r['task_id']:materialize_event(r) for r in planned}
    changed=[tid for tid,t in current.items() if t!=old[tid]]
    assert set(changed)=={'balanced-P01-08','balanced-P01-16'},changed
    assert not [(tid,public_selection_audit(t,standards[tid])) for tid,t in current.items()
                if public_selection_audit(t,standards[tid])]
    probes=[]
    for tid in ['balanced-P01-08','balanced-P02-08','balanced-P01-16','balanced-P02-16',
                'balanced-A01-08','balanced-A02-16','balanced-P06-03','balanced-P06-04']:
        issues=public_selection_audit(current[tid],standards[tid])
        probes.append(dict(task_id=tid,expected='passed',issues=issues))
        assert not issues
    # Retain a concrete mismatched join as a rejected counterexample.
    bad=deepcopy(current['balanced-P01-08'])
    import json
    origin=json.loads(bad.records[0].content)
    origin['目标路径']['Q1']=['夜批生成作业单','生成作业调用']
    bad.records[0].content=json.dumps(origin,ensure_ascii=False)
    issues=public_selection_audit(bad)
    assert '当前选择与目标路径不一致' in issues
    probes.append(dict(name='preview_selected_but_generation_path',expected='rejected',issues=issues))
    dump(logs/'selection_r4_probes.json',dict(before_checked=600,after_checked=600,
        original_conflicts=conflicts,changed_tasks=changed,probes=probes,
        scope='结构化选择/用途/关联直接检查；自然语言经同一执行助手逐族核对，不是通用解析或独立复核。'))
    print('selection regression passed: 2 original conflicts rejected; 2 corrected tasks; 598 public tasks unchanged')
