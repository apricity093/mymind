"""Build and verify the event-backed candidate and its recorded content reviews."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json

from agents.customer_policy.protocol import PublicTask, TerminalResult, VisibleTrajectory, FACT_TYPES
from rl.tasks import PrivateStandard, validate_task_pairs
from rl.grading import GRADING_REVISION, grade_result
from benchmarks.scripts.balanced_policy_design import profile_example, reference, fixture, dump, CLAUSES
from benchmarks.scripts.balanced_policy_candidates import classify, actual_subtype, SUBTYPES
from benchmarks.scripts.balanced_policy_events import DOSSIERS, materialize_event, read_event

REVISION='balanced-policy-v1-data-standards-r4-selection'
SPLITS=('train','validation','test')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def expectation_support(ex, evidence):
    policies=[g for g in ex['supports'] if not any(s.startswith('user:') for s in g)]
    a=ex['alternatives'][0]
    # These are policy definitions, independent of this event's state.
    if ex['section']=='facts' and a['type'] in {'timer_start','whitelist_required','whitelist_actor','enablement_actor'}:
        return policies
    return [[r] for r in evidence[a['target']]]+policies


def build(data):
    rows=load(data/'design/candidate_manifest.json')
    meta=[]; buckets={s:{'public':[],'private':[],'acceptance':[]} for s in SPLITS}
    for row in rows:
        if row['family_id'] not in DOSSIERS: continue
        task=materialize_event(row)
        states,evidence=read_event(task)
        sid,disposition=classify(task.role,states)
        sub=actual_subtype(sid,states)
        assert (sid,sub,disposition)==(row['stratum_id'],row['subtype_id'],row['expected_disposition']), task.task_id
        e=profile_example(sid,sub,states)
        e.expects=[ex for ex in e.expects if not (ex['section']=='missing_info' and
            all(states[a['target']].get(a['field'],{}).get('state')=='known' for a in ex['alternatives']))]
        for t,fields in states.items():
            for k,v in fields.items():
                if k not in FACT_TYPES or v['state']!='known': continue
                if k=='payment_channel' and v['value'] not in {'online_payment','bank_card','balance'}: continue
                if not any(x['section']=='facts' and any(a.get('target')==t and a.get('type')==k for a in x['alternatives']) for x in e.expects):
                    e.fact(t,k,v['value'],required=False)
        for ex in e.expects: ex['supports']=expectation_support(ex,evidence)
        standard=PrivateStandard(task_id=task.task_id,group_id=sid,split=row['candidate_split'],
            dispositions=[disposition],materials={m.reference_id:m.content for m in task.materials()},
            expectations=e.expects,severe_items=dict(e.severe),unprovided_user_facts=e.unprovided)
        b=buckets[row['candidate_split']]
        b['public'].append(task.model_dump()); b['private'].append(standard.model_dump())
        b['acceptance'].append(dict(task_id=task.task_id,source='同一执行助手AI离线参考，与标准同过程构造，非教师',result=reference(e),trajectory=fixture(task).model_dump()))
        meta.append(dict(task_id=task.task_id,family_id=row['family_id'],stratum_id=sid,subtype_id=sub,split=row['candidate_split'],
            role=task.role,style=row['style'],condition_states=states,evidence_dependencies=evidence,
            expected_disposition=disposition,standards_revision=REVISION,
            semantic_review='实际内容审阅结论见content_reviews.json；不由构造过程授予通过'))
    for split,b in buckets.items():
        for folder,items in b.items(): dump(data/f'{folder}/{split}.json',items)
    dump(data/'metadata/task_manifest.json',meta)
    public_by_id={t['task_id']:t for b in buckets.values() for t in b['public']}
    dump(data/'metadata/source_families.json',[
        dict(family_id=fid,split=next(r['candidate_split'] for r in rows if r['family_id']==fid),
             source='本会话AI编写合成材料',event=d,
             shared_facts='事件过程和可选业务记录共享；本次选择、目标路径和状态以各变体公开材料为准，不继承其他变体。',
             event_scope='event保存默认选择的事件；各反事实本次目标见counterfactuals.target_paths及对应公开材料。',
             counterfactuals=[dict(task_id=m['task_id'],stratum_id=m['stratum_id'],states=m['condition_states'],
                target_paths=json.loads(public_by_id[m['task_id']]['records'][0]['content'])['目标路径']) for m in meta if m['family_id']==fid],
             relation_to_dev='按事件重新编写，未使用dev逐题答案；共享政策、行为契约及推理模式。',
             other_sources='来源独立性及变体相容性结论见content_reviews.json；共享政策和表单不作为独立性证明。')
        for fid,d in DOSSIERS.items()])
    dump(data/'metadata/pending_slots.json',[
        dict(task_id=r['task_id'],family_id=r['family_id'],split=r['candidate_split'],stratum_id=r['stratum_id'],subtype_id=r['subtype_id'],
             reason='尚无经本次内容审阅的事件级原始业务材料；r1条件拷贝及背景不能作为独立实例纳入。')
        for r in rows if r['family_id'] not in DOSSIERS])
    dump(data/'metadata/version.json',dict(status='候选已构造；内容验收见acceptance_summary.json',
        standards_revision=REVISION,based_on='balanced-policy-v1-data-standards-r1',grading_revision=GRADING_REVISION,
        candidate_target=600,constructed=len(meta),test_frozen=False,seed=42,
        structure='保留原50族、600槽位及seed42划分；按事件原始记录构造，未改变整体规模。',
        costs=dict(teacher=0,model=0,gpu=0,external_api=0,retrieval=0,downloads=0,fee=0)))
    print(json.dumps(dict(constructed=len(meta),pending=600-len(meta)),ensure_ascii=False))


def duplicates(tasks, meta):
    """Exact body comparison and normalized decisive-state comparison, not a score."""
    index={m['task_id']:m for m in meta}
    result={}
    train=[t for t in tasks.values() if index[t.task_id]['split']=='train']
    def body(t): return [r.content for r in t.records[1:]]
    def core(t):
        states,_=read_event(t)
        # Identity/scoped names are intentionally removed. Retain states, boolean
        # values and business numbers so matches cannot disappear by relabeling.
        out=[]
        for fields in states.values():
            items=[]
            for k,v in sorted(fields.items()):
                val=v['value']
                if isinstance(val,dict): val=val.get('status','scope')
                elif k in {'outbound_ip','order_id','transaction_id'}: val='identity'
                items.append((k,v['state'],val))
            out.append(items)
        return out
    for split in ('validation','test'):
        selected=[t for t in tasks.values() if index[t.task_id]['split']==split]
        exact=[]; state_matches=[]
        for t in selected:
            peers=[p.task_id for p in train if body(t)==body(p)]
            if peers: exact.append(dict(task_id=t.task_id,train_peers=peers))
            peers=[p.task_id for p in train if core(t)==core(p)]
            if peers: state_matches.append(dict(task_id=t.task_id,train_peers=peers))
        result[split]=dict(total=len(selected),identical_object_bodies=exact,normalized_state_matches=state_matches,
            interpretation='状态同型允许复用政策推理；不能据正文不同推断独立。各事件的目标路径及业务关系另由AI审阅；状态匹配完整列出，不作为自动拒绝条件。')
    return result


def validate(data,logs):
    from benchmarks.scripts.balanced_policy_data_validation import semantic_audit
    meta=load(data/'metadata/task_manifest.json'); planned=load(data/'design/candidate_manifest.json')
    tasks={}; standards={}; refs={}; failures=[]; grades=[]
    for split in SPLITS:
        collections=[load(data/f'{folder}/{split}.json') for folder in ('public','private','acceptance')]
        expected={m['task_id'] for m in meta if m['split']==split}
        for rows in collections:
            assert len(rows)==len({r['task_id'] for r in rows}) and {r['task_id'] for r in rows}==expected
        tasks.update({r['task_id']:PublicTask.model_validate(r) for r in collections[0]})
        standards.update({r['task_id']:PrivateStandard.model_validate(r) for r in collections[1]})
        refs.update({r['task_id']:r for r in collections[2]})
    validate_task_pairs(tasks,standards,CLAUSES)
    reviews=[]
    review_path=data/'metadata/content_reviews.json'
    recorded=load(review_path) if review_path.exists() else []
    by_family={r['family_id']:r for r in recorded}
    by_plan={r['task_id']:r for r in planned}
    for m in meta:
        tid=m['task_id']; t=tasks[tid]; s=standards[tid]; r=refs[tid]
        states,evidence=read_event(t); local=semantic_audit(t,s,states)
        assert states==m['condition_states'] and evidence==m['evidence_dependencies']
        assert m['split']==s.split==by_plan[tid]['candidate_split']
        sid,disposition=classify(t.role,states)
        assert (sid,actual_subtype(sid,states),disposition)==(m['stratum_id'],m['subtype_id'],m['expected_disposition'])
        for ex in s.expectations:
            raw=ex.model_dump(); policies=[g for g in ex.supports if not any(x.startswith('user:') for x in g)]
            raw['supports']=[['user:1']]+policies
            assert ex.supports==expectation_support(raw,evidence)
        tr=VisibleTrajectory.model_validate(r['trajectory']); assert tr.materials==t.materials()
        TerminalResult.model_validate(r['result'])
        g=asdict(grade_result(s,r['result'],tr,CLAUSES)); grades.append(dict(task_id=tid,grade=g))
        if not g['complete_success']: local.append('固定评分参考失败')
        record=by_family.get(m['family_id'],{})
        content_status=record.get('status','unreviewed')
        issue=record.get('issues',[])
        if record.get('reviewed_public_tasks',{}).get(tid)!=t.model_dump():
            content_status='unreviewed'
            issue=['当前公开材料未被记录审阅，或已修改']
        exception=record.get('exceptions',{}).get(tid)
        if exception and content_status!='unreviewed':
            content_status=exception['status']; issue=exception['issues']
        reviews.append(dict(task_id=tid,status='passed' if not local else 'failed',issues=local,
            content_status=content_status,content_issues=issue,content_basis=record.get('basis','未审阅')))

        failures.extend(tid+':'+x for x in local)
    # Three relation-specific failures plus the positive direct-response control.
    probes=[]
    for label,tid,mode in [('missing_join','balanced-M03-02','remove'),('background_only','balanced-P02-01','background'),
                           ('wrong_object','balanced-M03-03','swap'),('direct_auth_without_background','balanced-A01-02','direct')]:
        r=deepcopy(refs[tid]); s=standards[tid]
        for section in ('facts','reasons','next_steps','missing_info'):
            for item in r['result'][section]:
                if mode in {'remove','direct'}: item['evidence']=[x for x in item['evidence'] if x!='user:2']
                elif mode=='background': item['evidence']=[x for x in item['evidence'] if not x.startswith('user:')]+['user:1']
                elif mode=='swap' and item['target']=='R2': item['evidence']=['user:3' if x=='user:4' else x for x in item['evidence']]
        g=asdict(grade_result(s,r['result'],VisibleTrajectory.model_validate(r['trajectory']),CLAUSES))
        expected=mode=='direct'
        if g['complete_success']!=expected: failures.append(label)
        probes.append(dict(name=label,task_id=tid,expected_success=expected,grade=g,raw_result=r['result']))
    # New response-derived authentication and historical receipt mechanisms.
    auth=tasks['balanced-A01-02'].model_copy(deep=True)
    body=json.loads(auth.records[1].content)
    body['本次认证记录']['脱敏响应']['error']='unauthorized'
    auth.records[1].content=json.dumps(body,ensure_ascii=False)
    changed,_=read_event(auth)
    assert changed['Q1']['token_expired']['state']=='explicit_unknown'
    probes.append(dict(name='current_auth_response_changes_expiration',passed=True))
    received=next(t for t in tasks.values() if any('到账核对' in r.content for r in t.records)).model_copy(deep=True)
    body=json.loads(received.records[1].content);body['到账核对']['实际到账时审核后完整工作日']+=1
    received.records[1].content=json.dumps(body,ensure_ascii=False)
    try:
        read_event(received)
    except AssertionError:
        probes.append(dict(name='receipt_time_must_match_actual_receipt',passed=True))
    else:
        failures.append('receipt_time_must_match_actual_receipt')
    for tid,t in tasks.items():
        exposed=json.dumps([t.model_dump(),refs[tid]['trajectory']],ensure_ascii=False)
        assert not any(k in exposed for k in ('stratum_id','subtype_id','family_id','expectations','severe_items','review_status'))
    dump(logs/'r3_relation_probes.json',probes)
    passed={r['task_id'] for r in reviews if r['status']=='passed'}
    coverage=[]
    for sid,names in SUBTYPES.items():
        for sub in names:
            coverage.append(dict(stratum_id=sid,subtype_id=sub,splits={split:dict(
                target=sum(r['stratum_id']==sid and r['subtype_id']==sub and r['candidate_split']==split for r in planned),
                constructed=sum(m['stratum_id']==sid and m['subtype_id']==sub and m['split']==split for m in meta),
                passed=sum(m['stratum_id']==sid and m['subtype_id']==sub and m['split']==split and m['task_id'] in passed for m in meta)
            ) for split in SPLITS}))
    content_passed={r['task_id'] for r in reviews if r['content_status']=='passed'}
    content_failed=[r['task_id'] for r in reviews if r['content_status']=='needs_revision']
    distribution_rows=[{**by_plan[m['task_id']],'decisive_conditions':{'objects':m['condition_states']}} for m in meta]
    from benchmarks.scripts.balanced_policy_validation import check_distribution
    distribution_failures,distribution=check_distribution(distribution_rows)
    failures.extend(distribution_failures)
    family_counts=Counter(m['family_id'] for m in meta)
    if len(family_counts)!=50 or set(family_counts.values())!={12}: failures.append('来源族未达到50族各12题')
    for c in coverage:
        for split in SPLITS:
            c['splits'][split]['content_passed']=sum(m['stratum_id']==c['stratum_id'] and m['subtype_id']==c['subtype_id'] and m['split']==split and m['task_id'] in content_passed for m in meta)
    summary=dict(status='可提交冻结审核' if len(tasks)==600 and len(content_passed)==600 and not failures else '需继续修订',automatic_checks='passed' if not failures else 'failed',
        grading_revision=GRADING_REVISION,standards_revision=REVISION,target=600,constructed=len(tasks),passed=len(passed),
        pending=600-len(tasks),content_revision_required=content_failed,
        content_reviewed_nonduplicate=len(content_passed),content_unreviewed=len(tasks)-len(content_passed)-len(content_failed),
        distribution=distribution,family_counts=dict(family_counts),
        freeze_accepted=0,failures=failures,subtype_coverage=coverage,
        split_report={split:dict(constructed=sum(m['split']==split for m in meta),passed=sum(m['split']==split and m['task_id'] in passed for m in meta),content_passed=sum(m['split']==split and m['task_id'] in content_passed for m in meta),
            **{k:dict(Counter(m[k] for m in meta if m['split']==split)) for k in ('stratum_id','subtype_id','role','style','expected_disposition')}) for split in SPLITS},
        duplication=duplicates(tasks,meta),reference_grades=grades,test_frozen=False,
        limits=['标准和参考同一构造过程；参考成功不代表模型能力或独立语义复核。',
                '自动检查表单解码、关系唯一定位、状态/标准一致性、政策要求、引用与固定评分；不是通用自然语言验收。',
                '事件自然语言、因果边界、同源关系由同一执行助手AI审阅；无人工金标、独立专家或盲审。',
                'test参与本次构造和同一AI内容审阅，未封存；独立复核及模型表现未知。'])
    family_splits={}
    for m in meta: family_splits.setdefault(m['family_id'],set()).add(m['split'])
    assert all(len(s)==1 for s in family_splits.values())
    dev=load(data/'design/examples_public.json')
    assert not set(tasks)&{t['task_id'] for t in dev}
    assert not {t.user_text for t in tasks.values()} & {t['user_text'] for t in dev}
    dump(data/'metadata/acceptance_summary.json',summary)
    dump(data/'metadata/semantic_review.json',reviews)
    index=len(list(logs.glob('validation_*.json')))+1
    dump(logs/f'validation_{index:03}.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','automatic_checks','constructed','passed','pending','failures')},ensure_ascii=False))
    if failures or len(content_passed)!=len(tasks): raise SystemExit(1)
