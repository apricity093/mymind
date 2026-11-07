"""Offline acceptance checks for the balanced-policy design artifacts."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

BACKEND=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BACKEND))

from pydantic import ValidationError
from agents.customer_policy.protocol import PublicTask, TerminalResult, VisibleTrajectory, SearchAction, Fact, FACT_TYPES
from rl.tasks import PrivateStandard, validate_task_pairs
from rl.grading import GRADING_REVISION, grade_result


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_distribution(rows):
    from benchmarks.scripts.balanced_policy_candidates import SUBTYPES, classify, actual_subtype
    failures=[]; families=defaultdict(set); strata=defaultdict(list)
    for row in rows:
        families[row['family_id']].add(row['candidate_split']);strata[row['stratum_id']].append(row)
        sid,disp=classify(row['role'],row['decisive_conditions']['objects'])
        if sid!=row['stratum_id']:failures.append(row['task_id']+':归层冲突')
        if disp!=row['expected_disposition']:failures.append(row['task_id']+':处置错误')
        if actual_subtype(sid,row['decisive_conditions']['objects'])!=row['subtype_id']:failures.append(row['task_id']+':子型条件不符')
    if any(len(v)!=1 for v in families.values()):failures.append('family跨集合')
    report={}
    for sid in SUBTYPES:
        items=strata[sid];counts=Counter(r['candidate_split'] for r in items)
        if counts!=dict(train=15,validation=5,test=5):failures.append(sid+':配额')
        details={}
        for split,n in [('train',15),('validation',5),('test',5)]:
            selected=[r for r in items if r['candidate_split']==split]
            expected={sub:n//len(SUBTYPES[sid])+int(i<n%len(SUBTYPES[sid])) for i,sub in enumerate(SUBTYPES[sid])}
            actual=Counter(actual_subtype(sid,r['decisive_conditions']['objects']) for r in selected)
            if actual!=expected:failures.append(sid+':实际子条件覆盖')
            cross=defaultdict(Counter)
            for r in selected:cross[r['subtype_id']][r['style']]+=1
            details[split]=dict(subtypes=dict(actual),styles=dict(Counter(r['style'] for r in selected)),style_by_subtype={sub:dict(c) for sub,c in cross.items()},required_quota=expected)
        report[sid]=dict(counts=dict(counts),coverage=details)
    return failures,report


def execution_status(grade):
    if grade['protocol_error']:
        return dict(protocol='failed',severe='confirmed' if grade['severe_error'] else 'unknown',evidence='not_evaluated',completeness='not_evaluated',disposition='not_evaluated')
    if grade['severe_error']:
        return dict(protocol='passed',severe='confirmed',evidence='not_evaluated',completeness='not_evaluated',disposition='not_evaluated')
    return dict(protocol='passed',severe='checked_no_error',evidence='evaluated',completeness='evaluated',disposition='evaluated')


def validate(design, logs):
    from benchmarks.scripts.balanced_policy_design import CLAUSES, KNOWLEDGE
    failures=[]; probes=[]
    tasks={r['task_id']:PublicTask.model_validate(r) for r in load(design/'examples_public.json')}
    standards={r['task_id']:PrivateStandard.model_validate(r) for r in load(design/'examples_private.json')}
    refs=load(design/'examples_reference.json'); meta=load(design/'examples_metadata.json')
    validate_task_pairs(tasks,standards,CLAUSES)
    for clause in CLAUSES.values():
        if ''.join(clause.text.split()) not in ''.join(KNOWLEDGE.split()): failures.append('政策正文缺失:'+clause.clause_id)
    grades=[]
    for row in refs:
        tid=row['task_id']; task=tasks[tid]; std=standards[tid]
        traj=VisibleTrajectory.model_validate(row['trajectory']); result=TerminalResult.model_validate(row['result'])
        assert traj.materials==task.materials()
        for search in traj.searches:
            SearchAction.model_validate(search.action)
            for snippet in search.observation.snippets:
                assert snippet.content in KNOWLEDGE
        # PublicTask and observation carry no metadata or private contract fields.
        exposed=json.dumps([task.messages(),[s.observation.model_dump() for s in traj.searches]],ensure_ascii=False)
        for hidden in ['stratum_id','family_id','expectations','severe_items','supports','review_status']:
            assert hidden not in exposed
        targets={alt['target'] for x in std.expectations for alt in x.alternatives}
        assert all(t=='case' or t in task.user_text for t in targets)
        grade=grade_result(std,result,traj,CLAUSES).to_dict()
        evaluated='not_evaluated' if grade['protocol_error'] or grade['severe_error'] else 'evaluated'
        grades.append(dict(grading_revision=GRADING_REVISION,task_id=tid,grade=grade,check_execution=execution_status(grade),completeness_check=evaluated,raw_result=row['result'],trajectory=row['trajectory']))
        if not grade['complete_success']: failures.append(tid+':reference_failed')

    def probe(label, tid, mutate, should_pass=False):
        base=deepcopy(next(r for r in refs if r['task_id']=='design-dev-'+tid))
        mutate(base)
        grade=grade_result(standards[base['task_id']],base['result'],VisibleTrajectory.model_validate(base['trajectory']),CLAUSES).to_dict()
        accepted=grade['complete_success']==should_pass
        probes.append(dict(name=label,task_id=base['task_id'],expected_complete_success=should_pass,passed=accepted,grade=grade,check_execution=execution_status(grade),
                           completeness_check='not_evaluated' if grade['protocol_error'] or grade['severe_error'] else 'evaluated',input=base))
        if not accepted: failures.append(label)

    def change_fact(row,kind,value):
        next(x for x in row['result']['facts'] if x['type']==kind)['value']=value
    def add(row,section,**payload): row['result'][section].append(dict(target='Q1',**payload,evidence=['user:1']))
    probe('未知审核不能断言within_window','R01',lambda r:r['result']['facts'].append(dict(target='R1',type='within_window',value=True,evidence=['user:1','search:1:1'])))
    probe('计时未知不能断言within_window','R02',lambda r:r['result']['facts'].append(dict(target='R1',type='within_window',value=True,evidence=['user:1','search:1:1'])))
    probe('已知false不能改unknown','P03',lambda r:change_fact(r,'feature_enabled',dict(project='测绘档案',environment='production',status='unknown')))
    probe('已知完整条件不能漏比较','R03',lambda r:r['result'].__setitem__('facts',[x for x in r['result']['facts'] if x['type']!='within_window']))
    probe('不能借用另一笔退款天数','M02',lambda r:change_fact(r,'elapsed_workdays',1))
    probe('功能环境错配','P06',lambda r:change_fact(r,'feature_enabled',dict(project='测绘档案',environment='production',status=True)))
    probe('白名单不能带环境范围','P02',lambda r:change_fact(r,'outbound_ip_allowed',dict(project='测绘档案',environment='production',outbound_ip='198.51.100.74',status=False)))
    probe('白名单已知不能重复索取','P04',lambda r:add(r,'missing_info',field='outbound_ip_allowed',purpose='decision'))
    probe('功能已知不能重复索取','P05',lambda r:add(r,'missing_info',field='feature_enabled',purpose='decision'))
    probe('严重影响不因缺日志延迟','F02',lambda r:r['result'].__setitem__('disposition','request_info'))
    probe('未提供频率不能断言single','F02',lambda r:add(r,'facts',type='occurrence',value='single'))
    probe('已恢复不排除数据丢失','F03',lambda r:r['result'].__setitem__('reasons',[dict(target='Q1',code='limited_recovered_fault',evidence=['user:1','search:1:2'])]))
    probe('401不能当作明确过期','A02',lambda r:change_fact(r,'token_expired',True))
    probe('对象不合法','R03',lambda r:r['result']['facts'][0].__setitem__('target','wrong-object'))
    probe('布尔字符串非法','P02',lambda r:change_fact(r,'auth_valid','true'))
    probe('枚举非法','R03',lambda r:change_fact(r,'payment_channel','银行卡'))
    probe('建议参数非法','R03',lambda r:next(x for x in r['result']['next_steps'] if x['code']=='verify_refund_if_overdue_unreceived')['params'].__setitem__('limit_workdays',8))
    probe('终止格式多字段','A01',lambda r:r['result'].__setitem__('answer','done'))
    probe('正确source_id不能代替正文','R03',lambda r:r['trajectory']['searches'][0]['observation']['snippets'][0].__setitem__('content','这是账单政策但不包含任何具体规则。'))
    probe('诊断字段顺序是等价表达','F01',lambda r:next(x for x in r['result']['next_steps'] if x['code']=='record_diagnostics')['params']['fields'].reverse(),True)
    probe('等价计时组合不必重复timer_start','R03',lambda r:r['result'].__setitem__('facts',[x for x in r['result']['facts'] if x['type']!='timer_start']),True)
    probe('等价审核组合可省审核项','R03',lambda r:r['result'].__setitem__('facts',[x for x in r['result']['facts'] if x['type']!='approval_status']),True)
    probe('可选正确建议允许输出','P04',lambda r:r['result']['next_steps'].append(dict(target='Q1',code='adjust_project_whitelist',params=dict(project='测绘档案',outbound_ip='198.51.100.74'),evidence=['user:1','search:1:2'])),True)
    probe('相互矛盾值不能同时接受','R03',lambda r:r['result']['facts'].append(dict(target='R1',type='within_window',value=False,evidence=['user:1','search:1:1'])))

    # Direct loader/pairing mutation checks; no alternate schemas.
    def reject(label,fn):
        try: fn()
        except (ValueError,ValidationError,TypeError): probes.append(dict(name=label,passed=True)); return
        probes.append(dict(name=label,passed=False)); failures.append(label)
    reject('任务标准不配对',lambda:validate_task_pairs(dict(list(tasks.items())[1:]),standards,CLAUSES))
    corrupt=deepcopy(standards); corrupt[next(iter(corrupt))].materials['user:1']='错误正文'
    reject('材料正文不一致',lambda:validate_task_pairs(tasks,corrupt,CLAUSES))
    corrupt=deepcopy(standards); corrupt[next(iter(corrupt))].expectations[0].supports=[['missing-policy']]
    reject('政策支持缺失',lambda:validate_task_pairs(tasks,corrupt,CLAUSES))
    reject('工具参数非法',lambda:SearchAction.model_validate(dict(query='政策',limit=10)))

    dump=lambda path,obj:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from benchmarks.scripts.balanced_policy_candidates import SUBTYPES, classify, actual_subtype, conditions_for, value
    from benchmarks.scripts.balanced_policy_design import main_sources
    rows=load(design/'candidate_manifest.json');families=load(design/'source_families.json')
    dist_fail,distribution=check_distribution(rows);failures.extend(dist_fail)
    meta_by_id={r['task_id']:r for r in meta}

    reviewed_public={r['task_id']:r for r in load(design/'v1/examples_public.json')}

    def annotation_errors(task, annotation):
        objects=annotation['condition_states']
        if annotation['kind']=='retained_main':
            expected=main_sources(annotation['stratum_id'])
            errors=[]
            if objects!=expected:errors.append('主例来源标注与独立题面注释不符')
            if task.user_text!=reviewed_public[task.task_id]['user_text']:errors.append('主例题面已改变，原独立来源注释须重新审核')
            return errors
        if annotation['kind']=='added_behavior_contract':
            errors=[]
            for target,fields in objects.items():
                for field,entry in fields.items():
                    prefix=f'{target}.{field}：'
                    if entry['state']=='not_provided':
                        if prefix in task.user_text:errors.append(field+':未提供但出现记录')
                    else:
                        wording='明确不知道' if entry['state']=='explicit_unknown' else json.dumps(entry['value'],ensure_ascii=False)
                        if prefix+wording+'。' not in task.user_text:errors.append(field+':来源状态与公开记录不符')
            return errors
        # Retained variants have separately authored annotations, including the R01 absence boundary.
        if 'approval_not_provided' in task.task_id:
            return [] if objects['R1']['approval_status']['state']=='not_provided' and '没有提供审核结果' in task.user_text else ['变体审核状态不符']
        return []

    source_state_count=0;semantic=[]
    for tid,task in tasks.items():
        m=meta_by_id[tid];objects=m['condition_states']
        errors=annotation_errors(task,m);failures.extend(tid+':'+e for e in errors)
        sid,disp=classify(task.role,objects)
        if sid!=m['stratum_id'] or disp not in standards[tid].dispositions:failures.append(tid+':契约归层/处置不符')
        if actual_subtype(sid,objects)!=m['subtype_id']:failures.append(tid+':契约实际子型不符')
        for target,fields in objects.items():
            for field,entry in fields.items():
                source_state_count+=1
                assert entry['state'] in {'known','explicit_unknown','not_provided'}
                if entry['state']=='not_provided':assert entry['value'] is None
                for item in next(r for r in refs if r['task_id']==tid)['result']['missing_info']:
                    if item['target']==target and item['field']==field and item['purpose']=='decision' and entry['state']=='known':failures.append(tid+':重复索取已知条件')
        semantic.append(dict(task_id=tid,source='同一执行助手AI语义审阅',review=m['semantic_review'],source_check_errors=errors,assigned_stratum=sid,disposition=disp))
    corrupt=deepcopy(meta_by_id['design-dev-R01']);corrupt['condition_states']['R1']['elapsed_workdays']['state']='explicit_unknown'
    assert annotation_errors(tasks['design-dev-R01'],corrupt)
    probes.append(dict(name='未提供天数不得标为明确未知',passed=True,evidence='R01题面只说审核不清楚；独立来源注释天数not_provided，污染该注释被拒绝'))
    revision_evidence=check_scoring_revision(design,logs,tasks,standards,refs)
    # Another supported manual escalation code must be accepted, while retaining the same severe-impact decision.
    base=deepcopy(next(r for r in refs if r['task_id']=='design-dev-F02'))
    next(item for item in base['result']['next_steps'] if item['code']=='escalate_technical')['code']='seek_human_review'
    assert grade_result(standards['design-dev-F02'],base['result'],VisibleTrajectory.model_validate(base['trajectory']),CLAUSES).complete_success
    probes.append(dict(name='人工升级等价建议允许',passed=True,input=base))
    # A source annotation is not a policy-derived fact.
    assert 'timer_start' not in meta_by_id['design-dev-R03']['condition_states']['R1']
    assert any(r['type']=='timer_start' for r in meta_by_id['design-dev-R03']['policy_derived_facts'])
    probes.append(dict(name='来源状态与政策推导分离',passed=True))
    bad=deepcopy(rows);bad[0]['candidate_split']='validation' if bad[0]['candidate_split']=='train' else 'train'
    assert 'family跨集合' in check_distribution(bad)[0]
    bad=deepcopy(rows)
    target=next(r for r in bad if r['stratum_id']=='P01')
    target['decisive_conditions']['objects']['Q1']['outbound_ip_allowed']['value']['status']=True
    try: conflict=check_distribution(bad)[0]
    except ValueError:conflict=['unsupported permission combination']
    assert conflict
    bad=deepcopy(rows);target=next(r for r in bad if r['stratum_id']=='F02' and r['subtype_id']=='payment');target['subtype_id']='corruption'
    assert any('子型条件' in e for e in check_distribution(bad)[0])
    probes.extend([dict(name='跨集合来源族拒绝',passed=True),dict(name='子型标签与实际条件冲突拒绝',passed=True),dict(name='权限条件改变须改变归属',passed=True)])
    f2=conditions_for('F02','payment_and_expired')
    assert classify('technical',f2)==('F02','escalate')
    for sid,sub in [('R01','approval_and_channel_unknown'),('R05','channel_and_time_unknown'),('P06','other_project'),('P04','feature_unknown_white_false')]:
        assert classify('billing' if sid.startswith('R') else 'technical',conditions_for(sid,sub))[0]==sid
    probes.append(dict(name='严重影响认证重叠与退款权限重叠唯一归属',passed=True))

    family_by_id={f['family_id']:f for f in families}
    assert len(family_by_id)==len(families)==50
    assert len({r['task_id'] for r in rows})==len(rows)==600
    for row in rows:
        f=family_by_id[row['family_id']]
        assert f['candidate_split']==row['candidate_split'] and f['role']==row['role']
        assert len(f['variants'])==12
        assert any(v['task_id']==row['task_id'] and v['changed_conditions']==row['decisive_conditions']['objects'] for v in f['variants'])
        assert row['variant']['retained_relations']==f['object_relations']
        assert len(f['original_records'])==2
        for target,fields in row['decisive_conditions']['objects'].items():
            for field,entry in fields.items():
                assert entry['state'] in {'known','explicit_unknown','not_provided'}
                if entry['state']=='not_provided':assert entry['value'] is None
                if entry['state']=='known' and field in FACT_TYPES:
                    if field=='payment_channel' and entry['value'] not in {'online_payment','bank_card','balance','unknown'}:continue
                    Fact.model_validate(dict(target=target,type=field,value=entry['value'],evidence=['user:1']))
    # Compare actual source processes/paired records; IDs and prose alone are not independence proof.
    assert len({(f['process'][0],f['object_relations']) for f in families})==50
    assert not ({m['family_id'] for m in meta}&set(family_by_id))
    for sid,names in SUBTYPES.items():
        signatures=[]
        for sub in names:
            objects=conditions_for(sid,sub)
            assert classify('billing' if sid[0] in 'RMB' else 'technical',objects)[0]==sid
            assert actual_subtype(sid,objects)==sub
            signatures.append(objects)
            assert any(m['stratum_id']==sid and m['subtype_id']==sub for m in meta)
        assert all(a!=b for i,a in enumerate(signatures) for b in signatures[i+1:])
    # Replay exactly one prespecified style mechanism; do not optimize its result.
    import random
    rng=random.Random(42)
    from benchmarks.scripts.balanced_policy_design import STYLES
    assert [r['style'] for r in rows]==[STYLES[rng.randrange(5)] for _ in rows]
    overall={}
    for split in ['train','validation','test']:
        selected=[r for r in rows if r['candidate_split']==split]
        cross_disp=defaultdict(Counter);cross_sub=defaultdict(Counter)
        for r in selected:
            cross_disp[r['expected_disposition']][r['style']]+=1
            cross_sub[r['stratum_id']+'/'+r['subtype_id']][r['style']]+=1
        overall[split]=dict(count=len(selected),roles=dict(Counter(r['role'] for r in selected)),dispositions=dict(Counter(r['expected_disposition'] for r in selected)),styles=dict(Counter(r['style'] for r in selected)),style_by_disposition={key:dict(c) for key,c in cross_disp.items()},style_by_subtype={key:dict(c) for key,c in cross_sub.items()},families=len({r['family_id'] for r in selected}))
    for suffix,change in [('not_checked','repeat_known_check'),('receipt_unknown','assert_receipt'),('received','current_verify')]:
        tid='design-dev-R04-'+suffix
        row=deepcopy(next(r for r in refs if r['task_id']==tid))
        if change=='repeat_known_check':row['result']['missing_info'].append(dict(target='R1',field='original_account_checked',purpose='decision',evidence=['user:1']))
        elif change=='assert_receipt':row['result']['facts'].append(dict(target='R1',type='received',value=False,evidence=['user:1']))
        else:row['result']['next_steps'].append(dict(target='R1',code='verify_refund',evidence=['user:1','search:1:1']))
        grade=grade_result(standards[tid],row['result'],VisibleTrajectory.model_validate(row['trajectory']),CLAUSES).to_dict()
        assert not grade['complete_success']
        probes.append(dict(name=change,passed=True,task_id=tid,grade=grade,input=row,check_execution=execution_status(grade)))
    overlap_id='design-dev-F02-payment_and_expired'
    overlap=deepcopy(next(r for r in refs if r['task_id']==overlap_id))
    overlap['result']['disposition']='answer'
    assert not grade_result(standards[overlap_id],overlap['result'],VisibleTrajectory.model_validate(overlap['trajectory']),CLAUSES).complete_success
    probes.append(dict(name='认证指导不能覆盖严重影响整体升级',passed=True,input=overlap))
    # Exercise command-level logging with one ordinary nonzero exit and one exception.
    # These are intentional fixtures, separately counted from acceptance failures.
    logging_probes=[]
    import subprocess
    from datetime import datetime
    from zoneinfo import ZoneInfo
    probe_log=logs/'logging_probes_v3.jsonl'
    for label,code,expected in [('normal_failure','raise SystemExit(3)',3),('exception','assert False, "intentional logging fixture"',1)]:
        command=[sys.executable,'-c',code]
        done=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        entry=dict(name=label,command=command,time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),exit_status=done.returncode,reason=done.stderr.strip() or 'intentional exit 3',intentional=True)
        with probe_log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
        assert done.returncode==expected
        assert json.loads(probe_log.read_text(encoding='utf-8').splitlines()[-1])==entry
        logging_probes.append(entry)
    previous=load(design/'v1/validation_results.json')
    old_ids={r['task_id'] for r in previous['references']}|{r['public_task']['task_id'] for r in previous['boundary_variants']}
    assert old_ids<=set(tasks)
    result=dict(grading_revision=GRADING_REVISION,standards_revision='balanced-policy-v1-standards-v4',status='passed' if not failures else 'failed',failures=failures,
        references=grades,reference_count=len(grades),retained_contracts=sorted(old_ids),new_contracts=sorted(set(tasks)-old_ids),probes=probes,logging_probes=logging_probes,
        semantic_reviews=semantic,distribution=distribution,overall=overall,source_state_annotations_checked=source_state_count,
        candidate_count=len(rows),source_family_count=len(families),subtype_count=sum(map(len,SUBTYPES.values())),accepted_formal_count=0,
        layer_gaps={sid:dict(candidate_spec_gap=0,final_task_acceptance_gap=25) for sid in SUBTYPES},
        historical_records=dict(successful_records=4,timestamps='unknown',unrecorded_attempts='unknown',evidence='v1/validation_commands.jsonl'),
        limitations=['600规格来自50个相关来源族，不能视作600独立事件','统一文体随机机制不保证小样本文体或交叉格相等；实际分布完整列示','来源规格及契约为同一执行助手AI审阅，非独立专家审核','参考由标准组装只证明相容性；语义审阅依据另存','最终任务与标准未构造验收，test未封存'],
        extra_semantics=dict(metric='独立于当前评分修订的完整成功',source_states=dict(denominator=len(tasks),failed=sum(bool(x['source_check_errors']) for x in semantic)),assignment=dict(denominator=len(tasks)+len(rows),result='见失败列表和逐例归属'),known_info_reasking=dict(denominator=len(tasks),result='逐契约decision待补对照来源known检查'),evidence='逐例semantic_reviews、probes及原始引用'),
        calls=dict(teacher=0,inference=0,gpu=0,retrieval=0,paid_api=0,dependency_download=0,cost=0),
        scoring_revision_checks=revision_evidence,unresolved_scoring_conflicts=[],recommendation='可进入完整数据构造' if not failures else '仍需修订')
    # Numbered results preserve the first result of every acceptance attempt.
    snapshots=sorted(logs.glob('validation_results_v3_*.json'))
    dump(logs/f'validation_results_v3_{len(snapshots)+1:02}.json',result)
    dump(logs/'validation_results.json',result);dump(design/'validation_results.json',result)
    print(json.dumps(dict(grading_revision=GRADING_REVISION,status=result['status'],contracts=len(grades),retained=len(old_ids),new=len(tasks)-len(old_ids),probes=len(probes),subtypes=result['subtype_count'],failures=failures,distribution={k:{field:v[field] for field in ['count','roles','styles','dispositions','families']} for k,v in overall.items()}),ensure_ascii=False))
    if failures:raise SystemExit(1)



def check_scoring_revision(design, logs, tasks, standards, refs):
    """Replay saved identical inputs; distinguish scorer changes from standard completion."""
    from benchmarks.scripts.balanced_policy_design import CLAUSES
    before=load(logs/'scoring_v13_before.json')
    by_id={r['task_id']:r for r in refs}
    comparisons=[]
    for old in before['cases']:
        tid=old['task_id'];raw=old['raw_result'];trajectory=VisibleTrajectory.model_validate(old['trajectory'])
        old_standard=PrivateStandard.model_validate(old['private_standard'])
        scorer_only=grade_result(old_standard,raw,trajectory,CLAUSES).to_dict()
        final=grade_result(standards[tid],raw,trajectory,CLAUSES).to_dict()
        comparisons.append(dict(task_id=tid,public_task=old['public_task'],raw_result=raw,trajectory=old['trajectory'],
            before=dict(grading_revision='v13',standard=old['private_standard'],grade=old['grade'],check_execution=execution_status(old['grade'])),
            scorer_only=dict(grading_revision=GRADING_REVISION,standard=old['private_standard'],grade=scorer_only,check_execution=execution_status(scorer_only)),
            after=dict(grading_revision=GRADING_REVISION,standard=standards[tid].model_dump(),grade=final,check_execution=execution_status(final)),
            attribution='评分修复去掉误判；标准补全允许可选表达' if 'P06' in tid else '退款标准补全；评分函数修改不改变此退款输入'))
        if 'P06' in tid:
            assert not scorer_only['severe_error'] and not scorer_only['complete_success'] and scorer_only['reward']==0.3
            assert final['complete_success'] and final['reward']==1
        else:
            assert scorer_only==old['grade']
            assert final['severe_error'] and final['reward']==-1
    scoped=[]
    def score_case(tid,name,raw,accept,severe=None):
        r=by_id[tid]
        grade=grade_result(standards[tid],raw,VisibleTrajectory.model_validate(r['trajectory']),CLAUSES).to_dict()
        assert grade['complete_success']==accept,(tid,name,grade)
        if severe is not None:assert grade['severe_error']==severe,(tid,name,grade)
        scoped.append(dict(task_id=tid,name=name,grading_revision=GRADING_REVISION,raw_result=raw,standard=standards[tid].model_dump(),public_task=tasks[tid].model_dump(),trajectory=r['trajectory'],grade=grade,check_execution=execution_status(grade)))
    for tid,field in [('design-dev-P06-other_project','feature_enabled'),('design-dev-P06-other_environment','feature_enabled'),('design-dev-P06-both_unknown','feature_enabled'),('design-dev-P06-both_unknown','outbound_ip_allowed')]:
        base=deepcopy(by_id[tid]['result'])
        expected=next(x for x in standards[tid].expectations if x.section=='facts' and x.alternatives[0].get('type')==field)
        assert not expected.required
        item=deepcopy(expected.alternatives[0]);item['evidence']=['user:1','search:1:2']
        with_unknown=deepcopy(base);with_unknown['facts'].append(item)
        score_case(tid,field+':有据unknown',with_unknown,True,False)
        score_case(tid,field+':合法省略',base,True,False)
        # Distinguish state assertion from scope. One covered field needs both boolean cases.
        states=[True,False] if tid.endswith('other_project') else [True]
        for status in states:
            wrong=deepcopy(with_unknown);wrong['facts'][-1]['value']['status']=status
            score_case(tid,field+':未提供断言'+str(status),wrong,False,True)
        scope_key='outbound_ip' if field=='outbound_ip_allowed' else 'environment' if tid.endswith('other_environment') else 'project'
        wrong=deepcopy(with_unknown);wrong['facts'][-1]['value'][scope_key]='test' if scope_key=='environment' else 'wrong-scope'
        score_case(tid,field+':错误'+scope_key,wrong,False)
        if tid.endswith('other_project'):
            wrong=deepcopy(with_unknown);wrong['facts'][-1]['target']='wrong-object'
            score_case(tid,'错误对象',wrong,False)
            wrong=deepcopy(with_unknown);wrong['missing_info']=[]
            score_case(tid,'可选事实不能代替必要待补',wrong,False)
            wrong=deepcopy(with_unknown);wrong['facts'][-1]['evidence']=['user:1']
            score_case(tid,'unknown推导缺少政策引用',wrong,False)
    for tid,status in [('design-dev-P02',False),('design-dev-P03','unknown')]:
        wrong=deepcopy(by_id[tid]['result'])
        next(x for x in wrong['facts'] if x['type']=='feature_enabled')['value']['status']=status
        score_case(tid,'篡改已知状态',wrong,False,True)
    # Both generation entry points must reject the same wrong waiting action.
    # Check every affected generated object with a known out-of-window comparison.
    refund_checks=[]
    for tid,standard in standards.items():
        for expected in standard.expectations:
            for item in expected.alternatives:
                if expected.section!='facts' or item.get('type')!='within_window' or item['value'] is not False:continue
                raw=deepcopy(by_id[tid]['result']);raw['next_steps'].append(dict(target=item['target'],code='wait_within_window',evidence=['user:1','search:1:1']))
                grade=grade_result(standard,raw,VisibleTrajectory.model_validate(by_id[tid]['trajectory']),CLAUSES).to_dict()
                assert grade['severe_error'] and grade['reward']==-1,(tid,item['target'],grade)
                refund_checks.append(dict(task_id=tid,target=item['target'],raw_result=raw,standard=standard.model_dump(),trajectory=by_id[tid]['trajectory'],grading_revision=GRADING_REVISION,grade=grade,check_execution=execution_status(grade)))
    for tid,target in [('design-dev-M03-other_elapsed_unknown','R2'),('design-dev-R01','R1')]:
        raw=deepcopy(by_id[tid]['result']);raw['next_steps'].append(dict(target=target,code='wait_within_window',evidence=['user:1','search:1:1']))
        score_case(tid,'未知对象错误等待不泛化为超时严重错误',raw,False,False)
    for tid in ['design-dev-R03','design-dev-R04-check_unknown','design-dev-M01']:
        score_case(tid,'正确等待或条件性核实未误伤',by_id[tid]['result'],True,False)
    result=dict(grading_revision=GRADING_REVISION,comparisons=comparisons,scope_and_controls=scoped,overdue_wait_checks=refund_checks,
                attribution='离线评分和标准一致性修复；没有模型行为变化或能力收益',unresolved=[])
    dump_path=logs/'scoring_v14_comparison.json'
    dump_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

def run_logged(action):
    """Record the command even if the child cannot import or parse its module."""
    import os
    import subprocess
    from datetime import datetime, timezone, timedelta
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    logs=root/'runs/balanced-policy-v1/data-design'
    logs.mkdir(parents=True,exist_ok=True)
    history=logs/'commands_v3.jsonl'
    entries=[json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()] if history.exists() else []
    number=len(entries)+1
    command=([sys.executable,'-m','pytest',str(BACKEND/'tests/test_rl_core.py'),'-q','--basetemp',str(logs/f'pytest_v3_{number:02}')] if action=='tests' else [sys.executable,str(Path(__file__).with_name('balanced_policy_design.py')),action,'--worker'])
    started=datetime.now(timezone(timedelta(hours=8))).isoformat()
    done=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    stdout=logs/f'command_v3_{number:02}_stdout.txt';stderr=logs/f'command_v3_{number:02}_stderr.txt'
    stdout.write_text(done.stdout,encoding='utf-8');stderr.write_text(done.stderr,encoding='utf-8')
    entry=dict(grading_revision=GRADING_REVISION,command=command,requested_command=[sys.executable,*sys.argv],started=started,finished=datetime.now(timezone(timedelta(hours=8))).isoformat(),exit_status=done.returncode,stdout=stdout.name,stderr=stderr.name,reason=(done.stderr.strip() or done.stdout.strip() or 'nonzero exit without output') if done.returncode else 'completed')
    with history.open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
    failure_details=[]
    for row in entries+[entry]:
        if row['exit_status']!=0:
            reason=row.get('reason','')
            if not reason and row.get('stdout'):
                reason=(logs/row['stdout']).read_text(encoding='utf-8').strip()
            failure_details.append(dict(command=row['command'],exit_status=row['exit_status'],reason=reason))
    summary=dict(grading_revision=GRADING_REVISION,failure_details=failure_details,recorded_commands=number,failed_commands=sum(r['exit_status']!=0 for r in entries+[entry]),failed_build_attempts=sum(r['exit_status']!=0 and 'build' in r['command'] for r in entries+[entry]),failed_validation_attempts=sum(r['exit_status']!=0 and 'validate' in r['command'] for r in entries+[entry]),failed_test_attempts=sum(r['exit_status']!=0 and ('pytest' in r['command'] or 'tests' in r['command']) for r in entries+[entry]),intentional_logging_probes='logging_probes_v3.jsonl单列，不计数据验收失败',historical_scope='v1只有四条成功记录，未记录尝试未知')
    (logs/'command_summary_v3.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(done.stdout,end='');print(done.stderr,end='',file=sys.stderr)
    print(json.dumps(dict(exit_status=done.returncode,recorded_commands=number,failed_commands=sum(r['exit_status']!=0 for r in entries+[entry]),historical_record_scope='v1四条成功记录，无时间；未记录历史未知'),ensure_ascii=False))
    return done.returncode


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','validate','tests'])
    raise SystemExit(run_logged(parser.parse_args().action))
