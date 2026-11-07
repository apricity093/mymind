"""Zero-call train contract checks and unchanged 72-response regrading."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from agents.customer_policy.protocol import VisibleTrajectory
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs
from rl.grading import (grade_result, _payload, _same_item, _supported, _fact_conflicts,
                        _unprovided_claim, _reason_conflicts, _step_conflicts, _compact, _equivalent_fact)
from benchmarks.scripts.teacher_contract_candidate import amend_standard, source_facts, REVISION
from benchmarks.scripts.balanced_policy_design import fixture


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def reference(standard, trajectory, clauses):
    result = dict(disposition=standard.dispositions[0], facts=[], reasons=[], next_steps=[], missing_info=[])
    for expected in standard.expectations:
        if not expected.required:
            continue
        item = deepcopy(expected.alternatives[0]); item['evidence'] = []
        for group in expected.supports:
            support = group[0]
            if support in standard.materials:
                item['evidence'].append(support)
            else:
                item['evidence'].append(next(s.reference_id for t in trajectory.searches for s in t.observation.snippets
                    if s.source_id == clauses[support].source_id and _compact(clauses[support].text) in _compact(s.content)))
        item['evidence'] = list(dict.fromkeys(item['evidence']))
        result[expected.section].append(item)
    return result


def verify(tasks, standards, clauses):
    probes = []
    def check(name, condition):
        assert condition, name
        probes.append(dict(name=name, passed=True))
    def has(tid, kind):
        return [e for e in standards[tid].expectations if e.section=='facts' and e.alternatives[0].get('type')==kind]
    for tid, standard in standards.items():
        tr = fixture(tasks[tid]); raw = reference(standard, tr, clauses)
        check(tid+':required_contract', grade_result(standard, raw, tr, clauses).complete_success)
        check(tid+':source_metadata', all(not _unprovided_claim(a,standard)
              for e in standard.expectations if e.section=='facts' for a in e.alternatives))
    for tid in ('balanced-R01-02', 'balanced-R02-02'):
        standard=standards[tid]; tr=fixture(tasks[tid]); raw=reference(standard,tr,clauses)
        optional=has(tid,'limit_workdays')[0]
        check(tid+':limit_optional', not optional.required)
        raw['facts'].append({**optional.alternatives[0], 'evidence':['user:2','user:3','search:1:1']})
        check(tid+':limit_allowed',grade_result(standard,raw,tr,clauses).complete_success)
        missing=deepcopy(raw); missing['missing_info']=[]
        check(tid+':limit_cannot_replace_missing',not grade_result(standard,missing,tr,clauses).complete_success)
        missing_fact=deepcopy(raw)
        missing_fact['facts']=[f for f in missing_fact['facts'] if f['type']!='elapsed_workdays']
        check(tid+':limit_cannot_replace_required_fact',not grade_result(standard,missing_fact,tr,clauses).complete_success)
        wrong=deepcopy(raw); wrong['facts'].append(dict(target='R1',type='within_window',value=True,evidence=['user:2','user:3','search:1:1']))
        check(tid+':no_comparison_without_clock',grade_result(standard,wrong,tr,clauses).severe_error)
    for tid in ('balanced-R05-02','balanced-R06-02'):
        check(tid+':no_channel_limit',not has(tid,'limit_workdays'))
        tr=fixture(tasks[tid]); raw=reference(standards[tid],tr,clauses)
        raw['facts'].append(dict(target='R1',type='limit_workdays',value=3,evidence=['user:2','user:3','search:1:1']))
        check(tid+':reject_invented_limit',not grade_result(standards[tid],raw,tr,clauses).complete_success)
    for tid,target,field,value in [('balanced-B02-02','O1','original_account_checked',False),
            ('balanced-R04-02','R1','received',False),('balanced-F03-02','Q1','data_corruption',False)]:
        tr=fixture(tasks[tid]); raw=reference(standards[tid],tr,clauses)
        raw['facts'].append(dict(target=target,type=field,value=value,evidence=['user:3']))
        check(tid+':unprovided_severe',grade_result(standards[tid],raw,tr,clauses).severe_error)
    for tid, field in [('balanced-P01-02','token_expired'),('balanced-F02-02','auth_valid'),('balanced-F03-02','data_loss'),('balanced-A01-02','environment')]:
        check(tid+':derived_'+field,field not in standards[tid].unprovided_user_facts['Q1'])
    tid='balanced-P02-02'; standard=standards[tid]; tr=fixture(tasks[tid]); raw=reference(standard,tr,clauses)
    no_join=deepcopy(raw)
    for section in ('facts','reasons','next_steps','missing_info'):
        for item in no_join[section]:item['evidence']=[r for r in item['evidence'] if r!='user:2']
    check('object_join_still_required',not grade_result(standard,no_join,tr,clauses).complete_success)
    wrong=deepcopy(raw); next(f for f in wrong['facts'] if f['type']=='feature_enabled')['value']['environment']='production'
    check('scope_still_checked',grade_result(standard,wrong,tr,clauses).severe_error)
    tid='balanced-A01-02'; tr=fixture(tasks[tid]); raw=reference(standards[tid],tr,clauses)
    check('direct_auth_no_background',all('user:2' not in f['evidence'] for f in raw['facts']) and grade_result(standards[tid],raw,tr,clauses).complete_success)
    return probes


def semantic(section, payload, standard, states):
    """Adjudicate actual unmatched items against public conditions; uncertainty stays explicit."""
    target=payload['target']; fields=states.get(target,{}); kind=payload.get('type'); code=payload.get('code'); field=payload.get('field')
    if section=='facts':
        if _fact_conflicts(payload,standard.expectations) or _unprovided_claim(payload,standard):
            return 'incorrect','公开状态矛盾或来源事实未提供；见source_facts及具体值'
        if kind=='timing_policy_covered':
            return 'incorrect','渠道未知不能断言政策未覆盖'
        if kind=='interface_name':
            return 'incorrect','执行单/调用记录标识不等于业务接口名称'
        if kind in {'request_scope','feature_enabled'}:
            return 'unresolved','公开映射可定位本次测试调用，但未给出可唯一核对的项目名称；不能仅以标准未匹配判错'
        if payload.get('value')=='unknown':
            return 'unresolved','未作具体断言；是否对当前任务有必要须限定，未自动放行unknown'
    if target=='case':
        return 'incorrect','公开对象为逐笔退款；整体case的扣款或技术严重影响无相应业务依据'
    if section=='reasons':
        if _reason_conflicts(payload,standard):return 'incorrect','理由含与公开状态冲突的事实主张'
        if code=='permission_state_unknown' and fields.get('outbound_ip_allowed',{}).get('state') in {'explicit_unknown','not_provided'}:
            return 'legal','白名单状态确实未知；不否认已知未开通，不代替当前开通升级'
        if code in {'policy_uncovered','suspected_duplicate_charge','severe_business_impact','within_refund_window'}:
            return 'incorrect','该理由的成立前提不满足；见当前渠道/扣款/影响/工作日材料'
        return 'unresolved','理由的任务相关性或条件含义未被当前公开契约唯一确定'
    if section=='next_steps':
        if code=='escalate_if_severe_impact':return 'legal','未来条件建议可保留，但已知严重影响仍必须给当前升级'
        if code=='check_original_account' and fields.get('original_account_checked',{}).get('value') is True:
            return 'incorrect','账户已核对，重复要求不能替代当前处置'
        if code in {'adjust_project_whitelist','request_platform_enablement','wait_within_window','verify_channel_timing','verify_duplicate_charge','reauthorize_then_verify'}:
            return 'incorrect','当前事实不足或矛盾，不能按已经成立的条件执行该建议'
        if code=='record_diagnostics':
            billing={'transaction_id','order_id','amount','currency','payment_channel','payment_time','approval_status','refund_stage','original_account_checked','received','transaction_summary'}
            if billing.intersection(payload.get('params',{}).get('fields',[])):
                return 'incorrect','技术调用诊断混入资金办理字段；本次无该资金对象'
            return 'legal','记录当前故障定位和影响字段有公开fault_handling依据；不替代升级或决定性待补'
        return 'unresolved','公开政策没有唯一确定此附加建议的相关性或适用条件'
    if section=='missing_info':
        entry=fields.get(field,{})
        if entry.get('state') in {'known','derived'}:
            return 'incorrect','公开材料已提供或可推导，不应再次作为缺失信息索取'
        if field in {'approval_status','elapsed_workdays','payment_channel','original_account_checked','received','refund_stage'}:
            return 'incorrect','当前决定性条件或业务对象不符；不能把实际办理或已知状态改成未知时效咨询'
        if field=='order_id':return 'incorrect','当前订单号已在公开业务表单给出'
        if field in {'transaction_id','amount','currency','payment_time','transaction_summary'}:
            return 'unresolved','真实人工核实可能需要，但非双扣情形的具体交接字段未由公开政策穷尽；本候选不放行'
        if field in {'request_id','event_time','response_summary','interface_name','occurrence','affected_users','recovered','request_read_only'}:
            if payload['purpose']=='decision':
                return 'unresolved','是否决定当前处置需看影响条件；不能把一般诊断字段一律升级为前置条件'
            return 'unresolved','一般诊断有价值，但本场景是否需要交接及该字段仍未唯一确定'
    return 'unresolved','当前公开规则不足以唯一裁定；保留原拒绝，不等同模型业务错误'


def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    runs=root/'runs/balanced-policy-v1'; old=runs/'teacher-collection'; new=runs/'teacher-diagnostic-guidance-v1'
    prior=runs/'teacher-diagnosis-v1'; output=runs/'teacher-contract-r6'; output.mkdir(exist_ok=True)
    config=read(old/'configuration.json'); tasks=load_public_tasks(root/config['tasks'])
    provenance = {}
    for label, folder in [('old', old), ('new', new)]:
        cfg = read(folder/'configuration.json')
        summary = read(folder/'summary-pilot.json')
        budget = read(folder/'budget.json')
        provenance[label] = dict(configuration_file=str((folder/'configuration.json').relative_to(root)),
            summary_file=str((folder/'summary-pilot.json').relative_to(root)),
            budget_file=str((folder/'budget.json').relative_to(root)),
            summary_attempts=len(summary['attempts']),requests=len(budget['requests']),totals=budget['totals'],
            generation=cfg['generation'],retrieval=cfg['retrieval'])
    # Existing diagnosis artifacts are evidence, never rewritten by this audit.
    comparison = read(prior/'standard_comparison.json')
    paired = read(prior/'paired_results.json')
    assert comparison['original_severe']==10 and comparison['revised_severe']==12
    assert paired['metrics']['new']['candidate_grade']['severe']==7
    frozen_policy = root/config['policy_support']
    assert frozen_policy.read_bytes()==(BACKEND/'rl/resources/policy_support.json').read_bytes()
    assert (frozen_policy.parent/'knowledge.md').read_bytes()==(BACKEND/'agents/customer_policy/resources/knowledge.md').read_bytes()
    versions={'r4':load_private_standards(root/config['standards']),
              'r5':load_private_standards(prior/'train_standards_candidate.json')}
    clauses=load_policy_support(root/config['policy_support']); revised={}; changes=[]
    for tid,task in tasks.items():
        revised[tid],change=amend_standard(task,versions['r5'][tid])
        assert [e.model_dump() for e in revised[tid].expectations[:len(versions['r5'][tid].expectations)]] == [e.model_dump() for e in versions['r5'][tid].expectations]
        assert revised[tid].severe_items==versions['r5'][tid].severe_items
        changes.append(dict(task_id=tid,**change))
    versions['r6']=revised; validate_task_pairs(tasks,revised,clauses)
    probes=verify(tasks,revised,clauses)
    write(output/'validation.json',dict(probes=probes,passed=len(probes),external_calls=0))
    write(output/'train_standards_candidate.json',[s.model_dump() for s in revised.values()])
    write(output/'standard_changes.json',dict(revision=REVISION,grading_changed=False,required_changed=False,severe_items_changed=False,changes=changes))
    old_notes={(r['task_id'],r['candidate']):r for r in read(prior/'attribution.json')}
    new_notes={r['task_id']:r for r in read(prior/'new_semantic_review.json')}
    records=[]
    for group,folder,candidate in [('old_first',old,1),('old_second',old,2),('new',new,1)]:
        for tid in config['pilot_task_ids']:
            path=folder/f'{tid}-candidate-{candidate}.json'; before=path.read_bytes(); row=read(path)
            tr=VisibleTrajectory.model_validate(row['trajectory']); raw=row['raw_result']; raw=json.loads(raw) if isinstance(raw,str) else raw
            grades={v:grade_result(ss[tid],row['raw_result'],tr,clauses).to_dict() for v,ss in versions.items()}
            assert grades['r5' if group=='new' else 'r4']==row['grade']
            states,deps=source_facts(tasks[tid]); standard=revised[tid]; items=[]; content_matches=set(); supported_facts=[]
            for section in ('facts','reasons','next_steps','missing_info'):
                for index,item in enumerate(raw[section]):
                    payload=_payload(item)
                    matches=[(i,e) for i,e in enumerate(standard.expectations) if e.section==section and any(_same_item(payload,a,e.required) for a in e.alternatives)]
                    if matches:
                        status,basis='legal','与由公开材料和政策导出的允许表达一致；引用质量另列'
                        content_matches.update(i for i,e in matches)
                        if section=='facts':supported_facts.append(payload)
                    else:status,basis=semantic(section,payload,standard,states)
                    support_ok=any(_supported(e,item['evidence'],tr,standard,clauses) for i,e in matches)
                    missing_groups=[]
                    for i,e in matches:
                        for g in e.supports:
                            if not _supported(e.model_copy(update={'supports':[g]}),item['evidence'],tr,standard,clauses):missing_groups.append(g)
                    old_matches={v:any(e.section==section and any(_same_item(payload,a,e.required) for a in e.alternatives) for e in versions[v][tid].expectations) for v in ('r4','r5')}
                    items.append(dict(section=section,index=index,payload=payload,evidence=item['evidence'],legality=status,basis=basis,
                        public_condition=states.get(payload['target'],{}).get(payload.get('type') or payload.get('field')),
                        matches_r6=bool(matches),matches_old=old_matches,support_ok=support_ok,missing_support_groups=missing_groups))
            for i,e in enumerate(standard.expectations):
                if e.section=='facts' and any(_equivalent_fact(a,supported_facts) for a in e.alternatives):content_matches.add(i)
            missing_content=[e.model_dump() for i,e in enumerate(standard.expectations) if e.required and i not in content_matches]
            note=new_notes[tid]['note'] if group=='new' else old_notes[(tid,candidate)]['semantic_review_note']
            categories=dict(fact_reason_action_error=any(i['legality']=='incorrect' for i in items) or raw['disposition'] not in standard.dispositions,
                required_content_omission=bool(missing_content),object_scope_error=any(i['legality']=='incorrect' and (i['payload']['target']=='case' or i['payload'].get('type') in {'interface_name','request_scope','feature_enabled'}) for i in items),
                evidence_chain_omission=any(any(any(s.startswith('user:') for s in g) for g in i['missing_support_groups']) for i in items),
                cited_text_not_supporting=any(any(any(s in clauses for s in g) for g in i['missing_support_groups']) or
                    (i['section']=='facts' and i['legality']=='incorrect' and any(ref in standard.materials for ref in i['evidence'])) for i in items),
                legal_expression_rejected_r4=any(i['legality']=='legal' and not i['matches_old']['r4'] for i in items),
                severity_missed_r4=not grades['r4']['severe_error'] and grades['r6']['severe_error'],
                protocol_error=grades['r6']['protocol_error'],unresolved=any(i['legality']=='unresolved' for i in items))
            records.append(dict(task_id=tid,group=group,candidate=candidate,source=str(path.relative_to(root)),grades=grades,
                source_facts=states,evidence_dependencies=deps,items=items,missing_required_content=missing_content,
                inherited_review=note,categories=categories,raw_unchanged=path.read_bytes()==before))
    assert len(records)==72 and all(r['raw_unchanged'] for r in records)
    metrics={g:{v:{k:sum(r['grades'][v][k] for r in records if r['group']==g) for k in ('complete_success','severe_error','protocol_error')} for v in versions} for g in ('old_first','old_second','new')}
    transitions={}
    for a,b in [('r4','r5'),('r5','r6'),('r4','r6')]:
        transitions[a+'->'+b]={k:{direction:[dict(task_id=r['task_id'],group=r['group'],before=r['grades'][a],after=r['grades'][b]) for r in records if bool(r['grades'][a][k])==before and bool(r['grades'][b][k])!=before] for direction,before in [('added',False),('removed',True)]} for k in ('complete_success','severe_error','protocol_error')}
    result=dict(revision=REVISION,metrics=metrics,transitions=transitions,attempts=72,external_calls=0,fee=0,
        category_counts={g:{k:sum(r['categories'][k] for r in records if r['group']==g) for k in records[0]['categories']} for g in metrics},
        limits=['逐项合法性与引用质量独立；未确认不自动判业务错。','协议错误不妨碍离线查看原始条目，但评分早退未检查内容仍不称已通过。'])
    by_group={(r['group'],r['task_id']):r for r in records}
    result['main_pairing']={v:{metric:{direction:[tid for tid in config['pilot_task_ids']
        if bool(by_group[('old_first',tid)]['grades'][v][metric])==before and
           bool(by_group[('new',tid)]['grades'][v][metric])!=before]
        for direction,before in [('added',False),('removed',True)]}
        for metric in ('complete_success','severe_error','protocol_error')} for v in versions}
    write(output/'recomputed_results.json',result); write(output/'attribution.json',records)
    write(output/'provenance.json',provenance)
    write(output/'unresolved_items.json',[dict(task_id=r['task_id'],group=r['group'],**i) for r in records for i in r['items'] if i['legality']=='unresolved'])
    print(json.dumps(dict(metrics=metrics,probes=len(probes),categories=result['category_counts']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
