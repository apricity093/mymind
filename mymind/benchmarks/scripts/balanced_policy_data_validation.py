"""Read final public records independently of reference assembly and audit their rules."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json

from benchmarks.scripts.balanced_policy_build import (
    load, dump, public_states, SPLITS, CLAUSES, KNOWLEDGE,
    REVISION, GRADING_REVISION, PublicTask, PrivateStandard, TerminalResult,
    VisibleTrajectory, validate_task_pairs, grade_result, execution_status,
    check_distribution, SUBTYPES,
)
from agents.customer_policy.protocol import FACT_TYPES


def public_selection_audit(task, standard=None):
    """Check the two authored task-selection forms, not arbitrary prose."""
    origin=json.loads(task.records[0].content)
    records=origin.get('原始业务摘录',{})
    if '作业清单' in records:
        selected=records['用户选中']
        choices={'预览作业':('预览','夜批预览作业单','预览作业调用'),
                 '生成作业':('生成','夜批生成作业单','生成作业调用')}
        choice,document,call=choices[selected]
        purpose=next(r['用途'] for r in records['作业清单'] if r['作业']==choice)
    elif '用途分工' in records:
        selected=records['当前选择']
        document,call={'日常任务':('日常任务选择单','日常任务调用'),
                       '审计任务':('审计任务选择单','审计任务调用')}[selected]
        purpose=records['用途分工'][selected]
    else:
        return []
    # These purposes belong to the concrete preview/export jobs above.
    # This is not a rule that a read-only request must be basic_query.
    capability={'只读展示':'basic_query','文件生成':'export',
                '预览账簿':'basic_query','导出账簿':'export'}[purpose]
    issues=[]
    if origin['目标路径']['Q1']!=[document,call]:
        issues.append('当前选择与目标路径不一致')
    if {'业务项':document,'对应调用':call} not in origin['业务关联记录']:
        issues.append('当前选择与业务关联记录不一致')
    body=next((json.loads(r.content) for r in task.records[1:]
               if json.loads(r.content).get('记录归属')==call or
               json.loads(r.content).get('本次认证记录',{}).get('实际调用')==call),None)
    if body is None:
        issues.append('当前选择无对应表单记录')
        return issues
    operation=body['业务表单'].get('操作名称')
    if operation is not None:
        actual={'基础查询':'basic_query','报表预览':'basic_query',
                '报表导出':'export','批量报表生成':'export'}[operation]
        if actual!=capability: issues.append('当前选中用途与表单操作能力冲突')
        if standard is not None:
            expected={'required_capability':capability,'platform_enablement_required':capability=='export'}
            for kind,value in expected.items():
                values=[a['value'] for ex in standard.expectations if ex.section=='facts'
                        for a in ex.alternatives if a.get('target')=='Q1' and a.get('type')==kind]
                if values!=[value]: issues.append('当前用途与标准能力冲突:'+kind)
    return issues


def semantic_audit(task, standard, states):
    """A second, direct policy reading; never consumes the offline reference."""
    issues = public_selection_audit(task,standard)
    facts, missing, steps, reasons = {}, set(), set(), set()
    required_facts, required_missing, required_steps = set(), set(), set()
    decisions = []
    def fact(t, k, v, required=True):
        facts[t, k] = v
        if required:
            required_facts.add((t, k))
    def need(t, k, purpose='decision', condition=None):
        item = (t, k, purpose, condition)
        missing.add(item); required_missing.add(item)
    def step(t, k, required=True):
        steps.add((t, k))
        if required:
            required_steps.add((t, k))
    for t, fields in states.items():
        def get(k):
            e = fields.get(k, {})
            if e.get('state') != 'known':
                return None
            v = e['value']
            return v.get('status') if k in {'feature_enabled','outbound_ip_allowed','user_can_configure'} and isinstance(v, dict) else v
        for k, e in fields.items():
            if k in FACT_TYPES and e['state'] == 'known':
                if k == 'payment_channel' and e['value'] not in {'online_payment','bank_card','balance'}:
                    continue
                fact(t, k, e['value'], False)
        if t.startswith('R'):
            ch, approved, days = get('payment_channel'), get('approval_status'), get('elapsed_workdays')
            fact(t, 'approval_status', approved or 'unknown')
            if ch in {'online_payment','bank_card','balance'} or ch is None:
                fact(t, 'payment_channel', ch or 'unknown')
            if ch is None:
                need(t, 'payment_channel')
            fact(t, 'elapsed_workdays', days if approved == 'approved' and days is not None else 'unknown')
            if approved != 'approved':
                need(t,'approval_status'); need(t,'elapsed_workdays',condition='if_approved')
                reasons.add((t,'refund_state_unknown')); decisions.append('request_info'); continue
            fact(t,'timer_start','approval')
            if days is None:
                need(t,'elapsed_workdays')
            if ch is not None and ch not in {'online_payment','bank_card'}:
                fact(t,'timing_policy_covered',False); reasons.add((t,'policy_uncovered'))
                step(t,'verify_channel_timing'); decisions.append('escalate'); continue
            if ch is None or days is None:
                reasons.add((t,'refund_state_unknown')); decisions.append('request_info'); continue
            limit = {'online_payment':3,'bank_card':5}[ch]
            fact(t,'limit_workdays',limit); fact(t,'within_window',days <= limit)
            if days <= limit:
                reasons.add((t,'within_refund_window')); step(t,'wait_within_window'); step(t,'verify_refund_if_overdue_unreceived')
                decisions.append('answer'); continue
            reasons.add((t,'refund_overdue'))
            if get('received') is True:
                decisions.append('answer'); continue
            if get('received') is False and get('original_account_checked') is True:
                step(t,'verify_refund'); decisions.append('escalate'); continue
            for k in ('received','original_account_checked'):
                if get(k) is None:
                    need(t,k)
            if get('original_account_checked') is not True:
                step(t,'check_original_account')
            step(t,'verify_refund_if_overdue_unreceived')
            decisions.append('request_info' if get('received') is None or get('original_account_checked') is None else 'answer')
        elif t == 'O1':
            decisions.append('escalate')
            if get('reported_charge_count') == 2:
                fact(t,'reported_charge_count',2); fact(t,'charges_same_order',True)
                reasons.add((t,'suspected_duplicate_charge')); step(t,'verify_duplicate_charge')
                for k in ('transaction_id','payment_time','amount','payment_channel','transaction_summary'):
                    if get(k) is None:
                        need(t,k,'handoff')
            else:
                reasons.add((t,'evidence_insufficient')); step(t,'seek_human_review')
        elif get('http_status') == 403:
            scope, ip = get('request_scope'), get('outbound_ip')
            basic = get('interface_name') in {'基础查询','报表预览'}
            fact(t,'required_capability','basic_query' if basic else 'export')
            fact(t,'whitelist_required',True); fact(t,'platform_enablement_required',not basic)
            for k in ('interface_name','request_scope','outbound_ip','auth_valid','user_can_configure','http_status'):
                required_facts.add((t,k))
            white, feature = get('outbound_ip_allowed'), get('feature_enabled')
            fact(t,'outbound_ip_allowed',dict(project=scope['project'],outbound_ip=ip,status=white if white is not None else 'unknown'),fields['outbound_ip_allowed']['state'] != 'not_provided')
            if not basic:
                fact(t,'feature_enabled',dict(**scope,status=feature if feature is not None else 'unknown'), fields['feature_enabled']['state'] != 'not_provided')
            if not basic and feature is False:
                decisions.append('escalate'); reasons.add((t,'platform_enablement_required')); step(t,'request_platform_enablement')
                if white is False:
                    step(t,'adjust_project_whitelist',False)
                if white is None:
                    missing.add((t,'outbound_ip_allowed','decision',None))
            elif white is False and (basic or feature is True):
                decisions.append('answer'); reasons.add((t,'user_config_fixable')); step(t,'adjust_project_whitelist')
            else:
                decisions.append('request_info')
                ready = feature is True and white is True
                reasons.add((t,'evidence_insufficient' if ready else 'permission_state_unknown'))
                if not basic and feature is None:
                    need(t,'feature_enabled')
                if white is None:
                    need(t,'outbound_ip_allowed')
                if white is False:
                    step(t,'adjust_project_whitelist',False)
                if ready:
                    for k in ('request_id','event_time','response_summary'):
                        need(t,k)
                    step(t,'record_diagnostics',False)
        elif any(get(k) is True for k in ('payment_affected','data_loss','data_corruption')) or get('environment') == 'production' and get('widespread_outage') is True:
            decisions.append('escalate'); reasons.add((t,'severe_business_impact'))
            step(t,'escalate_technical'); step(t,'seek_human_review',False)
            for k in ('request_id','event_time','response_summary'):
                if get(k) is None:
                    need(t,k,'handoff')
            if get('token_expired') is True:
                reasons.add((t,'token_expired')); step(t,'reauthorize_then_verify')
        elif get('http_status') == 401:
            if get('token_expired') is True:
                decisions.append('answer'); fact(t,'auth_valid',False)
                reasons.add((t,'token_expired')); step(t,'reauthorize_then_verify')
            else:
                decisions.append('request_info'); reasons.add((t,'evidence_insufficient'))
                for k in ('token_expired','auth_valid'):
                    fact(t,k,'unknown'); need(t,k)
        elif get('recovered') is True and get('request_read_only') is True and get('impact_scope'):
            decisions.append('answer'); reasons.add((t,'limited_recovered_fault'))
            for k in ('record_diagnostics','low_risk_checks','escalate_if_severe_impact'):
                step(t,k)
        else:
            decisions.append('request_info'); reasons.add((t,'evidence_insufficient')); step(t,'escalate_if_severe_impact')
            for k in ('recovered','request_read_only','payment_affected','data_loss','data_corruption','affected_users'):
                if get(k) is None:
                    fact(t,k,'unknown'); need(t,k)
    expected_disposition = 'escalate' if 'escalate' in decisions else 'request_info' if 'request_info' in decisions else 'answer'
    if standard.dispositions != [expected_disposition]:
        issues.append('整体处置与逐对象政策推导不一致')
    seen_facts, seen_missing, seen_steps, seen_reasons = set(), set(), set(), set()
    for ex in standard.expectations:
        for a in ex.alternatives:
            t = a['target']
            if ex.section == 'facts':
                key = (t,a['type'])
                if key not in facts or facts[key] != a['value']:
                    issues.append('无据事实/错误作用域:'+str(a))
                if ex.required:
                    seen_facts.add(key)
            elif ex.section == 'missing_info':
                key = (t,a['field'],a['purpose'],a.get('condition'))
                if key not in missing:
                    issues.append('冗余或错误待补:'+str(a))
                if ex.required:
                    seen_missing.add(key)
            elif ex.section == 'next_steps':
                key = (t,a['code'])
                if key not in steps:
                    issues.append('错误建议:'+str(a))
                if ex.required:
                    seen_steps.add(key)
                if a['code'] == 'adjust_project_whitelist':
                    q = states[t]
                    wanted = dict(project=q['request_scope']['value']['project'],outbound_ip=q['outbound_ip']['value'])
                    if a.get('params') != wanted:
                        issues.append('白名单参数范围错误')
                if a['code'] == 'request_platform_enablement' and a.get('params') != states[t]['request_scope']['value']:
                    issues.append('功能参数范围错误')
            elif ex.section == 'reasons':
                if (t,a['code']) not in reasons:
                    issues.append('错误理由:'+str(a))
                if ex.required:
                    seen_reasons.add((t,a['code']))
    for label, required, seen in [('必要事实',required_facts,seen_facts),('必要待补',required_missing,seen_missing),('必要建议',required_steps,seen_steps),('必要理由',reasons,seen_reasons)]:
        if required - seen:
            issues.append(label+'缺少:'+str(sorted(required-seen)))
    for t, fields in states.items():
        absent = [k for k,e in fields.items() if e['state'] == 'not_provided']
        if standard.unprovided_user_facts.get(t) != absent:
            issues.append('未提供事实标注不一致:'+t)
        ch = fields.get('payment_channel',{}).get('value')
        days = fields.get('elapsed_workdays',{}).get('value')
        overdue = ch in {'online_payment','bank_card'} and fields.get('approval_status',{}).get('value') == 'approved' and type(days) is int and days > (3 if ch == 'online_payment' else 5)
        marked = dict(target=t,code='wait_within_window') in standard.severe_items.get('next_steps',[])
        if overdue != marked:
            issues.append('超时等待严重项不一致:'+t)
    return issues


def validate(data, logs):
    from benchmarks.scripts.balanced_policy_content import validate as validate_content
    validate_content(data, logs)
