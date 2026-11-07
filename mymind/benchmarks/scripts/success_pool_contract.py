"""有限成功池契约：事实、当前决策与可选交接材料。"""
from copy import deepcopy
import json

from agents.customer_policy.protocol import FACT_TYPES, SCOPED_FACTS, USER_FACTS, Fact
from benchmarks.scripts.balanced_policy_events import LABELS
from rl.tasks import Expectation
from benchmarks.scripts.teacher_contract_candidate import source_facts

REVISION = 'success-pool-contract-v1'
CONSISTENCY_REVISION = 'success-pool-contract-v2'
EVIDENCE_REVISION = 'success-pool-evidence-v5'
BILLING_LOCATION = {'transaction_id', 'payment_time', 'amount', 'currency', 'transaction_summary'}
TECH_LOCATION = {'request_id', 'event_time', 'response_summary', 'interface_name', 'occurrence'}
IMPACT = {'affected_users', 'recovered', 'request_read_only', 'payment_affected',
          'write_affected', 'data_loss', 'data_corruption', 'widespread_outage'}


def component_sources(task):
    """解码当前业务表单及公开路由的四种复合字段，不从历史记录继承状态。"""
    states, _ = source_facts(task)
    origin = json.loads(task.records[0].content)
    raw = origin.get('原始业务摘录', {})
    result = {}
    for target, fields in states.items():
        record = next(r for r in task.records[1:] if json.loads(r.content).get('对象') == target
                      or json.loads(r.content).get('记录归属') == origin['目标路径'][target][-1])
        body = json.loads(record.content)
        direct = [record.reference_id] if body.get('对象') == target else [task.records[0].reference_id, record.reference_id]
        scope = deepcopy(fields.get('request_scope', {}).get('value') or {})
        scope_refs = {k:direct for k in scope}
        if not scope.get('project'):
            project = raw.get('书签解析', {}).get('当前项目') or raw.get('演练路由', {}).get('目标项目')
            if not project and raw.get('本次租户'):
                project = next((row['项目'] for row in raw.get('租户绑定', [])
                                if row['租户'] == raw['本次租户']), None)
            if project:
                scope['project'] = project; scope_refs['project'] = [task.records[0].reference_id]
        if not scope.get('environment') and fields.get('environment', {}).get('value') in {'test', 'production'}:
            scope['environment'] = fields['environment']['value']
            scope_refs['environment'] = [task.records[0].reference_id]
        # r6已从关系正文得到部分环境，不能误记成同记录表单直接提供。
        form = {LABELS[k]:v for k,v in body['业务表单'].items()}
        if 'request_scope' not in form:
            for k in scope_refs: scope_refs[k] = [task.records[0].reference_id]
        kinds = {}
        for kind, keys in SCOPED_FACTS.items():
            supplied = form.get(kind)
            supplied = supplied if isinstance(supplied,dict) else {}
            parts = {}
            for key in sorted(keys):
                value = supplied.get(key)
                refs = direct
                state = 'known'
                if value is None and key in {'project','environment'}:
                    value = scope.get(key); refs = scope_refs.get(key, [])
                    state = 'derived'
                if value is None and key == 'outbound_ip':
                    value = fields.get('outbound_ip', {}).get('value')
                if value is None:
                    state = 'explicit_unknown' if fields.get(kind,{}).get('state') == 'explicit_unknown' else 'not_provided'
                elif value == 'unknown':
                    state = 'explicit_unknown'
                parts[key] = dict(state=state,value=value,supports=refs)
            kinds[kind] = parts
        result[target] = kinds
    return result


def align_workflow(task, standard):
    """恢复审核未知时的可选计时事实；保留审核状态和条件性待补要求。"""
    result = reconcile(task, standard)
    pending = {a['target'] for e in result.expectations if e.section == 'facts'
               for a in e.alternatives
               if a.get('type') == 'approval_status' and a.get('value') == 'unknown'}
    expired = {a['target'] for e in result.expectations if e.section == 'facts'
               for a in e.alternatives if a.get('type') == 'token_expired' and a.get('value') is True}
    within = {a['target'] for e in result.expectations if e.section == 'facts'
              for a in e.alternatives if a.get('type') == 'within_window' and a.get('value') is True}
    for expected in result.expectations:
        if all((expected.section == 'facts' and (
                (a.get('target') in expired and a.get('type') == 'auth_valid' and a.get('value') is False)
                or (a.get('target') in within and a.get('type') in {'received','original_account_checked'})))
                or (expected.section == 'reasons' and a.get('target') in within
                    and a.get('code') == 'within_refund_window') for a in expected.alternatives):
            expected.required = False
            expected.decisive = False
        if expected.section == 'facts' and all(
                a.get('target') in pending and a.get('type') == 'elapsed_workdays'
                and a.get('value') == 'unknown' for a in expected.alternatives):
            expected.required = False
            expected.decisive = False
    for target in pending:
        channel = next((a['value'] for e in result.expectations if e.section == 'facts'
                        for a in e.alternatives if a.get('target') == target
                        and a.get('type') == 'payment_channel'), None)
        if channel in {'online_payment', 'bank_card'}:
            payload = dict(target=target, code='verify_refund_if_overdue_unreceived',
                           params={'limit_workdays': 3 if channel == 'online_payment' else 5})
            if not any(payload in e.alternatives for e in result.expectations):
                refs = assertion_records(task, 'next_steps', payload, result.user_fact_components)
                result.expectations.append(Expectation(section='next_steps', alternatives=[payload],
                    supports=[[r] for r in refs]+[['refund_window']], required=False))
    if task.role == 'technical' and result.dispositions == ['escalate']:
        for target in result.user_fact_components:
            payload = dict(target=target, code='record_diagnostics', params={'fields': sorted(TECH_LOCATION)})
            refs = assertion_records(task, 'next_steps', payload, result.user_fact_components)
            result.expectations.append(Expectation(section='next_steps', alternatives=[payload],
                supports=[[r] for r in refs]+[['fault_handling']], required=False))
    return result


def align_evidence(standard):
    """统一两类条件性规则常量的直接支持；组合表达由评分器限定验证。"""
    from rl.grading import POLICY_CONSTANT_RULES
    result = standard.model_copy(deep=True)
    for expected in result.expectations:
        if expected.section != 'facts':
            continue
        groups = []
        for alt in expected.alternatives:
            rule = POLICY_CONSTANT_RULES.get(alt.get('type'))
            if rule is None:
                break
            prerequisite, values = rule
            premise = next((e for e in result.expectations if e.section == 'facts'
                and any(a.get('target') == alt['target'] and a.get('type') == prerequisite
                        and a.get('value') in values and values[a['value']][0] == alt['value']
                        for a in e.alternatives)), None)
            if premise is None:
                raise ValueError(f"规则常量缺少适用前提：{alt}")
            value = next(a['value'] for a in premise.alternatives
                         if a.get('target') == alt['target'] and a.get('type') == prerequisite)
            groups.extend(premise.supports + [[values[value][1]]])
        else:
            expected.supports = [list(g) for g in dict.fromkeys(tuple(g) for g in groups)]
    return result


def assertion_records(task, section, payload, components):
    """只收集该断言实际使用的记录；政策支持由原业务规则独立保留。"""
    origin = json.loads(task.records[0].content)
    target = payload['target']
    targets = list(origin['目标路径']) if target == 'case' else [target]
    refs = []
    for current in targets:
        record = next(r for r in task.records[1:] if json.loads(r.content).get('对象') == current
                      or json.loads(r.content).get('记录归属') == origin['目标路径'][current][-1])
        body = json.loads(record.content)
        base = [record.reference_id] if body.get('对象') == current else [task.records[0].reference_id,record.reference_id]
        kind = payload.get('type') if section == 'facts' else payload.get('field') if section == 'missing_info' else None
        if kind == 'environment' and '运行环境' not in body['业务表单'] and '请求归属' not in body['业务表单']:
            needed = components[current]['request_scope']['environment']['supports']
            refs.extend(needed or base)
        elif kind in SCOPED_FACTS and section == 'facts':
            for key in payload['value']:
                refs.extend(components[current][kind][key]['supports'])
            if not refs: refs.extend(base)
        else:
            refs.extend(base)
    return list(dict.fromkeys(refs))


def reconcile(task, standard):
    """在相同公开输入上修订私有分量来源及逐断言证据，不改必要业务项。"""
    result = standard.model_copy(deep=True)
    result.materials = {'user:1':task.user_text, **{r.reference_id:r.content for r in task.records}}
    result.user_fact_components = component_sources(task)
    for target, kinds in result.user_fact_components.items():
        partial = {kind for kind, parts in kinds.items()
                   if any(p['state'] in {'known','derived'} for p in parts.values())}
        result.unprovided_user_facts[target] = [k for k in result.unprovided_user_facts.get(target,[]) if k not in partial]
    for expected in result.expectations:
        if not any(any(ref.startswith('user:') for ref in group) for group in expected.supports):
            continue
        records = []
        for alt in expected.alternatives:
            records.extend(assertion_records(task,expected.section,alt,result.user_fact_components))
        policy_groups = [g for g in expected.supports if not any(ref.startswith('user:') for ref in g)]
        expected.supports = [[r] for r in dict.fromkeys(records)] + policy_groups
    return align_fact_sources(task, result)


def align_fact_sources(task, standard):
    """直接状态及有限语义推导只需实际材料；规则应用保留政策支持。"""
    result = standard.model_copy(deep=True)
    states, _ = source_facts(task)
    for expected in result.expectations:
        if expected.section != 'facts':
            continue
        direct = True
        for alt in expected.alternatives:
            kind, value = alt['type'], alt['value']
            entry = states.get(alt['target'], {}).get(kind, {})
            if kind not in USER_FACTS:
                direct = False; break
            if kind in SCOPED_FACTS:
                parts = result.user_fact_components[alt['target']][kind]
                # 同记录给出范围和状态（含明确未知），不涉及其他记录适用性。
                refs = {tuple(p['supports']) for p in parts.values()}
                supplied = all(p['state'] in {'known', 'explicit_unknown'} for p in parts.values())
                # 明确未知的整项状态使用本记录请求范围，不把缺失记录当作直接状态。
                if entry.get('state') == 'explicit_unknown' and entry.get('value') is None:
                    supplied = all(p['state'] in {'known', 'derived'} for k,p in parts.items() if k != 'status')
                matches = all(value.get(k) == (p['value'] if p['value'] is not None else 'unknown')
                              for k,p in parts.items())
                scope = states[alt['target']].get('request_scope', {}).get('value') or {}
                matches = matches and all(value[k] == scope[k] for k in ('project','environment') if k in value and k in scope)
                matches = matches and ('outbound_ip' not in value or value['outbound_ip'] == states[alt['target']].get('outbound_ip', {}).get('value'))
                direct = supplied and matches and len(refs) == 1 and entry.get('state') in {'known','explicit_unknown'}
            else:
                direct = (entry.get('state') in {'known','explicit_unknown','derived'}
                          and value == (entry.get('value') if entry.get('value') is not None else 'unknown')
                          and not (entry.get('state') == 'derived' and kind == 'request_read_only'))
            if not direct:
                break
        if direct:
            refs = [r for alt in expected.alternatives
                    for r in assertion_records(task, 'facts', alt, result.user_fact_components)]
            expected.supports = [[r] for r in dict.fromkeys(refs)]
    return result


def handoff_fields(task, standard, target):
    if 'escalate' not in standard.dispositions:
        return set()
    if task.role == 'technical':
        return TECH_LOCATION | IMPACT
    # 不向另一笔仍在常规范围内的退款索取升级材料。
    relevant = any(e.section == 'next_steps' and any(a.get('target') == target and
        a.get('code') in {'verify_refund', 'verify_duplicate_charge', 'seek_human_review',
                          'verify_channel_timing'} for a in e.alternatives) for e in standard.expectations)
    return BILLING_LOCATION if relevant else set()


def amend(task, standard):
    result = standard.model_copy(deep=True)
    states, evidence = source_facts(task)

    def add(section, payload, supports):
        if not any(e.section == section and payload in e.alternatives for e in result.expectations):
            result.expectations.append(Expectation(section=section, alternatives=[payload],
                supports=supports, required=False, decisive=False))

    for target, fields in states.items():
        refs = [[r] for r in evidence[target]]
        for kind, state in fields.items():
            if kind not in FACT_TYPES or state['state'] not in {'known', 'derived'}:
                continue
            payload = dict(target=target, type=kind, value=deepcopy(state['value']))
            try:
                Fact.model_validate({**payload, 'evidence': []})
            except ValueError:
                continue  # 不完整复合范围不制造项目名。
            supports = refs
            if state['state'] == 'derived' and kind in {'token_expired', 'auth_valid'}:
                supports = refs + [['auth_expired']]
            if state['state'] == 'derived' and kind == 'request_read_only':
                supports = refs + [['capabilities']]
            add('facts', payload, supports)
        for field in handoff_fields(task, result, target):
            if fields.get(field, {}).get('state') not in {'known', 'derived'}:
                add('missing_info', dict(target=target, field=field, purpose='handoff'),
                    refs + [['fault_handling' if task.role == 'technical' else 'billing_handoff']])
        # 500/超时的频率可作为诊断事实；unknown不替代任何必需影响项。
        if task.role == 'technical' and any(k in fields for k in ('http_status', 'timeout_observed')):
            if fields.get('http_status', {}).get('value') == 500 or 'timeout_observed' in fields:
                if fields.get('occurrence', {}).get('state') in {'not_provided', 'explicit_unknown'}:
                    add('facts', dict(target=target, type='occurrence', value='unknown'), refs)
        if task.role == 'billing' and target == 'O1' and 'escalate' in result.dispositions:
            if fields.get('funds_request', {}).get('value') == '实际退款':
                add('next_steps', dict(target=target, code='verify_refund'), refs + [['billing_handoff']])
    result.split = 'dev'
    return result


def adjudicate(item, task, standard):
    """对保留未决项应用同一有限规则；合法性不等于证据已完整。"""
    p, section = item['payload'], item['section']
    target = p['target']
    states, _ = source_facts(task)
    fields = states.get(target, {})
    revised = amend(task, standard)
    if any(e.section == section and p in e.alternatives for e in revised.expectations):
        return 'legal', '有限契约允许；仍须逐项证据及当前必要处置完整'
    if section == 'facts':
        if p['type'] in {'request_scope', 'feature_enabled', 'outbound_ip_allowed'}:
            return 'incorrect', '未提供的项目/接口/出口不得用执行单、记录或描述标签替代；unknown不豁免范围'
        return 'incorrect', '无据、矛盾或与当前任务无关的附加事实；未知不等于可任意扩展范围'
    if section == 'missing_info':
        field = p['field']
        if fields.get(field, {}).get('state') in {'known', 'derived'}:
            return 'incorrect', '已有或公开可推导信息不得重复索取'
        if p['purpose'] == 'handoff' and field in handoff_fields(task, revised, target):
            return 'legal', '当前已需升级，可选交接资料不作为升级前提'
        return 'incorrect', '当前无需该交接或非决定性字段；不得以诊断字段延迟已确定的处置'
    return 'incorrect', '当前公开状态不支持该理由或即时动作；不能用未决命名掩盖已知条件冲突'
