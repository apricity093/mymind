"""Train-only r6 contract candidate; no frozen files or scoring code are changed."""
from copy import deepcopy
import json

from agents.customer_policy.protocol import USER_FACTS
from benchmarks.scripts.balanced_policy_events import read_event
from benchmarks.scripts.balanced_policy_design import profile_example
from benchmarks.scripts.balanced_policy_candidates import classify, actual_subtype
from benchmarks.scripts.balanced_policy_content import expectation_support
from rl.tasks import Expectation

REVISION = 'balanced-policy-v1-train-contract-r6-candidate'
BILLING_FACTS = {'payment_channel', 'refund_stage', 'approval_status', 'elapsed_workdays',
                 'original_account_checked', 'received', 'reported_charge_count', 'charges_same_order'}
TECHNICAL_FACTS = USER_FACTS - BILLING_FACTS


def source_facts(task):
    """Classify the finite supported business fields after resolving public joins.

    Absence is recorded only after the current response, scope and event records
    have been considered. It does not add optional unknowns to the answer.
    """
    states, evidence = read_event(task)
    origin = json.loads(task.records[0].content)
    for target, fields in states.items():
        for entry in fields.values():
            if entry['state'] == 'known':
                entry['basis'] = 'public_form_or_current_record'
        def derive(field, value, basis):
            if fields.get(field, {}).get('state') in {None, 'not_provided'}:
                fields[field] = dict(state='derived', value=value, basis=basis)
        if fields.get('auth_valid', {}).get('value') is True:
            derive('token_expired', False, '本次凭据有效蕴含本次未过期；字段语义，不表示已恢复认证')
        if fields.get('token_expired', {}).get('value') is True:
            derive('auth_valid', False, '本次凭据明确过期蕴含本次无效；不从401单独推断')
        scope = fields.get('request_scope', {}).get('value')
        if isinstance(scope, dict):
            derive('environment', scope['environment'], '本次请求归属.environment')
        if fields.get('interface_name', {}).get('value') in {'基础查询', '报表预览'}:
            derive('request_read_only', True, '公开操作属于基础查询能力；capabilities')
        # These public records explicitly locate the selected call's environment.
        raw = origin.get('原始业务摘录', {})
        route = raw.get('演练路由', {})
        if route.get('目标资源环境') in {'test', 'production'}:
            derive('environment', route['目标资源环境'], '公开演练路由.目标资源环境（非客户端环境）')
        if '部署表将蓝实例关联测试项目' in origin.get('关系说明', ''):
            derive('environment', 'test', '本次蓝实例回放的公开部署关联')
        if fields.get('environment', {}).get('state') in {'known', 'derived'} and not isinstance(scope, dict):
            derive('request_scope', {'environment': fields['environment']['value']},
                   '公开关联仅确定目标环境；项目名称仍未给出，不将整个复合字段标为未提供')
        impact = fields.get('impact_scope', {}).get('value')
        if impact == '仅本次派生读取，原业务及其他任务正常':
            derive('widespread_outage', False, '本次影响有限且其他任务正常；仅当前对象范围')
        if fields.get('data_loss', {}).get('state') == 'known':
            body = next(json.loads(r.content) for r in task.records[1:]
                        if json.loads(r.content).get('对象') == target or
                        json.loads(r.content).get('记录归属') == origin['目标路径'][target][-1])
            if '恢复后对账' in body:
                fields['data_loss']['state'] = 'derived'
                fields['data_loss']['basis'] = '当前批次最终记录实有数小于应有数，已排除展示延迟'
        for field in sorted(BILLING_FACTS if task.role == 'billing' else TECHNICAL_FACTS):
            fields.setdefault(field, dict(state='not_provided', value=None, basis='当前公开材料及规则无成立依据'))
    return states, evidence


def amend_standard(task, standard):
    """Apply r6 to a saved r5 standard, preserving required and severe entries."""
    result = standard.model_copy(deep=True)
    states, evidence = source_facts(task)
    decoded, _ = read_event(task)
    sid, _ = classify(task.role, decoded)
    generated = profile_example(sid, actual_subtype(sid, decoded), decoded)
    additions = []
    for expected in generated.expects:
        alt = expected['alternatives'][0]
        if expected['section'] != 'facts' or alt.get('type') != 'limit_workdays' or expected['required']:
            continue
        expected = deepcopy(expected)
        expected['supports'] = expectation_support(expected, evidence)
        additions.append(expected)
    for target, fields in states.items():
        for field, entry in fields.items():
            if entry['state'] != 'derived' or field in {'data_loss', 'request_scope'}:
                continue
            supports = [[ref] for ref in evidence[target]]
            if field == 'environment' and not isinstance(fields.get('request_scope', {}).get('value'), dict):
                supports = [['user:2']]
            if field in {'auth_valid', 'token_expired'}:
                supports += [['auth_expired']]
            if field == 'request_read_only':
                supports += [['capabilities']]
            additions.append(dict(section='facts', alternatives=[dict(target=target,type=field,value=entry['value'])],
                                  supports=supports,required=False,decisive=False))
    if sid.startswith(('R', 'M')):
        for target in states:
            additions.append(dict(section='facts', alternatives=[dict(target=target,type='timer_start',value='approval')],
                                  supports=[['refund_clock']],required=False,decisive=False))
    # Currency qualifies the amount requested for duplicate-charge verification.
    if sid == 'B01' and decoded['O1'].get('amount', {}).get('state') == 'not_provided':
        additions.append(dict(section='missing_info', alternatives=[dict(target='O1',field='currency',purpose='handoff')],
                              supports=[[ref] for ref in evidence['O1']]+[['billing_handoff']],required=False,decisive=False))
    if sid in {'F01', 'F02', 'F03'}:
        supports = [[ref] for ref in evidence['Q1']] + [['fault_handling']]
        for field in ('event_time', 'interface_name', 'request_id', 'occurrence', 'response_summary'):
            if states['Q1'].get(field, {}).get('state') in {None, 'not_provided'}:
                additions.append(dict(section='missing_info',alternatives=[dict(target='Q1',field=field,purpose='handoff')],
                                      supports=supports,required=False,decisive=False))
        for field in ('occurrence', 'affected_users', 'request_read_only', 'recovered'):
            if states['Q1'].get(field, {}).get('state') == 'not_provided':
                additions.append(dict(section='facts',alternatives=[dict(target='Q1',type=field,value='unknown')],
                                      supports=[[ref] for ref in evidence['Q1']],required=False,decisive=False))
    added = []
    for expected in additions:
        alt = expected['alternatives'][0]
        if not any(e.section == expected['section'] and alt in e.alternatives for e in result.expectations):
            result.expectations.append(Expectation.model_validate(expected)); added.append(expected)
    result.unprovided_user_facts = {
        target: sorted(field for field, entry in fields.items() if entry['state'] == 'not_provided')
        for target, fields in states.items()}
    return result, dict(additions=added, unprovided_before=standard.unprovided_user_facts,
                        unprovided_after=result.unprovided_user_facts, source_facts=states)
