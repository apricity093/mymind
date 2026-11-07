"""实际公开材料的事实来源边界；副本只作反例。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from agents.customer_policy.protocol import VisibleTrajectory
from benchmarks.scripts.success_pool_contract import align_workflow
from benchmarks.scripts.teacher_contract_candidate import source_facts
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support
from rl.grading import grade_result

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/success-pool-v1/teacher-expression-pair-v1'
if not RUN.exists():
    pytest.skip('需要单独保留的 teacher-expression-pair-v1 实验原件', allow_module_level=True)
TASKS = load_public_tasks(RUN/'inputs/public/B.json')
STANDARDS = load_private_standards(RUN/'inputs/private/B.json')
CLAUSES = load_policy_support(RUN/'inputs/policy/policy_support.json')
TID = 'mainline-balanced-P04-22'


def case(tid=TID):
    row=json.loads((RUN/(tid+'-B.json')).read_text(encoding='utf-8'))
    return align_workflow(TASKS[tid],STANDARDS[tid]),deepcopy(row['raw_result']),VisibleTrajectory.model_validate(row['trajectory'])


def test_direct_scope_unknown_and_valid_credential():
    s,raw,tr=case()
    # user:3明确给出当前对象、项目、环境、出口、登记状态和凭据有效。
    assert grade_result(s,raw,tr,CLAUSES).complete_success
    for kind in ('feature_enabled','outbound_ip_allowed','token_expired'):
        assert next(e for e in s.expectations if e.section=='facts' and e.alternatives[0].get('type')==kind).supports==[['user:3']]


@pytest.mark.parametrize('mutation', ['unknown_false','other_project','other_environment','wrong_object','wrong_reference','bad_duplicate','current_enablement'])
def test_wrong_assertions_still_fail(mutation):
    s,raw,tr=case()
    f=next(f for f in raw['facts'] if f['type']=='feature_enabled')
    if mutation=='unknown_false': f['value']['status']=False
    elif mutation=='other_project': f['value']['project']='其他项目'
    elif mutation=='other_environment': f['value']['environment']='test'
    elif mutation=='wrong_object': f['target']='Q2'
    elif mutation=='wrong_reference': f['evidence']=['user:2']
    elif mutation=='bad_duplicate':
        wrong=deepcopy(f);wrong['evidence']=['user:2'];raw['facts'].append(wrong)
    else: raw['next_steps'].append(dict(target='Q1',code='request_platform_enablement',params={k:f['value'][k] for k in ('project','environment')},evidence=['user:3']))
    assert not grade_result(s,raw,tr,CLAUSES).complete_success


def test_policy_comparison_requires_policy_and_future_action_remains_required():
    tid='mainline-M02-billing-event-split-tender'
    s,raw,tr=case(tid)
    grade=grade_result(s,raw,tr,CLAUSES)
    assert not grade.complete_success and any('verify_refund_if_overdue_unreceived' in m for m in grade.missing)
    f=next(f for f in raw['facts'] if f['type']=='within_window')
    f['evidence']=['user:3' if f['target']=='R1' else 'user:4']
    assert any('within_window' in e for e in grade_result(s,raw,tr,CLAUSES).errors)


def test_unprovided_recovery_remains_severe():
    s,raw,tr=case('mainline-balanced-F02-21')
    assert grade_result(s,raw,tr,CLAUSES).severe_error


@pytest.mark.parametrize('form', [{'凭据仍有效':False},{'凭据仍有效':'持有人明确表示不清楚'},{'响应码':401},{'响应码':403},{'响应明确过期':False}])
def test_no_reverse_or_status_code_auth_inference(form):
    task=TASKS[TID].model_copy(deep=True)
    body=json.loads(task.records[1].content)
    body['业务表单']=form
    task.records[1].content=json.dumps(body,ensure_ascii=False)
    fields=source_facts(task)[0]['Q1']
    if '响应明确过期' not in form:
        assert fields['token_expired']['state']=='not_provided'
    assert fields['auth_valid']['value'] is not True


def test_expired_implies_invalid_with_actual_premise_only():
    tid=next(t for t in load_public_tasks(ROOT/'runs/success-pool-v1/teacher-mainline-v1/inputs/tasks/candidates.json') if '-A01-' in t)
    tasks=load_public_tasks(ROOT/'runs/success-pool-v1/teacher-mainline-v1/inputs/tasks/candidates.json')
    standards=load_private_standards(ROOT/'runs/success-pool-v1/teacher-mainline-v1/inputs/standards/standards.json')
    s=align_workflow(tasks[tid],standards[tid])
    e=next(e for e in s.expectations if e.section=='facts' and e.alternatives[0].get('type')=='auth_valid')
    assert e.alternatives[0]['value'] is False and e.supports==[['user:3']]


def test_cross_record_scope_application_keeps_policy():
    task=TASKS[TID].model_copy(deep=True)
    body=json.loads(task.records[1].content)
    body['业务表单'].pop('功能登记')
    body['业务表单']['异范围功能登记']={'project':'其他项目','environment':'test','status':True}
    task.records[1].content=json.dumps(body,ensure_ascii=False)
    s=align_workflow(task,STANDARDS[TID])
    e=next(e for e in s.expectations if e.section=='facts' and e.alternatives[0].get('type')=='feature_enabled')
    assert ['config_scope'] in e.supports
