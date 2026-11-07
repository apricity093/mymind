"""有限政策常量例外的零调用反例；修改副本仅作测试，不导出示范。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from agents.customer_policy.protocol import VisibleTrajectory
from benchmarks.scripts.success_pool_contract import align_evidence, align_workflow
from benchmarks.scripts.balanced_policy_design import fixture
from benchmarks.scripts.teacher_contract_audit import reference
from rl.grading import grade_result
from rl.tasks import load_private_standards, load_public_tasks, load_policy_support

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'agentic_runs/workflow-control-v3-originals'
if not RUN.exists():
    pytest.skip('需要单独保留的 workflow-control-v3-originals 实验原件', allow_module_level=True)
STANDARDS=load_private_standards(RUN/'inputs/standards/dev.json')
CLAUSES=load_policy_support(RUN/'inputs/policy_support/policy_support.json')


def case(tid='dev-balanced-R01-12-clear'):
    row=json.loads((RUN/(tid+'-B.json')).read_text(encoding='utf-8'))
    return align_evidence(STANDARDS[tid]),deepcopy(row['raw_result']),VisibleTrajectory.model_validate(row['trajectory'])


def evaluate(standard,raw,trajectory):
    return grade_result(standard,raw,trajectory,CLAUSES)


def test_both_original_r01_and_full_joint_expression():
    for tid in ('M6-R01-001','dev-balanced-R01-12-clear'):
        s,raw,tr=case(tid)
        assert evaluate(s,raw,tr).complete_success
        channel=next(f for f in raw['facts'] if f['type']=='payment_channel')
        limit=next(f for f in raw['facts'] if f['type']=='limit_workdays')
        limit['evidence']+=channel['evidence']
        assert evaluate(s,raw,tr).complete_success


@pytest.mark.parametrize('mutation',[
    'wrong_channel','wrong_limit','missing_channel','another_object',
    'wrong_channel_reference','wrong_constant_user_reference','wrong_policy','wrong_duplicate'])
def test_wrong_value_or_explicit_reference_is_not_hidden(mutation):
    s,raw,tr=case()
    channel=next(f for f in raw['facts'] if f['type']=='payment_channel')
    limit=next(f for f in raw['facts'] if f['type']=='limit_workdays')
    if mutation=='wrong_channel': channel['value']='bank_card'
    elif mutation=='wrong_limit': limit['value']=5
    elif mutation=='missing_channel': raw['facts'].remove(channel)
    elif mutation=='another_object': channel['target']='R2'
    elif mutation=='wrong_channel_reference': channel['evidence']=['user:2']
    elif mutation=='wrong_constant_user_reference': limit['evidence'].append('user:2')
    elif mutation=='wrong_policy':
        # 可见片段只保留另一渠道规则，不再包含本渠道适用条款。
        tr.searches[0].observation.snippets[0].content=CLAUSES['refund_bank_limit'].text
    elif mutation=='wrong_duplicate':
        wrong=deepcopy(limit);wrong['evidence']=['search:1:2'];raw['facts'].append(wrong)
    assert not evaluate(s,raw,tr).complete_success


def test_unknown_or_uncovered_channel_cannot_select_limit():
    cfg=json.loads((ROOT/'runs/balanced-policy-v1/teacher-collection/configuration.json').read_text(encoding='utf-8'))
    tasks=load_public_tasks(ROOT/cfg['tasks'])  # 仅旧train，不读取旧test。
    standards=load_private_standards(ROOT/'runs/balanced-policy-v1/teacher-contract-r6/train_standards_candidate.json')
    for group in ('R05','R06'):
        tid=next(t for t,s in standards.items() if s.group_id==group)
        s=align_evidence(align_workflow(tasks[tid],standards[tid]))
        tr=fixture(tasks[tid]);raw=reference(s,tr,CLAUSES)
        assert evaluate(s,raw,tr).complete_success
        target=next(f['target'] for f in raw['facts'] if f['type']=='payment_channel')
        raw['facts'].append(dict(target=target,type='limit_workdays',value=3,evidence=['search:1:1']))
        assert not evaluate(s,raw,tr).complete_success


def test_constants_do_not_supply_comparison_or_current_escalation():
    s,raw,tr=case()
    raw['facts'].append(dict(target='R1',type='within_window',value=True,evidence=['search:1:1']))
    assert evaluate(s,raw,tr).severe_error
    s,raw,tr=case()
    raw['disposition']='escalate'
    raw['next_steps'].append(dict(target='R1',code='verify_refund',evidence=['search:1:1']))
    assert not evaluate(s,raw,tr).complete_success
    # 同一常量不能替代policy_covered条目自身的用户依据。
    s,raw,tr=case()
    next(f for f in raw['facts'] if f['type']=='timing_policy_covered')['evidence']=['search:1:1']
    assert not evaluate(s,raw,tr).complete_success


def test_capability_rule_uses_only_its_own_supported_prerequisite():
    base=ROOT/'runs/success-pool-v1/source-dev-v1'
    tasks=load_public_tasks(base/'public/dev.json')
    standards=load_private_standards(base/'private/dev.json')
    tid='dev-balanced-P04-10'
    s=align_evidence(align_workflow(tasks[tid],standards[tid]))
    tr=fixture(tasks[tid]);raw=reference(s,tr,CLAUSES)
    assert evaluate(s,raw,tr).complete_success
    constant=next(f for f in raw['facts'] if f['type']=='platform_enablement_required')
    constant['evidence']=['search:1:2']
    assert evaluate(s,raw,tr).complete_success
    wrong=deepcopy(raw)
    next(f for f in wrong['facts'] if f['type']=='required_capability')['target']='Q2'
    assert not evaluate(s,wrong,tr).complete_success
    constant['value']=False
    assert evaluate(s,raw,tr).severe_error
