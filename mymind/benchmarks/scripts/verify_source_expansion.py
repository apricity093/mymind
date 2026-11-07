"""从公开材料核对业务语义，参考评分仅作辅助；不调用服务。"""
import os
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.scripts.prepare_mainline_candidates import read, write
from benchmarks.scripts.prepare_source_expansion import REL
from benchmarks.scripts.balanced_policy_design import fixture
from benchmarks.scripts.teacher_contract_audit import reference
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs
from rl.grading import grade_result, GRADING_REVISION


def verify(root):
    out=root/REL
    tasks=load_public_tasks(out/'public/candidates.json')
    standards=load_private_standards(out/'private/standards.json')
    clauses=load_policy_support(out/'policy/policy_support.json')
    meta=read(out/'metadata/task_manifest.json')
    families=read(out/'metadata/source_families.json')
    eligibility=read(out/'metadata/source_eligibility.json')
    assert len(tasks)==len(meta)==38 and len(families)==16
    assert Counter(m['stratum_id'] for m in meta)==dict(A01=6,B01=10,F02=6,M02=5,P04=6,R01=5)
    assert sorted(f['task_count'] for f in families)==[1]*5+[3]*11
    assert not {f['family_id'] for f in families}&set(eligibility['excluded_dev_families'])
    assert not {f['family_id'] for f in families}&{f['family_id'] for f in eligibility['sealed_families']}
    assert all(m['split'] is None and m['formal_pool_eligible'] for m in meta)
    assert all(s.split=='candidate' for s in standards.values())
    old_manifest=read(root/'runs/success-pool-v1/mainline-candidates-v1/metadata/task_manifest.json')
    assert not {m['family_id'] for m in meta}&{m['family_id'] for m in old_manifest}
    for item in read(out/'private/source_transformations.json'):
        before=read_content_object(item['original']['records'][1])['业务表单']
        after=read_content_object(item['candidate']['records'][1])['业务表单']
        assert before==after  # 包括订单号、false、未知及计时原值，不修改状态来利用来源。
    validate_task_pairs(tasks,standards,clauses)
    reviews=[]; references=[]
    for m in meta:
        tid=m['task_id']; task=tasks[tid]; s=standards[tid]; layer=m['stratum_id']
        origin=read_content(task.records[0]); bodies=[read_content(r) for r in task.records[1:]]
        facts={(a['target'],a['type']):a['value'] for e in s.expectations if e.section=='facts' and e.required for a in e.alternatives}
        steps={(a['target'],a['code']) for e in s.expectations if e.section=='next_steps' and e.required for a in e.alternatives}
        missing={(a['target'],a['field'],a.get('purpose'),a.get('condition')) for e in s.expectations if e.section=='missing_info' and e.required for a in e.alternatives}
        assert {b['对象'] for b in bodies}==set(origin['目标路径'])
        for b in bodies:
            target=b['对象']; path=origin['目标路径'][target]
            assert b.get('记录归属',path[-1])==path[-1]
            assert any(row['业务项']==path[0] and path[-1] in row.values() for row in origin['业务关联记录'])
        form=bodies[0]['业务表单']; target=bodies[0]['对象']
        if layer=='A01':
            auth=bodies[0]['本次认证记录']
            assert auth['实际调用']==origin['目标路径']['Q1'][-1]
            assert auth['客户端与凭据来源'] and auth['脱敏响应']['error']=='token_expired'
            assert facts['Q1','token_expired'] is True and facts['Q1','http_status']==401
            assert ('Q1','reauthorize_then_verify') in steps and s.dispositions==['answer']
            assert all(not e.required for e in s.expectations if any(a.get('type')=='auth_valid' for a in e.alternatives))
            basis='认证记录直接给本次调用、凭据归属及token_expired响应；不由401单独推断过期。'
        elif layer=='B01':
            assert form['显示扣款条数']==2 and form['两条对应同一订单'] is True
            assert ('O1','verify_duplicate_charge') in steps and s.dispositions==['escalate']
            for label,kind in [('流水号','transaction_id'),('交易时间','payment_time'),('金额','amount'),('支付类别','payment_channel'),('流水摘要','transaction_summary'),('订单号','order_id')]:
                assert (('O1',kind,'handoff',None) in missing)==(label not in form)
            assert not any(kind in {'approval_status','within_window','elapsed_workdays'} for t,kind in facts)
            basis='两条报告绑定同一订单，仍只是疑似；已知定位资料不再索取，真实缺项仅handoff，不引入退款审核计时。'
        elif layer=='M02':
            assert len(bodies)==2 and s.dispositions==['escalate']
            assert len({b['支付绑定']['对应资金记录'] for b in bodies})==2
            windows=[]
            for b in bodies:
                t=b['对象']; f=b['业务表单']; limit={'online_payment':3,'bank_card':5}[f['支付类别']]
                within=f['通过后完整工作日']<=limit; windows.append(within)
                assert f['审核回执']=='approved' and f['查过原支付账户'] is True and f['原账户收到款'] is False
                assert facts[t,'limit_workdays']==limit and facts[t,'within_window']==within
                assert facts[t,'elapsed_workdays']==f['通过后完整工作日'] and facts[t,'payment_channel']==f['支付类别']
                assert (t,'wait_within_window' if within else 'verify_refund') in steps
                if within:
                    assert any(a.get('code')=='verify_refund_if_overdue_unreceived' and a['target']==t for e in s.expectations for a in e.alternatives)
                    assert all(not e.required for e in s.expectations if e.section=='facts' and any(a.get('target')==t and a.get('type') in {'received','original_account_checked'} for a in e.alternatives))
            assert windows==[False,True]
            basis='两笔独立实付绑定逐笔审核计时，R1超过自身渠道阈值且核对未到账，R2窗口内；整体升级，R2保留未来条件建议。'
        elif layer=='P04':
            scope=form['请求归属']; ip=form['实际出口']
            assert form['凭据仍有效'] is True and form['响应码']==403
            assert form['操作名称']=='批量报表生成'
            assert form['出口登记']['project']==scope['project'] and form['出口登记']['outbound_ip']==ip
            assert facts['Q1','required_capability']=='export'
            assert facts['Q1','feature_enabled']==dict(**scope,status='unknown')
            assert facts['Q1','outbound_ip_allowed']==form['出口登记']
            assert ('Q1','feature_enabled','decision',None) in missing and s.dispositions==['request_info']
            assert not any(code=='request_platform_enablement' for t,code in steps)
            basis='当前项目环境的导出功能明确未知，白名单按项目+实际出口已知true/false；补功能，false时可选自助修正，不立即要求开通。'
        elif layer=='F02':
            assert s.dispositions==['escalate'] and ('Q1','escalate_technical') in steps
            severe={'大面积不可用':'widespread_outage','核对发现丢失':'data_loss','核对发现错乱':'data_corruption','支付链路异常':'payment_affected'}
            assert any(form.get(k) is True for k in severe)
            for label,kind in severe.items():
                if label in form: assert facts['Q1',kind]==form[label]
            if form.get('大面积不可用'): assert form['运行环境']=='production'
            for label,kind in [('请求定位号','request_id'),('发生时间','event_time'),('脱敏摘要','response_summary')]:
                assert (('Q1',kind,'handoff',None) in missing)==(label not in form)
            assert bodies[0]['业务影响归属']['调用']==origin['目标路径']['Q1'][-1]
            basis='严重影响由本次调用的业务批次记录给出；生产大面积不可用/未恢复丢失/错乱/支付异常独立标注，日志不齐不延迟升级。'
        else:
            assert form['申请登记']=='accepted' and form['支付类别'] in {'bank_card','online_payment'}
            assert '通过后完整工作日' not in form
            assert facts['R1','approval_status']=='unknown' and ('R1','elapsed_workdays') not in facts
            assert ('R1','approval_status','decision',None) in missing
            assert ('R1','elapsed_workdays','decision','if_approved') in missing
            assert s.dispositions==['request_info'] and not any(k=='within_window' for t,k in facts)
            basis='受理不是审核通过；审核明确未知，计时未提供，仅在审核通过条件下补完整工作日；未知计时事实可选。'
        if layer=='F02' and '恢复记录' not in form:
            assert not any(a.get('type')=='recovered' and a.get('value') is True for e in s.expectations for a in e.alternatives)
        # 全项公开支持关系逐条可审阅；该记录不由参考评分产生。
        assertions=[]
        for e in s.expectations:
            assertions.append(dict(section=e.section,alternatives=e.alternatives,required=e.required,
                sources=[[dict(reference=ref,text=s.materials.get(ref) or clauses[ref].text) for ref in group] for group in e.supports]))
        reviews.append(dict(task_id=tid,branch_basis=basis,status='passed',review_kind='固定规则的AI内容/标准一致性审阅；非独立盲审或人工金标',assertions=assertions))
        tr=fixture(task); raw=reference(s,tr,clauses); grade=grade_result(s,raw,tr,clauses)
        assert grade.complete_success,(tid,grade.to_dict())
        references.append(dict(task_id=tid,result=raw,grade=grade.to_dict(),use='离线标准相容性辅助；非教师、非训练数据'))
    write(out/'checks/public_basis_review.json',reviews)
    write(out/'checks/offline_references.json',references)
    result=dict(status='passed',tasks=38,families=16,by_branch=dict(Counter(m['stratum_id'] for m in meta)),
        public_basis_reviews=len(reviews),auxiliary_reference_passes=len(references),
        grading_revision=GRADING_REVISION,standards_revision='success-pool-evidence-v5',
        model_calls=0,real_retrieval=0,deployments=0,training=False,formal_success_pool=0,
        scope='来源、结构、对象/支付绑定、当前范围、必需/可选、当前/条件动作及公开私有对应；既有v17常量机制不重复全量历史测试',
        issues='10条旧账单M02/R01范围不符排除并保留原件；B01原订单号不删；生成期字段标签修正见metadata/issues.json',
        teacher_recovery='F02/M02/P04/A01当前主线清晰输入的真实恢复未知；候选通过不表示教师成功')
    write(out/'validation.json',result)
    return result


def read_content(record):
    import json
    return json.loads(record.content)


def read_content_object(record):
    import json
    return json.loads(record['content'])


if __name__=='__main__':
    print(verify(Path(os.environ['MYMIND_WORK_ROOT']).resolve()))
