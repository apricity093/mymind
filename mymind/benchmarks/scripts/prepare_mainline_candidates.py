"""构造六分支主线候选；仅本地数据构造，不连接服务。"""
import json
import os
from pathlib import Path
import shutil
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from agents.customer_policy.protocol import PublicTask, FACT_TYPES
from rl.tasks import PrivateStandard, Expectation, validate_task_pairs
from benchmarks.scripts.balanced_policy_events import read_event, UNKNOWN
from benchmarks.scripts.balanced_policy_candidates import classify, actual_subtype
from benchmarks.scripts.balanced_policy_design import profile_example, CLAUSES
from benchmarks.scripts.balanced_policy_content import expectation_support
from benchmarks.scripts.teacher_standard_candidate import amend_standard as r5
from benchmarks.scripts.teacher_contract_candidate import amend_standard as r6
from benchmarks.scripts.success_pool_contract import amend, align_workflow, align_evidence

REL = 'runs/success-pool-v1/mainline-candidates-v1'
LAYERS = ('A01', 'B01', 'F02', 'M02', 'P04', 'R01')
TECH = ('13', '15', '17', '18', '21', '22')
BILLING = ('16', '17', '19', '23')
INTENTS = {
    'A01': '请根据Q1本次认证响应说明当前认证问题与下一步。',
    'B01': '请核实O1同一订单疑似两次扣款，说明当前处置及实际缺少的定位材料。',
    'F02': '请判断Q1本次调用关联业务批次的严重影响与当前处置，说明升级所需待补材料。接口可用与业务损失恢复分别判断。',
    'M02': '请逐笔判断R1和R2两笔既有退款的时效与下一步，并给出整体处置。',
    'P04': '请判断Q1本次目标项目、环境、出口及用途下的权限状态，说明待补与当前可作的修正。',
    'R01': '请判断R1退款当前审核与计时还缺什么，并说明下一步。',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def mark_missing(body, layer):
    labels = {'B01':['订单号','流水号','交易时间','金额','支付类别','流水摘要'],
              'F02':['请求定位号','发生时间','脱敏摘要','操作名称','出现频率'],
              'R01':['通过后完整工作日']}.get(layer,[])
    if labels:
        body['本次未提供']=[label for label in labels if label not in body['业务表单']]
    return body


def new_events():
    # 先定义事件和实体连接；三个分支是同事件反事实，不能跨集合拆族。
    return [
        dict(family_id='billing-event-split-tender',
             event='培训报名取消：同一报名订单由公司与学员分别实付，取消分摊表明确把两项退款分配到各自原支付。',
             entities={'order':'报名订单-CT731', 'allocation':'取消分摊表-CT731',
                       'payments':['公司实付-CP731', '学员实付-SP731']},
             links={'R1':['公司分摊行-CT731', '公司实付-CP731'], 'R2':['学员分摊行-CT731', '学员实付-SP731']},
             relation='分摊表通过付款主体及原支付编号逐行绑定；公司与学员两笔真实支付不能合成一笔。',
             channel='bank_card', overdue=7, below=3, duplicate_complete=True),
        dict(family_id='billing-event-settlement-reversal',
             event='两家商户分别收款的联营配送取消：聚合结算批次下两条撤销行通过原捕获编号指向两笔商户实付；批次日期不是审核日期。',
             entities={'order':'联营订单-JD842', 'allocation':'撤销批次-RV842',
                       'payments':['商户甲捕获-CA842', '商户乙捕获-CB842']},
             links={'R1':['撤销行甲-RV842', '商户甲捕获-CA842'], 'R2':['撤销行乙-RV842', '商户乙捕获-CB842']},
             relation='撤销行以原捕获编号关联商户实付；聚合批次下的退款分别审核计时，不以批次总额替代。',
             channel='online_payment', overdue=5, below=1, duplicate_complete=False),
    ]


def synthetic_task(event, layer):
    links = event['links']
    if layer == 'B01':
        paths = {'O1':[event['entities']['order'], '该订单扣款报告']}
        form = {'订单号':event['entities']['order'], '显示扣款条数':2, '两条对应同一订单':True}
        if event['duplicate_complete']:
            form.update({'流水号':['公司捕获-CP731-A','公司捕获-CP731-B'], '交易时间':['2026-11-01T09:01:00+08:00','2026-11-01T09:04:00+08:00'],
                         '金额':[480,480], '支付类别':event['channel'], '流水摘要':'同报名订单的公司付款项报告两条扣款；尚未核实是否重复。'})
        forms = {'O1':form}
    else:
        paths = {t:p for t,p in links.items() if layer == 'M02' or t == 'R1'}
        forms = {}
        for target in paths:
            channel = event['channel'] if target == 'R1' else ('online_payment' if event['channel']=='bank_card' else 'bank_card')
            forms[target] = {'支付类别':channel, '审核回执':'approved',
                '通过后完整工作日':event['overdue'] if target=='R1' else event['below'],
                '查过原支付账户':True, '原账户收到款':False}
            if layer == 'R01':
                forms[target].update({'审核回执':UNKNOWN, '申请登记':'accepted'})
                forms[target].pop('通过后完整工作日')
    origin = {'目标路径':paths, '关系说明':event['relation'],
              '业务关联记录':[{'业务项':p[0], '对应资金记录':p[-1]} for p in paths.values()]}
    records = [dict(reference_id='user:2', content=json.dumps(origin,ensure_ascii=False,indent=2))]
    for i,(target,form) in enumerate(forms.items(),3):
        body = {'对象':target, '记录归属':paths[target][-1], '业务表单':form,
                '支付绑定':{'业务项':paths[target][0], '对应资金记录':paths[target][-1]}}
        records.append(dict(reference_id=f'user:{i}',content=json.dumps(mark_missing(body,layer),ensure_ascii=False,indent=2)))
    return PublicTask(task_id='mainline-'+layer+'-'+event['family_id'],role='billing',user_text=INTENTS[layer],records=records)


def clarify(task, layer):
    t = task.model_copy(deep=True)
    origin = json.loads(t.records[0].content)
    # 路径与业务连接保留；直接事实合并对象标识，去掉无关跨业务背景。
    kept = {k:origin[k] for k in ('目标路径','业务关联记录') if k in origin}
    for k in ('补充经过','本次观察边界'):
        if k in origin and layer=='F02': kept[k]=origin[k]
    t.records[0].content = json.dumps(kept,ensure_ascii=False,indent=2)
    for record in t.records[1:]:
        body = json.loads(record.content)
        target = body.get('对象') or next(k for k,p in origin['目标路径'].items() if p[-1]==body['记录归属'])
        body['对象'] = target
        if layer in {'M02','R01'}:
            body['支付绑定'] = {'业务项':origin['目标路径'][target][0], '对应资金记录':origin['目标路径'][target][-1]}
        record.content = json.dumps(mark_missing(body,layer),ensure_ascii=False,indent=2)
    t.user_text = INTENTS[layer]
    t.task_id = 'mainline-' + task.task_id
    return t


def standard_for(task, layer):
    states,evidence = read_event(task)
    sid, disposition = classify(task.role,states)
    assert sid == layer
    example = profile_example(sid,actual_subtype(sid,states),states)
    example.expects = [e for e in example.expects if not (e['section']=='missing_info' and
        all(states[a['target']].get(a['field'],{}).get('state')=='known' for a in e['alternatives']))]
    for target,fields in states.items():
        for kind,entry in fields.items():
            if kind in FACT_TYPES and entry['state']=='known' and not any(e['section']=='facts' and any(a.get('target')==target and a.get('type')==kind for a in e['alternatives']) for e in example.expects):
                example.fact(target,kind,entry['value'],required=False)
    for e in example.expects: e['supports']=expectation_support(e,evidence)
    standard = PrivateStandard(task_id=task.task_id,group_id=layer,split='candidate',dispositions=[disposition],
        materials={m.reference_id:m.content for m in task.materials()},expectations=example.expects,
        severe_items=dict(example.severe),unprovided_user_facts=example.unprovided)
    standard = r5(task,standard)[0]
    standard = r6(task,standard)[0]
    standard = align_evidence(align_workflow(task,amend(task,standard)))
    if layer=='B01' and '订单号' not in json.loads(task.records[1].content)['业务表单']:
        standard.expectations.append(Expectation(section='missing_info',
            alternatives=[dict(target='O1',field='order_id',purpose='handoff')],supports=[['user:3'],['billing_handoff']]))
    standard.split = 'candidate'
    return standard


def main():
    root = Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    out = root/REL
    if (out/'public/candidates.json').exists(): raise ValueError('候选已存在，不覆盖')
    selection=read(root/'runs/success-pool-v1/source-dev-v1/source_selection.json')
    remaining=read(root/'runs/success-pool-v1/source-dev-v1/remaining_capacity.json')
    meta=read(root/'data/agentic_rl/balanced-policy-v1/metadata/task_manifest.json')
    eligible=set(remaining['families'])
    chosen={'technical-dossier-'+n for n in TECH}|{'billing-dossier-'+n for n in BILLING}
    assert chosen <= eligible
    oldcfg=read(root/'runs/balanced-policy-v1/teacher-preparation/collection.json')
    oldtasks={t['task_id']:t for t in read(root/oldcfg['tasks'])}
    selected=[m for m in meta if m['split']=='train' and m['family_id'] in chosen and m['stratum_id'] in LAYERS]
    tasks=[]; metadata=[]; sources=[]; issues=[]
    for m in selected:
        original=PublicTask.model_validate(oldtasks[m['task_id']]); layer=m['stratum_id']
        task=clarify(original,layer); tasks.append(task)
        metadata.append(dict(task_id=task.task_id,original_task_id=original.task_id,family_id=m['family_id'],stratum_id=layer,
            original_subtype=m['subtype_id'],source_kind='reused_old_train',original_split='train',split=None,
            formal_pool_eligible=True,review_use='固定规范内容与标准一致性审阅；未用于教师/提示调整或生成失败分析'))
        sources.append(dict(task_id=task.task_id,original_task=original.model_dump(),original_metadata=m,
            retained='目标路径、支付绑定、认证响应/凭据归属、当前表单、严重业务批次关联',
            removed='无关事件背景及其他范围状态；B01保留公开表单原订单号'))
        if layer=='B01': issues.append(dict(task_id=task.task_id,issue='原订单号曾仅因名称形式被删除，无相反材料依据',
            original_value=json.loads(original.records[1].content)['业务表单'].get('订单号'),
            resolution='恢复公开表单原样订单标识，取消新增order_id待补；保留同订单两扣款关系、立即人工核验及其他真实缺项',status='resolved'))
    events=new_events()
    write(out/'metadata/new_events.json',events)
    for event in events:
        for layer in ('B01','M02','R01'):
            task=synthetic_task(event,layer); tasks.append(task)
            metadata.append(dict(task_id=task.task_id,original_task_id=None,family_id=event['family_id'],stratum_id=layer,
                source_kind='new_synthetic_event',original_split=None,split=None,formal_pool_eligible=True,
                review_use='事件/实体/记录先定义；固定规范构造审阅；同事件三个反事实整族绑定'))
    # 六轮、层轮转；同层按来源族名称排序，顺序与结果无关。
    by_id={t.task_id:t for t in tasks}; ordered=[]
    for i in range(6):
        for layer in LAYERS:
            ordered.append(sorted([m for m in metadata if m['stratum_id']==layer],key=lambda m:m['family_id'])[i])
    metadata=ordered; tasks=[by_id[m['task_id']] for m in metadata]
    standards=[standard_for(t,m['stratum_id']) for t,m in zip(tasks,metadata)]
    validate_task_pairs({t.task_id:t for t in tasks},{s.task_id:s for s in standards},CLAUSES)
    write(out/'public/candidates.json',[t.model_dump() for t in tasks])
    write(out/'private/standards.json',[s.model_dump() for s in standards])
    write(out/'metadata/task_manifest.json',metadata)
    write(out/'private/source_transformations.json',sources)
    write(out/'metadata/issues.json',issues)
    families=[]
    for fid in sorted({m['family_id'] for m in metadata}):
        entries=[m for m in metadata if m['family_id']==fid]
        families.append(dict(family_id=fid,task_ids=[m['task_id'] for m in entries],task_count=len(entries),
            relation='同事件不同业务分支/反事实；共享对象和记录路径，不互相继承状态；全部同源变体整族划分',
            source_kind=entries[0]['source_kind'],historical_limit='旧train有历史AI构造与审阅，无完整访问审计；未称全新或未访问' if fid in chosen else '本次AI合成与AI审阅；非真实客户材料或人工金标'))
    write(out/'metadata/source_families.json',families)
    excluded=set(selection['prior_dev'])|set(selection['families'])
    write(out/'metadata/source_eligibility.json',dict(eligible_old_train=remaining,
        selected_old_train=sorted(chosen),excluded_dev_families=sorted(excluded),
        sealed_families=[dict(family_id=fid,splits=sorted({m['split'] for m in meta if m['family_id']==fid})) for fid in sorted({m['family_id'] for m in meta if m['split']!='train'})],
        exposed_test_metadata_exclusion='billing-dossier-01；沿用已曝光摘要事实，本次未读取摘要或test题面/标准',
        old_train_not_selected='billing其余5族M02两笔均超时且R01渠道未知；technical其余3族无需扩充当前六族配额',
        capacity_decision='每分支6族；旧train满足技术6族，账单目标状态仅4族，补2个独立合成事件；共36题/12族',
        test_access='仅必要分割元数据；不打开旧test公开题面/私有标准',
        qualification='固定规范构造审阅保持候选资格；教师/提示调整或细致生成失败分析整族转dev；夹具与参考只在checks中'))
    for name in ('knowledge.md','policy_support.json'):
        dest=out/'policy'/name; dest.parent.mkdir(exist_ok=True); shutil.copy2(root/'runs/success-pool-v1/source-dev-v1/policy'/name,dest)
    print('36 candidates / 12 families (30 reused old-train tasks + 6 new synthetic tasks); zero calls')


if __name__=='__main__': main()
