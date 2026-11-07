"""准备固定六分支扩充材料，不执行检索、部署或采集。"""
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.customer_policy.protocol import PublicTask
from benchmarks.scripts.prepare_mainline_candidates import read, write, clarify, standard_for, INTENTS, mark_missing

REL = 'runs/success-pool-v1/source-expansion-v1'
LAYERS = ('A01', 'B01', 'F02', 'M02', 'P04', 'R01')


def events():
    # 先定义事件实体及连接；三个分支是同事件的独立条件视图，不互相继承状态。
    return [
        dict(family_id='technical-event-period-close', role='technical',
             event='月结任务按账期绑定导出作业；本次选择最终结账作业，不是试算预览。',
             entities=dict(item='六月最终结账单', record='最终结账导出调用', project='结账服务'),
             links=[['六月最终结账单','导出作业-C61'],['导出作业-C61','最终结账导出调用']],
             relation='结账单上的导出作业号与调用记录的作业号相同；认证、影响和权限均定位这次最终结账调用。',
             auth='最终结账作业使用作业凭据槽；本次响应对应该槽携带的凭据，无其他作业授权结果。',
             impact={'响应码':500,'运行环境':'production','大面积不可用':True}, white=False),
        dict(family_id='technical-event-receipt-upload', role='technical',
             event='仓库收货单关联一次设备上传；上传回执与该单入库对账表通过回执号连接。',
             entities=dict(item='到货单-R82', record='收货上传调用', project='收货登记'),
             links=[['到货单-R82','上传回执-U82'],['上传回执-U82','收货上传调用']],
             relation='收货单保存上传回执号，回执注明实际调用；对账仅覆盖这张收货单，接口可用性与业务条目一致性分别记录。',
             auth='仓库设备的上传进程使用设备专用凭据；响应摘录来自本次上传请求，不借用管理后台凭据。',
             impact={'响应码':500,'运行环境':'production','核对发现错乱':True,'恢复记录':True}, white=True),
        dict(family_id='technical-event-booking-payment', role='technical',
             event='场地预订完成单按支付授权号连接确认回执；本次确认调用的业务影响来自该授权的处理记录。',
             entities=dict(item='预订完成单-B93', record='预订支付确认调用', project='场地预订'),
             links=[['预订完成单-B93','支付授权-A93'],['支付授权-A93','预订支付确认调用']],
             relation='完成单和调用记录都标明支付授权A93；支付影响只针对这次授权，不以接口其他请求的成功替代。',
             auth='支付确认客户端使用独立服务凭据；摘录指本次授权A93请求携带的凭据，不含完整凭据值。',
             impact={'响应码':500,'支付链路异常':True}, white=False),
        dict(family_id='billing-event-cycle-ledger', role='billing',
             event='订阅退订分别撤销两个已结算账期，退款行以账期编号连接各自原扣款。',
             order='订阅订单-S41', links={'R1':['账期撤销行-六月','六月实付-C41'],'R2':['账期撤销行-七月','七月实付-C42']},
             relation='撤销行上的账期号匹配原扣款账期；两笔均真实支付，不能按退订日期共用审核计时。',
             channel='online_payment', days=[6,2], complete=True),
        dict(family_id='billing-event-ticket-exchange', role='billing',
             event='机票改签后原票退差价与原服务费退费分别受理，改签单分别引用票款捕获号和服务费捕获号。',
             order='原票订单-T52', links={'R1':['原票退差行','票款实付-T52'],'R2':['服务费退费行','服务费实付-F52']},
             relation='改签单的两类退款按原捕获号区分，均为已发生的独立实付；新票记录不作为原退款的计时起点。',
             channel='bank_card', days=[8,1], complete=False),
        dict(family_id='billing-event-escrow-release', role='billing',
             event='项目终止单分别释放托管保证金与撤销已付里程碑款，两类清算回执各指向原支付。',
             order='项目保证金订单-E63', links={'R1':['保证金释放回执','托管实付-E63'],'R2':['里程碑撤销回执','里程碑实付-M63']},
             relation='终止单分别引用保证金释放号与里程碑撤销号，回执各带原支付编号；项目终止时间不证明两笔同时审核。',
             channel='online_payment', days=[4,5], complete=True),
        dict(family_id='billing-event-credit-note', role='billing',
             event='两张发票调整贷项各自冲回一次原支付，贷项行通过发票号查得对应支付凭据。',
             order='发票调整订单-I74', links={'R1':['发票甲贷项行','发票甲实付-I74'],'R2':['发票乙贷项行','发票乙实付-I75']},
             relation='贷项行绑定发票号，发票支付表逐行绑定原实付；贷项开具日不是退款审核通过日，也不把两张发票合并计时。',
             channel='bank_card', days=[6,3], complete=False),
        dict(family_id='billing-event-return-components', role='billing',
             event='退货授权分别关联商品实付与另行支付的安装费，两条退款以原支付凭据连接，运单不充当支付凭据。',
             order='商品订单-R85', links={'R1':['商品退款授权行','商品实付-R85'],'R2':['安装费退款授权行','安装费实付-A85']},
             relation='授权表注明退款费用类型及各自原支付号，安装费确为单独实付；签收日期不能替代逐笔退款审核日期。',
             channel='online_payment', days=[7,2], complete=False),
    ]


def task_for(event, layer):
    import json
    if event['role']=='technical':
        ent=event['entities']; paths={'Q1':[ent['item'],ent['record']]}
        origin={'目标路径':paths,'业务关联记录':[{'业务项':ent['item'],'对应调用':ent['record']}],
                '关系说明':event['relation'],'记录连接':event['links']}
        body={'对象':'Q1','记录归属':ent['record']}
        if layer=='A01':
            body.update(业务表单={'响应码':401},本次认证记录={'调用选择':ent['item'],'实际调用':ent['record'],
                '客户端与凭据来源':event['auth'],'脱敏响应':{'status':401,'error':'token_expired','说明':'服务端明确指本次携带凭据已过期；尚未重新授权验证。'},'凭据披露':'仅为脱敏来源说明，无完整凭据。'})
        elif layer=='F02':
            body.update(业务表单=event['impact'],业务影响归属={'调用':ent['record'],'业务批次':ent['item'],
                '观察范围':'本记录仅覆盖上述业务批次。'})
        else:
            scope={'project':ent['project'],'environment':'production'}
            body['业务表单']={'响应码':403,'凭据仍有效':True,'操作名称':'批量报表生成','请求归属':scope,
                '实际出口':'198.51.100.161','配置权登记':{'project':ent['project'],'status':True},
                '出口登记':{'project':ent['project'],'outbound_ip':'198.51.100.161','status':event['white']},
                '功能登记':'持有人明确表示不清楚'}
        bodies=[body]
    else:
        paths={'O1':[event['order'],'该订单扣款报告']} if layer=='B01' else {t:p for t,p in event['links'].items() if layer=='M02' or t=='R1'}
        origin={'目标路径':paths,'关系说明':event['relation'],
            '业务关联记录':[{'业务项':p[0],'对应资金记录':p[-1]} for p in paths.values()]}
        bodies=[]
        for t,p in paths.items():
            if layer=='B01':
                form={'订单号':event['order'],'显示扣款条数':2,'两条对应同一订单':True}
                origin['关系说明']='该订单下报告两条扣款，两条均指这一个订单，尚未核验是否重复。'
                if event['complete']:
                    form.update({'流水号':[event['order']+'-C1',event['order']+'-C2'],'交易时间':['09:12','09:17'],
                                 '金额':[320,320],'支付类别':event['channel'],'流水摘要':'两条报告均指本次订单，尚待人工核实是否重复。'})
            else:
                ch=event['channel'] if t=='R1' else ('bank_card' if event['channel']=='online_payment' else 'online_payment')
                form={'支付类别':ch,'审核回执':'approved','通过后完整工作日':event['days'][0 if t=='R1' else 1],
                      '查过原支付账户':True,'原账户收到款':False}
                if layer=='R01':
                    form.update({'申请登记':'accepted','审核回执':'持有人明确表示不清楚'});form.pop('通过后完整工作日')
            b={'对象':t,'记录归属':p[-1],'业务表单':form}
            if layer!='B01':b['支付绑定']={'业务项':p[0],'对应资金记录':p[-1]}
            bodies.append(b)
    records=[dict(reference_id='user:2',content=json.dumps(origin,ensure_ascii=False,indent=2))]
    records.extend(dict(reference_id=f'user:{i}',content=json.dumps(mark_missing(body,layer),ensure_ascii=False,indent=2)) for i,body in enumerate(bodies,3))
    return PublicTask(task_id='expansion-'+layer+'-'+event['family_id'],role=event['role'],user_text=INTENTS[layer],records=records)


def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve();out=root/REL
    if (out/'public/candidates.json').exists(): raise ValueError('候选已存在，不覆盖')
    base=root/'runs/success-pool-v1'
    available=read(base/'unified-pool-v5-v1/available_sources.json')
    eligibility=read(base/'mainline-candidates-v1/metadata/source_eligibility.json')
    dev=set(available['excluded_dev'])
    sealed={m['family_id'] for m in available['sealed']}
    oldcfg=read(root/'runs/balanced-policy-v1/teacher-preparation/collection.json')
    assert Path(oldcfg['tasks']).name=='train.json'
    original={t['task_id']:t for t in read(root/oldcfg['tasks'])}
    tasks=[];meta=[];transform=[];issues=[]
    for m in available['tasks']:
        layer=m['stratum_id'];fid=m['family_id']
        if fid.startswith('billing-') and layer!='B01':
            issues.append(dict(task_id=m['task_id'],family_id=fid,status='excluded_from_this_scope',
                reason='M02双超时' if layer=='M02' else 'R01渠道未知',original_unchanged=True));continue
        assert fid not in dev|sealed
        old=PublicTask.model_validate(original[m['task_id']]);t=clarify(old,layer)
        t.task_id='expansion-'+old.task_id
        tasks.append(t)
        meta.append(dict(task_id=t.task_id,original_task_id=old.task_id,family_id=fid,stratum_id=layer,
            source_kind='reused_old_train',original_split='train',split=None,formal_pool_eligible=True,
            qualification='既有资格元数据可复用且不在dev/封存名单；固定规范内容审阅，无本次教师开发反馈；缺完整历史访问审计，不称从未访问'))
        transform.append(dict(task_id=t.task_id,original=old.model_dump(),candidate=t.model_dump(),
            transformation='沿用已验收clarify：清晰意图、直接对象、必要支付绑定；原业务状态与已提供订单号保留'))
    new=events()
    for event in new:
        for layer in (('A01','F02','P04') if event['role']=='technical' else ('B01','M02','R01')):
            t=task_for(event,layer);tasks.append(t)
            meta.append(dict(task_id=t.task_id,original_task_id=None,family_id=event['family_id'],stratum_id=layer,
                source_kind='new_synthetic_event',original_split=None,split=None,formal_pool_eligible=True,
                qualification='本次事件实体关系先行的合成来源；仅固定规范构造/内容审阅；未作模型开发小试，非真实客户材料或人工金标'))
    # 各层按族名排序，固定层轮转；短层耗尽后跳过，第二遍沿用此顺序过滤失败。
    by_layer={layer:sorted([m for m in meta if m['stratum_id']==layer],key=lambda m:m['family_id']) for layer in LAYERS}
    ordered=[by_layer[layer][i] for i in range(max(map(len,by_layer.values()))) for layer in LAYERS if i<len(by_layer[layer])]
    by_id={t.task_id:t for t in tasks};tasks=[by_id[m['task_id']] for m in ordered]
    standards=[standard_for(t,m['stratum_id']) for t,m in zip(tasks,ordered)]
    write(out/'public/candidates.json',[t.model_dump() for t in tasks]);write(out/'private/standards.json',[s.model_dump() for s in standards])
    write(out/'metadata/task_manifest.json',ordered);write(out/'metadata/new_events.json',new)
    write(out/'private/source_transformations.json',transform);write(out/'metadata/issues.json',issues)
    write(out/'metadata/source_eligibility.json',dict(excluded_dev_families=sorted(dev),sealed_families=available['sealed'],
        qualification='固定规范内容审阅不自动转dev；教师/提示开发及开发性失败分析整族dev；资格不明待定或排除，不能只放train绕过',
        historical_access=available['historical_access'],formal_split_created=False))
    families=[]
    for fid in sorted({m['family_id'] for m in ordered}):
        entries=[m for m in ordered if m['family_id']==fid]
        families.append(dict(family_id=fid,task_ids=[m['task_id'] for m in entries],task_count=len(entries),
            source_kind=entries[0]['source_kind'],relation='同事件实体与记录路径共享；分支条件视图不互相继承状态；全部同源变体整族绑定',split=None))
    write(out/'metadata/source_families.json',families)
    for name in ('knowledge.md','policy_support.json'):
        dest=out/'policy'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(base/'mainline-candidates-v1/policy'/name,dest)
    print(f'{len(tasks)} candidates / {len(families)} families; no calls')


if __name__=='__main__':main()
