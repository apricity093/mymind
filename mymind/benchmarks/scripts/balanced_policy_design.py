"""Build and check the balanced-policy design without models or retrieval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from agents.customer_policy.protocol import PublicTask, TerminalResult, VisibleTrajectory, SearchAction
from rl.tasks import PrivateStandard, load_policy_support, validate_task_pairs
from rl.grading import GRADING_REVISION, grade_result

CLAUSES = load_policy_support(BACKEND / 'rl/resources/policy_support.json')
KNOWLEDGE_PATH = BACKEND / 'agents/customer_policy/resources/knowledge.md'
KNOWLEDGE = KNOWLEDGE_PATH.read_text(encoding='utf-8')
BILLING, TECHNICAL = KNOWLEDGE.split('### 接口认证、授权与故障咨询', 1)
STYLES = ['用户短问', '事件复盘', '交接便笺', '问答汇总', '记录对照']
# Each event is a new synthetic dossier specification, not an observed customer case.
BILLING_EVENTS = [
'组合商品缺货后拆单：主件交付、附件取消，各有独立支付流水',
'年度订阅切换月付：旧订阅退余额，新订阅另付，不合并计时',
'预售尾款订单取消：定金保留争议与尾款退款分开核对',
'多人拼单解散：发起人与参与者支付记录分别保留',
'物流拒收后部分回仓：已回仓退款与待验收货品分开',
'双币种采购取消：原币支付与展示币金额分别记录',
'分期商品退货：首期退款与后续期停止扣款是不同诉求',
'酒店连续住宿缩短：已住夜数与取消夜数分别结算',
'活动改期后退单：门票款与服务费两条资金记录',
'仓库分批发货退货：两个仓库独立确认退回',
'企业席位减员：停用席位退款与存续席位续费并列',
'家庭套餐拆分：主账户退款与子账户新购分别记账',
'运输险理赔到账：保险赔付不替代商品退款记录',
'充值后撤销服务：充值流水与消费撤销流水分别列示',
'团体课程停课：已上课扣除与未上课退费分项展示',
'迁移云资源：旧资源销毁退款与新资源预付独立',
'优惠券失效重下单：优惠差额与原支付退款分开',
'线下自提未取：商品款退款与押金退回分别处理',
'跨店满减拆单：保留店铺金额与退款分摊的关系',
'换货失败转退款：补差价支付与原订单退款分开',
'授权预扣与正式扣款并存：用户报告两条记录需区分',
'账单合并后用户关闭自动续费：关闭动作与已扣流水分开',
'租赁提前归还：租金结算与设备押金退回独立',
'批量采购部分验收不合格：合格品保留、问题批次退款',
'支付重试后订单成功：失败提示对应流水与成功流水待核对',
]
TECH_EVENTS = [
'定时报表任务在发布后从共享出口改为专用出口',
'蓝绿切换后旧实例与新实例分别发起请求',
'分公司合并项目后仍保留原项目的功能开通单',
'灾备演练中测试服务访问生产报表资源',
'队列消费者扩容后不同节点分别记录请求结果',
'企业代理轮换出口后部分请求经过旧代理',
'只读看板从直接调用改为网关转发',
'夜间批处理把预览与导出拆成两个作业',
'跨区域迁移后源区域和目标区域配置并存',
'新租户启用隔离项目，旧租户记录仍可见',
'值班交接只转交响应摘要，定位日志尚未转交',
'证书轮换窗口内不同客户端保留不同凭据',
'容器重建后缓存凭据与配置页面不一致',
'企业报表从手工触发改为日历触发',
'支付回调从串行改为队列处理，前台状态与落库分开',
'审计导出与普通业务查询共享项目但用途不同',
'数据修复任务恢复服务后还需对账缺失记录',
'滚动发布只影响一组实例，其他实例正常',
'内网链路切到公网出口后需要核对实际出口记录',
'项目所有者交接后配置权限与功能状态分别记录',
'服务网格新增转发跳点，请求标识沿链路传递',
'多客户端共享报表任务，其中移动端与后台分别调用',
'上游支付成功而通知任务失败，需要核对业务影响',
'恢复快照后读服务可用，但业务记录仍需核对',
'批量写入与只读查询并行运行，两类影响分别汇报',
]

# Stable strata. The full executable requirements are the per-object expectations below.
LAYERS = [('R01', '单笔退款', '审核未知', 'request_info', ['refund_accepted', 'refund_clock']),
 ('R02', '单笔退款', '审核后计时未知', 'request_info', ['refund_clock']),
 ('R03', '单笔退款', '覆盖渠道且未超过上限', 'answer', ['refund_window', 'refund_clock']),
 ('R04', '单笔退款', '覆盖渠道且超过上限', 'escalate', ['refund_overdue', 'refund_window']),
 ('R05', '单笔退款', '渠道未知', 'request_info', ['refund_scope']),
 ('R06', '单笔退款', '渠道已知但不覆盖', 'escalate', ['refund_scope', 'channel_handoff']),
 ('M01', '多笔退款', '两笔均在时限内', 'answer', ['refund_multi']),
 ('M02', '多笔退款', '一笔可核实另一笔完整', 'escalate', ['refund_multi', 'refund_overdue']),
 ('M03', '多笔退款', '一笔可升级另一笔待补', 'escalate', ['refund_multi']),
 ('M04', '多笔退款', '无已成立升级且存在待补', 'request_info', ['refund_multi']),
 ('B01', '扣款与资金核实', '疑似重复扣款', 'escalate', ['billing_handoff']),
 ('B02', '扣款与资金核实', '实际资金操作请求', 'escalate', ['billing_handoff']),
 ('P01', '权限白名单范围', '基础能力白名单可自助', 'answer', ['capabilities', 'config_actors', 'config_scope']),
 ('P02', '权限白名单范围', '导出已开通仅缺白名单', 'answer', ['capabilities', 'config_actors']),
 ('P03', '权限白名单范围', '当前范围明确未开通', 'escalate', ['capabilities', 'config_actors']),
 ('P04', '权限白名单范围', '白名单已知功能未知', 'request_info', ['permission_unknown', 'config_scope']),
 ('P05', '权限白名单范围', '功能已知白名单未知', 'request_info', ['permission_unknown', 'config_scope']),
 ('P06',
  '权限白名单范围',
  '异范围记录或其他配置不足',
  'request_info',
  ['config_scope', 'unmatched_scope', 'permission_unknown']),
 ('F01', '故障影响未知', '有限且恢复的只读故障', 'answer', ['limited_fault']),
 ('F02', '故障影响未知', '严重影响已成立', 'escalate', ['severe_fault']),
 ('F03', '故障影响未知', '恢复但数据丢失', 'escalate', ['severe_fault']),
 ('F04', '故障影响未知', '影响信息不足', 'request_info', ['fault_handling']),
 ('A01', '认证凭据', '明确Token过期', 'answer', ['auth_expired']),
 ('A02', '认证凭据', '401原因未定', 'request_info', ['auth_expired'])]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class Example:
    def __init__(self, sid, role):
        self.task = dict(task_id='design-dev-'+sid, role=role, user_text='', records=[])
        self.expects = []
        self.states = {}
        self.notes = []
        self.disposition = 'answer'
        self.severe = defaultdict(list)
        self.unprovided = {}

    def item(self, section, target, payload, policies=(), required=True, decisive=False, alternatives=()):
        alt = dict(target=target, **payload)
        if section=='next_steps' and payload.get('code')=='escalate_technical':
            alternatives=(*alternatives,dict(target=target,code='seek_human_review'))
        self.expects.append(dict(section=section, alternatives=[alt, *alternatives],
            supports=[['user:1'], *[[p] for p in policies]], required=required, decisive=decisive))

    def fact(self, target, kind, value, policies=(), required=True):
        self.item('facts', target, dict(type=kind, value=value), policies, required)

    def reason(self, target, code, policies):
        self.item('reasons', target, dict(code=code), policies, decisive=True)

    def step(self, target, code, policies, params=None, required=True):
        self.item('next_steps', target, dict(code=code, **({'params':params} if params else {})), policies, required)

    def missing(self, target, field, condition=None, purpose='decision', required=True):
        self.item('missing_info', target, dict(field=field, purpose=purpose, **({'condition':condition} if condition else {})), required=required)


def refund(e, t, channel='online_payment', days=2, mode='normal'):
    chtext = {'online_payment':'线上支付渠道','bank_card':'银行卡渠道','unknown':'未知渠道','balance':'平台余额渠道'}[channel]
    e.fact(t,'payment_channel',channel)
    if mode == 'approval':
        text = f'{t}的{chtext}退款仅有申请接收记录，审核结果明确不清楚。'
        e.fact(t,'approval_status','unknown')
        e.fact(t,'elapsed_workdays','unknown',required=False)
        e.reason(t,'refund_state_unknown',['refund_accepted','refund_clock'])
        e.missing(t,'approval_status'); e.missing(t,'elapsed_workdays','if_approved')
        return text
    e.fact(t,'approval_status','approved')
    e.fact(t,'timer_start','approval',['refund_clock'])
    if mode == 'clock':
        e.fact(t,'elapsed_workdays','unknown'); e.missing(t,'elapsed_workdays')
        e.reason(t,'refund_state_unknown',['refund_clock'])
        return f'{t}为{chtext}退款，审核已通过，但通过后经过多少完整工作日明确不知道。'
    e.fact(t,'elapsed_workdays',days)
    if channel == 'unknown':
        e.missing(t,'payment_channel'); e.reason(t,'refund_state_unknown',['refund_scope'])
        return f'{t}退款审核已通过{days}个完整工作日，渠道明确不知道。'
    if channel == 'balance':
        e.fact(t,'timing_policy_covered',False,['refund_scope'])
        e.reason(t,'policy_uncovered',['refund_scope','channel_handoff'])
        e.step(t,'verify_channel_timing',['channel_handoff'])
        return f'{t}明确为平台余额渠道退款，审核已通过{days}个完整工作日，想核实适用时效。'
    limit = 3 if channel == 'online_payment' else 5
    e.fact(t,'timing_policy_covered',True,['refund_scope'],False)
    e.fact(t,'limit_workdays',limit,['refund_window'])
    e.fact(t,'within_window',days<=limit,['refund_window','refund_clock'])
    if mode != 'unchecked':
        e.fact(t,'original_account_checked',True); e.fact(t,'received',False)
    else:
        e.unprovided[t] = ['original_account_checked','received']
        e.missing(t,'original_account_checked'); e.missing(t,'received')
    if days<=limit:
        e.reason(t,'within_refund_window',['refund_window','refund_clock'])
        e.step(t,'wait_within_window',['refund_wait'])
        e.step(t,'verify_refund_if_overdue_unreceived',['refund_window'],{'limit_workdays':limit})
        e.step(t,'check_original_account',['refund_wait'],required=False)
    else:
        e.reason(t,'refund_overdue',['refund_window','refund_clock'])
        if mode != 'unchecked':
            e.step(t,'verify_refund',['refund_overdue'])
        else:
            e.step(t,'check_original_account',['refund_window'])
            e.step(t,'verify_refund_if_overdue_unreceived',['refund_window'],{'limit_workdays':limit})
        e.severe['next_steps'].append(dict(target=t, code='wait_within_window'))
    return f'{t}为{chtext}退款，已审核通过{days}个完整工作日，当前渠道处理中；' + ('未说明原账户核对及到账状态。' if mode=='unchecked' else '已核对该笔原支付账户，仍未到账。')


def examples():
    out=[]
    for sid,group,name,disp,policies in LAYERS:
        role='billing' if sid[0] in 'RMB' else 'technical'
        e=Example(sid,role); e.disposition=disp
        if sid.startswith('R'):
            args={'R01':dict(mode='approval',channel='bank_card'), 'R02':dict(mode='clock'),
                  'R03':dict(days=5,channel='bank_card'), 'R04':dict(days=6,channel='bank_card'),
                  'R05':dict(channel='unknown'), 'R06':dict(channel='balance')}
            text=refund(e,'R1',**args[sid])
            text='设备保养服务的未使用部分已申请退费。'+text
        elif sid.startswith('M'):
            configs={'M01':(dict(days=3),dict(days=5,channel='bank_card')),
                     'M02':(dict(days=7,channel='bank_card'),dict(days=1)),
                     'M03':(dict(days=5),dict(mode='clock',channel='bank_card')),
                     'M04':(dict(days=2),dict(days=6,channel='bank_card',mode='unchecked'))}
            a,b=configs[sid]
            text='独立订购的培训辅导和考核服务分别退费。'+refund(e,'R1',**a)+refund(e,'R2',**b)
        elif sid=='B01':
            text='订单 O-DS 是设备校准服务。用户提供的两条付款通知均标注订单 O-DS，各显示一次扣款，怀疑重复扣款；尚无交易号、时间、金额、渠道及流水摘要。请核实。'
            e.fact('O-DS','reported_charge_count',2); e.fact('O-DS','charges_same_order',True)
            e.reason('O-DS','suspected_duplicate_charge',['billing_handoff'])
            e.step('O-DS','verify_duplicate_charge',['billing_handoff'])
            for f in ['transaction_id','payment_time','amount','payment_channel','transaction_summary']:
                e.missing('O-DS',f,purpose='handoff')
            e.unprovided['O-DS']=['payment_channel']
        elif sid=='B02':
            text='订单 O-DS2 的按次校准服务已取消；我要求实际办理退回未使用服务费，订单号已提供，交易流水和金额稍后给。'
            e.reason('O-DS2','evidence_insufficient',['billing_handoff'])
            e.step('O-DS2','seek_human_review',['billing_handoff'])
            for f in ['transaction_id','amount']: e.missing('O-DS2',f,purpose='handoff')
        elif sid.startswith('P'):
            scope=dict(project='测绘档案',environment='production'); ip='198.51.100.74'
            basic=sid=='P01'; feature={'P02':True,'P03':False,'P04':'unknown','P05':True,'P06':'unknown'}.get(sid)
            white='unknown' if sid=='P05' else False
            text=f'请求 Q1：测绘档案项目生产环境的'+('报表预览' if basic else '批量报表生成')+f'返回403。认证有效，用户有测绘档案项目配置权，当前出口为{ip}。'
            text+=('当前出口在该项目白名单中的状态明确不知道。' if white=='unknown' else '该项目白名单明确不含当前出口。')
            e.fact('Q1','interface_name','报表预览' if basic else '批量报表生成')
            e.fact('Q1','request_scope',scope); e.fact('Q1','outbound_ip',ip)
            e.fact('Q1','auth_valid',True); e.fact('Q1','http_status',403,required=False)
            e.fact('Q1','user_can_configure',dict(project=scope['project'],status=True))
            e.fact('Q1','required_capability','basic_query' if basic else 'export',['capabilities'])
            e.fact('Q1','whitelist_required',True,['capabilities'])
            e.fact('Q1','platform_enablement_required',not basic,['capabilities'])
            e.fact('Q1','outbound_ip_allowed',dict(project=scope['project'],outbound_ip=ip,status=white),['config_scope'])
            if not basic:
                text+= '当前项目生产环境导出功能'+({True:'明确已开通。',False:'明确未开通。','unknown':'状态明确不知道。'}[feature])
                if sid=='P06': text+='只能看到同项目测试环境已开通记录；未见当前生产环境记录。'
                e.fact('Q1','feature_enabled',dict(**scope,status=feature), ['config_scope','unmatched_scope'] if sid=='P06' else [])
            if sid in {'P01','P02'}:
                e.reason('Q1','user_config_fixable',['capabilities','config_actors','config_scope'])
                e.step('Q1','adjust_project_whitelist',['config_actors','config_scope'],dict(project=scope['project'],outbound_ip=ip))
            elif sid=='P03':
                e.reason('Q1','platform_enablement_required',['capabilities','config_actors','config_scope'])
                e.step('Q1','request_platform_enablement',['config_actors','config_scope'],scope)
                e.step('Q1','adjust_project_whitelist',['config_actors','config_scope'],dict(project=scope['project'],outbound_ip=ip),False)
            else:
                e.reason('Q1','permission_state_unknown',['capabilities','permission_unknown','config_scope'])
                e.missing('Q1','outbound_ip_allowed' if sid=='P05' else 'feature_enabled')
                if white is False:
                    e.step('Q1','adjust_project_whitelist',['config_actors','config_scope'],dict(project=scope['project'],outbound_ip=ip),False)
        elif sid.startswith('F'):
            if sid=='F01':
                text='请求 Q1 是档案目录只读预览，测试环境返回500后已经恢复，影响明确限于该预览任务，不涉及支付、写入、数据丢失或错乱；未提供频率及人数。'
                vals=dict(request_read_only=True,environment='test',http_status=500,recovered=True,payment_affected=False,write_affected=False,data_loss=False,data_corruption=False)
                e.reason('Q1','limited_recovered_fault',['limited_fault'])
                e.step('Q1','low_risk_checks',['limited_fault'],dict(directions=['network','recent_config_changes']))
                e.step('Q1','record_diagnostics',['limited_fault'],dict(fields=['event_time','interface_name','request_id','occurrence','response_summary']))
                e.step('Q1','escalate_if_severe_impact',['fault_handling'])
            elif sid=='F02':
                text='请求 Q1 的订金支付链路已明确受影响，返回500；没有提供发生频率、影响人数、只读属性、恢复状态及request_id。'
                vals=dict(payment_affected=True,http_status=500)
                e.reason('Q1','severe_business_impact',['severe_fault']); e.step('Q1','escalate_technical',['severe_fault'])
                for f in ['request_id','event_time','response_summary']: e.missing('Q1',f,purpose='handoff')
            elif sid=='F03':
                text='请求 Q1 的档案读取服务已经恢复，但对账确认仍有数据丢失，尚无定位request_id；没有提供人数、频率、支付与写入影响。'
                vals=dict(recovered=True,data_loss=True)
                e.reason('Q1','severe_business_impact',['severe_fault']); e.step('Q1','escalate_technical',['severe_fault'])
                for f in ['request_id','event_time','response_summary']: e.missing('Q1',f,purpose='handoff')
            else:
                text='请求 Q1 的档案服务返回500。目前只知道错误码，恢复、只读、支付影响、数据丢失、错乱及影响范围均明确不知道。'
                vals=dict(http_status=500,recovered='unknown',request_read_only='unknown',payment_affected='unknown',data_loss='unknown',data_corruption='unknown',affected_users='unknown')
                e.reason('Q1','evidence_insufficient',['fault_handling'])
                for f in ['recovered','request_read_only','payment_affected','data_loss','data_corruption','affected_users']: e.missing('Q1',f)
                e.step('Q1','escalate_if_severe_impact',['fault_handling'])
            for k,v in vals.items(): e.fact('Q1',k,v)
            e.unprovided['Q1']=[k for k in ['occurrence','affected_users','request_read_only','payment_affected','write_affected','data_loss','data_corruption','recovered','environment'] if k not in vals]
        else:
            text='请求 Q1 为档案签收服务，返回401。'+('脱敏响应明确提示Token过期，尚未重新授权。' if sid=='A01' else '响应没有提供原因，凭据是否过期明确不知道，有效性也明确不知道。')
            e.fact('Q1','http_status',401)
            if sid=='A01':
                e.fact('Q1','token_expired',True); e.fact('Q1','auth_valid',False,['auth_expired'])
                e.reason('Q1','token_expired',['auth_expired']); e.step('Q1','reauthorize_then_verify',['auth_expired'])
            else:
                e.fact('Q1','token_expired','unknown'); e.fact('Q1','auth_valid','unknown')
                e.reason('Q1','evidence_insufficient',['auth_expired'])
                e.missing('Q1','token_expired'); e.missing('Q1','auth_valid')
                e.missing('Q1','response_summary',required=False)
        e.task['user_text']=text+'请说明当前处置与必要后续。'
        out.append(e)
    return out


def fixture(task):
    return VisibleTrajectory.model_validate(dict(task_id=task.task_id,materials=[m.model_dump() for m in task.materials()], searches=[dict(action=dict(query='离线政策契约检查'),observation=dict(request_number=1,status=dict(reference_id='search:1:status',state='candidates',detail='离线fixture；公开政策原文；非实际检索'),snippets=[dict(reference_id='search:1:1',source_id='billing-policy',title='公开账单政策离线fixture',section_path=[],content=BILLING),dict(reference_id='search:1:2',source_id='technical-policy',title='公开技术政策离线fixture',section_path=[],content=TECHNICAL)]))]))


def reference(e):
    result=dict(disposition=e.disposition,reasons=[],facts=[],next_steps=[],missing_info=[])
    for x in e.expects:
        if not x['required']: continue
        item=deepcopy(x['alternatives'][0]); refs=[]
        for group in x['supports']:
            support=group[0]
            ref=support if support.startswith(('user:','role:')) else ('search:1:1' if CLAUSES[support].source_id=='billing-policy' else 'search:1:2')
            if ref not in refs: refs.append(ref)
        result[x['section']].append(dict(**item,evidence=refs))
    return result


def main_sources(sid):
    """Independent source annotations authored from the 24 public stories, not outputs."""
    from benchmarks.scripts.balanced_policy_candidates import values, known, absent, refund_fields, permission_fields
    if sid.startswith('R'):
        r = {'R01': refund_fields('bank_card', 0), 'R02': refund_fields(),
             'R03': refund_fields('bank_card',5), 'R04': refund_fields('bank_card',6),
             'R05': refund_fields(), 'R06': refund_fields('balance',2)}[sid]
        if sid in {'R01','R02','R05','R06'}:
            r.pop('original_account_checked'); r.pop('received')
        if sid=='R01': r.update(approval_status=absent(True),elapsed_workdays=absent(),refund_stage=known('accepted'))
        if sid=='R02': r['elapsed_workdays']=absent(True)
        if sid=='R05': r['payment_channel']=absent(True)
        return {'R1':r}
    if sid.startswith('M'):
        a,b={'M01':(refund_fields('online_payment',3),refund_fields('bank_card',5)),
             'M02':(refund_fields('bank_card',7),refund_fields('online_payment',1)),
             'M03':(refund_fields('online_payment',5),refund_fields('bank_card',0)),
             'M04':(refund_fields('online_payment',2),refund_fields('bank_card',6))}[sid]
        if sid=='M03':
            b['elapsed_workdays']=absent(True);b.pop('original_account_checked');b.pop('received')
        if sid=='M04': b.update(original_account_checked=absent(),received=absent())
        return {'R1':a,'R2':b}
    if sid=='B01':
        r=values(reported_charge_count=2,charges_same_order=True)
        r.update({k:absent() for k in ['transaction_id','payment_time','amount','payment_channel','transaction_summary']})
        return {'O-DS':r}
    if sid=='B02': return {'O-DS2':values(funds_request='实际退款',order_id='O-DS2') | {'transaction_id':absent(),'amount':absent()}}
    if sid.startswith('P'):
        r=permission_fields(sid=='P01')
        r['interface_name']=known('报表预览' if sid=='P01' else '批量报表生成')
        r['request_scope']['value']['project']='测绘档案'; r['outbound_ip']=known('198.51.100.74')
        r['user_can_configure']['value']['project']='测绘档案'
        r['outbound_ip_allowed']['value'].update(project='测绘档案',outbound_ip='198.51.100.74')
        if 'feature_enabled' in r:r['feature_enabled']['value']['project']='测绘档案'
        if sid=='P03':r['feature_enabled']['value']['status']=False
        if sid in {'P04','P06'}:r['feature_enabled']=absent(True)
        if sid=='P05':r['outbound_ip_allowed']=absent(True)
        if sid=='P06':r['other_feature_record']=known(dict(project='测绘档案',environment='test',status=True))
        return {'Q1':r}
    if sid.startswith('F'):
        r={k:absent() for k in ['occurrence','affected_users','request_read_only','payment_affected','write_affected','data_loss','data_corruption','recovered','environment']}
        if sid=='F01': r.update(values(request_read_only=True,environment='test',http_status=500,recovered=True,payment_affected=False,write_affected=False,data_loss=False,data_corruption=False,impact_scope='仅限该预览任务'))
        elif sid=='F02':r.update(values(payment_affected=True,http_status=500))
        elif sid=='F03':r.update(values(recovered=True,data_loss=True))
        else:
            r['http_status']=known(500)
            for k in ['recovered','request_read_only','payment_affected','data_loss','data_corruption','affected_users']:r[k]=absent(True)
        return {'Q1':r}
    r=values(http_status=401)
    if sid=='A01':r['token_expired']=known(True)
    else:r.update(token_expired=absent(True),auth_valid=absent(True))
    return {'Q1':r}


def refund_contract(e, target, fields):
    from benchmarks.scripts.balanced_policy_candidates import value
    channel=value(fields,'payment_channel'); approval=value(fields,'approval_status'); days=value(fields,'elapsed_workdays')
    if channel in {'online_payment','bank_card'}:
        e.fact(target,'timing_policy_covered',True,['refund_scope'],required=False)
        # The channel fixes the policy limit even before an individual clock is known.
        if approval != 'approved' or days is None:
            e.fact(target,'limit_workdays',3 if channel=='online_payment' else 5,
                   ['refund_window'],required=False)
    if channel in {'online_payment','bank_card','balance'}:e.fact(target,'payment_channel',channel)
    elif channel is None:
        e.fact(target,'payment_channel','unknown');e.missing(target,'payment_channel')
    e.fact(target,'approval_status','approved' if approval=='approved' else 'unknown')
    if approval!='approved':
        e.fact(target,'elapsed_workdays','unknown',required=False);e.missing(target,'approval_status');e.missing(target,'elapsed_workdays','if_approved')
        e.reason(target,'refund_state_unknown',['refund_clock','refund_accepted'])
        return
    e.fact(target,'timer_start','approval',['refund_clock'])
    e.fact(target,'elapsed_workdays',days if days is not None else 'unknown')
    if days is None:e.missing(target,'elapsed_workdays')
    if channel is not None and channel not in {'online_payment','bank_card'}:
        e.fact(target,'timing_policy_covered',False,['refund_scope']);e.reason(target,'policy_uncovered',['refund_scope','channel_handoff']);e.step(target,'verify_channel_timing',['channel_handoff']);return
    if channel is None or days is None:
        e.reason(target,'refund_state_unknown',['refund_clock','refund_scope']);return
    limit=3 if channel=='online_payment' else 5
    e.fact(target,'limit_workdays',limit,['refund_window']);e.fact(target,'within_window',days<=limit,['refund_window','refund_clock'])
    for field in ['original_account_checked','received']:
        v=value(fields,field)
        if v is not None:e.fact(target,field,v)
    if days<=limit:
        e.reason(target,'within_refund_window',['refund_window','refund_clock']);e.step(target,'wait_within_window',['refund_wait']);e.step(target,'verify_refund_if_overdue_unreceived',['refund_window'],{'limit_workdays':limit});return
    e.severe['next_steps'].append(dict(target=target,code='wait_within_window'))
    e.reason(target,'refund_overdue',['refund_window','refund_clock'])
    if value(fields,'received') is True:return
    checked=value(fields,'original_account_checked');received=value(fields,'received')
    if checked is True and received is False:e.step(target,'verify_refund',['refund_overdue']);return
    if checked is None:e.missing(target,'original_account_checked')
    if received is None:e.missing(target,'received')
    if checked is False or checked is None:e.step(target,'check_original_account',['refund_window'])
    e.step(target,'verify_refund_if_overdue_unreceived',['refund_window'],{'limit_workdays':limit})


def profile_example(sid, subtype, objects):
    """Derive a dev contract from explicit source conditions and unchanged clauses."""
    from benchmarks.scripts.balanced_policy_candidates import classify, value
    from agents.customer_policy.protocol import FACT_TYPES
    role='billing' if sid[0] in 'RMB' else 'technical'
    e=Example(sid+'-'+subtype,role);e.states=deepcopy(objects)
    _,e.disposition=classify(role,objects)
    parts=[]
    for target,fields in objects.items():
        parts.append(f'对象 {target} 的用户记录：')
        for field,entry in fields.items():
            if entry['state']=='not_provided':continue
            wording='明确不知道' if entry['state']=='explicit_unknown' else json.dumps(entry['value'],ensure_ascii=False)
            parts.append(f'{target}.{field}：{wording}。')
    e.task['user_text']='\n'.join(parts)+'\n只判断已报告条件的当前处置及下一步，不推断其他用户事实。'
    if sid.startswith(('R','M')):
        for target,fields in objects.items():refund_contract(e,target,fields)
    elif sid=='B01':
        target,fields=next(iter(objects.items()))
        e.fact(target,'reported_charge_count',2);e.fact(target,'charges_same_order',True)
        e.reason(target,'suspected_duplicate_charge',['billing_handoff']);e.step(target,'verify_duplicate_charge',['billing_handoff'])
        for field in ['transaction_id','payment_time','amount','payment_channel','transaction_summary']:
            if value(fields,field) is None:e.missing(target,field,purpose='handoff')
        if value(fields,'payment_channel') is not None:e.fact(target,'payment_channel',value(fields,'payment_channel'))
    elif sid=='B02':
        target=next(iter(objects));e.reason(target,'evidence_insufficient',['billing_handoff']);e.step(target,'seek_human_review',['billing_handoff'])
    elif sid.startswith('P'):
        q=objects['Q1'];scope=value(q,'request_scope');ip=value(q,'outbound_ip');basic=sid=='P01'
        for field in ['interface_name','request_scope','outbound_ip','auth_valid','user_can_configure','http_status']:
            if value(q,field) is not None:e.fact('Q1',field,q[field]['value'])
        e.fact('Q1','required_capability','basic_query' if basic else 'export',['capabilities']);e.fact('Q1','whitelist_required',True,['capabilities']);e.fact('Q1','platform_enablement_required',not basic,['capabilities'])
        e.fact('Q1','whitelist_actor','project_user',['config_actors'],required=False)
        if not basic:e.fact('Q1','enablement_actor','platform',['config_actors'],required=False)
        if value(q,'auth_valid') is True:
            e.fact('Q1','token_expired',False,['auth_expired'],required=False)
        white=value(q,'outbound_ip_allowed');feature=value(q,'feature_enabled')
        e.fact('Q1','outbound_ip_allowed',dict(project=scope['project'],outbound_ip=ip,status=white if white is not None else 'unknown'),['config_scope'],required=q['outbound_ip_allowed']['state']!='not_provided')
        if not basic:e.fact('Q1','feature_enabled',dict(**scope,status=feature if feature is not None else 'unknown'),['config_scope','unmatched_scope'] if value(q,'other_feature_record') else ['config_scope'],required=q['feature_enabled']['state']!='not_provided')
        if sid in {'P01','P02'}:
            e.reason('Q1','user_config_fixable',['capabilities','config_actors','config_scope']);e.step('Q1','adjust_project_whitelist',['config_actors','config_scope'],dict(project=scope['project'],outbound_ip=ip))
        elif sid=='P03':
            e.reason('Q1','platform_enablement_required',['capabilities','config_actors','config_scope']);e.step('Q1','request_platform_enablement',['config_actors','config_scope'],scope)
            if white is False:e.step('Q1','adjust_project_whitelist',['config_actors','config_scope'],dict(project=scope['project'],outbound_ip=ip),False)
            if white is None:e.missing('Q1','outbound_ip_allowed',required=False)
        else:
            e.reason('Q1','evidence_insufficient' if feature is True and white is True else 'permission_state_unknown',['permission_unknown','capabilities','config_scope'])
            if not basic and feature is None:e.missing('Q1','feature_enabled')
            if white is None:e.missing('Q1','outbound_ip_allowed')
            if feature is True and white is True:
                for field in ['request_id','event_time','response_summary']:e.missing('Q1',field)
                e.step('Q1','record_diagnostics',['permission_unknown'],{'fields':['request_id','event_time','response_summary']},False)
            if white is False:e.step('Q1','adjust_project_whitelist',['config_actors','config_scope'],dict(project=scope['project'],outbound_ip=ip),False)
    else:
        q=objects['Q1']
        for field,entry in q.items():
            if field in FACT_TYPES and entry['state']=='known':e.fact('Q1',field,entry['value'])
        if sid=='F01':
            e.reason('Q1','limited_recovered_fault',['limited_fault']);e.step('Q1','record_diagnostics',['limited_fault'],{'fields':['event_time','interface_name','request_id','occurrence','response_summary']});e.step('Q1','low_risk_checks',['limited_fault'],{'directions':['network','recent_config_changes']});e.step('Q1','escalate_if_severe_impact',['fault_handling'])
        elif sid in {'F02','F03'}:
            e.reason('Q1','severe_business_impact',['severe_fault']);e.step('Q1','escalate_technical',['severe_fault'])
            for field in ['request_id','event_time','response_summary']:e.missing('Q1',field,purpose='handoff')
            if value(q,'token_expired') is True:
                e.reason('Q1','token_expired',['auth_expired']);e.step('Q1','reauthorize_then_verify',['auth_expired'])
        elif sid=='F04':
            e.reason('Q1','evidence_insufficient',['fault_handling'])
            for field in ['recovered','request_read_only','payment_affected','data_loss','data_corruption','affected_users']:
                if value(q,field) is None:e.fact('Q1',field,'unknown');e.missing('Q1',field)
            e.step('Q1','escalate_if_severe_impact',['fault_handling'])
        elif sid=='A01':
            e.fact('Q1','auth_valid',False,['auth_expired']);e.reason('Q1','token_expired',['auth_expired']);e.step('Q1','reauthorize_then_verify',['auth_expired'])
        else:
            for field in ['token_expired','auth_valid']:e.fact('Q1',field,'unknown');e.missing('Q1',field)
            e.reason('Q1','evidence_insufficient',['auth_expired'])
    for target,fields in objects.items():
        e.unprovided[target]=[field for field,entry in fields.items() if entry['state']=='not_provided']
    return e


def example_record(e, sid, subtype, kind):
    task=PublicTask.model_validate(e.task)
    standard=PrivateStandard.model_validate(dict(task_id=task.task_id,group_id=sid,split='dev',dispositions=[e.disposition],materials={m.reference_id:m.content for m in task.materials()},expectations=e.expects,severe_items=dict(e.severe),unprovided_user_facts=e.unprovided))
    derived=[dict(target=alt['target'],type=alt['type'],value=alt['value'],supports=x['supports']) for x in e.expects if x['section']=='facts' for alt in x['alternatives'] if any(s in CLAUSES for group in x['supports'] for s in group)]
    return dict(public_task=task.model_dump(),private_standard=standard.model_dump(),result=reference(e),trajectory=fixture(task).model_dump(),
                metadata=dict(grading_revision=GRADING_REVISION,standards_revision='balanced-policy-v1-standards-v4',task_id=task.task_id,stratum_id=sid,subtype_id=subtype,kind=kind,family_id=('dev-family-single-refund' if sid.startswith('R') else 'dev-family-two-refunds' if sid.startswith('M') else 'dev-family-permission' if sid.startswith('P') else 'dev-family-'+sid),
                              condition_states=e.states,policy_derived_facts=derived,
                              review_source='同一执行助手AI编写与AI语义审阅，非人工金标或独立专家',
                              semantic_review=dict(public_basis=task.user_text,rule_basis=sorted({s for x in e.expects for group in x['supports'] for s in group if s in CLAUSES}),
                                                   inference='来源条件按公开记录独立标注；本层直接归属规则决定主层与整体处置，逐对象要求由列明条款推出。'+('已到账只比较历史经过天数，不再触发当前未到账核实。' if subtype=='received' else '')+('严重影响先升级，同时保留重新授权指导。' if subtype=='payment_and_expired' else ''),
                                                   limits='参考由标准组装，仅证明相容性；可选输出和等价表达以当前评分修订及已验收边界为限'),
                              formal_exclusion='dev及其摘录、改写和反事实全部排除正式池'))


def legacy_variants(design):
    """Retain the seven previously accepted full contracts with corrected source annotations."""
    old=json.loads((design/'v1/validation_results.json').read_text(encoding='utf-8'))['boundary_variants']
    from benchmarks.scripts.balanced_policy_candidates import known, absent, classify, actual_subtype, refund_fields
    result=[]
    for row in old:
        name=row['name']
        if name in {'online_exact','online_overdue','bank_below','overdue_account_status_missing'}:
            channel='bank_card' if name in {'bank_below','overdue_account_status_missing'} else 'online_payment'
            days={'online_exact':3,'online_overdue':4,'bank_below':4,'overdue_account_status_missing':6}[name]
            objects={'R1':refund_fields(channel,days)}
            if name=='overdue_account_status_missing':objects['R1'].update(original_account_checked=absent(),received=absent())
        else:
            sid={'feature_other_project':'P06','whitelist_cross_environment':'P02','approval_not_provided':'R01'}[name]
            objects=main_sources(sid)
            if name=='feature_other_project':objects['Q1']['other_feature_record']=known(dict(project='其他项目',environment='production',status=True))
            if name=='approval_not_provided':objects['R1']['approval_status']=absent()
        sid,disp=classify(row['public_task']['role'],objects)
        e=Example(name,row['public_task']['role']);e.task=row['public_task'];e.states=objects;e.disposition=disp
        e.expects=row['private_standard']['expectations'];e.severe=row['private_standard']['severe_items'];e.unprovided=row['private_standard'].get('unprovided_user_facts',{})
        record=example_record(e,sid,actual_subtype(sid,objects),'retained_boundary')
        record['result']=row['result'];result.append(record)
    return result


DEFINITIONS = {
'R01':'单笔退款审核状态未知；渠道可已知或同时未知，计时不作已通过断言。',
'R02':'单笔退款渠道明确且政策覆盖、审核已通过，审核后天数未知。',
'R03':'单笔退款渠道覆盖、审核通过、完整工作日数已知且不超过对应上限。',
'R04':'单笔退款渠道覆盖、审核通过且天数超过上限；到账与原账户核对决定当前处置。',
'R05':'单笔退款审核通过，但渠道未知；天数可已知或同时未知。',
'R06':'单笔退款审核通过且渠道明确为政策未覆盖渠道；不得套用3或5天。',
'M01':'两笔退款均完整可判断且在常规范围，无决定性缺失或已成立人工条件。',
'M02':'两笔中至少一笔已成立人工条件；另一笔无决定性缺失。',
'M03':'两笔中至少一笔已成立人工条件，另一笔仍有决定性缺失；整体升级。',
'M04':'两笔均无已成立人工条件且至少一笔缺决定性信息。',
'B01':'用户报告同一订单疑似两次扣款；不认定已确认重复。',
'B02':'用户要求实际退款、补偿或调账；排除仅咨询时效与B01疑似重复。',
'P01':'认证有效、基础能力、当前项目出口白名单false且有该项目配置权。',
'P02':'认证有效、导出功能当前项目环境true、项目出口白名单false且用户可配置。',
'P03':'认证有效、导出功能在当前项目环境明确false；白名单可true/false/未知。',
'P04':'认证有效、当前功能未知且当前白名单已知，无仅异范围开通记录；排除P03、P06。',
'P05':'认证有效、当前功能true、当前项目出口白名单未知；排除当前功能false。',
'P06':'认证有效、导出请求：仅异范围功能记录，或功能和白名单均未知，或全部必要配置满足仍403。',
'F01':'无已成立严重影响；只读、已恢复、影响有限有据，500或超时。',
'F02':'支付异常、数据丢失/错乱或生产大面积不可用已成立；排除恢复且数据丢失的F03；认证问题不阻断升级。',
'F03':'服务已恢复但仍确认数据丢失；保留其他严重影响与认证建议。',
'F04':'没有已成立严重影响，且不足以确认F01的有限恢复只读；询问实际缺失的影响信息。',
'A01':'没有已成立严重影响且响应明确Token过期；保留重新授权后验证指导。',
'A02':'没有已成立严重影响且401原因未定，过期未确认；不能由401断言过期。',
}
PRIORITY='billing：实际操作/重复扣款→退款对象数；单笔审核未知R01→渠道未知R05→已知不覆盖R06→计时未知R02→比较R03/R04。technical：已恢复且丢数据F03→其他已成立严重影响F02→认证A01/A02→权限P01-P06→一般故障F01/F04。P03优先；仅异范围/两状态缺失/全部满足仍403归P06，其余功能未知白名单已知归P04。主层唯一，其他已知事实和必要建议仍保留。'


def build(design):
    from benchmarks.scripts.balanced_policy_candidates import SUBTYPES, conditions_for, classify, actual_subtype, source_family
    records=[]
    for layer,e in zip(LAYERS,examples()):
        sid=layer[0];e.states=main_sources(sid)
        classified,disp=classify(e.task['role'],e.states)
        if classified!=sid or disp!=e.disposition:raise ValueError(('main assignment',sid,classified,disp))
        records.append(example_record(e,sid,actual_subtype(sid,e.states),'retained_main'))
    records.extend(legacy_variants(design))
    def pattern(objects):
        result=[]
        for target,fields in objects.items():
            for key,entry in sorted(fields.items()):
                v=entry['value']
                state=entry['state'] if entry['state']=='known' else 'unknown'
                detail=(v.get('status') if isinstance(v,dict) else v) if type(v) is bool or isinstance(v,dict) and 'status' in v else None
                result.append((target,key,state,detail))
        return result
    covered={(r['metadata']['stratum_id'],r['metadata']['subtype_id']) for r in records}
    for sid,names in SUBTYPES.items():
        for subtype in names:
            objects=conditions_for(sid,subtype)
            matches=[r for r in records if r['metadata']['stratum_id']==sid and r['metadata']['subtype_id']==subtype and pattern(r['metadata']['condition_states'])==pattern(objects)]
            if not matches:
                records.append(example_record(profile_example(sid,subtype,conditions_for(sid,subtype)),sid,subtype,'added_behavior_contract'))
    # Actual known-false vs absent distinction, in addition to the retained two-absent example.
    # Each retained subtype has at least one full runnable contract; pure style changes add none.
    for name,key in [('examples_public.json','public_task'),('examples_private.json','private_standard')]:dump(design/name,[r[key] for r in records])
    dump(design/'examples_reference.json',[dict(task_id=r['public_task']['task_id'],result=r['result'],trajectory=r['trajectory']) for r in records])
    dump(design/'examples_metadata.json',[r['metadata'] for r in records])
    conditions=[]
    for layer in LAYERS:
        sid,group,name,disp,policies=layer
        clause_ids={s for r in records if r['metadata']['stratum_id']==sid for x in r['private_standard']['expectations'] for g in x['supports'] for s in g if s in CLAUSES}
        sources=[dict(clause_id=k,source_id=CLAUSES[k].source_id,text=CLAUSES[k].text,source_path='mymind/agents/customer_policy/resources/knowledge.md',line=next(i for i,line in enumerate(KNOWLEDGE.splitlines(),1) if CLAUSES[k].text.strip().split('\n')[0] in line)) for k in sorted(clause_ids)]
        names=SUBTYPES[sid];quota={split:{sub:count//len(names)+int(i<count%len(names)) for i,sub in enumerate(names)} for split,count in [('train',15),('validation',5),('test',5)]}
        conditions.append(dict(stratum_id=sid,role='billing' if sid[0] in 'RMB' else 'technical',scenario=group,business_question=name,
            necessary_conditions=DEFINITIONS[sid],exclusions='不满足本层必要条件，或在统一优先规则中已归其他层。未列权限配置组合不作为本版支持输入。',overlap_rule=PRIORITY,
            assignment_implementation='balanced_policy_candidates.classify / actual_subtype；直接检查字段与对象关系',
            subtypes=[dict(subtype_id=sub,conditions=conditions_for(sid,sub),contract_ids=[r['public_task']['task_id'] for r in records if r['metadata']['stratum_id']==sid and r['metadata']['subtype_id']==sub]) for sub in names],
            subtype_quota=quota,material_variation='来源明确未知/未提供、记录呈现位置、接口同能力名称与跨环境白名单记录为独立材料轴，不冒充额外决定子型。已知事实的变化如影响必要输出需单独契约。',
            style_rule='seed42独立随机流，对稳定600行顺序逐行randrange(5)；不读取split/处置/子型',policies=sources,
            capacity=dict(candidate_specs=25,final_accepted=0,final_task_gap=25),
            expected_disposition_rule='classify返回当前处置；R04明确未核对且未到账为answer指导核对，未知条件为request_info；当前资金核实条件齐全才escalate。'))
    dump(design/'conditions.json',conditions)
    families=[]
    for role,titles in [('billing',BILLING_EVENTS),('technical',TECH_EVENTS)]:
        families.extend(source_family(role,i,title) for i,title in enumerate(titles))
    rng=random.Random(42);assignments={};positions={}
    for role in ['billing','technical']:
        ids=[f['family_id'] for f in families if f['role']==role];rng.shuffle(ids)
        for j,fid in enumerate(ids):
            assignments[fid]='train' if j<15 else 'validation' if j<20 else 'test'
            positions[fid]=j if j<15 else j-15 if j<20 else j-20
    styles=random.Random(42);manifest=[]
    for c in conditions:
        sid=c['stratum_id'];names=SUBTYPES[sid]
        for family in [f for f in families if f['role']==c['role']]:
            fid=family['family_id'];subtype=names[positions[fid]%len(names)]
            objects=conditions_for(sid,subtype)
            assignment,disposition=classify(c['role'],objects)
            if assignment!=sid:raise ValueError(('candidate assignment',sid,subtype,assignment))
            # Source occurrence descriptions are independent of the output contract.
            variant=dict(variant_id=sid+'-'+subtype,retained_source_records=['origin:1','origin:2'],
                         retained_relations=family['object_relations'],
                         changed_conditions=objects,
                         change_scope='仅将这次事件的退款状态/渠道、请求用途/配置或观察到的影响改为明确列值；未列条件不继承其他变体。',
                         compatibility_review='该过程提供可核对的资金或请求记录；本变体是该过程的假设性政策咨询，不新增合同权利或根因判断。',
                         original_record_use='原始过程记录只支持对象关系，不支持可变状态；最终题面另给本变体的用户状态记录。')
            manifest.append(dict(task_id=f'balanced-{sid}-{fid.rsplit("-",1)[1]}',stratum_id=sid,subtype_id=subtype,family_id=fid,role=c['role'],
                                 decisive_conditions=dict(objects=objects),variant=variant,object_relations=family['object_relations'],
                                 source_description=family['process'],expected_disposition=disposition,style=STYLES[styles.randrange(5)],
                                 candidate_split=assignments[fid],review_status='AI_spec_reviewed_not_final_task',
                                 unresolved=['完整公开题面、标准尚未构造验收','来源过程及记录关系已经AI规格审阅，最终材料仍须核查同源性']))
    for family in families:
        family['candidate_split']=assignments[family['family_id']]
        family['variants']=[dict(task_id=r['task_id'],**r['variant']) for r in manifest if r['family_id']==family['family_id']]
    dump(design/'source_families.json',families);dump(design/'candidate_manifest.json',manifest)
    mappings=[('审核未知与审核后计时未知',['R01','R02']),('恰好与超过时限',['R03','R04']),('渠道未知与未覆盖',['R05','R06']),('超时到账及核对组合',['R04']),('一笔升级另一笔待补',['M03']),('白名单已知功能未知及反向',['P04','P05']),('功能项目/环境错配与白名单环境独立',['P06','P01','P02']),('严重影响与频率人数未提供',['F02']),('恢复但仍丢数据',['F03']),('明确过期与401未定',['A01','A02']),('严重影响与认证同时出现',['F02'])]
    dump(design/'boundary_mapping.json',[dict(boundary=b,stratum_ids=ids,contracts=[r['public_task']['task_id'] for r in records if r['metadata']['stratum_id'] in ids]) for b,ids in mappings])
    print(json.dumps(dict(contracts=len(records),candidates=len(manifest),families=len(families)),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['build','validate'])
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.worker:
        from benchmarks.scripts.balanced_policy_validation import run_logged
        raise SystemExit(run_logged(args.action))
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    design=root/'data/agentic_rl/balanced-policy-v1/design';logs=root/'runs/balanced-policy-v1/data-design'
    design.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    if args.action=='build':build(design)
    else:
        from benchmarks.scripts.balanced_policy_validation import validate
        validate(design,logs)


if __name__=='__main__':
    main()
