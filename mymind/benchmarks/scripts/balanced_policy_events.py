"""Authored synthetic events and finite business-form decoding."""
from copy import deepcopy
import json

from agents.customer_policy.protocol import PublicTask

UNKNOWN = '持有人明确表示不清楚'
LABELS = {
    '支付类别': 'payment_channel', '审核回执': 'approval_status',
    '通过后完整工作日': 'elapsed_workdays', '查过原支付账户': 'original_account_checked',
    '原账户收到款': 'received', '申请登记': 'refund_stage',
    '响应码': 'http_status', '凭据仍有效': 'auth_valid', '响应明确过期': 'token_expired',
    '操作名称': 'interface_name', '请求归属': 'request_scope', '实际出口': 'outbound_ip',
    '配置权登记': 'user_can_configure', '功能登记': 'feature_enabled',
    '出口登记': 'outbound_ip_allowed', '异范围功能登记': 'other_feature_record',
    '只读调用': 'request_read_only', '恢复记录': 'recovered', '支付链路异常': 'payment_affected',
    '写业务异常': 'write_affected', '核对发现丢失': 'data_loss', '核对发现错乱': 'data_corruption',
    '影响边界': 'impact_scope', '运行环境': 'environment', '大面积不可用': 'widespread_outage',
    '观察超时': 'timeout_observed', '影响人数': 'affected_users', '出现频率': 'occurrence',
    '显示扣款条数': 'reported_charge_count', '两条对应同一订单': 'charges_same_order',
    '流水号': 'transaction_id', '交易时间': 'payment_time', '金额': 'amount',
    '流水摘要': 'transaction_summary', '订单号': 'order_id', '本次诉求': 'funds_request',
    '请求定位号':'request_id', '发生时间':'event_time', '脱敏摘要':'response_summary',
}

# Actual joins differ: return line -> payment; contract -> cycle; settlement
# component -> captured payment; scheduler -> route; deployment -> pod; message
# -> consumer. These relations are retained within each counterfactual family.
DOSSIERS = {
 'billing-dossier-01': dict(
    story='主机先签收，扩展坞缺货取消。随后主机因价保另建一笔已存在的差价退款。这里只问这两笔退款的进度，不据签收或取消推定审核通过。',
    links={'R1': ['取消行-扩展坞', '支付-附件'], 'R2': ['价保行-主机', '支付-主件']},
    relation='取消单行回指附件的原支付；价保单行回指主机原支付。签收日期不是退款审核起点。',
    channel='bank_card', days=6, below=2, project=None),
 'billing-dossier-02': dict(
    story='年付合同终止后改月付；月付首期随后也撤销。两份终止结算各自引用原周期支付，页面上的新首付不表示旧年费退款到账。',
    links={'R1': ['年付终止合同', '年周期实付'], 'R2': ['月付撤销合同', '月首期实付']},
    relation='合同终止结算表通过周期栏关联实付流水；年周期和月首期是两笔独立支付。',
    channel='online_payment', days=4, below=1, project=None),
 'billing-dossier-03': dict(
    story='预售先付定金再付尾款。取消后已经分别建立两笔退款，当前只问进度，不讨论定金资格。合计结算页没有单独审核时点。',
    links={'R1': ['结算页尾款分项', '尾款捕获'], 'R2': ['结算页定金分项', '定金捕获']},
    relation='结算分项借助捕获用途栏定位原支付；尾款排在定金之前，不能按付款顺序分配本次标识。',
    channel='bank_card', days=9, below=3, project=None),
 'technical-dossier-01': dict(
    story='日历调度发布后使用专用出口，旧手工任务仍走共享出口。本次提交日历触发的请求；旧手工任务成功不能代表本次。',
    links={'Q1': ['日历触发执行单', '专用路由调用']},
    relation='执行单的触发器栏指向日历调度，路由表再将该执行单关联专用出口调用。',
    project='排班报表', environment='production', ip='198.51.100.81'),
 'technical-dossier-02': dict(
    story='蓝绿切换时入口指向绿实例，但一项回放仍指定蓝实例。本次咨询指定回放调用；入口切换记录不能替代回放的实例归属。',
    links={'Q1': ['指定蓝实例回放单', '蓝实例调用']},
    relation='回放单显式指定蓝实例，部署表将蓝实例关联测试项目；绿实例生产开通记录不继承给回放。',
    project='回放沙箱', environment='test', ip='203.0.113.18'),
 'technical-dossier-05': dict(
    story='队列扩容后有两组消费者。用户追踪一条重投消息，首投在旧节点，重投落在新增节点。本次问题指重投；首投结果不替代重投结果。',
    links={'Q1': ['消息重投序号2', '新增消费者调用']},
    relation='投递清单用消息及重投序号定位消费者，消费者运行表关联当前请求；首投属于另一调用。',
    project='消息结算', environment='production', ip='192.0.2.42'),
}

from benchmarks.scripts.balanced_policy_dossiers import complete_dossiers, select_event
complete_dossiers(DOSSIERS)

AUTH_CONTEXT = {
 1:'调度器本周期沿用昨日缓存凭据；手工任务已换新凭据。本次摘录来自日历触发执行，尚未重试。',
 2:'回放客户端显式选择蓝实例凭据槽，入口绿实例的授权成功不属于本次回放；脱敏响应同时标明客户端和凭据槽。',
 3:'合并后的项目书签仍从旧项目客户端读取凭据；开通单只涉及功能，不证明这份凭据的期限。',
 4:'演练客户端使用演练凭据访问生产资源，响应来自生产端；测试端登录成功不是本次响应。',
 5:'消息首次投递已结束，本次重投落在新增消费者；响应日志标明重投序号2及该消费者的缓存凭据槽。',
 6:'旧代理保留连接携带轮换前凭据；新代理探测使用另一份凭据，本次响应属于旧连接。',
 7:'网关未转发浏览器会话，而用服务端凭据发起下游请求；摘录是下游鉴权响应。',
 8:'生成作业从批处理凭据槽读取，预览作业使用交互登录；响应标明生成作业，不能沿用预览的认证结果。',
 9:'目标区新实例复制了源区凭据槽；源区续期操作未同步到该实例，摘录对应目标区首次调用。',
 10:'隔离租户使用独立服务凭据，共享租户的续期记录属于另一凭据；本次响应来自隔离项目。',
 11:'交接仅包含当前响应摘录及客户端别名，原始凭据和管理页面均未转交；不将日志索引当有效性证明。',
 12:'旧采集客户端继续从本地缓存读取，桌面版本已经更新；摘录显示旧采集版本，尚无旧客户端换证后调用。',
 13:'启动日志确认容器读取挂载卷缓存，而非配置页的新凭据；响应对应这次重建后的首次请求。',
 14:'日历执行使用机器人账户，手工运行使用操作员账户；本次响应属于机器人，不能借手工成功判断。',
 15:'队列消费者使用通知服务凭据，前台支付会话使用用户凭据；摘录来自消费者处理通知时的接口。',
 16:'审计任务使用审计服务账户，普通查询使用值班账户；本次仅提供审计任务的脱敏响应。',
 17:'修复作业结束后验收脚本单独取凭据，修复作业的认证记录不属于验收脚本；摘录对应验收。',
 18:'换版实例加载随版本打包的凭据，未换版组加载旧挂载；本次响应标明换版组，另一组结果不替代。',
 19:'公网连接使用出口侧注入的服务凭据，内网会话的登录记录不适用于它；摘录来自公网实际调用。',
 20:'所有者变更只移交配置权，当前调用仍使用原服务账户；转交单没有服务凭据有效期。',
 21:'网格最后一跳换用下游服务凭据，入口凭据已被交换；鉴权摘录来自最后一跳而非入口。',
 22:'移动端保留本地会话，后台使用定时续期服务账户；摘录来自移动端本次执行，后台成功另列。',
 23:'支付账本写入已完成，通知消费者另持服务凭据访问通知接口；本次401不来自支付扣款接口。',
 24:'快照恢复带回了快照内的凭据缓存，恢复后尚未刷新；摘录来自恢复后核对调用，快照创建时成功不代表现在。',
 25:'只读查询进程与批量写进程使用不同凭据槽；当前摘录只标识查询进程，写进程的授权结果不继承。',
}


def refund_form(channel, days):
    return {'支付类别': channel, '审核回执': 'approved', '通过后完整工作日': days,
            '查过原支付账户': True, '原账户收到款': False}


def refund_variant(sid, subtype, dossier):
    """Write a hypothetical progress return for existing, event-bound refunds."""
    ch = dossier['channel']; overdue = dossier['days']
    a = refund_form(ch, dossier['below'])
    b = refund_form('online_payment' if ch == 'bank_card' else 'bank_card', 2)
    if sid == 'R01':
        a.update({'审核回执': UNKNOWN, '申请登记': 'accepted'})
        a.pop('通过后完整工作日')
        if subtype == 'approval_and_channel_unknown': a['支付类别'] = UNKNOWN
    elif sid == 'R02': a['通过后完整工作日'] = UNKNOWN
    elif sid == 'R03':
        a['支付类别'] = 'online_payment' if subtype.startswith('online') else 'bank_card'
        limit = 3 if a['支付类别'] == 'online_payment' else 5
        a['通过后完整工作日'] = limit if subtype.endswith('exact') else min(dossier['below'], limit-1)
    elif sid == 'R04':
        a['通过后完整工作日'] = overdue
        key = {'check_unknown':'查过原支付账户','receipt_unknown':'原账户收到款'}.get(subtype)
        if key: a.pop(key)
        if subtype == 'not_checked': a['查过原支付账户'] = False
        if subtype == 'received': a['原账户收到款'] = True
    elif sid == 'R05':
        a['支付类别'] = UNKNOWN
        if subtype == 'channel_and_time_unknown': a.pop('通过后完整工作日')
    elif sid == 'R06': a['支付类别'] = 'balance' if subtype == 'balance_enum' else '线下汇款渠道'
    elif sid.startswith('M'):
        if sid in {'M02','M03'}: a['通过后完整工作日'] = overdue
        if subtype == 'both_overdue': b['通过后完整工作日'] = 8
        if sid in {'M03','M04'}:
            key = {'other_approval_unknown':'审核回执','other_elapsed_unknown':'通过后完整工作日',
                   'other_channel_unknown':'支付类别','other_check_unknown':'查过原支付账户',
                   'other_receipt_unknown':'原账户收到款','both_incomplete':'审核回执'}[subtype]
            b[key] = UNKNOWN
            if key == '审核回执': b.pop('通过后完整工作日')
            if key in {'查过原支付账户','原账户收到款'}: b['通过后完整工作日'] = 8
            if subtype == 'both_incomplete': a.pop('通过后完整工作日')
    return {'R1': a, **({'R2': b} if sid.startswith('M') else {})}


def technical_variant(sid, subtype, d):
    if sid.startswith('P'):
        scope = dict(project=d['project'], environment=d['environment'])
        a = {'响应码':403, '凭据仍有效':True, '操作名称':'报表预览' if sid=='P01' else '批量报表生成',
             '请求归属':scope, '实际出口':d['ip'], '配置权登记':dict(project=d['project'],status=True),
             '出口登记':dict(project=d['project'],outbound_ip=d['ip'],status=False)}
        if sid != 'P01': a['功能登记'] = dict(**scope,status=True)
        if sid == 'P03':
            a['功能登记']['status'] = False
            a['出口登记']['status'] = True if subtype.endswith('true') else False if subtype.endswith('false') else 'unknown'
        if sid == 'P04':
            a['功能登记'] = UNKNOWN; a['出口登记']['status'] = subtype.endswith('true')
        if sid == 'P05': a['出口登记'] = UNKNOWN
        if sid == 'P06':
            a['出口登记']['status'] = True
            if subtype == 'both_unknown':
                a.pop('功能登记'); a.pop('出口登记')
            elif subtype != 'all_ready_still_403':
                a.pop('功能登记')
                a['异范围功能登记'] = dict(project=d.get('other_project','历史'+d['project']) if subtype=='other_project' else d['project'],
                    environment=('test' if d['environment']=='production' else 'production') if subtype=='other_environment' else d['environment'],status=True)
        return {'Q1':a}
    a = {}
    if sid == 'F01':
        a.update({'只读调用':True,'恢复记录':True,'支付链路异常':False,'写业务异常':False,
                  '核对发现丢失':False,'核对发现错乱':False,'影响边界':'仅本次派生读取，原业务及其他任务正常'})
        a['观察超时' if subtype=='timeout' else '响应码'] = True if subtype=='timeout' else 500
    elif sid == 'F02':
        a['响应码'] = 500
        if subtype in {'payment','payment_and_expired'}: a['支付链路异常'] = True
        if subtype == 'production_widespread': a.update({'运行环境':'production','大面积不可用':True})
        if subtype == 'corruption': a['核对发现错乱'] = True
        if subtype == 'data_loss_not_recovered': a.update({'核对发现丢失':True,'恢复记录':False})
        if subtype == 'payment_and_expired': a.update({'响应码':401,'响应明确过期':True})
    elif sid == 'F03': a.update({'恢复记录':True,'核对发现丢失':True})
    elif sid == 'F04':
        a['响应码'] = 500
        for k in ['恢复记录','只读调用','支付链路异常','核对发现丢失','核对发现错乱','影响人数']: a[k] = UNKNOWN
        if subtype == 'read_only_known_impact_unknown': a['只读调用'] = True
    else:
        a['响应码'] = 401
        if sid == 'A01': a['响应明确过期'] = True
        else: a.update({'响应明确过期':UNKNOWN,'凭据仍有效':UNKNOWN})
    return {'Q1':a}


def materialize_event(row):
    d = DOSSIERS[row['family_id']]; sid=row['stratum_id']; sub=row['subtype_id']
    d = select_event(d,row['family_id'],sid)
    if sid.startswith(('R','M')): forms=refund_variant(sid,sub,d)
    elif sid.startswith('B'):
        # A separate hypothetical capture anomaly on the same event's selected
        # component; never reinterpret two legitimate component payments as a duplicate.
        a={'订单号':d['links']['R1'][0], '本次诉求':'实际退款'}
        if sid == 'B01':
            a={'订单号':d['links']['R1'][0], '显示扣款条数':2,'两条对应同一订单':True}
            if sub == 'handoff_materials_complete':
                n=int(row['family_id'].rsplit('-',1)[1])
                amount={4:180,9:320,11:260,13:260,15:260,18:240,20:300,22:260}.get(n,260)
                a.update({'流水号':['捕获甲','捕获乙'],'交易时间':['10:12','10:17'],'金额':[amount,amount],
                          '支付类别':d['channel'],'流水摘要':'用户提供选定分项下的两条捕获记录，尚未核验是否重复'})
        forms={'O1':a}
    else: forms=technical_variant(sid,sub,d)
    n=int(row['family_id'].rsplit('-',1)[1])
    if sid in {'F02','F03'} and n not in {1,2,5,11} and n%3==0:
        forms['Q1'].update({'请求定位号':d['links']['Q1'][-1]+'-本次', '发生时间':'2026-10-28T10:20:00+08:00',
            '脱敏摘要':'本次调用响应及业务影响见同记录表单；未附完整日志或凭据。'})
    links={t:d['links'].get(t,d['links'].get('R1')) for t in forms}
    if sid.startswith('B'):
        links={'O1':[d['links']['R1'][0], '该分项本次订单资金材料']}
    origin={'事件经过':d['story'].replace('这里只问这两笔退款的进度，','').replace('当前只问进度，',''),
            '关系说明':d['relation'],'目标路径':links}
    origin['业务关联记录']=[{'业务项':path[0],'对应资金记录' if t.startswith(('R','O')) else '对应调用':path[-1]} for t,path in links.items()]
    if 'records' in d: origin['原始业务摘录']=deepcopy(d['records'])
    if row['family_id'] in {'technical-dossier-17','technical-dossier-24'}:
        origin['本次观察边界']='修复或快照启动记录先于本次调用；后附本次调用及业务核对记录独立给出当前恢复和完整性，不从启动成功继承。'
    if sid=='B01': origin['补充经过']='假设选定分项下又显示两条扣款，二者均指该分项订单；不是把两个分项的正常支付合并为双扣。'
    if sid=='B02': origin['补充经过']='本变体诉求改为实际办理退款，不只是咨询既有退款的进度。'
    if sid in {'F02','F03'}:
        origin['补充经过']='运行单将本次调用关联到业务处理批次，后附影响记录描述该批次；恢复仅指接口可用，业务对账另列。'
    records=[dict(reference_id='user:2',content=json.dumps(origin,ensure_ascii=False,indent=2))]
    for i,(t,form) in enumerate(forms.items(),3):
        direct=sid.startswith('A')
        body={'对象' if direct else '记录归属':t if direct else links[t][-1], '业务表单':form}
        if sid=='R04' and sub=='received':
            body['到账核对']={'核对对象':links[t][-1], '实际到账时审核后完整工作日':form['通过后完整工作日'],
                '当前审核后完整工作日':form['通过后完整工作日']+2,'说明':'表单计时描述实际到账时点，当前已到账，不要求继续追未到账。'}
        if direct:
            n=int(row['family_id'].rsplit('-',1)[1])
            body['本次认证记录']={'调用选择':links[t][0],'实际调用':links[t][-1],
                '客户端与凭据来源':AUTH_CONTEXT[n],
                '脱敏响应':({'status':401,'error':'token_expired','说明':'服务端明确指本次携带凭据已过期；尚未重新授权验证。'} if sid=='A01' else
                    {'status':401,'error':'unauthorized','说明':'未返回过期原因；持有人明确不清楚本次凭据是否过期及是否有效。'}),
                '凭据披露':'仅提供客户端及凭据槽来源，无完整凭据。'}
            form.pop('响应明确过期',None)
            form.pop('凭据仍有效',None)
        if sid.startswith('F'):
            body['业务影响归属']={'调用':links[t][-1], '业务批次':links[t][0],
                '观察范围':'只描述本次批次；其他节点、客户端或历史执行不在本表核对范围。'}
        if sid=='F03':
            expected,actual={'technical-dossier-01':(24,22),'technical-dossier-02':(8,7),
                             'technical-dossier-05':(16,13)}.get(row['family_id'],(40,36))
            body['业务表单'].pop('核对发现丢失')
            body['恢复后对账']={'关联调用':links[t][-1], '应有业务记录':expected,'实际业务记录':actual,
                              '核对范围':'本次处理批次的最终业务记录，已排除仅展示延迟；缺项仍未恢复'}
        records.append(dict(reference_id=f'user:{i}',content=json.dumps(body,ensure_ascii=False,indent=2)))
    prompts={'用户短问':'请帮我看附件，','事件复盘':'我们复盘后把原始材料列在附件，',
             '交接便笺':'客服交接：客户提供了附件，','问答汇总':'客服问这次看哪份？用户答：按附件的目标路径看，',
             '记录对照':'请对照目标路径和业务表单，'}
    targets='、'.join(forms)
    text=prompts[row['style']]+f'请分别说明{targets}目前能判断什么、下一步怎么办及还缺什么。'
    return PublicTask(task_id=row['task_id'],role=row['role'],user_text=text,records=records)


def read_event(task):
    """Decode the supported authored forms, including the actual association join."""
    origin=json.loads(task.records[0].content)
    states={}; evidence={}
    for t,path in origin['目标路径'].items():
        candidates=[r for r in task.records[1:] if (json.loads(r.content).get('对象')==t or json.loads(r.content).get('记录归属')==path[-1])]
        if len(candidates)!=1: raise ValueError(f'{t}: 对象关系不能唯一定位业务记录')
        r=candidates[0]; body=json.loads(r.content); fields={}
        if '业务关联记录' in origin:
            assert any(x['业务项']==path[0] and path[-1] in x.values() for x in origin['业务关联记录'])
        for label,v in body['业务表单'].items():
            k=LABELS[label]
            fields[k]=dict(state='explicit_unknown' if v==UNKNOWN or isinstance(v,dict) and v.get('status')=='unknown' else 'known',value=None if v==UNKNOWN else deepcopy(v))
        if '本次认证记录' in body:
            auth=body['本次认证记录']
            assert auth['实际调用']==path[-1] and auth['调用选择']==path[0]
            assert auth['脱敏响应']['status']==fields['http_status']['value']
            expired=auth['脱敏响应']['error']=='token_expired'
            fields['token_expired']=dict(state='known' if expired else 'explicit_unknown',value=True if expired else None)
            if not expired: fields['auth_valid']=dict(state='explicit_unknown',value=None)
        if '到账核对' in body:
            audit=body['到账核对']
            assert audit['核对对象']==path[-1] and fields['received']['value'] is True
            assert audit['实际到账时审核后完整工作日']==fields['elapsed_workdays']['value']
        if '恢复后对账' in body:
            audit=body['恢复后对账']
            if audit['关联调用']!=path[-1]: raise ValueError('对账不属于本次调用')
            fields['data_loss']=dict(state='known',value=audit['实际业务记录']<audit['应有业务记录'])
        states[t]=fields
        evidence[t]=[r.reference_id] if body.get('对象')==t else ['user:2',r.reference_id]
    # The relevant omitted user fields are explicit in this finite data contract.
    from benchmarks.scripts.balanced_policy_candidates import absent
    for t,fields in states.items():
        if t.startswith('R'): universe=['payment_channel','approval_status','elapsed_workdays','original_account_checked','received']
        elif t=='O1': universe=['transaction_id','payment_time','amount','payment_channel','transaction_summary']
        elif 'request_scope' in fields: universe=['feature_enabled','outbound_ip_allowed']
        else: universe=['occurrence','affected_users','request_read_only','recovered','payment_affected','write_affected','data_loss','data_corruption']
        for k in universe: fields.setdefault(k,absent())
    return states,evidence
