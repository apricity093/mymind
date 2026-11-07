"""Typed source conditions, direct assignment rules, and synthetic family specifications."""
from copy import deepcopy


def known(value):
    return {'state': 'known', 'value': value}


def absent(explicit=False):
    return {'state': 'explicit_unknown' if explicit else 'not_provided', 'value': None}


def values(**items):
    return {key: known(value) for key, value in items.items()}


def value(fields, key):
    item = fields.get(key, absent())
    if item['state'] != 'known':
        return None
    result = item['value']
    return result.get('status') if key in {'feature_enabled','outbound_ip_allowed','user_can_configure'} and isinstance(result,dict) else result


def refund_fields(channel='online_payment', days=2):
    return values(payment_channel=channel, approval_status='approved', elapsed_workdays=days,
                  original_account_checked=True, received=False)


def permission_fields(basic=False):
    scope = {'project': '目标项目', 'environment': 'production'}
    fields = values(auth_valid=True, http_status=403, interface_name='基础查询' if basic else '报表导出',
                    request_scope=scope, outbound_ip='198.51.100.26',
                    user_can_configure={'project': scope['project'], 'status': True},
                    outbound_ip_allowed={'project': scope['project'], 'outbound_ip': '198.51.100.26', 'status': False})
    if not basic:
        fields['feature_enabled'] = known({**scope, 'status': True})
    return fields


# Names describe actual conditions. Source presentation and prose are separate axes.
SUBTYPES = {
    'R01': ['approval_unknown_channel_known', 'approval_and_channel_unknown'],
    'R02': ['elapsed_unknown'],
    'R03': ['online_below', 'online_exact', 'bank_below', 'bank_exact'],
    'R04': ['unreceived_checked', 'check_unknown', 'receipt_unknown', 'not_checked', 'received'],
    'R05': ['channel_unknown_time_known', 'channel_and_time_unknown'],
    'R06': ['balance_enum', 'other_channel_literal'],
    'M01': ['both_within'],
    'M02': ['overdue_and_within', 'both_overdue'],
    'M03': ['other_approval_unknown', 'other_elapsed_unknown', 'other_channel_unknown', 'other_check_unknown', 'other_receipt_unknown'],
    'M04': ['other_approval_unknown', 'other_elapsed_unknown', 'other_channel_unknown', 'other_check_unknown', 'both_incomplete'],
    'B01': ['handoff_materials_missing', 'handoff_materials_complete'],
    'B02': ['actual_funds_operation'],
    'P01': ['basic_whitelist_fix'],
    'P02': ['export_whitelist_fix'],
    'P03': ['feature_false_white_true', 'feature_false_white_false', 'feature_false_white_unknown'],
    'P04': ['feature_unknown_white_true', 'feature_unknown_white_false'],
    'P05': ['feature_true_white_unknown'],
    'P06': ['other_project', 'other_environment', 'both_unknown', 'all_ready_still_403'],
    'F01': ['http500', 'timeout'],
    'F02': ['payment', 'production_widespread', 'corruption', 'data_loss_not_recovered', 'payment_and_expired'],
    'F03': ['recovered_data_loss'],
    'F04': ['impact_and_recovery_unknown', 'read_only_known_impact_unknown'],
    'A01': ['explicit_expiration'],
    'A02': ['auth_cause_unknown'],
}


def conditions_for(sid, subtype):
    if isinstance(subtype, int):
        subtype = SUBTYPES[sid][subtype]
    if subtype not in SUBTYPES[sid]:
        raise ValueError((sid, subtype))
    if sid.startswith('R'):
        r = refund_fields()
        if sid == 'R01':
            r.update(approval_status=absent(True), elapsed_workdays=absent(), refund_stage=known('accepted'))
            if subtype == 'approval_and_channel_unknown': r['payment_channel'] = absent(True)
        elif sid == 'R02': r['elapsed_workdays'] = absent(True)
        elif sid == 'R03':
            channel, days = {'online_below': ('online_payment', 1), 'online_exact': ('online_payment', 3),
                             'bank_below': ('bank_card', 4), 'bank_exact': ('bank_card', 5)}[subtype]
            r = refund_fields(channel, days)
        elif sid == 'R04':
            r = refund_fields('bank_card', 7)
            if subtype == 'check_unknown': r['original_account_checked'] = absent()
            if subtype == 'receipt_unknown': r['received'] = absent()
            if subtype == 'not_checked': r['original_account_checked'] = known(False)
            if subtype == 'received':
                r['received'] = known(True)
                r['receipt_timing_note'] = known('本次刚刚到账，到账时审核后已过7个完整工作日')
        elif sid == 'R05':
            r['payment_channel'] = absent(True)
            if subtype == 'channel_and_time_unknown': r['elapsed_workdays'] = absent()
        else:
            r['payment_channel'] = known('balance' if subtype == 'balance_enum' else '线下汇款渠道')
        return {'R1': r}
    if sid.startswith('M'):
        a, b = refund_fields('online_payment', 3), refund_fields('bank_card', 5)
        if sid in {'M02', 'M03'}: a['elapsed_workdays'] = known(6)
        if subtype == 'both_overdue': b['elapsed_workdays'] = known(8)
        if sid in {'M03', 'M04'}:
            field = {'other_approval_unknown': 'approval_status', 'other_elapsed_unknown': 'elapsed_workdays',
                     'other_channel_unknown': 'payment_channel', 'other_check_unknown': 'original_account_checked',
                     'other_receipt_unknown': 'received', 'both_incomplete': 'approval_status'}[subtype]
            b[field] = absent(True)
            if field == 'approval_status': b['elapsed_workdays'] = absent()
            if field in {'original_account_checked', 'received'}: b['elapsed_workdays'] = known(8)
            if subtype == 'both_incomplete': a['elapsed_workdays'] = absent()
        return {'R1': a, 'R2': b}
    if sid == 'B01':
        r = values(reported_charge_count=2, charges_same_order=True)
        fields = ['transaction_id', 'payment_time', 'amount', 'payment_channel', 'transaction_summary']
        for field in fields: r[field] = absent()
        if subtype == 'handoff_materials_complete':
            r.update(values(transaction_id=['T-A', 'T-B'], payment_time=['09:00', '09:04'], amount=[180, 180],
                            payment_channel='bank_card', transaction_summary='用户提供两条已扣款记录，是否重复尚未核验'))
        return {'O1': r}
    if sid == 'B02': return {'O1': values(funds_request='实际退款', order_id='O1', transaction_id='T-F', amount=180)}
    if sid.startswith('P'):
        r = permission_fields(sid == 'P01')
        if sid == 'P03':
            r['feature_enabled']['value']['status'] = False
            status = {'feature_false_white_true': True, 'feature_false_white_false': False, 'feature_false_white_unknown': 'unknown'}[subtype]
            r['outbound_ip_allowed']['value']['status'] = status
            if status == 'unknown': r['outbound_ip_allowed']['state'] = 'explicit_unknown'
        elif sid == 'P04':
            r['feature_enabled'] = absent(True)
            r['outbound_ip_allowed']['value']['status'] = subtype.endswith('white_true')
        elif sid == 'P05': r['outbound_ip_allowed'] = absent(True)
        elif sid == 'P06':
            r['outbound_ip_allowed']['value']['status'] = True
            if subtype == 'both_unknown':
                r.update(feature_enabled=absent(), outbound_ip_allowed=absent())
            elif subtype != 'all_ready_still_403':
                r['feature_enabled'] = absent()
                r['other_feature_record'] = known({'project': '其他项目' if subtype == 'other_project' else '目标项目',
                                                   'environment': 'test' if subtype == 'other_environment' else 'production', 'status': True})
        return {'Q1': r}
    if sid.startswith('F'):
        r = {key: absent() for key in ['occurrence', 'affected_users', 'request_read_only', 'recovered', 'payment_affected', 'write_affected', 'data_loss', 'data_corruption']}
        if sid == 'F01':
            r.update(values(request_read_only=True, recovered=True, payment_affected=False, write_affected=False,
                            data_loss=False, data_corruption=False, impact_scope='明确仅限局部任务，其他业务正常'))
            r['timeout_observed' if subtype == 'timeout' else 'http_status'] = known(True if subtype == 'timeout' else 500)
        elif sid == 'F02':
            if subtype in {'payment', 'payment_and_expired'}: r['payment_affected'] = known(True)
            if subtype == 'production_widespread': r.update(values(environment='production', widespread_outage=True))
            if subtype == 'corruption': r['data_corruption'] = known(True)
            if subtype == 'data_loss_not_recovered': r.update(values(data_loss=True,recovered=False))
            r['http_status'] = known(401 if subtype == 'payment_and_expired' else 500)
            if subtype == 'payment_and_expired': r['token_expired'] = known(True)
        elif sid == 'F03': r.update(values(recovered=True, data_loss=True))
        else:
            r['http_status'] = known(500)
            for field in ['recovered', 'request_read_only', 'payment_affected', 'data_loss', 'data_corruption', 'affected_users']: r[field] = absent(True)
            if subtype == 'read_only_known_impact_unknown': r['request_read_only'] = known(True)
        return {'Q1': r}
    r = values(http_status=401)
    if sid == 'A01': r['token_expired'] = known(True)
    else: r.update(token_expired=absent(True), auth_valid=absent(True))
    return {'Q1': r}


def refund_decision(r):
    channel, approval, days = (value(r, k) for k in ['payment_channel', 'approval_status', 'elapsed_workdays'])
    if approval != 'approved': return 'R01', 'request_info'
    if channel is None: return 'R05', 'request_info'
    if channel not in {'online_payment', 'bank_card'}: return 'R06', 'escalate'
    if days is None: return 'R02', 'request_info'
    if days <= (3 if channel == 'online_payment' else 5): return 'R03', 'answer'
    if value(r, 'received') is True: return 'R04', 'answer'
    if value(r, 'original_account_checked') is False: return 'R04', 'answer' if value(r, 'received') is not None else 'request_info'
    if value(r, 'original_account_checked') is None or value(r, 'received') is None: return 'R04', 'request_info'
    return 'R04', 'escalate'


def classify(role, objects):
    """Assign the supported design inputs once; preserve secondary advice separately."""
    if role == 'billing':
        first = next(iter(objects.values()))
        if value(first, 'funds_request') is not None: return 'B02', 'escalate'
        if value(first, 'reported_charge_count') == 2: return 'B01', 'escalate'
        decisions = [refund_decision(r) for r in objects.values()]
        if len(decisions) == 1: return decisions[0]
        if any(d == 'escalate' for _, d in decisions):
            return ('M03' if any(d == 'request_info' for _, d in decisions) else 'M02'), 'escalate'
        if any(d == 'request_info' for _, d in decisions): return 'M04', 'request_info'
        return 'M01', 'answer'
    q = objects['Q1']
    if value(q, 'recovered') is True and value(q, 'data_loss') is True: return 'F03', 'escalate'
    if any(value(q, k) is True for k in ['payment_affected', 'data_loss', 'data_corruption']) or (value(q, 'environment') == 'production' and value(q, 'widespread_outage') is True): return 'F02', 'escalate'
    if value(q, 'http_status') == 401 or value(q, 'token_expired') is True:
        return ('A01', 'answer') if value(q, 'token_expired') is True else ('A02', 'request_info')
    if value(q, 'http_status') == 403:
        feature, white = value(q, 'feature_enabled'), value(q, 'outbound_ip_allowed')
        if value(q, 'interface_name') in {'基础查询', '报表预览'} and white is False and value(q, 'user_can_configure') is True: return 'P01', 'answer'
        if feature is False: return 'P03', 'escalate'
        if value(q, 'other_feature_record') is not None or (feature is None and white is None) or (feature is True and white is True): return 'P06', 'request_info'
        if feature is None and white is not None: return 'P04', 'request_info'
        if feature is True and white is None: return 'P05', 'request_info'
        if feature is True and white is False and value(q, 'user_can_configure') is True: return 'P02', 'answer'
        raise ValueError('权限条件不在本设计支持范围内')
    if value(q, 'request_read_only') is True and value(q, 'recovered') is True and value(q, 'impact_scope') is not None: return 'F01', 'answer'
    return 'F04', 'request_info'


def actual_subtype(sid, objects):
    """Inspect actual state values and relationships, never subtype_index."""
    q = next(iter(objects.values()))
    if sid == 'R01': return 'approval_and_channel_unknown' if value(q, 'payment_channel') is None else 'approval_unknown_channel_known'
    if sid in {'R02','M01','B02','P01','P02','P05','F03','A01','A02'}: return SUBTYPES[sid][0]
    if sid == 'R03':
        channel = 'online' if value(q, 'payment_channel') == 'online_payment' else 'bank'
        return channel + ('_exact' if value(q, 'elapsed_workdays') == (3 if channel == 'online' else 5) else '_below')
    if sid == 'R04':
        if value(q, 'received') is True: return 'received'
        if value(q, 'original_account_checked') is False: return 'not_checked'
        if value(q, 'original_account_checked') is None: return 'check_unknown'
        if value(q, 'received') is None: return 'receipt_unknown'
        return 'unreceived_checked'
    if sid == 'R05': return 'channel_and_time_unknown' if value(q, 'elapsed_workdays') is None else 'channel_unknown_time_known'
    if sid == 'R06': return 'balance_enum' if value(q, 'payment_channel') == 'balance' else 'other_channel_literal'
    if sid == 'M02': return 'both_overdue' if all(refund_decision(r)[1] == 'escalate' for r in objects.values()) else 'overdue_and_within'
    if sid in {'M03','M04'}:
        incomplete = [r for r in objects.values() if refund_decision(r)[1] == 'request_info']
        if len(incomplete) == 2: return 'both_incomplete'
        r = incomplete[0]
        for field, name in [('approval_status','approval'),('payment_channel','channel'),('elapsed_workdays','elapsed'),('original_account_checked','check'),('received','receipt')]:
            if value(r, field) is None: return 'other_' + name + '_unknown'
    if sid == 'B01': return 'handoff_materials_complete' if all(value(q,k) is not None for k in ['transaction_id','payment_time','amount','payment_channel','transaction_summary']) else 'handoff_materials_missing'
    if sid in {'P03','P04'}:
        white = value(q,'outbound_ip_allowed')
        return ('feature_false_white_' if sid=='P03' else 'feature_unknown_white_') + ('unknown' if white is None else 'true' if white else 'false')
    if sid == 'P06':
        other=value(q,'other_feature_record')
        if other:
            return 'other_project' if other['project'] != value(q,'request_scope')['project'] else 'other_environment'
        return 'both_unknown' if value(q,'feature_enabled') is None else 'all_ready_still_403'
    if sid == 'F01': return 'timeout' if value(q,'timeout_observed') is True else 'http500'
    if sid == 'F02':
        if value(q,'token_expired') is True and value(q,'payment_affected') is True: return 'payment_and_expired'
        if value(q,'payment_affected') is True: return 'payment'
        if value(q,'data_corruption') is True: return 'corruption'
        if value(q,'data_loss') is True: return 'data_loss_not_recovered'
        return 'production_widespread'
    if sid == 'F04': return 'read_only_known_impact_unknown' if value(q,'request_read_only') is True else 'impact_and_recovery_unknown'
    raise ValueError((sid,objects))


# The process and the relationship between primary records differ between dossiers.
BILLING_RECORDS = [
('主件已签收、附件取消','主件交付单与附件取消单、两条分账支付流水'),
('旧年费周期结算后改月付','旧合同剩余期结算单与新月付首期流水'),
('定金先付、尾款后付再取消','定金凭证与尾款清算凭证，退款只定位已捕获款项'),
('团购解散后逐支付人结算','发起人汇总单与参与人各自支付凭证'),
('拒收货分两批回仓','第一批签收单与第二批待验收入库单'),
('外币采购取消后以原币结算','原币流水与展示币换算截图分别保留'),
('分期合同终止并结算首期','首期已扣流水与后续停止扣款确认单'),
('住宿提前离店后重算未住夜数','实际入住单与未住夜数结算单'),
('改期活动取消服务','票款凭证和服务费支付记录分别记账'),
('两仓退货分别验收','两个仓库签收时间各自对应原支付子单'),
('席位停用并结算存续席位','停用席位清单和续用席位续费流水'),
('家庭套餐拆开新购','主账户旧合同与子账户新购凭证'),
('商品退货同时保险赔付','保险理赔编号与商品退款编号不相互替代'),
('充值后撤销已订服务','充值入账凭证与服务撤销结算单'),
('课程中止后结算已学部分','出勤明细与未授课退费结算记录'),
('旧云资源释放后新建','旧资源释放单与新资源预付流水'),
('优惠失效后重新付款','旧订单实付凭证与新订单价格明细'),
('自提逾期后结算押金','商品支付单与押金支付单独立'),
('跨店拆分满减后部分退货','店铺原始实付分摊表与子单退款请求'),
('换货失败终止订单','原货支付与换货补差价凭证分开'),
('预授权最终捕获后发现两条通知','预扣通知与最终捕获结算单，是否重复待财务核对'),
('关闭续费时核对已出账期','关闭自动续费确认与关闭前已捕获流水'),
('提前归还设备后结算','归还验收单与押金/租金两类凭证'),
('一部分采购批次验收失败','验收合格清单与退回批次明细'),
('支付重试后核对两条交易','失败响应截图与订单成功结算记录，是否两次捕获待核'),
]
TECH_RECORDS = [
('定时报表改走专用出口','任务调度记录与共享/专用出口访问记录'),
('蓝绿实例切换','旧实例与新实例请求标识、部署批次对应表'),
('分公司项目合并','旧项目开通单与合并后的目标项目请求记录'),
('灾备演练访问主站资源','演练路由单与目标环境资源访问摘要'),
('消费者队列扩容','节点消费记录与每条业务消息请求映射'),
('代理轮换','旧代理和新代理的出口记录与请求时间对应'),
('看板接入网关','看板发起记录与网关转发记录配对'),
('预览和导出作业拆分','预览作业单、导出任务单，记录用途而非凭名称推根因'),
('区域迁移','源区域配置快照与目标区域新实例记录'),
('租户项目隔离','旧共享项目开通单与新隔离项目请求'),
('值班交接','上班摘要与下班尚未移交的原始日志索引'),
('凭据轮换','客户端版本与脱敏凭据有效期记录'),
('容器重建','配置页快照与容器缓存凭据的脱敏状态'),
('日历触发替代手工触发','手工任务单与日历任务请求标识'),
('支付回调入队','前台支付确认与通知落库记录单独追踪'),
('审计导出并行普通查询','审计任务和业务查询的接口用途记录'),
('数据修复后恢复读服务','恢复时间记录与缺失业务记录对账单'),
('单组实例滚动发布','受影响实例组与正常实例组的请求摘要'),
('内网切公网出口','链路切换单与实际公网出口映射'),
('项目所有者交接','配置权交接单与平台开通记录分别保存'),
('服务网格增加跳点','入口请求标识与各跳转发标识对应表'),
('移动端后台共用任务','移动客户端和后台客户端请求来源分别记录'),
('支付成功通知失败','支付账本与通知消费状态单独记录'),
('快照恢复后对账','快照恢复时间和恢复后的业务记录核对结果'),
('写入查询并行','写作业与只读作业的调用及影响记录分开'),
]


def source_family(role, index, title):
    process, records = (BILLING_RECORDS if role == 'billing' else TECH_RECORDS)[index]
    return dict(family_id=f'{role}-dossier-{index+1:02}', role=role, event=title,
                process=[process, '过程发生后，参与者针对本流程中的资金或接口提出本次咨询，提交配对记录', '客服仅依据本次咨询所报告状态解释，不核验后台或执行操作'],
                participants=['提出咨询的支付人或业务操作人', '持有原始流水/请求记录的业务方', '可核验但本阶段不调用的人工处理方'],
                object_relations=records, original_records=[dict(record_id='origin:1',content=process,record_kind='业务过程单的规格',observable_fields=['发起动作','参与主体角色','原始业务对象'],binding='只说明起因；不提供本次R1/R2或Q1状态'), dict(record_id='origin:2',content=records,record_kind='成对用户记录的规格',observable_fields=['记录种类','彼此对应关系','用户可提供的原始凭证'],binding='具体状态由variant另给；不得把历史作业已恢复当成本次Q1已恢复')],
                shared_facts=[process,records],
                target_binding='billing：R1/R2对应本流程资金记录的两笔既有退款，B01/O1是同一流程订单的疑似双扣反事实；technical：Q1是本流程中本次咨询的目标接口调用，历史作业或其他节点状态不代表Q1状态。',
                mutable_fields=['本次退款存在及目标对象映射','渠道与审核后状态','本次请求用途/项目环境/出口','本次配置与认证状态','本次请求业务影响及恢复状态'],
                counterfactual_rule='保留事件过程和记录间关系；每层显式更改退款/请求用途、状态与观察到的业务影响。origin记录不含这些可变判断，不可沿用被改动状态。',
                business_scope='背景仅定位事件，不推导退款资格、金额、保险/税汇/合同责任或故障根因；只判断现有政策规定的时效、能力条件和升级。',
                dev_relation='非dev记录的摘录或改写；共享通用政策不等于同源。AI按过程与原始记录关系审阅；最终自然语言构造后仍需复核。',
                other_family_relation='50个计划事件按过程及配对记录关系区分；共同角色/政策不证明独立。发现同源必须合族重算。',
                review=dict(source='同一执行助手AI审阅，非独立专家或人工金标',basis=process+'；'+records,status='source_specification_reviewed_not_final_data'))
