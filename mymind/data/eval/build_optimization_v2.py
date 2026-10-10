"""生成独立合成评测材料；事实和划分在检索前确定。"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
VERSION = "synthetic-service-v2-20261009"

INTENTS = {
    "query": [("积分兑换有什么门槛？", "想了解积分能换什么"), ("你们几点开始营业？", "周末服务窗口开到几点"), ("会员分几个等级？", "能介绍一下会员档位吗"), ("这个系列有哪些颜色？", "我想知道这款现货的配色"), ("门店位置在哪里？", "去实体店应该走到哪个地址"), ("支持哪些语言？", "产品的界面可以切换成哪些文字")],
    "complaint": [("客服态度让我很不舒服", "接待的人很不耐烦我很生气"), ("一直没人处理我的问题", "等了大半天仍然无人回应"), ("你们承诺的服务没有兑现", "说好的售后根本没做到"), ("连续几次答复互相矛盾", "每个人都给我不同的说法实在受不了"), ("这次服务体验太差了", "整个办理过程让我非常失望"), ("工单被无故关闭我要投诉", "问题没解决你们就结案这不合理")],
    "request": [("请帮我取消尚未支付的订单", "我不买了把待付款单子撤掉吧"), ("替我修改还未发货的收货地址", "包裹没出库请换个收件地点"), ("把收货人改成我母亲", "这单还没寄请更换收件姓名"), ("请合并这两笔未发货订单", "两个待出库订单能一起寄吗"), ("替我暂停自动续订", "下个周期先不要自动续上"), ("请延后这笔预约到周六", "帮忙把服务预约挪到周末")],
    "greeting": [("你好呀", "嗨在吗"), ("早上好客服", "早安今天辛苦了"), ("下午好", "午后好呀有人在线吗"), ("晚上好", "晚安之前来打个招呼"), ("hello", "hi there"), ("第一次来向你问好", "刚进来先说声你好")],
    "escalation": [("请主管复核之前的处理决定", "这个结论希望上级重新审核"), ("我要与你们经理谈", "请安排负责经理接待"), ("需要升级到上一级处理", "目前权限不够请向上升级"), ("让门店负责人给我答复", "希望由门店管理人员处理这件事"), ("请上级复查我的工单证据", "要求更高权限的负责人重看材料"), ("我要求管理层书面解释", "请让管理团队说明决定依据")],
    "technical": [("接口超时时间怎么配置", "SDK等待时长应该填在哪里"), ("导出的文件格式不正确", "下载报表的列格式总是不对"), ("数据同步延迟很大", "两端数据隔很久才一致"), ("批量导入字段映射怎么设置", "上传表格的列如何对应系统字段"), ("回调地址如何验证", "Webhook接收地址在哪测试"), ("打印布局错位如何调整", "报表打出来表头跑到下一页了")],
    "billing": [("订阅套餐按月还是按年计费", "这个服务的收费周期怎么算"), ("账单周期从哪天开始", "每期费用的结算时间怎么定"), ("使用量费用明细在哪里查", "想看本期各项服务花了多少钱"), ("套餐升级差价如何折算", "月中换高档套餐剩余天数怎么收费"), ("试用结束后的费用标准是什么", "免费期过后每月应付多少"), ("账单中的税费怎么算", "费用结算时税额采用什么比例")],
    "account": [("我要修改绑定邮箱", "旧邮箱不用了请换个联系方式"), ("我想注销账户", "以后不用了帮我关闭个人账号"), ("更新我的昵称和头像", "个人资料展示信息想换一下"), ("如何合并两个个人账户", "我注册了两个号能合并资料吗"), ("请解绑不再使用的手机号", "旧号码停用了要解除关联"), ("我要更改账户所在地区", "个人账号地区设置如何调整")],
    "feedback": [("这次接待很棒谢谢", "服务让我非常满意"), ("给刚才的客服一个好评", "工作人员耐心解决了问题值得表扬"), ("处理速度超出预期", "这么快解决我很惊喜"), ("说明写得很清楚", "帮助文档很实用赞一个"), ("配送安排很贴心", "你们这次安排得很周到谢谢"), ("新版使用体验好多了", "更新之后真的顺手很多")],
    "order_status": [("订单现在到哪一步了", "购买记录目前是什么处理状态"), ("我的单子已经发货了吗", "仓库是否把这笔订单发出了"), ("付款后订单还显示待处理", "我想确认已付订单是否进入备货"), ("订单号AB1024有被取消吗", "查一下AB1024是否仍然有效"), ("预售订单什么时候开始备货", "这笔预订目前进入生产了吗"), ("我这笔订单是否已经完成", "购买流程现在算结单了吗")],
    "logistics": [("物流一直不更新", "运单跟踪停在前天没动"), ("快递什么时候到", "包裹预计哪天送达"), ("配送员联系不到怎么办", "承运人员没接电话怎么查件"), ("包裹显示签收但我没收到", "承运记录写送达可我没拿到"), ("能查一下包裹走到哪里吗", "货物现在经过哪个运输站点"), ("偏远地区配送要多久", "寄到山区运输会多花几天")],
    "refund": [("我要申请退款", "这件商品不想要了钱能退回来吗"), ("退货后多久退回钱", "寄回商品之后款项几天到账"), ("有质量问题怎么退货退款", "收到坏的产品想退钱流程是什么"), ("退款申请卡在审核", "退钱单一直等待审批怎么处理"), ("退款可以改到账户余额吗", "原路退的钱能改成平台余额吗"), ("定制商品能否退回退款", "刻了名字的设备可以退钱吗")],
    "invoice": [("帮我开电子发票", "这笔消费需要票据"), ("发票抬头写错了", "票据上的公司名字怎么改"), ("发票税号怎么填写", "开票时纳税识别号放在哪里"), ("电子发票下载入口在哪里", "已开的票从哪儿取到文件"), ("能换成增值税专用发票吗", "普通票如何改成专票"), ("发票金额与订单不一致", "票据显示的总额少了一部分")],
    "payment_issue": [("支付失败但卡里余额够", "结账被拒绝可账户有钱"), ("同一单被重复扣款", "付款后发现扣了两遍钱"), ("扣钱了页面却说未支付", "银行已经扣款订单仍待付款"), ("支付验证码通过却没完成", "银行卡验证后付款一直转圈"), ("这个月莫名多扣了一笔", "没买新东西怎么又扣费了"), ("付款时显示交易限额", "收银台说超过支付额度怎么办")],
    "account_security": [("账户被盗了", "发现别人进入了我的账号"), ("收到异常登录提醒", "异地设备登入通知不是我本人"), ("我要重置密码", "忘了口令希望换个新的"), ("如何开启两步验证", "想给账号加第二道验证"), ("密码可能泄露需要保护账户", "凭据落到别人手里怎样紧急处理"), ("有人擅自改了我的绑定号码", "账户手机号被陌生人换了")],
    "technical_login": [("登录一直报401", "进入账户时提示认证401失败"), ("验证码收不到", "等登录短信很久仍没到"), ("无法登录账号", "输入资料后一直进不了系统"), ("登录页面不断循环跳转", "验证完又回到登入页怎么办"), ("单点登录授权失败", "SSO进入系统时卡在认证环节"), ("正确密码也提示登录过期", "口令没错但登入总说会话失效")],
    "technical_crash": [("应用一直崩溃", "软件打开就退出了"), ("页面报500错误", "浏览业务页面出现服务器500"), ("更新后软件闪退", "安装新版后程序突然关掉"), ("点击保存导致整个应用退出", "提交按钮一按程序就挂了"), ("桌面客户端启动黑屏随后崩溃", "电脑端打开黑一下便自动关闭"), ("系统报错503打不开", "访问服务返回503无法使用")],
    "human_handoff": [("转人工客服", "给我真人接待"), ("我要找人工", "请安排在线工作人员"), ("不要机器人请转人", "能不能让真正的人跟我聊"), ("我需要人工协助填写", "安排客服本人帮我办理吧"), ("请接通电话客服", "让我直接和服务人员说话"), ("机器人理解不了换真人", "这个问题我想交给人工沟通")],
    "other": [("帮我写一首关于月亮的诗", "写几句月光主题诗歌吧"), ("今天的球赛谁赢了", "昨晚那场比赛比分是多少"), ("世界最高的山叫什么", "哪座山峰海拔世界第一"), ("推荐一个晚饭菜谱", "晚上做什么家常菜好"), ("计算三角形面积公式", "怎样求三角形的面积"), ("陪我玩个猜数字游戏", "我们来猜一个你心里的数")],
}


def build():
    corpus, questions, family = [], [], 0
    products = ["星河相机", "青岚路由器", "北辰音箱", "云舟报表", "山海同步", "晨光打印"]
    channels = ["官网零售", "企业采购", "授权门店", "教育专供", "试用转正"]
    topics = ["质量退款", "权限导出", "接口重试", "保修送检", "账单复核", "账号迁移", "物流查件", "票据更正", "设备换新", "积分补记"]
    for i in range(60):
        product, channel, topic = products[i % 6], channels[(i // 6) % 5], topics[i // 6]
        version, code = f"V{2+i//30}", f"E{610+i}"
        title = f"{product}{version}{channel}{topic}"
        sid = f"v2-policy-{i:02d}"
        condition = f"本条仅适用于{product}{version}的{channel}订单，须由原订单申请人提供订单号和完整操作记录。"
        fact1 = f"遇到{code}时先停止重复提交，进入{topic}页面核对{channel}凭证，再上传记录；材料齐全后{2+i%7}个工作日内给出复核结论。"
        fact2 = f"{product}{version}{channel}的{topic}申请超过{10+i%9}个自然日后，不再受理常规申请；经确认的系统故障须提交故障发生日期证明，转专门复核，不能承诺直接批准。"
        long = i < 16
        procedures = []
        stages = ["核对产品与版本", "辨认交易渠道", "准备订单材料", "保存操作记录", "检查凭证有效期", "填写事实经过", "提交支持附件", "确认受理回执", "等待复核结论", "检查执行结果", "申请再次复核", "归档办理记录"]
        for j, stage in enumerate(stages if long else stages[:2]):
            procedures.append(f"步骤{j+1}：{stage}。申请人在办理{product}的{topic}时，应核实当前记录与原始订单对应，不能将其他渠道或另一版本的办理说明作为本订单结论。经办人员在这一环节填写办理时间、观察现象和材料来源，区分用户推测与能够查证的事实。若信息不全，先保留本次记录并通知补充；在补充完成之前不计入材料齐全后的处理期限。回执只说明材料已经收到，不表示审核通过。对记录有异议时，补充说明具体条目，不能重复创建同一事项。")
        middle = len(procedures)//2
        body = f"{condition}\n\n{fact1}\n\n" + "\n\n".join(procedures[:middle]) + f"\n\n{fact2}\n\n" + "\n\n".join(procedures[middle:])
        content = f"# {title}\n\n## 办理条件和执行流程\n{body}\n\n## 适用边界\n这份规则优先用于标题指定的版本和渠道，其他版本、其他渠道不能类推。不得混用不同产品的错误码处理期限；常规期限与系统故障例外分别判断。"
        corpus.append(dict(source_id=sid, title=title, content=content, format="md", synthetic=True, long_section=long))
        entries = [("处理", [condition, fact1], [f"{product}{version}通过{channel}出现{code}，该怎样办，材料和复核时长是什么？", f"我的{product}是{channel}买的{version}版本，屏幕提示{code}，能说说受理要求以及后续步骤吗？"])]
        if i < 15:
            entries.append(("例外", [condition, fact2], [f"{product}{version}{channel}的{topic}超过常规期限，若是系统故障还能直接批准吗？", f"给{product}{version}办{topic}已经超过规定天数了，{channel}订单有系统故障证明时该如何处理？"]))
        for kind, facts, pair in entries:
            split = "dev" if family % 3 == 0 else "test"
            groups = [dict(fact_id=f"{sid}-{kind}-{n}", alternatives=[dict(source_id=sid, quote=fact, start=content.index(fact), end=content.index(fact)+len(fact))]) for n, fact in enumerate(facts)]
            for v, q in enumerate(pair):
                questions.append(dict(id=f"rag-v2-{family:03d}-{v}", family_id=f"{sid}-{kind}", split=split, question=q, answerable=True,
                    tags=["condition", "multi_fact", "long_section" if long else "short_section", "exception" if kind=="例外" else "exact_code"], evidence_groups=groups,
                    required_conditions=[condition], forbidden_claims=["所有版本和渠道均适用", "提交即可直接批准"], synthetic=True))
            family += 1
    # 全库仅规定办理时长和记录，以下 15 个事实不存在。
    absent = ["线下现金补偿金额", "外币退款汇率", "运输丢件赔付倍数", "免费上门次数", "海外仓所在城市", "永久会员赠送额度", "账号出售回购价格", "纸质票据邮寄费用", "积分兑换现金比例", "加急审批收费", "维修零件品牌", "数据导出压缩算法", "第三方保险免赔额", "软件开源许可证", "跨境海关税率"]
    for n, missing in enumerate(absent):
        split = "dev" if n % 3 == 0 else "test"
        for v, q in enumerate([f"星河相机的{missing}是多少？", f"你们材料里有说明{missing}的具体标准吗？"]):
            questions.append(dict(id=f"rag-v2-absent-{n:02d}-{v}", family_id=f"absent-{n:02d}", split=split, question=q,
                answerable=False, tags=["unanswerable"], evidence_groups=[], required_conditions=[], forbidden_claims=[f"编造{missing}的具体值"],
                absence_reason=f"全库未规定{missing}，办理时长不能推出该标准", absent_topic=missing, confusing_sources=["v2-policy-00"], synthetic=True))
    answers = [q["id"] for q in questions if q["split"]=="test" and q["answerable"]][:24]+[q["id"] for q in questions if q["split"]=="test" and not q["answerable"]][:6]
    rag = dict(version=VERSION, synthetic=True, evaluation_scope="共享知识库的问题泛化，非未见文档或真实用户泛化", corpus=corpus, questions=questions, answer_ids=answers,
               legacy_regression_datasets=["gemini_rag_eval.json", "gemini_rag_boundary_eval.json"])
    cases=[]
    for label, families in INTENTS.items():
        for j, pair in enumerate(families):
            for v, text in enumerate(pair):
                cases.append(dict(id=f"intent-v2-{label}-{j}-{v}", family_id=f"{label}-{j}", split="dev" if j<2 else "test", message=text, intent=label, synthetic=True))
    diagnostic=[]
    for n, (text, labels) in enumerate([
        ("退款迟迟没到而且没人回复，我很生气", ["refund", "complaint"]), ("账户被盗以后现在也登不进去", ["account_security", "technical_login"]),
        ("扣了两遍钱我要退回多收那笔", ["payment_issue", "refund"]), ("帮我查包裹顺便开个发票", ["logistics", "invoice"]),
        ("找真人然后让经理复核", ["human_handoff", "escalation"]), ("程序闪退导致支付失败", ["technical_crash", "payment_issue"]),
        ("我想注销账号并退掉订阅", ["account", "refund"]), ("服务太差我要找主管", ["complaint", "escalation"]),
        ("账单金额不对而且发票也有问题", ["billing", "invoice"]), ("订单发没发，发了以后多久送到", ["order_status", "logistics"])]):
        for v in range(2):
            diagnostic.append(dict(id=f"diagnostic-{n}-{v}", family_id=f"diagnostic-{n}", message=text if v==0 else "麻烦处理一下："+text, allowed_intents=labels, scoring="diagnostic_only_no_new_route_rule"))
    intent=dict(version=VERSION, labels=list(INTENTS), cases=cases, diagnostics=diagnostic, synthetic=True)
    for filename, data in [("rag_optimization_v2.json",rag),("intent_optimization_v2.json",intent)]:
        (ROOT/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"documents={len(corpus)} rag={len(questions)} intent={len(cases)} diagnostic={len(diagnostic)}")


if __name__ == "__main__":
    build()
