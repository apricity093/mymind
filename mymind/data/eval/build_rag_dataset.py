"""
构建 check.md 要求的 RAG 检索评测数据集。

输出到 mymind/data/eval/：
  - rag_corpus.json   8 个业务主题的 Markdown 语料（含多级章节）
  - rag_dataset.json  128 条标注查询（8 类 × 16 条）

标注规则：
  - relevant_source_ids / relevant_sections / must_recall_facts 均由模板显式声明；
  - 每个模板的 4 个改写共享 partition，避免改写泄漏到不同数据分区；
  - no_answer 类查询 relevant_source_ids 为空；
  - relevance 以 source_id -> 0/1/2 存储，供 nDCG 分级使用。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.retrieval import source_id_for

OUT_DIR = Path(__file__).resolve().parent

CORPUS = [
    {
        "title": "退款与售后政策",
        "content": """# 退款与售后政策

## 七天无理由退款
用户在购买后 7 天内可以申请无理由退款，商品需保持完好且不影响二次销售。

## 退款审核时效
退款申请提交后，系统会在 1-3 个工作日内完成审核；大促期间审核时间可能延长到 3-5 个工作日。

## 款项到账时效
审核通过后，款项将在 5-7 个工作日内退回原支付账户，具体到账时间以银行入账为准。

## 已发货订单的退货流程
如果商品已发货，需要先完成退货流程才能退款；退货单号生成后请在 7 天内寄回商品。

## 退货运费规则
退货运费由用户承担，除非是商品质量问题；质量问题需在签收后 48 小时内提供照片证据。

## 超期退款
超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能申请退款；超过 30 天不支持退款。
""",
    },
    {
        "title": "物流与配送说明",
        "content": """# 物流与配送说明

## 标准配送
标准配送时效为 3-5 个工作日，订单满 99 元免运费，不满 99 元收取 8 元运费。

## 加急配送
加急配送时效为 1-2 个工作日，运费 15 元；每日 16:00 前支付可在当天发出。

## 物流信息更新
物流信息通常在发货后 24 小时内更新；如果超过 24 小时仍无记录，请先确认商家是否已经发货。

## 超时查件
订单显示已发货但超过 7 天未收到，可以联系客服申请查件，查件处理时效为 2 个工作日。

## 同城配送
同城配送支持当日达或次日达，运费 10 元；配送时间为每天 9:00-18:00，节假日可能延迟。

## 修改收货地址
如果需要修改收货地址，请在发货前联系客服；发货后地址不可修改，只能拒收后重新下单。
""",
    },
    {
        "title": "支付与账单说明",
        "content": """# 支付与账单说明

## 支持的支付方式
平台支持银行卡、微信支付和支付宝；单笔订单金额上限为 50000 元。

## 重复扣款处理
同一订单被重复扣款时，系统会在 1 个工作日内自动原路退回多扣款项，用户无需主动申请。

## 支付失败原因
支付失败请确认银行卡余额充足，并检查是否已开通网上支付功能；连续失败 3 次会临时锁定支付 10 分钟。

## 账单查询
用户可以在订单详情中下载最近 12 个月的账单，账单包含每笔交易的时间和支付渠道。

## 电子发票
电子发票可在支付成功后 24 小时内申请，开票信息提交后不支持修改，请仔细核对抬头和税号。

## 退款与优惠券
退款时已使用的优惠券不退回；若订单发生部分退款，优惠金额按商品价格比例分摊。
""",
    },
    {
        "title": "账户与安全说明",
        "content": """# 账户与安全说明

## 密码安全
建议用户定期修改密码，密码长度至少 8 位，必须同时包含字母和数字。

## 忘记密码
如果忘记密码，可以通过绑定的手机号或邮箱重置；重置链接有效期为 30 分钟。

## 两步验证
用户可以在安全设置中开启两步验证，开启后每次登录都需要输入 6 位动态验证码。

## 异常登录保护
发现账户异常登录时，系统会自动锁定账户并发送通知；用户可凭绑定手机号申请解除锁定。

## 修改绑定邮箱
修改账户邮箱需要先完成身份验证，新邮箱验证通过后 24 小时内生效。

## 账户注销
账户注销后所有订单与积分数据将永久删除且不可恢复，请先处理未完成的退款和售后申请。
""",
    },
    {
        "title": "订阅与会员说明",
        "content": """# 订阅与会员说明

## 自动续费规则
连续包月订阅默认开启自动续费，系统会在到期前 24 小时从原支付渠道扣款。

## 取消订阅
用户可以在订阅管理页面自助取消自动续费；已扣费周期内权益继续有效，不支持按天退订。

## 会员等级
会员等级分为普通会员、银卡会员和金卡会员；累计消费满 1000 元升级银卡，满 5000 元升级金卡。

## 会员折扣
银卡会员享受 95 折优惠，金卡会员享受 9 折优惠，折扣不可与部分限时活动叠加。

## 积分累积
每消费 1 元累积 1 积分；生日当月消费可获得双倍积分，积分到账时间为订单完成后 24 小时。

## 积分有效期
积分有效期为 1 年，过期自动清零；积分可在下单时抵扣，100 积分等于 1 元。
""",
    },
    {
        "title": "开放平台 API 接入文档",
        "content": """# 开放平台 API 接入文档

## 认证方式
所有 API 请求必须在请求头携带 Authorization: Bearer <access_token>；access_token 有效期为 7200 秒。

## 请求格式
接口使用 HTTPS 与 JSON 格式，时间字段统一使用 ISO 8601 标准；请求体大小不能超过 1MB。

## 速率限制
每个应用默认每分钟最多调用 120 次；超出限制会返回 429 状态码和 Retry-After 响应头。

## 沙箱环境
沙箱环境地址为 https://sandbox.example.com，沙箱与生产环境数据隔离，适合联调退款和下单流程。

## 回调验签
Webhook 回调使用 HMAC-SHA256 签名，验签失败时请不要处理该回调；回调重试间隔为 5 分钟，最多 3 次。

## 分页规范
列表接口使用 page 和 page_size 分页，page_size 最大为 100；全量拉取请使用游标接口避免漏单。
""",
    },
    {
        "title": "错误码与故障排查",
        "content": """# 错误码与故障排查

## 401 未认证
接口返回 401 表示认证失败，请检查 access_token 是否缺失、过期或签名错误；重新获取 token 后再试。

## 403 无权限
接口返回 403 表示当前应用没有访问该资源的权限，需要在开放平台申请对应权限点并由管理员审批。

## 500 服务器错误
接口返回 500 表示服务端内部错误，请稍后重试；同一请求连续 3 次返回 500 请记录 request_id 并联系技术支持。

## 网络超时
客户端网络超时建议先检查代理和防火墙；SDK 默认连接超时为 10 秒，读超时为 30 秒。

## 429 限流
遇到 429 请按 Retry-After 退避重试，不要在 1 秒内突发重试，否则会被熔断 5 分钟。

## 应用崩溃
App 端应用崩溃请先清除缓存并重启；如果问题持续，请更新到最新版本并提交崩溃日志。
""",
    },
    {
        "title": "退款常见问题速查",
        "content": """# 退款常见问题速查

## 虚拟商品退款
虚拟商品（会员码、课程）一经兑换不支持退款，请在下单前确认商品类型。

## 退款次数说明
同一账户每月最多发起 5 次无理由退款；超过次数后当月只能申请质量退款。

## 跨境订单退款
跨境订单退款审核需要 5-7 个工作日，款项到账时间取决于发卡行清算周期。

## 退款失败处理
退款失败时系统会发送站内信，用户需要补充正确的原支付账户信息后重新提交。
""",
    },
    {
        "title": "物流与配送补充说明",
        "content": """# 物流与配送补充说明

## 大件配送
大件商品（冰箱、洗衣机）配送时效为 5-7 个工作日，送货前配送员会电话预约。

## 生鲜配送
生鲜订单使用冷链配送，发货后 2 小时内可查询温度记录，签收时请当场验收。

## 物流异常类型
物流异常包括包裹破损、面单脱落和超卖未发；破损件请在签收后 24 小时内拍照反馈。

## 配送范围查询
部分偏远地区不支持配送，下单前可在商品页输入地址查询是否在配送范围内。
""",
    },
    {
        "title": "支付与账单补充说明",
        "content": """# 支付与账单补充说明

## 分期支付
订单满 300 元可使用 3/6/12 期分期，分期服务费由用户承担。

## 对公转账
企业用户可选择对公转账，转账单需在 24 小时内完成支付，否则订单自动关闭。

## 账单争议
对账单有异议可在交易发生后 60 天内发起争议，平台将在 3 个工作日内反馈核查结果。

## 支付安全风控
新设备首次支付超过 2000 元会触发风控验证，验证通过后 24 小时内同一设备免验证。
""",
    },
    {
        "title": "账户与安全补充说明",
        "content": """# 账户与安全补充说明

## 登录设备管理
用户可在设备管理页查看最近 90 天的登录设备，并远程下线不再使用的设备。

## 实名认证
提现和跨境交易需要完成实名认证，认证信息与支付账户不一致会被驳回。

## 账号申诉
账号被盗时可提交申诉材料，申诉处理时效为 1-3 个工作日，通过后恢复原绑定关系。

## 隐私授权管理
第三方授权可在隐私中心逐项查看和撤销，撤销后相关服务立即停止同步数据。
""",
    },
    {
        "title": "订阅与会员补充说明",
        "content": """# 订阅与会员补充说明

## 家庭共享
金卡会员可邀请最多 3 位家庭成员共享会员权益，被邀请人不能再次转赠。

## 会员日活动
每月 8 日为会员日，会员日当天积分抵扣比例翻倍，部分商品额外 95 折。

## 订阅暂停
连续包月订阅支持暂停 1 个月，暂停期间不扣费，恢复后按原周期继续计费。

## 等级有效期
会员等级每年 1 月 1 日重新计算，以过去 12 个月累计消费金额为准。
""",
    },
    {
        "title": "开放平台 API 接入补充说明",
        "content": """# 开放平台 API 接入补充说明

## OAuth 授权流程
第三方应用使用 OAuth 2.0 授权码模式获取用户授权，授权码有效期为 10 分钟。

## 事件订阅
事件订阅支持订单支付成功、退款完成和物流签收三类事件，订阅后即时推送。

## 接口幂等
创建订单和退款接口支持幂等键，重复请求携带相同幂等键不会产生重复单据。

## 日志与监控
开放平台提供 API 调用日志查询，日志保留 30 天，可下载为 CSV 文件用于对账。
""",
    },
    {
        "title": "错误码与故障排查补充说明",
        "content": """# 错误码与故障排查补充说明

## 400 请求错误
接口返回 400 表示请求参数格式错误，请检查 JSON 结构和必填字段是否完整。

## 404 资源不存在
接口返回 404 表示请求的资源不存在或已被删除，请核对资源 ID 是否拼写正确。

## 503 服务不可用
接口返回 503 表示服务正在维护或过载，请等待 5 分钟后再试，不要高频重试。

## 签名时间戳错误
回调验签报时间戳超时，请先校准服务器时间，时间偏差超过 300 秒会被拒绝。
""",
    },
    {
        "title": "退款与售后补充条款",
        "content": """# 退款与售后补充条款

## 退款申请入口
退款申请可在订单详情页提交，提交后不可修改退款原因和退款金额。

## 退款进度查询
退款进度可在售后中心查询，状态分为待审核、审核中、退款中和已到账。

## 退款短信通知
退款状态变化会发送短信通知，短信发送到下单手机号，海外号码可能延迟。

## 退款重新提交
退款被驳回后可以补充材料重新提交，同一订单最多重新提交 2 次。
""",
    },
    {
        "title": "配送服务补充条款",
        "content": """# 配送服务补充条款

## 配送时效承诺
配送时效从仓库出库后开始计算，预售商品以页面承诺的发货时间为准。

## 配送延迟补偿
超过承诺时效 48 小时仍未送达的订单，可申请 5 元无门槛优惠券补偿。

## 配送签收规则
配送员会通过电话或短信联系收件人，连续 3 次联系不上将安排退回仓库。

## 配送投诉处理
对配送服务不满意可在订单页发起投诉，配送投诉处理时效为 1 个工作日。
""",
    },
    {
        "title": "支付渠道补充条款",
        "content": """# 支付渠道补充条款

## 支付限额调整
银行卡快捷支付单笔限额由发卡行决定，平台不修改银行侧限额。

## 支付渠道切换
订单提交后支付渠道不可更换，如需换渠道请取消订单后重新下单。

## 支付结果通知
支付成功或失败都会在 1 分钟内推送结果通知，网络异常时可到订单页手动刷新。

## 支付凭证下载
支付成功后可在订单详情下载电子支付凭证，凭证保留时间为 5 年。
""",
    },
    {
        "title": "账户安全补充条款",
        "content": """# 账户安全补充条款

## 密码修改周期
平台建议每 90 天修改一次密码，连续 5 次输错密码会临时锁定账户 1 小时。

## 安全验证方式
安全验证支持短信验证码、邮箱验证码和人脸识别，人脸识别仅限已实名账户。

## 账户找回材料
账户找回需要提供注册手机号、最近一笔订单号和身份证明，材料不符将被拒绝。

## 登录保护开关
用户可关闭异常登录自动锁定，但关闭后账户被盗风险由用户自行承担。
""",
    },
    {
        "title": "订阅会员补充条款",
        "content": """# 订阅会员补充条款

## 订阅扣费顺序
订阅扣费优先使用余额，余额不足时按绑定的默认支付渠道顺序扣款。

## 订阅权益说明
连续包月订阅按自然月计算权益，月中开通也按整月计费，不支持按天折算。

## 订阅到期提醒
订阅到期前 3 天会发送站内信和短信提醒，关闭提醒后不再发送通知。

## 订阅历史查询
订阅历史保留最近 24 个月，更早记录可在帮助中心申请人工查询。
""",
    },
    {
        "title": "开放平台 API 补充条款",
        "content": """# 开放平台 API 补充条款

## Token 刷新策略
access_token 过期后使用 refresh_token 刷新，refresh_token 有效期为 30 天。

## 接口版本管理
接口使用 /v1/ 路径前缀管理版本，旧版本在停用前会提前 90 天公告。

## 错误重试建议
网络层错误建议指数退避重试，首次等待 1 秒，最大等待不超过 60 秒。

## 测试环境数据
测试环境数据每日凌晨清理一次，生产环境数据不会被测试请求修改。
""",
    },
    {
        "title": "错误码与故障补充条款",
        "content": """# 错误码与故障补充条款

## 401 常见原因
401 错误常见于 token 过期、签名密钥轮换和请求时间戳偏差过大。

## 403 权限审批
权限点审批由应用管理员在开放平台提交，审批通过后最长 10 分钟生效。

## 500 排查清单
遇到 500 错误请保留完整请求体、响应体和 request_id，三者缺一会影响排查。

## 错误码总表
错误码总表可在开放平台文档中心下载，表格每季度更新一次。
""",
    },
    {
        "title": "平台服务公告",
        "content": """# 平台服务公告

## 客服时间
在线客服服务时间为每天 8:00-23:00；人工客服高峰期平均等待时间为 3 分钟。

## 投诉与建议
投诉与建议可在帮助中心提交，普通投诉处理时效为 2 个工作日，紧急投诉 4 小时内响应。

## 隐私与数据
平台不会向第三方出售用户数据；用户可在隐私中心导出或删除自己的行为数据。

## 安全提醒
客服人员不会索要用户密码或短信验证码，遇到此类要求请立即终止对话并举报。

## 服务中断通知
计划内系统维护会提前 48 小时通过站内信通知，维护期间下单与支付功能可能不可用。

## 版本更新
客户端每 4 周发布一个稳定版本；自动更新失败时可前往官网手动下载最新安装包。
""",
    },
]

# 模板：每个分区 4 个改写，标签完全一致。
TEMPLATES = [
    # refund
    {"id": "refund-01", "category": "refund", "query_type": "factoid", "source": "退款与售后政策", "sections": ["退款与售后政策/七天无理由退款"], "facts": ["购买后 7 天内", "无理由退款"], "grades": {"退款与售后政策": 2}, "queries": ["购买后多少天内可以无理由退款？", "无理由退款的期限是几天？", "我想知道七天无理由退款从什么时候开始算？", "刚买的商品，无理由退款的窗口期是多久？"]},
    {"id": "refund-02", "category": "refund", "query_type": "factoid", "source": "退款与售后政策", "sections": ["退款与售后政策/退款审核时效"], "facts": ["1-3 个工作日"], "grades": {"退款与售后政策": 2}, "queries": ["退款审核一般需要多长时间？", "提交退款申请后多久审核完？", "退款审核要等几个工作日？", "我申请退款了，审核周期是多久？"]},
    {"id": "refund-03", "category": "refund", "query_type": "factoid", "source": "退款与售后政策", "sections": ["退款与售后政策/款项到账时效"], "facts": ["5-7 个工作日"], "grades": {"退款与售后政策": 2}, "queries": ["退款审核通过后钱多久到账？", "退款金额几个工作日能退回银行卡？", "显示退款成功，款项什么时候到原支付账户？", "退款到账一般要等几天？"]},
    {"id": "refund-04", "category": "refund", "query_type": "procedural", "source": "退款与售后政策", "sections": ["退款与售后政策/已发货订单的退货流程", "退款与售后政策/退货运费规则"], "facts": ["先完成退货流程", "48 小时内提供照片证据"], "grades": {"退款与售后政策": 2}, "queries": ["商品已经发货了，想退款要怎么操作？", "已发货订单申请退款需要先做什么？", "东西在路上了，退款流程是什么？", "发货后还能退款吗？具体步骤是什么？"]},
    # logistics
    {"id": "logistics-01", "category": "logistics", "query_type": "factoid", "source": "物流与配送说明", "sections": ["物流与配送说明/标准配送"], "facts": ["3-5 个工作日", "满 99 元免运费"], "grades": {"物流与配送说明": 2}, "queries": ["标准配送几天能送到？", "普通快递的配送时效是多久？", "不选加急的话，几天能收到货？", "标准配送要等几个工作日？"]},
    {"id": "logistics-02", "category": "logistics", "query_type": "factoid", "source": "物流与配送说明", "sections": ["物流与配送说明/物流信息更新"], "facts": ["发货后 24 小时内更新"], "grades": {"物流与配送说明": 2}, "queries": ["发货后多久能看到物流信息？", "物流信息一般什么时候更新？", "订单发货了，为什么还查不到物流记录？", "物流轨迹要等多久才会更新？"]},
    {"id": "logistics-03", "category": "logistics", "query_type": "troubleshooting", "source": "物流与配送说明", "sections": ["物流与配送说明/超时查件"], "facts": ["超过 7 天未收到", "申请查件"], "grades": {"物流与配送说明": 2}, "queries": ["订单显示已发货，超过 7 天还没收到怎么办？", "快递一周了还没到，可以申请什么？", "物流显示已发货但迟迟收不到货，怎么处理？", "包裹超过 7 天没送到，能查件吗？"]},
    {"id": "logistics-04", "category": "logistics", "query_type": "procedural", "source": "物流与配送说明", "sections": ["物流与配送说明/修改收货地址"], "facts": ["发货前联系客服", "发货后地址不可修改"], "grades": {"物流与配送说明": 2}, "queries": ["下单后发现地址写错了，怎么改收货地址？", "发货后还能修改收货地址吗？", "我想改一下配送地址，找谁处理？", "地址填错了，什么时候之前能改？"]},
    # payment
    {"id": "payment-01", "category": "payment", "query_type": "factoid", "source": "支付与账单说明", "sections": ["支付与账单说明/支持的支付方式"], "facts": ["银行卡、微信支付和支付宝", "50000 元"], "grades": {"支付与账单说明": 2}, "queries": ["平台支持哪些支付方式？", "可以用支付宝付款吗？还有哪些支付渠道？", "下单付款支持微信和银行卡吗？", "这个平台能用的支付方式有哪些？"]},
    {"id": "payment-02", "category": "payment", "query_type": "troubleshooting", "source": "支付与账单说明", "sections": ["支付与账单说明/重复扣款处理"], "facts": ["1 个工作日内自动原路退回"], "grades": {"支付与账单说明": 2}, "queries": ["同一订单被重复扣款了怎么办？", "银行卡被扣了两次钱，会自动退吗？", "重复扣款需要我主动申请退款吗？", "一笔订单扣了两次款，多久能退回来？"]},
    {"id": "payment-03", "category": "payment", "query_type": "troubleshooting", "source": "支付与账单说明", "sections": ["支付与账单说明/支付失败原因"], "facts": ["银行卡余额充足", "开通网上支付功能"], "grades": {"支付与账单说明": 2}, "queries": ["支付一直失败，可能是什么原因？", "下单付款提示失败，要检查什么？", "银行卡支付失败该怎么排查？", "为什么我的订单总是支付不成功？"]},
    {"id": "payment-04", "category": "payment", "query_type": "procedural", "source": "支付与账单说明", "sections": ["支付与账单说明/电子发票"], "facts": ["支付成功后 24 小时内申请", "不支持修改"], "grades": {"支付与账单说明": 2}, "queries": ["电子发票怎么申请？", "付款后多久可以开发票？", "开票信息提交后还能改吗？", "我想要电子发票，具体怎么操作？"]},
    # account
    {"id": "account-01", "category": "account", "query_type": "procedural", "source": "账户与安全说明", "sections": ["账户与安全说明/忘记密码"], "facts": ["手机号或邮箱重置", "30 分钟"], "grades": {"账户与安全说明": 2}, "queries": ["忘记密码了怎么找回？", "登录密码忘了，如何重置？", "没有密码登不进去，怎么重置账户密码？", "忘记登录密码应该怎么办？"]},
    {"id": "account-02", "category": "account", "query_type": "factoid", "source": "账户与安全说明", "sections": ["账户与安全说明/密码安全"], "facts": ["至少 8 位", "字母和数字"], "grades": {"账户与安全说明": 2}, "queries": ["设置密码有什么要求？", "账户密码需要多长，要包含什么字符？", "平台对密码长度和格式有要求吗？", "改密码的时候要满足什么规则？"]},
    {"id": "account-03", "category": "account", "query_type": "procedural", "source": "账户与安全说明", "sections": ["账户与安全说明/两步验证"], "facts": ["6 位动态验证码"], "grades": {"账户与安全说明": 2}, "queries": ["怎么开启两步验证？", "账户想加一层安全验证，在哪里设置？", "两步验证开启后每次登录要做什么？", "如何给账户开启二次验证保护？"]},
    {"id": "account-04", "category": "account", "query_type": "troubleshooting", "source": "账户与安全说明", "sections": ["账户与安全说明/异常登录保护"], "facts": ["自动锁定账户", "绑定手机号申请解除锁定"], "grades": {"账户与安全说明": 2}, "queries": ["提示账户有异常登录，账号被锁了怎么办？", "系统检测到异常登录自动锁定了账户，如何解锁？", "账户被自动锁定，能自己解除吗？", "收到异常登录通知，账户被锁定怎么处理？"]},
    # subscription
    {"id": "subscription-01", "category": "subscription", "query_type": "procedural", "source": "订阅与会员说明", "sections": ["订阅与会员说明/取消订阅"], "facts": ["订阅管理页面", "已扣费周期内权益继续有效"], "grades": {"订阅与会员说明": 2}, "queries": ["怎么取消连续包月的自动续费？", "不想续订了，在哪里关闭订阅？", "如何停止会员自动扣费？", "取消订阅的具体入口在哪里？"]},
    {"id": "subscription-02", "category": "subscription", "query_type": "factoid", "source": "订阅与会员说明", "sections": ["订阅与会员说明/自动续费规则"], "facts": ["到期前 24 小时", "原支付渠道扣款"], "grades": {"订阅与会员说明": 2}, "queries": ["自动续费会在什么时候扣款？", "连续包月订阅什么时候自动扣下一次费用？", "自动续费提前多久从支付渠道扣钱？", "会员到期前什么时候会自动续费？"]},
    {"id": "subscription-03", "category": "subscription", "query_type": "factoid", "source": "订阅与会员说明", "sections": ["订阅与会员说明/会员等级"], "facts": ["1000 元升级银卡", "5000 元升级金卡"], "grades": {"订阅与会员说明": 2}, "queries": ["会员等级怎么划分的？", "消费多少能升级银卡或金卡会员？", "银卡和金卡的升级门槛是多少？", "会员等级有哪些，分别要消费多少钱？"]},
    {"id": "subscription-04", "category": "subscription", "query_type": "factoid", "source": "订阅与会员说明", "sections": ["订阅与会员说明/积分有效期"], "facts": ["1 年", "100 积分等于 1 元"], "grades": {"订阅与会员说明": 2}, "queries": ["积分会过期吗？有效期多久？", "账户里的积分能保留多长时间？", "积分什么时候会被清零？", "积分有效期是多久，过期后还能用吗？"]},
    # api
    {"id": "api-01", "category": "api", "query_type": "procedural", "source": "开放平台 API 接入文档", "sections": ["开放平台 API 接入文档/认证方式"], "facts": ["Authorization: Bearer", "7200 秒"], "grades": {"开放平台 API 接入文档": 2}, "queries": ["调用 API 要怎么带认证信息？", "接口请求头里 access_token 怎么传？", "接入开放平台接口，认证头应该怎么写？", "API 调用的 token 有效期是多久？"]},
    {"id": "api-02", "category": "api", "query_type": "factoid", "source": "开放平台 API 接入文档", "sections": ["开放平台 API 接入文档/速率限制"], "facts": ["每分钟最多调用 120 次", "429"], "grades": {"开放平台 API 接入文档": 2}, "queries": ["API 的调用频率限制是多少？", "每分钟最多能请求多少次接口？", "开放平台接口限流是多少 QPS？", "API 调用超过次数限制会怎样？"]},
    {"id": "api-03", "category": "api", "query_type": "procedural", "source": "开放平台 API 接入文档", "sections": ["开放平台 API 接入文档/回调验签"], "facts": ["HMAC-SHA256", "最多 3 次"], "grades": {"开放平台 API 接入文档": 2}, "queries": ["Webhook 回调怎么验证签名？", "收到回调通知后要验签吗？用什么算法？", "回调验签失败该怎么处理？", "Webhook 通知的重试策略是什么？"]},
    {"id": "api-04", "category": "api", "query_type": "procedural", "source": "开放平台 API 接入文档", "sections": ["开放平台 API 接入文档/分页规范"], "facts": ["page 和 page_size", "page_size 最大为 100"], "grades": {"开放平台 API 接入文档": 2}, "queries": ["列表接口怎么分页拉取数据？", "API 返回列表分页参数怎么传？", "接口分页每页最多返回多少条？", "拉取全量列表数据应该用什么方式？"]},
    # error_codes
    {"id": "error-01", "category": "error_codes", "query_type": "troubleshooting", "source": "错误码与故障排查", "sections": ["错误码与故障排查/401 未认证"], "facts": ["401 表示认证失败", "重新获取 token"], "grades": {"错误码与故障排查": 2}, "queries": ["接口返回 401 是什么问题？", "调用接口遇到 401 错误该怎么处理？", "401 状态码代表什么，如何解决？", "请求 API 报 401，应该检查什么？"]},
    {"id": "error-02", "category": "error_codes", "query_type": "troubleshooting", "source": "错误码与故障排查", "sections": ["错误码与故障排查/403 无权限"], "facts": ["403 表示当前应用没有访问该资源的权限", "申请对应权限点"], "grades": {"错误码与故障排查": 2}, "queries": ["接口返回 403 权限不足怎么办？", "调用 API 报 403 错误，如何申请权限？", "403 错误是什么原因？", "遇到 403 状态码应该怎么处理？"]},
    {"id": "error-03", "category": "error_codes", "query_type": "troubleshooting", "source": "错误码与故障排查", "sections": ["错误码与故障排查/500 服务器错误"], "facts": ["服务端内部错误", "记录 request_id"], "grades": {"错误码与故障排查": 2}, "queries": ["接口返回 500 错误怎么办？", "调用 API 出现 500，是服务端问题吗？", "连续报 500 错误应该提供什么信息？", "500 状态码要怎么排查？"]},
    {"id": "error-04", "category": "error_codes", "query_type": "troubleshooting", "source": "错误码与故障排查", "sections": ["错误码与故障排查/429 限流"], "facts": ["Retry-After", "熔断 5 分钟"], "grades": {"错误码与故障排查": 2}, "queries": ["接口返回 429 限流了怎么办？", "遇到 429 错误要如何重试？", "API 触发限流后要等多久再试？", "429 状态码应该怎么退避重试？"]},
    # no_answer
    {"id": "noanswer-01", "category": "no_answer", "query_type": "no_answer", "source": "", "sections": [], "facts": [], "grades": {}, "queries": ["今天北京会下雨吗？", "明天上海的天气怎么样？", "帮我查一下深圳现在的气温", "杭州这周有台风吗？"]},
    {"id": "noanswer-02", "category": "no_answer", "query_type": "no_answer", "source": "", "sections": [], "facts": [], "grades": {}, "queries": ["推荐一部最近好看的科幻电影", "有什么适合周末看的电影？", "最近有什么热门电视剧推荐？", "想看喜剧片，有什么推荐吗？"]},
    {"id": "noanswer-03", "category": "no_answer", "query_type": "no_answer", "source": "", "sections": [], "facts": [], "grades": {}, "queries": ["附近哪家川菜馆最好吃？", "帮我推荐一家本帮菜餐厅", "这个城市有什么特色小吃？", "晚上去哪家烧烤店比较合适？"]},
    {"id": "noanswer-04", "category": "no_answer", "query_type": "no_answer", "source": "", "sections": [], "facts": [], "grades": {}, "queries": ["感冒了吃什么药好得快？", "嗓子疼应该吃什么药？", "失眠有什么非处方药推荐？", "皮肤过敏抹什么药膏合适？"]},
]


def build_dataset() -> tuple[list[dict], list[dict]]:
    source_ids = {
        doc["title"]: source_id_for(doc["title"], doc["content"])
        for doc in CORPUS
    }
    rows: list[dict] = []
    for template in TEMPLATES:
        relevant_source_ids = [source_ids[template["source"]]] if template["source"] else []
        relevance = {
            source_ids[source_title]: grade
            for source_title, grade in template["grades"].items()
        }
        for variant_index, query in enumerate(template["queries"], start=1):
            rows.append({
                "id": f"{template['id']}-{variant_index}",
                "partition": template["id"],
                "query": query,
                "category": template["category"],
                "query_type": template["query_type"],
                "relevant_source_ids": relevant_source_ids,
                "relevant_sections": list(template["sections"]),
                "must_recall_facts": list(template["facts"]),
                "relevance": relevance,
                "no_answer": template["category"] == "no_answer",
                # noanswer-01/02 专用于无答案阈值标定，不计入测试指标，避免在测试集上调阈值。
                "role": "calibration" if template["id"] in {"noanswer-01", "noanswer-02"} else "test",
            })
    return list(CORPUS), rows


def main() -> None:
    corpus, rows = build_dataset()
    (OUT_DIR / "rag_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "rag_dataset.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 自检：每个事实都必须出现在对应源文档中，防止标签与语料脱节。
    corpus_by_title = {doc["title"]: doc["content"] for doc in corpus}
    for row in rows:
        if row["no_answer"]:
            assert row["relevant_source_ids"] == [] and row["must_recall_facts"] == []
            continue
        title = next(title for title, sid in {
            doc["title"]: source_id_for(doc["title"], doc["content"]) for doc in corpus
        }.items() if sid == row["relevant_source_ids"][0])
        for fact in row["must_recall_facts"]:
            assert re.sub(r"\s+", "", fact) in re.sub(r"\s+", "", corpus_by_title[title]), (row["id"], fact)

    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    print(f"corpus docs: {len(corpus)}")
    print(f"queries: {len(rows)}")
    print(f"categories: {json.dumps(category_counts, ensure_ascii=False)}")
    print(f"partitions: {len({row['partition'] for row in rows})}")


if __name__ == "__main__":
    main()
