"""公开结果字段和处理代码的共同语义。"""

REASONS = {
    "refund_state_unknown": "退款判断所需状态尚不完整",
    "within_refund_window": "仍在常规退款时效内", "refund_overdue": "超过常规退款时效",
    "policy_uncovered": "已见政策未覆盖该渠道时效",
    "suspected_duplicate_charge": "用户报告疑似重复扣款，需人工核实",
    "user_config_fixable": "可由用户修正配置后验证",
    "platform_enablement_required": "需要平台核实功能开通",
    "permission_state_unknown": "权限判断所需状态尚不完整",
    "token_expired": "用户提供的响应提示凭据过期",
    "limited_recovered_fault": "用户所述影响有限且已恢复",
    "severe_business_impact": "用户报告的业务影响需人工或二线处理",
    "evidence_insufficient": "当前未取得充分适用依据",
    "retrieval_unavailable": "检索服务不可用",
}

# reason 的确定断言与 facts 使用相同的事实核查；目录和公开题面保持不变。
REASON_FACT_CLAIMS = {
    "within_refund_window": (("within_window", True),),
    "refund_overdue": (("within_window", False),),
    "policy_uncovered": (("timing_policy_covered", False),),
    "token_expired": (("token_expired", True), ("auth_valid", False)),
    "limited_recovered_fault": (("recovered", True),),
    "platform_enablement_required": (("feature_enabled", {"status": False}),),
}

FACT_LABELS = {
    "payment_channel": "支付渠道", "refund_stage": "退款阶段",
    "approval_status": "审核状态", "timer_start": "计时起点",
    "elapsed_workdays": "审核通过后完整工作日数", "limit_workdays": "常规时效上限（工作日）",
    "within_window": "是否在常规时效内", "original_account_checked": "是否核对原账户",
    "received": "是否到账", "timing_policy_covered": "时效政策是否覆盖",
    "reported_charge_count": "用户报告扣款笔数", "charges_same_order": "是否同一订单",
    "interface_name": "接口名称", "required_capability": "所需能力",
    "request_scope": "请求范围", "outbound_ip": "请求出口地址", "auth_valid": "认证是否有效",
    "user_can_configure": "用户配置权", "feature_enabled": "功能开通记录",
    "outbound_ip_allowed": "白名单状态", "platform_enablement_required": "是否需平台开通",
    "whitelist_required": "是否需访问白名单", "enablement_actor": "开通处理主体",
    "whitelist_actor": "白名单处理主体", "http_status": "响应码",
    "timeout_observed": "是否超时", "token_expired": "凭据是否过期", "environment": "运行环境",
    "request_read_only": "是否只读请求", "recovered": "是否恢复", "occurrence": "发生频率",
    "affected_users": "影响用户范围", "widespread_outage": "是否大面积不可用",
    "payment_affected": "支付是否受影响", "write_affected": "写入是否受影响",
    "data_loss": "是否报告数据丢失", "data_corruption": "是否报告数据错乱",
    "order_id": "订单号", "transaction_id": "交易号", "payment_time": "交易时间",
    "amount": "金额", "currency": "币种", "transaction_summary": "交易记录摘要",
    "request_id": "request_id", "event_time": "发生时间", "response_summary": "响应摘要",
}

STEPS = {
    "wait_within_window": "在常规范围内条件性等待，不能保证到账",
    "check_original_account": "核对原支付账户",
    "verify_refund": "请人工核实退款流水", "verify_channel_timing": "请人工核实该渠道适用时效",
    "verify_duplicate_charge": "请人工或财务核实疑似重复扣款",
    "reauthorize_then_verify": "重新授权或更新有效凭据后验证请求结果",
    "escalate_if_severe_impact": "若出现严重业务影响，请人工或二线处理",
    "escalate_technical": "请人工或二线介入核实", "seek_human_review": "请人工核实处理",
}
