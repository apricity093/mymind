"""只呈现策略显式提交的事实、条件和建议。"""

from __future__ import annotations

from agents.customer_policy.protocol import parse_result
from agents.customer_policy.vocabulary import REASONS, FACT_LABELS, STEPS


DISPOSITIONS = {"answer": "回答与指导", "request_info": "请求补充信息",
                "escalate": "建议人工核实"}
VALUES = {
    "unknown": "未知", "online_payment": "线上支付渠道", "bank_card": "银行卡渠道",
    "balance": "平台余额渠道", "accepted": "申请受理", "approved": "审核通过",
    "channel_processing": "渠道处理中", "approval": "审核通过", "test": "测试",
    "production": "生产", "basic_query": "基础查询", "export": "导出",
    "platform": "平台", "project_user": "项目用户", "single": "一次",
    "repeated": "重复", "continuous": "持续", "one": "一人", "multiple": "多人",
    "widespread": "大面积",
}


def _value(value):
    if type(value) is bool:
        return "是" if value else "否"
    if isinstance(value, str):
        return VALUES.get(value, value)
    if isinstance(value, dict):
        labels = {"project": "项目", "environment": "环境", "outbound_ip": "出口地址", "status": "状态"}
        return "，".join(f"{labels[key]}：{_value(item)}" for key, item in value.items())
    return str(value)


def _step(step):
    params = step.params or {}
    if step.code == "verify_refund_if_overdue_unreceived":
        return f"审核通过后超过 {params['limit_workdays']} 个完整工作日且仍未到账时，请人工核实退款流水"
    if step.code == "adjust_project_whitelist":
        return f"由用户调整项目 {params['project']} 的白名单，核对出口地址 {params['outbound_ip']} 后验证"
    if step.code == "request_platform_enablement":
        return f"请平台核实项目 {params['project']} 在{_value(params['environment'])}环境的功能开通"
    if step.code == "record_diagnostics":
        return "记录定位材料：" + "、".join(FACT_LABELS.get(f, f) for f in params["fields"])
    if step.code == "low_risk_checks":
        labels = {"network": "网络", "recent_config_changes": "近期配置变更"}
        return "低风险排查：" + "、".join(labels[d] for d in params["directions"])
    return STEPS[step.code]


def render_result(raw_result) -> str:
    result = parse_result(raw_result)
    lines = [f"处理建议：{DISPOSITIONS[result.disposition]}"]

    def add(item, text):
        refs = "、".join(item.evidence)
        lines.append(f"- {item.target}：{text}" + (f"（依据：{refs}）" if refs else ""))

    for reason in result.reasons:
        add(reason, REASONS[reason.code])
    for fact in result.facts:
        add(fact, f"{FACT_LABELS[fact.type]}：{_value(fact.value)}")
    for step in result.next_steps:
        add(step, _step(step))
    for missing in result.missing_info:
        condition = "若该笔已审核通过，" if missing.condition == "if_approved" else ""
        purpose = "用于判断" if missing.purpose == "decision" else "用于人工定位"
        add(missing, f"{condition}请补充{FACT_LABELS[missing.field]}（{purpose}）")
    return "\n".join(lines)
