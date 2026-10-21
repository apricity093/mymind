"""构造阶段 5 的 40 题与独立标准，不生成正式数据。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mymind"))
from agents.customer_policy.protocol import PublicTask
from experiments.agentic_rl.tasks import PolicyClause, PrivateStandard, load_policy_support, validate_task_pairs

DEST = REPO / "data/agentic_rl/calibration"
tasks, standards, sources = [], [], []
dev = {s["task_id"]: s for s in json.loads(
    (REPO / "mymind/experiments/agentic_rl/resources/dev_standards.json").read_text(encoding="utf-8"))}


def expected(section, payload, supports, required=True, decisive=False, alternatives=None):
    return dict(section=section, alternatives=alternatives or [payload],
                supports=[[s] if isinstance(s, str) else s for s in supports],
                required=required, decisive=decisive)


def fact(target, kind, value, supports=("user:1",), required=True, decisive=False):
    return expected("facts", dict(target=target, type=kind, value=value), supports, required, decisive)


def reason(target, code, supports, required=True, decisive=False):
    return expected("reasons", dict(target=target, code=code), supports, required, decisive)


def step(target, code, supports, required=True, params=None):
    payload = dict(target=target, code=code)
    if params is not None:
        payload["params"] = params
    return expected("next_steps", payload, supports, required)


def missing(target, field, purpose="decision", required=True, condition=None):
    payload = dict(target=target, field=field, purpose=purpose)
    if condition:
        payload["condition"] = condition
    return expected("missing_info", payload, ["user:1"], required)


def add(group, role, text, disposition, items, ancestor, variation, severe=None):
    tid = f"C{len(tasks)+1:02d}"
    task = PublicTask(task_id=tid, role=role, user_text=text)
    standard = dict(task_id=tid, group_id=group, split="train", dispositions=[disposition],
                    materials={m.reference_id: m.content for m in task.materials()},
                    expectations=items, severe_items=severe or {})
    # 这些题面只给退款记录，没有提供接口运行状态或实际扣款笔数。
    if group.startswith(("R", "D")):
        unprovided = ["environment", "request_read_only", "payment_affected", "write_affected",
                      "widespread_outage", "data_loss", "data_corruption", "timeout_observed",
                      "token_expired", "auth_valid", "occurrence", "affected_users",
                      "reported_charge_count", "charges_same_order"]
        targets = {a["target"] for e in items if e["section"] == "facts" for a in e["alternatives"]}
        standard["unprovided_user_facts"] = {target: unprovided for target in sorted(targets)}
    elif group == "T01":
        # 401/凭据过期不提供项目、环境、读写性质及配置记录。
        standard["unprovided_user_facts"] = {"Q1": ["environment", "request_read_only",
            "request_scope", "feature_enabled", "outbound_ip_allowed", "user_can_configure"]}
    elif group == "F01":
        standard["unprovided_user_facts"] = {"Q1": ["auth_valid", "token_expired"]}
    elif group.startswith("P"):
        # 403 配置题未提供整体影响、发生频率、恢复或扣款情况；报告者不等于唯一受影响者。
        unprovided = ["payment_affected", "write_affected", "widespread_outage", "data_loss",
                      "data_corruption", "occurrence", "affected_users", "recovered",
                      "reported_charge_count", "charges_same_order"]
        # 查询／预览的只读用途可由接口名判断；导出／生成未说明读写性质。
        if group != "P01":
            unprovided.append("request_read_only")
        standard["unprovided_user_facts"] = {"Q1": unprovided}
    tasks.append(task.model_dump())
    standards.append(standard)
    sources.append(dict(task_id=tid, group_id=group, split="train", origin="stage5-20261012-current-session",
        ancestor=ancestor, source_family=group, variation=variation,
        same_source_excluded_from=["validation", "test"], executor_review="待核查", human_review="待人工核查"))


def refund(target, channel, days, checked=True, stage="channel_processing"):
    limit = 3 if channel == "online_payment" else 5
    clause = "refund_online_limit" if limit == 3 else "refund_bank_limit"
    within = days <= limit
    result = [fact(target, "payment_channel", channel), fact(target, "approval_status", "approved"),
        fact(target, "elapsed_workdays", days), fact(target, "original_account_checked", checked, required=not within),
        fact(target, "received", False, required=not within), fact(target, "refund_stage", stage, required=False),
        fact(target, "timer_start", "approval", ["refund_clock"]),
        fact(target, "limit_workdays", limit, [clause]),
        fact(target, "within_window", within, ["user:1", "refund_clock", clause], decisive=True),
        fact(target, "timing_policy_covered", True, ["user:1", "refund_scope"], required=False),
        reason(target, "within_refund_window" if within else "refund_overdue", ["user:1", "refund_clock", clause])]
    if within:
        result += [step(target, "wait_within_window", ["user:1", clause, "refund_wait"]),
            step(target, "verify_refund_if_overdue_unreceived", ["user:1", clause, "refund_overdue"],
                 params={"limit_workdays": limit}),
            step(target, "check_original_account", ["user:1", "refund_wait"], required=checked is not True)]
    else:
        result += [step(target, "verify_refund", ["user:1", clause, "refund_overdue"]),
            step(target, "check_original_account", ["user:1", "refund_overdue"], required=False)]
        result += [missing(target, field, "handoff", False) for field in
                   ("order_id", "transaction_id", "payment_time", "amount", "currency", "transaction_summary")]
    return result


def untimed(target, channel, reason_required=True):
    clause = "refund_online_limit" if channel == "online_payment" else "refund_bank_limit"
    return [fact(target, "payment_channel", channel), fact(target, "approval_status", "approved"),
        fact(target, "elapsed_workdays", "unknown", decisive=True),
        fact(target, "received", False, required=False), fact(target, "original_account_checked", True, required=False),
        fact(target, "refund_stage", "channel_processing", required=False),
        fact(target, "timer_start", "approval", ["refund_clock"], required=False),
        fact(target, "limit_workdays", 3 if channel == "online_payment" else 5, [clause], required=False),
        fact(target, "timing_policy_covered", True, ["user:1", "refund_scope"], required=False),
        reason(target, "refund_state_unknown", ["user:1", "refund_clock"], required=reason_required),
        missing(target, "elapsed_workdays"),
        step(target, "verify_refund_if_overdue_unreceived", ["user:1", "refund_clock", clause, "refund_overdue"],
             required=False, params={"limit_workdays": 3 if channel == "online_payment" else 5})]


for channel, text in [
    ("bank_card", "退款记录 R1：银行卡渠道。通知只写申请已受理，未看到审核结果，原账户没有收到钱。我能从受理那天算到账时间吗？"),
    ("online_payment", "我为 R1 申请的是线上支付渠道退款，受理通知已经保存。最新审核状态还不知道，也不知道通过后过了多少工作日，目前未到账。请告诉我需要补哪些状态才能判断。")]:
    items = deepcopy(dev["S01"]["expectations"])
    for e in items:
        for a in e["alternatives"]:
            if a.get("type") == "payment_channel": a["value"] = channel
            if a.get("type") == "limit_workdays": a["value"] = 3 if channel == "online_payment" else 5
        e["supports"] = [["refund_online_limit"]] if any(a.get("type") == "limit_workdays" for a in e["alternatives"]) and channel == "online_payment" else e["supports"]
    items.append(fact("R1", "elapsed_workdays", "unknown", required=False))
    items += [fact("R1", "original_account_checked", "unknown", required=False),
        step("R1", "verify_refund_if_overdue_unreceived",
             ["user:1", "refund_clock", "refund_bank_limit" if channel == "bank_card" else "refund_online_limit", "refund_overdue"],
             required=False, params={"limit_workdays": 5 if channel == "bank_card" else 3})]
    add("R01", "billing", text, "request_info", items, "S01",
        "询问计时起点并补充原账户未到账描述；受理与审核仍分开" if channel == "bank_card" else
        "渠道和询问重点变化；受理与审核仍分开")

for group, channel, cases in [
    ("R02", "online_payment", [(2, False, "R1 线上支付渠道退款，审核已通过 2 个完整工作日，现显示渠道处理中；还没到账，原支付账户还未逐项核对。现在该怎么办？"),
                               (3, True, "退款 R1 的渠道标识是线上支付渠道。审核通过后恰好满 3 个完整工作日，进入渠道处理中后没有另算时间，我核对原账户仍未到账。请说明时效和下一步。")]),
    ("R03", "bank_card", [(4, True, "R1 是银行卡渠道退款，虽然付款在线完成，渠道标识仍为银行卡。审核通过起已过 4 个完整工作日，渠道处理中，已核对原账户未到账。是否超过时限？"),
                          (5, False, "银行卡渠道的 R1 已审核通过，完整工作日数现在为 5，渠道处理中且还没到账。我尚未核对原账户，请给下一步建议。")]),
    ("R04", "online_payment", [(4, True, "R1：线上支付渠道退款，审核通过后 4 个完整工作日，仍显示渠道处理中。已经核对原账户，未到账，订单和交易号暂时找不到。请建议处理。"),
                               (7, True, "关于退款记录 R1，我反复核对过原支付账户没有退款入账。渠道标识线上支付渠道，审核已通过 7 个完整工作日，状态渠道处理中。需要继续等吗？")]),
    ("R05", "bank_card", [(6, True, "银行卡渠道退款 R1 已审核通过，过了 6 个完整工作日，渠道处理中；核对原银行卡账户仍未到账，没有带订单号和交易号。该怎么处理？"),
                          (9, True, "R1 的退款渠道是银行卡。原账户查过仍未到账，审核通过至今 9 个完整工作日，状态渠道处理中。请说明是否需核实及要准备什么。")])]:
    for days, checked, text in cases:
        add(group, "billing", text, "answer" if group in {"R02", "R03"} else "escalate",
            refund("R1", channel, days, checked), {"R02":"S02", "R03":"S04", "R04":"S03", "R05":"R04-rule-derivation"}[group],
            "边界／同区间天数、账户核对和材料表达变化")

for days, text in [(4, "退款记录 R1 是平台余额渠道，审核已通过 4 个完整工作日，核对原余额账户后未到账。我只想确认这个渠道的常规时效。"),
                   (8, "R1 已审核通过 8 个完整工作日，钱仍没有回到核对过的原平台余额账户。支付渠道标识为平台余额渠道。现有时效说明是否适用，下一步是什么？")]:
    items = deepcopy(dev["S13"]["expectations"])
    for e in items:
        for a in e["alternatives"]:
            if a.get("type") == "elapsed_workdays": a["value"] = days
    items.append(fact("R1", "timer_start", "approval", ["refund_clock"], required=False))
    add("R06", "billing", text, "escalate", items, "S13", "经过日数与范围咨询表达变化，均不填时限",
        severe=deepcopy(dev["S13"]["severe_items"]))

double_cases = [
    ("D01", "answer", [("R1","online_payment",2),("R2","bank_card",4)], "请分别说明两笔退款和整体建议：R1 线上支付渠道，审核通过后 2 个完整工作日；R2 银行卡渠道，审核通过后 4 个完整工作日。两笔都显示渠道处理中，分别核对原账户都未到账。"),
    ("D01", "answer", [("R2","bank_card",5),("R1","online_payment",3)], "先列银行卡退款 R2：审核通过后恰好 5 个完整工作日。再列线上支付退款 R1：审核通过后恰好 3 个完整工作日。两笔均渠道处理中，原账户均核对未到账。需要怎样分别处理？"),
    ("D02", "escalate", [("R1","online_payment",4),("R2","bank_card",2)], "退款 R1 是线上支付渠道，通过审核后 4 个完整工作日；R2 是银行卡渠道，通过后 2 个完整工作日。两笔渠道处理中，已各自核对原账户未到账，请分别判断并给整体处理。"),
    ("D02", "escalate", [("R2","bank_card",5),("R1","online_payment",6)], "同一次咨询中 R2 银行卡退款通过审核已满 5 个完整工作日，R1 线上支付退款已满 6 个完整工作日。两笔渠道处理中，核对各自原账户都未到账。请不要混用两笔的时间，分别说明下一步。"),
    ("D05", "request_info", [("R1","online_payment",3),("R2","bank_card",None)], "R1 是线上支付渠道退款，审核通过后 3 个完整工作日。R2 为银行卡渠道，也明确审核通过，但我不记得通过后有多少完整工作日。两笔渠道处理中、原账户均核对未到账。请分别判断，需要我补什么？"),
    ("D05", "request_info", [("R2","bank_card",4),("R1","online_payment",None)], "记录先列 R2：银行卡退款，审核通过后 4 个完整工作日；再列 R1：线上支付退款，审核已通过，但经过的完整工作日不清楚。两笔渠道处理中，各自原账户核对未到账。整体如何处理？"),
    ("D06", "escalate", [("R1","online_payment",5),("R2","bank_card",None)], "R1 线上支付退款审核通过后 5 个完整工作日，R2 银行卡退款审核已通过但不清楚经过多少完整工作日。两笔均渠道处理中，已核对原账户未到账。请分别说明并给整体建议，订单和交易号没带。"),
    ("D06", "escalate", [("R2","bank_card",7),("R1","online_payment",None)], "R2 银行卡渠道退款，通过审核后 7 个完整工作日；R1 线上支付渠道退款已通过审核，但天数记不清了。两笔渠道处理中，分别核对原账户未到账。能先确定整体下一步吗？")]
for group, disposition, records, text in double_cases:
    items = []
    for target, channel, days in records:
        items += untimed(target, channel, reason_required=disposition == "request_info") if days is None else refund(target, channel, days)
    if disposition == "escalate":
        for item in items:
            if item["section"] == "reasons" and item["alternatives"][0]["code"] == "within_refund_window":
                # 另一笔的有据比较、等待和条件性后续仍必需；它不构成整体升级理由。
                item["required"] = False
    if disposition == "escalate":
        for item in items:
            if item["section"] == "reasons" and item["alternatives"][0]["code"] == "within_refund_window":
                # 另一笔的有据比较、等待和条件性后续仍必需；它不构成整体升级理由。
                item["required"] = False
    code = {"answer":"within_refund_window", "request_info":"refund_state_unknown", "escalate":"refund_overdue"}[disposition]
    case_supports = ["user:1", "refund_multi", "refund_clock"]
    for _, channel, days in records:
        limit = 3 if channel == "online_payment" else 5
        if days is not None and (disposition != "escalate" or days > limit):
            case_supports.append("refund_online_limit" if limit == 3 else "refund_bank_limit")
    items.append(reason("case", code, case_supports, required=False))
    add(group, "billing", text, disposition, items, "D02" if group == "D02" else "plan-4.3/7.2",
        "双笔順序、边界日数及已知／未知项的渠道归属变化")

for full, text in [(False, "订单 A102 看起来扣了两次钱，两笔都是 99 元，第二笔交易号找不到，也没有带两笔时间、渠道及交易记录。请帮我核实处理。"),
                   (True, "订单 A102 疑似两笔扣款，都是 49 元人民币。记录一：10:01、线上支付渠道、交易号 TX101；记录二：10:03、线上支付渠道、交易号 TX102；两条扣款记录截图都在。请人工核实是否重复并处理，我还不能确认。")]:
    items = [fact("A102", "reported_charge_count", 2), fact("A102", "charges_same_order", True),
        fact("A102", "payment_channel", "online_payment" if full else "unknown", required=False),
        reason("A102", "suspected_duplicate_charge", ["user:1", ["role:1", "billing_handoff"]], decisive=True),
        step("A102", "verify_duplicate_charge", ["user:1", ["role:1", "billing_handoff"]])]
    if not full:
        items += [missing("A102", k, "handoff", k != "transaction_id") for k in
                  ["payment_time", "payment_channel", "transaction_summary", "transaction_id"]]
    add("B01", "billing", text, "escalate", items, "S05", "定位材料缺失与齐全两种状态")

for group in ["P01", "P02", "P03", "P04", "P05", "P06"]:
    for variant in range(2):
        project, env, ip = ("Orchid", "production", "203.0.113.21") if variant == 0 else ("Cedar", "test", "198.51.100.42")
        name = (["基础查询", "报表预览"] if group == "P01" else ["报表导出", "批量报表生成"])[variant]
        env_text = "生产" if env == "production" else "测试"
        current = {"P01":None,"P02":True,"P03":False,"P04":"unknown","P05":"unknown","P06":"unknown"}[group]
        allowed = "unknown" if group == "P04" else (True if group in {"P03","P05","P06"} and variant else False)
        status_text = {True:"包含",False:"不包含","unknown":"是否包含还不清楚"}[allowed]
        text = f"请求 Q1：{project} 项目{env_text}环境的{name}接口返回 403，认证有效。我有 {project} 项目配置权，当前出口地址 {ip}。目标项目白名单{status_text}该地址。"
        other = None
        if group in {"P02","P03"}:
            text += f"配置页明确显示 {project} 项目{env_text}环境的对应导出功能{'已开通' if current else '未开通'}。"
        elif group == "P04":
            text += "当前功能是否开通不知道，没有其他项目或环境的开通记录。"
        elif group == "P05":
            other = dict(project=project, environment="test" if env == "production" else "production", status=True)
            text += f"我仅有 {project} 项目{'测试' if env == 'production' else '生产'}环境的已开通记录，没有当前环境记录。"
        elif group == "P06":
            other = dict(project="Birch", environment=env, status=True)
            text += f"我仅有 Birch 项目{env_text}环境的已开通记录，没有目标项目的开通记录。"
        text += "请判断我能如何处理，以及还需确认什么。" if variant == 0 else "我该自行调整还是请平台处理？"
        cap = "basic_query" if group == "P01" else "export"
        items = [fact("Q1","request_scope",dict(project=project,environment=env)), fact("Q1","outbound_ip",ip),
            fact("Q1","auth_valid",True), fact("Q1","user_can_configure",dict(project=project,status=True)),
            fact("Q1","http_status",403,required=False), fact("Q1","environment",env,required=False),
            expected("facts",None,["user:1"],alternatives=[dict(target="Q1",type="interface_name",value=n) for n in [name,name+"接口"]]),
            fact("Q1","required_capability",cap,["user:1","capabilities"]),
            fact("Q1","whitelist_required",True,["capabilities"]),
            fact("Q1","platform_enablement_required",group != "P01",["capabilities"]),
            fact("Q1","whitelist_actor","project_user",["config_actors"],required=False),
            fact("Q1","outbound_ip_allowed",dict(project=project,outbound_ip=ip,status=allowed))]
        if current is not None:
            items += [fact("Q1","feature_enabled",dict(project=project,environment=env,status=current),
                ["user:1","enablement_scope","unmatched_scope"] if other else ["user:1"], decisive=bool(other)),
                fact("Q1","enablement_actor","platform",["config_actors"],required=False)]
        if other: items += [fact("Q1","feature_enabled",other,required=False)]
        if group in {"P01","P02"}:
            disposition, code = "answer","user_config_fixable"
        elif group == "P03":
            disposition, code = "escalate","platform_enablement_required"
        else:
            disposition, code = "request_info","permission_state_unknown"
        capability_support = ["capabilities", "permission_unknown"] if disposition == "request_info" else (
            ["capabilities", "config_fix_allowed"] if group == "P02" else
            ["capabilities", "config_enablement_handoff"] if group == "P03" else "capabilities")
        items += [reason("Q1",code,["user:1",capability_support,"unmatched_scope"] if other else
            ["user:1",capability_support,"config_actors"] if disposition != "request_info" else ["user:1",capability_support], decisive=True)]
        if allowed is False:
            items += [step("Q1","adjust_project_whitelist",["user:1","config_actors"],
                required=disposition == "answer",params=dict(project=project,outbound_ip=ip))]
        if group == "P03": items += [step("Q1","request_platform_enablement",["user:1","config_actors"],params=dict(project=project,environment=env))]
        if current == "unknown": items += [missing("Q1","feature_enabled")]
        if allowed == "unknown": items += [missing("Q1","outbound_ip_allowed")]
        add(group,"technical",text,disposition,items,{"P01":"S06","P02":"S07","P03":"S08","P04":"S09","P05":"P05","P06":"plan-4.5/7.3"}[group],
            "同能力接口、当前环境、白名单状态及配置记录关系变化；不增加额外记录")

for text in ["请求 Q1 的接口返回 401，响应摘要是 Token 已过期，我还没重新授权。应该先做什么？",
             "我调用接口 Q1 时收到 401，响应明确写着认证 Token 过期。目前还未换凭据或重新授权，请给安全的验证步骤。"]:
    items = deepcopy(dev["S10"]["expectations"])
    if "调用接口 Q1" in text:
        items.append(fact("Q1","interface_name","Q1",required=False))
    add("T01","technical",text,"answer",items,"S10","错误报告与操作咨询表达变化")

for group, texts, ancestor in [
    ("F01", ["请求 Q1：测试环境只读查询刚发生一次超时，后来已恢复，其他请求正常，没有支付或数据写入受影响。我没带发生时间、接口、request_id 和响应摘要，想知道应记录什么及如何初步排查。",
             "Q1 是测试环境的一次只读查询超时，后续已经恢复，其他请求正常，无支付或写入影响。时间、接口、request_id、响应摘要暂时没整理。我最近改过客户端配置，该记录哪些信息并如何低风险排查？"],"S11"),
    ("F02", ["请求 Q1：生产支付接口持续返回 500，多个用户无法完成支付，目前还没拿到 request_id、发生时间、具体接口和响应摘要，请给处理建议。",
             "Q1 的现场描述：多个用户的生产支付流程受影响，支付接口连续返回 500，仍未恢复。request_id、具体接口、时间和响应摘要暂未收齐。是否应等日志齐全再处理？"],"S12")]:
    for text in texts:
        items = deepcopy(dev[ancestor]["expectations"])
        if group == "F01":
            items += [fact("Q1","widespread_outage",False,required=False),
                      fact("Q1","affected_users","unknown",required=False)]
            items += [missing("Q1",field,"handoff",False) for field in
                      ("event_time","interface_name","request_id","response_summary")]
        else:
            items += [fact("Q1","recovered",False,required=False),
                      fact("Q1","widespread_outage","unknown",required=False),
                      missing("Q1","response_summary","handoff",False),
                      fact("Q1","interface_name","支付接口",required=False)]
            diagnostic = deepcopy(next(e for e in dev["S11"]["expectations"] if
                e["section"] == "next_steps" and e["alternatives"][0]["code"] == "record_diagnostics"))
            diagnostic.update(required=False, supports=[["fault_handling"]])
            diagnostic["alternatives"].append(dict(target="Q1",code="record_diagnostics",params={"fields":
                ["request_id","interface_name","event_time","response_summary","occurrence","affected_users"]}))
            items.append(diagnostic)
        add(group,"technical",text,"answer" if group == "F01" else "escalate",items,ancestor,
            "定位材料未齐但一般指导／已成立升级条件，叙述与配置背景变化")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source_path = DEST / "sources.json"
    previous_sources = {s["task_id"]: s for s in json.loads(source_path.read_text(encoding="utf-8"))} if source_path.exists() else {}
    for source in sources:
        previous = previous_sources.get(source["task_id"], {})
        for key in ("executor_review", "human_review", "human_conclusion", "ai_review"):
            if key in previous:
                source[key] = previous[key]
    task_map = {t["task_id"]: PublicTask.model_validate(t) for t in tasks}
    standard_map = {s["task_id"]: PrivateStandard.model_validate(s) for s in standards}
    clauses = load_policy_support(REPO / "mymind/experiments/agentic_rl/resources/policy_support.json")
    for key, text in [
        ("config_fix_allowed", "当前开通满足而只缺白名单时，可指导自助修正并验证；"),
        ("config_enablement_handoff", "当前范围明确未开通时，应建议人工核实开通，白名单建议可一并说明，但不能承诺只改白名单就能使用导出。")]:
        clauses[key] = PolicyClause(clause_id=key,source_id="technical-policy",text=text)
    validate_task_pairs(task_map, standard_map, clauses)
    assert len(tasks) == 40 and len({s["group_id"] for s in standards}) == 20
    for name, value in [("public_tasks.json",tasks),("private_standards.json",standards),("sources.json",sources)]:
        (DEST / name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    (DEST / "policy_support.json").write_text(json.dumps([c.model_dump() for c in clauses.values()],ensure_ascii=False,indent=2),encoding="utf-8")
    generation_path = DEST / "generation_record.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.exists() else dict(date="2026-10-12",scope="stage5-only",
        source="合成客服政策与固定训练侧场景组合",
        method="逐组固定条件后构造两题；标准来自既定规则，不由教师结果反推",
        counts=dict(tasks=40,groups=20,billing=22,technical=18),
        instruction="40 题，仅开发／训练组合；公开和私有分开；非标识替换副本；不新增业务规则",
        human_review="待人工核查")
    generation["human_review"] = (f"{sum(s['human_review'] == '已确认审阅结论' for s in sources)} 题人工确认；"
                                  f"{sum(bool(s.get('ai_review')) for s in sources)} 题已记录用户授权的 AI 代审；整体验收待独立技术审核，通过后按用户预先条件授权办理冻结")
    generation_path.write_text(json.dumps(generation,ensure_ascii=False,indent=2),encoding="utf-8")
    print("prepared 40 calibration tasks / 20 training groups")


if __name__ == "__main__":
    main()
