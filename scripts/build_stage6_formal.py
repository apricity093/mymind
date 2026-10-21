"""按既定35组构造正式实例、独立规则标准与来源记录。"""
from collections import Counter
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mymind"))
from agents.customer_policy.protocol import PublicTask
from experiments.agentic_rl.tasks import PrivateStandard, load_policy_support, validate_task_pairs

DATA = REPO / "data/agentic_rl/formal"
RUN = REPO / "agentic_runs/stage6-20261013"
GROUPS = {
    "R01": ("train", 10), "R02": ("train", 10), "R03": ("train", 10),
    "R04": ("train", 10), "R05": ("train", 10), "R06": ("train", 10),
    "R07": ("validation", 10), "R08": ("test", 15),
    "D01": ("train", 25), "D02": ("train", 25), "D03": ("test", 15),
    "D04": ("validation", 15), "D05": ("train", 25), "D06": ("train", 25),
    "D07": ("validation", 15), "D08": ("test", 15), "D09": ("test", 15),
    **{f"P{i:02d}": ("train", 20) for i in range(1, 7)},
    **{f"P{i:02d}": ("validation", 10) for i in range(7, 10)},
    **{f"P{i:02d}": ("test", 15) for i in range(10, 13)},
    "B01": ("train", 40), "T01": ("train", 20), "F01": ("train", 30),
    "F02": ("train", 30), "F03": ("validation", 10), "F04": ("test", 15),
}
CHANNEL = {"online_payment": "线上支付渠道", "bank_card": "银行卡渠道", "balance": "平台余额渠道", "unknown": "渠道未确认"}
ENV = {"production": "生产", "test": "测试"}
DEV = {s["task_id"]: s for s in json.loads((REPO / "mymind/experiments/agentic_rl/resources/dev_standards.json").read_text(encoding="utf-8"))}
CLAUSES = load_policy_support(REPO / "data/agentic_rl/calibration/policy_support.json")


def expected(section, payload, supports, required=True, decisive=False, alternatives=None):
    return dict(section=section, alternatives=alternatives or [payload],
        supports=[[s] if isinstance(s, str) else s for s in supports], required=required, decisive=decisive)


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


def refund_items(record, overall):
    target, channel, state = record["target"], record["channel"], record["state"]
    days, checked = record["days"], record["checked"]
    limit = 3 if channel == "online_payment" else 5
    clause = "refund_online_limit" if limit == 3 else "refund_bank_limit"
    result = [fact(target, "payment_channel", channel), fact(target, "received", False, required=False),
        fact(target, "original_account_checked", checked, required=False)]
    if state == "unapproved":
        result += [fact(target, "refund_stage", "accepted"), fact(target, "approval_status", "unknown", decisive=True),
            fact(target, "elapsed_workdays", "unknown", required=False),
            fact(target, "timer_start", "approval", ["refund_clock"], required=False),
            reason(target, "refund_state_unknown", ["user:1", "refund_accepted", "refund_clock"]),
            missing(target, "approval_status"), missing(target, "elapsed_workdays", condition="if_approved")]
    else:
        result += [fact(target, "approval_status", "approved"), fact(target, "refund_stage", record["stage"], required=False),
            fact(target, "elapsed_workdays", days if days is not None else "unknown", decisive=state == "untimed")]
        if state == "uncovered":
            return result + [fact(target, "timing_policy_covered", False, ["user:1", "refund_scope"], decisive=True),
                fact(target, "timer_start", "approval", ["refund_clock"], required=False),
                reason(target, "policy_uncovered", ["user:1", "refund_scope"]),
                step(target, "verify_channel_timing", ["user:1", "refund_scope", "channel_handoff"])]
        if state == "unchanneled":
            return result + [reason(target, "refund_state_unknown", ["user:1", "refund_scope"], decisive=True),
                missing(target, "payment_channel"), fact(target, "timer_start", "approval", ["refund_clock"], required=False)]
        if state == "untimed":
            result += [fact(target, "timer_start", "approval", ["refund_clock"], required=False),
                       reason(target, "refund_state_unknown", ["user:1", "refund_clock"], required=overall != "escalate"),
                       missing(target, "elapsed_workdays")]
        else:
            within = days <= limit
            result += [fact(target, "timer_start", "approval", ["refund_clock"]),
                fact(target, "limit_workdays", limit, [clause]),
                fact(target, "timing_policy_covered", True, ["user:1", "refund_scope"], required=False),
                fact(target, "within_window", within, ["user:1", "refund_clock", clause], decisive=True),
                reason(target, "within_refund_window" if within else "refund_overdue", ["user:1", "refund_clock", clause],
                       required=not within or overall != "escalate")]
            if within:
                result += [step(target, "wait_within_window", ["user:1", clause, "refund_wait"]),
                    step(target, "check_original_account", ["user:1", "refund_wait"], required=not checked)]
            else:
                for e in result:
                    if e["alternatives"][0].get("type") in {"received", "original_account_checked"}:
                        e["required"] = True
                result += [step(target, "verify_refund", ["user:1", clause, "refund_overdue"])]
                result += [step(target, "check_original_account", ["user:1", "refund_overdue"], required=False)]
                result += [missing(target, field, "handoff", False) for field in
                    ("order_id", "transaction_id", "payment_time", "amount", "currency", "transaction_summary")
                    if field not in record["provided_location"]]
    result += [fact(target, "timing_policy_covered", True, ["user:1", "refund_scope"], required=False),
        fact(target, "limit_workdays", limit, [clause], required=False)] if state in {"unapproved", "untimed"} else []
    if state in {"within", "untimed", "unapproved"}:
        result.append(step(target, "verify_refund_if_overdue_unreceived",
            ["user:1", "refund_clock", clause, "refund_overdue"], required=state == "within", params={"limit_workdays": limit}))
    if state in {"untimed", "unapproved"}:
        result.append(step(target, "check_original_account", ["user:1", "refund_wait"], required=False))
    return result


def refund_body(record, i):
    target, channel, state = record["target"], record["channel"], record["state"]
    checked = "我已核对原支付账户" if record["checked"] else "原支付账户尚未核对"
    if state == "unapproved":
        status = ["通知只有申请已受理，最新审核结果我没有看到", "页面显示受理完成；审核有没有通过仍不清楚",
                  "已收到受理通知，但我还不知道审核状态", "申请受理后没有查到最新审核结果"][i % 4]
        timing = f"受理后我自己数了{2+i%7}个工作日；审核通过后的完整工作日数无法提供"
    elif state == "untimed":
        status = "审核已通过，目前渠道处理中"
        timing = ["没有记录审核通过后的完整工作日数", "从审核通过起经过多少个完整工作日，我还没核对",
                  "通过时间的消息找不到，暂不能给出此后经过的完整工作日数"][i % 3]
    else:
        status = "审核已通过" + ("，目前渠道处理中" if record["stage"] == "channel_processing" else "，页面仍显示审核通过")
        timing = f"从审核通过算，已过{record['days']}个完整工作日"
    channel_text = "我没有确认这笔的支付渠道标识" if channel == "unknown" else f"渠道标识是{CHANNEL[channel]}"
    details = [f"{target}：{channel_text}。{status}，{timing}；{checked}，目前未到账。",
        f"关于{target}，钱还没到账，{checked}。{timing}；{status}。{channel_text}。",
        f"我整理的退款记录{target}写着：{status}；{channel_text}；{timing}；{checked}，未到账。",
        f"{target}仍未到账。先说明计时：{timing}；状态：{status}。另外{channel_text}，{checked}。",
        f"退款{target}的信息分开记录：{channel_text}，{status}。我按审核状态整理的时间是：{timing}。{checked}，未到账。"]
    body = details[i % len(details)]
    if channel == "bank_card" and i % 4 == 1:
        body += "这笔付款虽在线完成，支付记录的渠道标识仍是银行卡渠道。"
    provided = record["provided_location"]
    if "order_id" in provided:
        body += f"对应订单编号ORD-{target}。"
    if "transaction_id" in provided:
        body += f"交易记录编号TX-{target}。"
    return body


def build_refund(group, i):
    if group.startswith("R"):
        channel = {"R02": "online_payment", "R03": "bank_card", "R04": "online_payment", "R05": "bank_card",
                   "R06": "balance", "R08": "unknown"}.get(group, "online_payment" if i % 2 else "bank_card")
        state = {"R01": "unapproved", "R02": "within", "R03": "within", "R04": "overdue", "R05": "overdue",
                 "R06": "uncovered", "R07": "untimed", "R08": "unchanneled"}[group]
        states = [(channel, state)]
    else:
        states = {"D01": [("online_payment", "within"), ("bank_card", "within")],
            "D02": [("online_payment", "overdue"), ("bank_card", "within")],
            "D03": [("online_payment", "within"), ("bank_card", "overdue")],
            "D04": [("online_payment", "overdue"), ("bank_card", "overdue")]} .get(group)
        if states is None:
            first = "within" if group in {"D05", "D07"} else "overdue" if group in {"D06", "D08"} else "unapproved"
            second = "unapproved" if group in {"D07", "D08"} else "untimed"
            channels = ("online_payment", "bank_card") if i % 2 else ("bank_card", "online_payment")
            states = list(zip(channels, (first, second)))
    disposition = "escalate" if any(s in {"overdue", "uncovered"} for _, s in states) else (
        "request_info" if any(s in {"unapproved", "untimed", "unchanneled"} for _, s in states) else "answer")
    records, items, severe = [], [], {}
    for j, (channel, state) in enumerate(states):
        limit = 3 if channel == "online_payment" else 5
        days = None if state in {"unapproved", "untimed"} else 4 if state == "unchanneled" else (
            (i + j) % (limit + 1) if state == "within" else limit + 1 + (i * 2 + j) % 8)
        record = dict(target=f"RF-{list(GROUPS).index(group)*50+i:04d}-{j+1}", channel=channel, state=state, days=days,
            checked=True if state == "overdue" else (i+j) % 3 != 0,
            stage="channel_processing" if (i+j) % 3 else "approved",
            provided_location=(["order_id", "transaction_id"] if i % 4 == 0 else ["order_id"] if i % 4 == 1 else []))
        if state == "uncovered":
            record["checked"] = True
        if state == "untimed":
            record["stage"] = "channel_processing"
        if state == "unapproved":
            record["stage"] = "accepted"
        records.append(record)
        items.extend(refund_items(record, disposition))
        if state == "uncovered":
            severe = deepcopy(DEV["S13"]["severe_items"])
            for entries in severe.values():
                for entry in entries:
                    entry["target"] = record["target"]
    ordered = list(reversed(records)) if i % 2 else records
    lead = ["我想咨询退款到账，请根据这些记录说明下一步。", "下面是我能提供的退款信息，想知道现在如何处理。",
            "我把现有退款材料列在下面，请说明时效与后续。", "资金还没到账，我需要分别理解这些记录。",
            "请帮我看退款记录中还需要确认哪些信息，以及下一步。"][i % 5]
    body = lead + "\n" + "\n".join(refund_body(r, i+j) for j, r in enumerate(ordered))
    if len(records) == 2:
        body += "\n这些是两笔独立退款，请分别说明后再给整体建议。"
        supports = ["user:1", "refund_multi", "refund_clock"]
        supports += sorted({"refund_online_limit" if r["channel"] == "online_payment" else "refund_bank_limit"
            for r in records if r["state"] in {"within", "overdue"}})
        code = "refund_overdue" if disposition == "escalate" else "refund_state_unknown" if disposition == "request_info" else "within_refund_window"
        items.append(reason("case", code, supports, required=False))
    body += "\n" + ["请解释现在能判断哪些内容。", "我希望先知道还缺哪些决定性信息。", "请按这些已有材料给处理建议。",
        "我想避免把受理时间和审核时间混在一起。", "请说明基于这些记录的下一步。", "我还没有联系人工，想先了解如何处理。",
        "我提供的是页面和账户记录中的情况，请据此说明。"][i % 7]
    return body, disposition, items, severe, {"refunds": records, "order": [r["target"] for r in ordered]}


def build_permission(group, i):
    target = f"REQ-{list(GROUPS).index(group)*50+i:04d}"
    project = ["Willow", "Juniper", "Maple", "Lotus", "Elm"][i % 5]
    env = "test" if i % 2 else "production"
    ip = f"203.0.113.{50+i}" if i % 2 else f"198.51.100.{60+i}"
    basic = group in {"P01", "P10"}
    name = (["基础查询", "报表预览"] if basic else ["报表导出", "批量报表生成"])[i % 2]
    current = None if basic else True if group in {"P02", "P07", "P09", "P12"} else False if group in {"P03", "P08"} else "unknown"
    allowed = "unknown" if group in {"P04", "P07", "P10"} else True if group == "P08" else (
        bool(i % 2) if group in {"P03", "P05", "P06", "P11"} else False)
    other = None
    if group in {"P05", "P08", "P12"}:
        other = dict(project=project, environment="production" if env == "test" else "test", status=group != "P12")
    elif group in {"P06", "P09"}:
        other = dict(project=f"Poplar-{i%4}", environment=env, status=group != "P09")
    head = [f"请求{target}在{project}项目{ENV[env]}环境调用{name}接口，返回403。",
            f"我收到403，出错请求记为{target}：{name}接口，{project}项目的{ENV[env]}环境。",
            f"{project}项目{ENV[env]}环境的{name}报403，请求标签是{target}。",
            f"调用{name}时返回403。当前请求{target}属于{project}项目，运行在{ENV[env]}环境。"][i % 4]
    common = f"当前认证有效，我有{project}项目配置权，本次请求的出口地址为{ip}。"
    whitelist = {True: f"{project}项目白名单明确包含这个出口地址。", False: f"我核对的{project}项目白名单不包含这个出口地址。",
                 "unknown": f"尚未查看{project}项目白名单，不知道它是否包含本次出口地址。"}[allowed]
    feature = ""
    if current is not None:
        if group in {"P05", "P06"}:
            feature = "没有当前目标项目和环境的开通记录，当前对应功能是否开通无法确认。"
        elif current == "unknown":
            feature = "对应功能在当前项目和环境是否开通我不知道，也没有其他项目或环境的开通记录。"
        else:
            feature = f"配置记录明确写着{project}项目{ENV[env]}环境对应导出功能{'已开通' if current else '未开通'}。"
    if other:
        feature += f"另有一条记录：{other['project']}项目{ENV[other['environment']]}环境对应导出功能{'已开通' if other['status'] else '未开通'}。"
    pieces = [common, whitelist, feature] if i % 3 else [feature, common, whitelist]
    question = ["请说明应怎样处理，还需要我确认什么。", "我想知道下一步由我配置还是请平台介入。", "请根据当前这些信息给建议。",
                "这些记录够不够判断下一步，请说明还需核对的内容。"][i % 4]
    body = head + "\n" + "".join(pieces) + question
    cap = "basic_query" if basic else "export"
    items = [fact(target, "request_scope", dict(project=project, environment=env)), fact(target, "outbound_ip", ip),
        fact(target, "auth_valid", True), fact(target, "user_can_configure", dict(project=project, status=True)),
        fact(target, "http_status", 403, required=False), fact(target, "environment", env, required=False),
        expected("facts", None, ["user:1"], alternatives=[dict(target=target, type="interface_name", value=n) for n in (name, name+"接口")]),
        fact(target, "required_capability", cap, ["user:1", "capabilities"]),
        fact(target, "whitelist_required", True, ["capabilities"]),
        fact(target, "platform_enablement_required", not basic, ["capabilities"]),
        fact(target, "whitelist_actor", "project_user", ["config_actors"], required=False),
        fact(target, "outbound_ip_allowed", dict(project=project, outbound_ip=ip, status=allowed))]
    if not basic:
        items += [fact(target, "feature_enabled", dict(project=project, environment=env, status=current),
            ["user:1", "enablement_scope", "unmatched_scope"] if group in {"P05", "P06"} else ["user:1"], decisive=group in {"P05", "P06"}),
            fact(target, "enablement_actor", "platform", ["config_actors"], required=False)]
    if other:
        items.append(fact(target, "feature_enabled", other, required=False))
    disposition = "escalate" if current is False else "answer" if allowed is False and (basic or current is True) else "request_info"
    code = {"answer": "user_config_fixable", "escalate": "platform_enablement_required", "request_info": "permission_state_unknown"}[disposition]
    support = ["user:1", "capabilities", "permission_unknown"] if disposition == "request_info" else ["user:1", "capabilities", "config_actors"]
    if group in {"P05", "P06"}:
        support += ["unmatched_scope"]
    elif disposition == "answer" and not basic:
        support += ["config_fix_allowed"]
    elif disposition == "escalate":
        support += ["config_enablement_handoff"]
    items.append(reason(target, code, support, decisive=True))
    if allowed is False:
        items.append(step(target, "adjust_project_whitelist", ["user:1", "config_actors"], required=disposition == "answer",
            params=dict(project=project, outbound_ip=ip)))
    if disposition == "escalate":
        items.append(step(target, "request_platform_enablement", ["user:1", "config_actors"], params=dict(project=project, environment=env)))
    if current == "unknown":
        items.append(missing(target, "feature_enabled"))
    if allowed == "unknown":
        items.append(missing(target, "outbound_ip_allowed"))
    return body, disposition, items, {}, dict(target=target, project=project, environment=env, outbound_ip=ip,
        interface=name, capability=cap, feature=current, allowed=allowed, other_record=other)


def build_duplicate(i):
    target = f"ORDER-{1700+i:04d}"
    amount = [39, 79, 128, 249, 560][i % 5]
    channel = "bank_card" if i % 2 else "online_payment"
    available = {"order_id", "amount", "currency"}
    if i % 4 in {0, 1}:
        available.add("payment_time")
    if i % 3 != 0:
        available.add("payment_channel")
    if i % 5 == 0:
        available.update({"transaction_id", "transaction_summary"})
    elif i % 5 == 1:
        available.add("transaction_summary")
    head = [f"订单{target}看起来有两笔扣款，两笔各{amount}元人民币，我还不能确认是否重复。",
            f"我想请人核实订单{target}：它出现两条各{amount}元人民币的扣款记录。",
            f"关于订单{target}，我怀疑被扣了两次，各{amount}元人民币，但这是我看到记录后的怀疑。",
            f"我在同一订单{target}下看到两条各{amount}元人民币的扣款，想核实后处理。",
            f"请帮我看订单{target}的两条扣款记录，金额均为{amount}元人民币，我无法确定是不是重复扣款。"][i % 5]
    details = []
    if "payment_time" in available:
        details.append(f"两笔扣款时间分别为09:{10+i%35:02d}和09:{12+i%35:02d}。")
    else:
        details.append("两笔扣款的准确时间尚未整理。")
    details.append(f"两笔渠道标识均为{CHANNEL[channel]}。" if "payment_channel" in available else "两笔的渠道标识还未核对。")
    if "transaction_id" in available:
        details.append(f"交易编号分别TX-{i}-A和TX-{i}-B，两条记录摘要都在。")
    elif "transaction_summary" in available:
        details.append("两条扣款记录截图在，但第二笔交易编号没有找到。")
    else:
        details.append("第二笔交易编号找不到，两笔完整扣款记录也暂未整理。")
    items = [fact(target, "reported_charge_count", 2), fact(target, "charges_same_order", True),
        fact(target, "payment_channel", channel if "payment_channel" in available else "unknown", required=False),
        reason(target, "suspected_duplicate_charge", ["user:1", ["role:1", "billing_handoff"]], decisive=True),
        step(target, "verify_duplicate_charge", ["user:1", ["role:1", "billing_handoff"]])]
    items += [missing(target, field, "handoff", required=field != "transaction_id") for field in
        ("payment_time", "payment_channel", "transaction_summary", "transaction_id") if field not in available]
    items.append(duplicate_recording(target, "user:1"))
    question = ["我希望先核实再谈处理。", "我担心材料不全会耽误核实，请说明后续。", "请告诉我还应提供哪些定位材料。",
        "我保留了现有账单信息，请据此给下一步。", "两条都是我看到的扣款记录，是否重复还需要核实。",
        "我还没有拿到核实结果，希望知道怎么处理。", "请根据目前的材料安排后续核实。"][i % 7]
    return head + "\n" + "".join(details) + question, "escalate", items, {}, dict(target=target, amount=amount, channel=channel,
        available_location=sorted(available), reported_charge_count=2, charges_same_order=True)


def duplicate_recording(target, user_reference):
    """记录既有和待补的核验材料，不将已知材料再列为 missing_info。"""
    return expected("next_steps", dict(target=target, code="record_diagnostics", params={
        "fields": ["order_id", "payment_time", "amount", "currency", "payment_channel",
                   "transaction_id", "transaction_summary"]}),
        [user_reference, ["role:1", "billing_handoff"]], required=False)


def revise_optional_recording(standard):
    """仅补入疑似重复扣款的可选记录项，必要判断和待补项保持。"""
    result = deepcopy(standard)
    for item in standard["expectations"]:
        if item["section"] != "reasons":
            continue
        for alternative in item["alternatives"]:
            if alternative.get("code") != "suspected_duplicate_charge":
                continue
            references = [ref for group in item["supports"] for ref in group if ref.startswith("user:")]
            if len(set(references)) != 1:
                raise ValueError("重复扣款记录项需要明确的对象用户材料")
            addition = duplicate_recording(alternative["target"], references[0])
            if addition not in result["expectations"]:
                result["expectations"].append(addition)
    return result


def revise_standards(source, output):
    values = json.loads(Path(source).read_text(encoding="utf-8"))
    revised = [revise_optional_recording(value) for value in values]
    for value in revised:
        PrivateStandard.model_validate(value)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(revised, stream, ensure_ascii=False, indent=2)
    print(f"revised {sum(a != b for a, b in zip(values, revised))}/{len(values)} standards: {target}")


def build_fault(group, i):
    target = f"CALL-{list(GROUPS).index(group)*50+i:04d}"
    name = ["库存查询", "报表读取", "订单详情查询", "状态查询", "目录查询"][i % 5] if group == "F01" else (
        ["支付确认", "支付提交", "支付结算"][i % 3] if group == "F02" else ["记录查询", "报表读取", "业务记录查询"][i % 3])
    supplied = {"interface_name"}
    if i % 3 == 0:
        supplied.add("request_id")
    if i % 4 in {0, 1}:
        supplied.add("event_time")
    if group == "T01":
        body = [f"调用{target}时收到401，响应摘要明确写Token已过期，目前我还没有重新授权。",
            f"请求{target}返回401，我看到的是认证凭据过期提示；还没换凭据或重新授权。",
            f"接口请求{target}的响应为401，摘要：Token已过期。我尚未重新授权，想知道先做什么。",
            f"我还没重新授权，请求{target}收到401，响应明确提示Token过期。"][i % 4]
        body += ["请给下一步验证建议。", "该先进行什么操作？", "我需要怎样处理并验证？"][i % 3]
        body += ["这是客户端日志里的响应摘要。", "我已把响应错误摘要抄在这里。", "同事把这个错误摘要转给我处理。",
            "我正在整理接口接入问题。", "我想确认重新发起请求前的处理顺序。", "我负责接入端，想根据该错误提示处理。",
            "我保留了这条错误响应，想按提示先处理凭据。"][i % 7]
        items = deepcopy(DEV["S10"]["expectations"])
        for e in items:
            for a in e["alternatives"]:
                a["target"] = target
        return body, "answer", items, {}, dict(target=target, http_status=401, token_expired=True, reauthorized=False)
    if group == "F01":
        core = f"{target}是测试环境的只读{name}请求，只出现一次超时，之后已经恢复，其他请求正常，没有支付或数据写入受影响。"
        items = deepcopy(DEV["S11"]["expectations"])
        disposition = "answer"
        extra_facts = dict(widespread_outage=False, data_loss=False, data_corruption=False)
        core += "没有发生数据丢失或错乱。"
    elif group == "F02":
        core = f"生产环境的{name}接口持续返回500，请求记录为{target}。多个用户的支付流程受到影响，当前仍未恢复。"
        items = deepcopy(DEV["S12"]["expectations"])
        disposition = "escalate"
        extra_facts = dict(recovered=False, widespread_outage="unknown", data_loss="unknown", data_corruption="unknown")
        core += "是否存在更大范围的不可用，以及数据是否受损，我还无法确认。"
    elif group == "F03":
        core = f"请求{target}所在的生产环境已经大面积不可用，多数用户都无法使用服务，目前仍未恢复；支付没有受影响，也没有发现数据丢失或错乱。受影响的接口是{name}。"
        disposition = "escalate"
        extra_facts = dict(environment="production", affected_users="widespread", widespread_outage=True,
            recovered=False, payment_affected=False, data_loss=False, data_corruption=False)
        items = [reason(target, "severe_business_impact", ["user:1", ["role:1", "severe_fault"]], decisive=True),
            step(target, "escalate_technical", ["user:1", ["role:1", "severe_fault"]])]
    else:
        core = f"生产环境的{name}请求{target}返回过500，后来服务已经恢复，但我明确发现业务记录数据丢失。不是仅仅显示慢；支付未受影响，也未发现数据错乱。"
        disposition = "escalate"
        extra_facts = dict(environment="production", http_status=500, recovered=True, data_loss=True,
            payment_affected=False, data_corruption=False, widespread_outage="unknown")
        core += "是否曾大面积不可用我不知道。"
        items = [reason(target, "severe_business_impact", ["user:1", ["role:1", "severe_fault"]], decisive=True),
            step(target, "escalate_technical", ["user:1", ["role:1", "severe_fault"]])]
    for e in items:
        for a in e["alternatives"]:
            a["target"] = target
    items = [e for e in items if not (e["section"] == "missing_info" and e["alternatives"][0]["field"] in supplied)]
    if group in {"F01", "F02", "F03", "F04"}:
        items.append(missing(target, "response_summary", "handoff", required=False))
    if group == "F01":
        items += [missing(target, field, "handoff", required=False) for field in
                  ("event_time", "request_id") if field not in supplied]
    if group in {"F02", "F03", "F04"}:
        items.append(expected("next_steps", None, ["fault_handling"], required=False, alternatives=[
            dict(target=target, code="record_diagnostics", params={"fields": fields}) for fields in (
                ["event_time", "interface_name", "request_id", "occurrence"],
                ["event_time", "interface_name", "request_id", "occurrence", "response_summary"],
                ["request_id", "interface_name", "event_time", "response_summary", "occurrence", "affected_users"])]))
    details = f"具体接口就是{name}。"
    details += f"已有request_id=rid-{list(GROUPS).index(group)*50+i:04d}。" if "request_id" in supplied else "request_id还没拿到。"
    details += f"发生时间记录为10:{10+i%45:02d}。" if "event_time" in supplied else "发生的准确时间暂未整理。"
    for field in extra_facts:
        items.append(fact(target, field, extra_facts[field], required=group in {"F03", "F04"}, decisive=field in {"data_loss", "widespread_outage"} and extra_facts[field] is True))
    items.append(expected("facts", None, ["user:1"], required=False,
        alternatives=[dict(target=target, type="interface_name", value=n) for n in (name, name+"接口")]))
    if group in {"F03", "F04"}:
        items += [missing(target, k, "handoff") for k in ("request_id", "event_time") if k not in supplied]
    questions = ["想知道下一步怎么处理以及应记录哪些材料。", "我没有整理完日志，现在该怎么做？", "请说明接下来应如何处理。",
        "这些是目前能提供的信息，请给后续建议。", "我想知道应先做什么，哪些材料需要补充。"][i % 5]
    body = (details + core if i % 3 == 0 else core + details) + questions
    if group == "F01" and i % 2:
        body += "最近我调整过客户端超时配置。"
    if group == "F01":
        body += ["我希望保留记录，便于以后对照。", "我想从客户端网络侧做初步核对。", "请说明低风险排查的顺序。",
            "我需要向同事交接这次异常的记录。", "我不确定该把哪些响应信息记下来。", "请根据这个已恢复的现象给记录建议。",
            "我准备整理问题记录并核对近期配置。"][i % 7]
    else:
        body += ["我需要把当前情况交给后续处理人。", "还有部分定位记录在整理中。", "请说明处理决定与待补材料。",
            "我准备向支持人员说明当前现场情况。", "我想确认目前材料不全时的处理路径。", "请根据已经描述的影响判断下一步。",
            "我提供的是现场观察，还没有后台核实结果。"][i % 7]
    return body, disposition, items, {}, dict(target=target, interface=name, supplied_location=sorted(supplied),
        facts={e["alternatives"][0]["type"]: e["alternatives"][0]["value"] for e in items if e["section"] == "facts"})


def unprovided(group, spec, items):
    if group.startswith(("R", "D")):
        fields = ["environment", "request_read_only", "payment_affected", "write_affected", "widespread_outage", "data_loss",
            "data_corruption", "timeout_observed", "token_expired", "auth_valid", "occurrence", "affected_users", "reported_charge_count", "charges_same_order"]
        return {r["target"]: fields for r in spec["refunds"]}
    target = spec["target"]
    if group.startswith("P"):
        fields = ["payment_affected", "write_affected", "widespread_outage", "data_loss", "data_corruption", "occurrence",
            "affected_users", "recovered", "reported_charge_count", "charges_same_order"]
        if spec["capability"] != "basic_query":
            fields.append("request_read_only")
    elif group == "T01":
        fields = ["environment", "request_read_only", "request_scope", "feature_enabled", "outbound_ip_allowed", "user_can_configure",
            "payment_affected", "write_affected", "affected_users", "widespread_outage", "data_loss", "data_corruption", "recovered", "occurrence"]
    elif group.startswith("F"):
        fields = ["auth_valid", "token_expired", "reported_charge_count", "charges_same_order"]
        if group != "F01":
            fields += ["write_affected", "request_read_only"]
        if group == "F01":
            fields.append("affected_users")
        if group == "F04":
            fields += ["affected_users", "occurrence"]
        if group == "F03":
            fields += ["http_status", "timeout_observed", "occurrence"]
    else:
        fields = ["environment", "auth_valid", "token_expired", "http_status", "timeout_observed", "widespread_outage", "data_loss", "data_corruption"]
    known = {a["type"] for e in items if e["section"] == "facts" for a in e["alternatives"]}
    return {target: [f for f in fields if f not in known]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    tasks, standards, sources, specs = [], [], [], []
    for group, (split, count) in GROUPS.items():
        for i in range(1, count+1):
            body, disposition, items, severe, spec = (build_refund(group, i) if group.startswith(("R", "D")) else
                build_permission(group, i) if group.startswith("P") else build_duplicate(i) if group == "B01" else build_fault(group, i))
            task_id = f"M6-{group}-{i:03d}"
            role = "billing" if group.startswith(("R", "D", "B")) else "technical"
            presentation = "attached_record" if i % 3 == 0 else "user_text"
            if presentation == "attached_record":
                task = PublicTask(task_id=task_id, role=role, user_text="我把目前掌握的咨询情况附在记录中，请根据这些信息给出下一步建议。",
                    records=[dict(reference_id="user:2", content=body)])
                for e in items:
                    e["supports"] = [["user:2" if s == "user:1" else s for s in group_support] for group_support in e["supports"]]
            else:
                task = PublicTask(task_id=task_id, role=role, user_text=body)
            standard = PrivateStandard(task_id=task_id, group_id=group, split=split, dispositions=[disposition],
                materials={m.reference_id: m.content for m in task.materials()}, expectations=items, severe_items=severe,
                unprovided_user_facts=unprovided(group, spec, items))
            source_id = f"stage6-constructed:{group}:{i:03d}"
            tasks.append(task.model_dump())
            standards.append(standard.model_dump())
            sources.append(dict(task_id=task_id, source_id=source_id, source_family=group, split=split, group_id=group,
                origin="stage6-20261013-C", ancestor=None, calibration_reuse=False, teacher_reuse=False,
                presentation=presentation, instance_spec=f"instance_specs.json#{task_id}",
                variation="决定性状态固定；组内参数、核对状态、定位材料、记录顺序和材料呈现按实例说明变化",
                ai_precheck="待核查", human_review="待人工核查"))
            specs.append(dict(task_id=task_id, source_id=source_id, group_id=group, split=split,
                disposition=disposition, presentation=presentation, instance=spec))
    task_map = {t["task_id"]: PublicTask.model_validate(t) for t in tasks}
    standard_map = {s["task_id"]: PrivateStandard.model_validate(s) for s in standards}
    validate_task_pairs(task_map, standard_map, CLAUSES)
    counts = Counter(s["split"] for s in standards)
    assert counts == {"train": 400, "validation": 80, "test": 120}
    assert len(task_map) == 600 and len({s["group_id"] for s in standards}) == 35
    for name, value in [("public_tasks.json", tasks), ("private_standards.json", standards), ("sources.json", sources),
                        ("instance_specs.json", specs), ("policy_support.json", [c.model_dump() for c in CLAUSES.values()]),
                        ("group_allocation.json", [{"group_id": g, "split": s, "count": n} for g, (s, n) in GROUPS.items()])]:
        write(DATA / name, value)
    for split in counts:
        ids = {s["task_id"] for s in standards if s["split"] == split}
        write(DATA / f"{split}_public_tasks.json", [t for t in tasks if t["task_id"] in ids])
        write(DATA / f"{split}_private_standards.json", [s for s in standards if s["task_id"] in ids])
    write(DATA / "generation_record.json", dict(date="2026-10-13", executor="合成任务构建器", model="gpt-6.1-sol",
        reasoning_effort="xhigh", source="合成客服政策与35个固定场景组合",
        method="先确定实例条件，再写自然语言；私有标准从固定规则构造，不使用教师解答反推真值",
        settings=dict(separate_generation_api=False, temperature=None, random_seed=None, export="确定性脚本"),
        instruction="agentic_runs/stage6-20261013/GENERATION.md", script="scripts/build_stage6_formal.py",
        task_counts=dict(counts), reused_calibration_tasks=0, reused_demonstrations=0, teacher_new_tasks=400,
        candidate_upper_bound=800, final_test_human_review="待逐题人工核查", data_frozen=False))
    print("constructed 600 tasks / 35 groups; train=400 validation=80 test=120")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revise-standards-from")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.revise_standards_from:
        if not args.output:
            parser.error("修订标准须指定新的 --output")
        revise_standards(args.revise_standards_from, args.output)
    elif args.output:
        parser.error("--output 需配合 --revise-standards-from")
    else:
        main()
