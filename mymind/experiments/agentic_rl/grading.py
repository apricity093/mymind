"""依据私有标准、提交结果与实际可见证据计算四档业务奖励。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError

from agents.customer_policy.protocol import VisibleTrajectory, parse_result
from agents.customer_policy.vocabulary import REASON_FACT_CLAIMS
from experiments.agentic_rl.tasks import PolicyClause, PrivateStandard


@dataclass
class Grade:
    reward: float
    complete_success: bool = False
    severe_error: bool = False
    protocol_error: bool = False
    errors: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _payload(item):
    return {key: value for key, value in item.items()
            if key != "evidence" and value is not None}


def _compact(text):
    return "".join(text.split())


def _same_item(item, alternative, required=True):
    left, right = dict(item), dict(alternative)
    for payload in (left, right):
        if "params" in payload:
            payload["params"] = dict(payload["params"])
            for key in ("fields", "directions"):
                if key in payload["params"]:
                    payload["params"][key] = sorted(set(payload["params"][key]))
    if left.get("code") == right.get("code") == "record_diagnostics":
        context = {"environment", "http_status", "timeout_observed", "recovered",
                   "request_read_only", "payment_affected", "write_affected",
                   "affected_users", "widespread_outage", "request_scope",
                   "data_loss", "data_corruption"}
        actual = set(left["params"]["fields"])
        needed = set(right["params"]["fields"])
        if not needed.intersection({"interface_name", "request_id", "event_time"}):
            context = set()
        if ((needed <= actual and actual - needed <= context) or
                (not required and actual and actual <= needed | context)):
            left["params"]["fields"] = right["params"]["fields"]
    return left == right


def _supported(expected, evidence, trajectory, standard, clauses):
    materials = {m.reference_id: m.content for m in trajectory.materials}
    snippets = {s.reference_id: s for turn in trajectory.searches
                for s in turn.observation.snippets}
    statuses = {turn.observation.status.reference_id: turn.observation.status
                for turn in trajectory.searches}
    if not evidence or any(ref not in materials and ref not in snippets and ref not in statuses
                           for ref in evidence):
        return False

    def visible(support):
        if support in standard.materials:
            return support in evidence and materials.get(support) == standard.materials[support]
        if support == "retrieval_unavailable":
            return any(ref in statuses and statuses[ref].state == "unavailable" for ref in evidence)
        clause: PolicyClause = clauses[support]
        texts = [snippet.content for ref, snippet in snippets.items()
                 if ref in evidence and snippet.source_id == clause.source_id]
        needle = _compact(clause.text)
        return any(needle in _compact(text) for text in texts) or needle in _compact("".join(texts))

    return all(any(visible(support) for support in group) for group in expected.supports)


def _fact_conflicts(item, expectations):
    if "value" not in item:
        return False
    if item.get("type") == "within_window" and type(item["value"]) is bool:
        prerequisites = {"approval_status", "elapsed_workdays", "payment_channel"}
        if any(alt.get("target") == item.get("target")
               and alt.get("type") in prerequisites and alt.get("value") == "unknown"
               for expected in expectations if expected.section == "facts"
               for alt in expected.alternatives):
            return True
    candidates = [alt for expected in expectations if expected.section == "facts"
                  for alt in expected.alternatives
                  if alt.get("target") == item.get("target") and alt.get("type") == item.get("type")]
    if (not candidates and item.get("type") == "http_status"
            and type(item["value"]) is int and item["value"] in {401, 403, 500}
            and any(alt.get("target") == item.get("target")
                    and alt.get("type") == "timeout_observed" and alt.get("value") is True
                    for expected in expectations if expected.section == "facts"
                    for alt in expected.alternatives)):
        return True
    # 缺少作用范围是协议遗漏；不能把正确状态的裸布尔值判成相反状态。
    values = [alt["value"] for alt in candidates]
    def comparable(left, right):
        return (type(left) is type(right) or
                (left == "unknown" and type(right) in {bool, int}) or
                (right == "unknown" and type(left) in {bool, int}))
    if values and all(isinstance(value, dict) for value in values):
        if not isinstance(item["value"], dict):
            return False
        return not any(all(item["value"][key] == value[key]
                           for key in value.keys() & item["value"].keys()
                           if comparable(item["value"][key], value[key])) for value in values)
    values = [value for value in values if comparable(item["value"], value)]
    return bool(values) and not any(item["value"] == value for value in values)


def _step_conflicts(item, expectations):
    protected = {"limit_workdays", "project", "environment", "outbound_ip"}
    params = item.get("params")
    if not isinstance(params, dict) or not protected.intersection(params):
        return False
    candidates = [alt for expected in expectations if expected.section == "next_steps"
                  for alt in expected.alternatives
                  if alt.get("target") == item.get("target") and alt.get("code") == item.get("code")]
    # 无参数建议上的多余参数仅造成协议失败；比较实际规定的业务参数。
    candidates = [alt for alt in candidates if protected.intersection(alt.get("params", {}))]
    return bool(candidates) and not any(all(
        params[key] == alt.get("params", {}).get(key) for key in protected.intersection(params)
    ) for alt in candidates)


def _reason_conflicts(item, standard):
    target, code = item.get("target"), item.get("code")
    for kind, value in REASON_FACT_CLAIMS.get(code, ()):
        claim = {"target": target, "type": kind, "value": value}
        if _unprovided_claim(claim, standard):
            return True
        if kind == "feature_enabled":
            # 平台处置理由针对当前请求，不能借用同一对象下的异范围开通记录。
            scope = next((alt["value"] for expected in standard.expectations
                          if expected.section == "facts" for alt in expected.alternatives
                          if alt.get("target") == target and alt.get("type") == "request_scope"), None)
            if scope is None:
                continue
            claim["value"] = {**scope, **value}
        if _fact_conflicts(claim, standard.expectations):
            return True
    if code == "user_config_fixable":
        # 当前项目配置权已明确时才比较；重新授权建议不声称具备项目配置权。
        return _fact_conflicts({"target": target, "type": "user_can_configure",
                                "value": {"status": True}}, standard.expectations)
    if code == "limited_recovered_fault":
        # “影响有限”与已报告的严重影响矛盾，不等于断言所有未列出的状态均为 false。
        return any(alt.get("target") == target and (
            (alt.get("type") in {"payment_affected", "data_loss", "data_corruption",
                                "widespread_outage"} and alt.get("value") is True) or
            (alt.get("type") == "affected_users" and alt.get("value") == "widespread"))
            for expected in standard.expectations if expected.section == "facts"
            for alt in expected.alternatives)
    return False


def _unprovided_claim(item, standard):
    unprovided = standard.unprovided_user_facts
    fields = unprovided.get(item.get("target"))
    if fields is None:
        # 实际输出会将引用编号错写成对象；错绑不使所有对象均未提供的事实变为已知。
        targets = {alt["target"] for expected in standard.expectations if expected.section == "facts"
                   for alt in expected.alternatives}
        fields = set.intersection(*(set(unprovided.get(target, [])) for target in targets)) if targets else set()
    if item.get("type") not in fields:
        return False
    value = item.get("value")
    if isinstance(value, dict):
        return any(v is not None and v != "unknown" for v in value.values())
    return value is not None and value != "unknown"


def _equivalent_fact(expected, supported_facts):
    facts = [f for f in supported_facts if f["target"] == expected["target"]]
    kind, value = expected["type"], expected["value"]
    if kind == "outbound_ip":
        return any(f["type"] == "outbound_ip_allowed" and f["value"]["outbound_ip"] == value for f in facts)
    if kind == "request_scope":
        return any(f["type"] == "feature_enabled" and all(f["value"][k] == v for k, v in value.items())
                   for f in facts)
    timed = any(f["type"] == "elapsed_workdays" and type(f["value"]) is int for f in facts)
    if kind == "approval_status" and value == "approved":
        return timed and any(f["type"] == "timer_start" and f["value"] == "approval" for f in facts)
    if kind == "timer_start" and value == "approval":
        # 已有据的时效比较包含审核起算规则；数值字段已表达审核后经过时间。
        return timed and any(f["type"] == "within_window" for f in facts)
    return False


def grade_result(standard: PrivateStandard, raw_result, trajectory: VisibleTrajectory,
                 clauses: dict[str, PolicyClause]) -> Grade:
    if trajectory.task_id != standard.task_id:
        raise ValueError("轨迹与判分标准的 task_id 不一致")
    severe, errors = [], []
    try:
        raw = raw_result
        if isinstance(raw_result, str):
            try:
                raw = json.loads(raw_result)
            except json.JSONDecodeError as exc:
                if exc.msg != "Extra data":
                    raise
                # 完整对象之后的多余文本仍是协议错误；这里只核查已写出的严重断言。
                raw, _ = json.JSONDecoder().raw_decode(raw_result)
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(exclude_none=True)
        if isinstance(raw, dict):
            for section in ("facts", "reasons", "next_steps", "missing_info"):
                entries = raw.get(section, [])
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    payload = _payload(item)
                    if section == "facts" and _fact_conflicts(payload, standard.expectations):
                        severe.append(f"facts:{payload.get('target')}:{payload.get('type')}:与用户事实或适用规则矛盾")
                    if section == "facts" and _unprovided_claim(payload, standard):
                        severe.append(f"facts:{payload.get('target')}:{payload.get('type')}:将题面未提供的用户事实写成已确认")
                    if section == "next_steps" and _step_conflicts(payload, standard.expectations):
                        severe.append(f"next_steps:{payload.get('target')}:{payload.get('code')}:时效或作用范围错误")
                    if section == "reasons" and _reason_conflicts(payload, standard):
                        severe.append(f"reasons:{payload.get('target')}:{payload.get('code')}:与已知事实、明确未知状态或适用规则矛盾")
                    if payload in standard.severe_items.get(section, []):
                        severe.append(f"{section}:{payload}:严重业务错误")
        result = parse_result(raw_result)
    except (ValidationError, ValueError, TypeError) as exc:
        return Grade(-1.0 if severe else 0.0, severe_error=bool(severe),
                     protocol_error=True, errors=severe + [f"协议错误:{exc}"])
    if severe:
        return Grade(-1.0, severe_error=True, errors=severe)

    matched = set()
    supported_facts = []
    for section in ("reasons", "facts", "next_steps", "missing_info"):
        for submitted in getattr(result, section):
            item = submitted.model_dump(exclude_none=True)
            payload = _payload(item)
            candidates = [(i, expected) for i, expected in enumerate(standard.expectations)
                          if expected.section == section and any(_same_item(payload, alt, expected.required)
                                                                  for alt in expected.alternatives)]
            accepted = [i for i, expected in candidates if _supported(
                expected, item["evidence"], trajectory, standard, clauses
            )]
            if accepted:
                matched.update(accepted)
                if section == "facts":
                    supported_facts.append(payload)
            else:
                errors.append(f"{section}:{payload}:" + ("引用不存在或正文不支持" if candidates
                                                        else "对象、值、范围或建议不符合任务"))
    # 只补足缺失字段的等价信息；已提交条目的值及独立引用仍按上面的核查结果处理。
    for i, expected in enumerate(standard.expectations):
        if expected.section != "facts":
            continue
        if any(_equivalent_fact(alt, supported_facts) for alt in expected.alternatives):
            matched.add(i)
    missing = [f"{expected.section}:{expected.alternatives}" for i, expected in enumerate(
        standard.expectations) if expected.required and i not in matched]
    if result.disposition not in standard.dispositions:
        errors.append("主要处置错误")
        return Grade(0.0, errors=errors, missing=missing)
    if not missing and not errors:
        return Grade(1.0, complete_success=True)
    decisive = any(standard.expectations[i].decisive or
                   (standard.expectations[i].section == "reasons" and
                    any(a["code"] not in {"evidence_insufficient", "retrieval_unavailable"}
                        for a in standard.expectations[i].alternatives)) for i in matched)
    rule_supported = any(
        standard.expectations[i].section in {"facts", "reasons"}
        and any(support in clauses or support == "role:1"
                for group in standard.expectations[i].supports for support in group)
        for i in matched
    )
    return Grade(0.3 if decisive and rule_supported else 0.0, errors=errors, missing=missing)
