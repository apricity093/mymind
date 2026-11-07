"""策略可见的任务、检索观察、轨迹和结构化终止结果。"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from agents.customer_policy.vocabulary import REASONS, FACT_LABELS, STEPS


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


FACT_ENUMS = {
    "payment_channel": {"online_payment", "bank_card", "balance", "unknown"},
    "refund_stage": {"accepted", "approved", "channel_processing", "unknown"},
    "approval_status": {"approved", "unknown"},
    "timer_start": {"approval"},
    "required_capability": {"basic_query", "export"},
    "enablement_actor": {"platform"},
    "whitelist_actor": {"project_user"},
    "environment": {"test", "production", "unknown"},
    "occurrence": {"single", "repeated", "continuous", "unknown"},
    "affected_users": {"one", "multiple", "widespread", "unknown"},
}
BOOLEAN_FACTS = {
    "original_account_checked", "received", "charges_same_order", "auth_valid",
    "timeout_observed", "token_expired", "request_read_only", "recovered",
    "widespread_outage", "payment_affected", "write_affected", "data_loss",
    "data_corruption",
}
RULE_BOOLEANS = {
    "within_window", "timing_policy_covered", "platform_enablement_required",
    "whitelist_required",
}
SCOPED_FACTS = {
    "request_scope": {"project", "environment"},
    "user_can_configure": {"project", "status"},
    "feature_enabled": {"project", "environment", "status"},
    "outbound_ip_allowed": {"project", "outbound_ip", "status"},
}
FACT_TYPES = tuple(sorted(
    set(FACT_ENUMS) | BOOLEAN_FACTS | RULE_BOOLEANS | set(SCOPED_FACTS)
    | {"elapsed_workdays", "limit_workdays", "reported_charge_count",
       "interface_name", "outbound_ip", "http_status"}
))

ReasonCode = Literal[
    "refund_state_unknown", "within_refund_window", "refund_overdue",
    "policy_uncovered", "suspected_duplicate_charge", "user_config_fixable",
    "platform_enablement_required", "permission_state_unknown", "token_expired",
    "limited_recovered_fault", "severe_business_impact", "evidence_insufficient",
    "retrieval_unavailable",
]
StepCode = Literal[
    "wait_within_window", "check_original_account", "verify_refund",
    "verify_refund_if_overdue_unreceived", "verify_channel_timing",
    "verify_duplicate_charge", "adjust_project_whitelist",
    "request_platform_enablement", "reauthorize_then_verify", "record_diagnostics",
    "low_risk_checks", "escalate_if_severe_impact", "escalate_technical",
    "seek_human_review",
]
LOCATION_FIELDS = {
    "order_id", "transaction_id", "payment_time", "amount", "currency",
    "transaction_summary", "request_id", "event_time", "response_summary",
}
USER_FACTS = (set(FACT_ENUMS) - {"timer_start", "required_capability",
                              "enablement_actor", "whitelist_actor"}) | BOOLEAN_FACTS | {
    "elapsed_workdays", "reported_charge_count", "interface_name", "outbound_ip",
    "request_scope", "user_can_configure", "feature_enabled", "outbound_ip_allowed",
    "http_status",
}
MISSING_FIELDS = USER_FACTS | LOCATION_FIELDS
STEP_PARAMS = {
    "verify_refund_if_overdue_unreceived": {"limit_workdays": "策略填写的 3 或 5"},
    "adjust_project_whitelist": {"project": "目标项目", "outbound_ip": "本次请求出口地址"},
    "request_platform_enablement": {"project": "目标项目", "environment": "test 或 production"},
    "record_diagnostics": {"fields": "必要定位字段列表"},
    "low_risk_checks": {"directions": ["network", "recent_config_changes"]},
}


def public_value_catalog() -> dict:
    return {
        "reason_codes": REASONS,
        "fact_labels": {key: FACT_LABELS[key] for key in FACT_TYPES},
        "step_codes": {**STEPS,
                       "verify_refund_if_overdue_unreceived": "超过所述上限且仍未到账时才人工核实",
                       "adjust_project_whitelist": "用户调整目标项目及出口地址白名单后验证",
                       "request_platform_enablement": "平台核实目标项目及环境的功能开通",
                       "record_diagnostics": "记录所列定位字段",
                       "low_risk_checks": "按所列低风险方向排查"},
        "fact_enums": {key: sorted(values) for key, values in FACT_ENUMS.items()},
        "user_boolean_facts": sorted(BOOLEAN_FACTS),
        "rule_boolean_facts": sorted(RULE_BOOLEANS),
        "scoped_facts": {key: {field: ([True, False, "unknown"] if field == "status" else
                            ["test", "production"] if field == "environment" else "题面原值")
                              for field in sorted(values)} for key, values in SCOPED_FACTS.items()},
        "numeric_facts": {"elapsed_workdays": "非负整数或 unknown",
                          "limit_workdays": [3, 5], "reported_charge_count": "非负整数",
                          "http_status": [401, 403, 500]},
        "step_params": {code: STEP_PARAMS.get(code) for code in StepCode.__args__},
        "diagnostic_fields": sorted(MISSING_FIELDS),
        "diagnostic_labels": {key: FACT_LABELS[key] for key in sorted(MISSING_FIELDS)},
    }


class EvidenceItem(ProtocolModel):
    target: str = Field(min_length=1)
    evidence: list[str]


class Reason(EvidenceItem):
    code: ReasonCode


def _state(value):
    return type(value) is bool or value == "unknown"


class Fact(EvidenceItem):
    type: Literal[FACT_TYPES]
    value: JsonValue

    @model_validator(mode="after")
    def validate_value(self):
        value, kind = self.value, self.type
        valid = False
        if kind in FACT_ENUMS:
            valid = isinstance(value, str) and value in FACT_ENUMS[kind]
        elif kind in BOOLEAN_FACTS:
            valid = _state(value)
        elif kind in RULE_BOOLEANS:
            valid = type(value) is bool
        elif kind in SCOPED_FACTS:
            valid = isinstance(value, dict) and set(value) == SCOPED_FACTS[kind]
            if valid:
                valid = all(isinstance(v, str) and bool(v) for k, v in value.items()
                            if k != "status")
                if "environment" in value:
                    valid = valid and value["environment"] in {"test", "production"}
                if "status" in value:
                    valid = valid and _state(value["status"])
        elif kind in {"elapsed_workdays", "reported_charge_count"}:
            valid = type(value) is int and value >= 0
            if kind == "elapsed_workdays":
                valid = valid or value == "unknown"
        elif kind == "limit_workdays":
            valid = type(value) is int and value in {3, 5}
        elif kind == "http_status":
            valid = type(value) is int and value in {401, 403, 500}
        elif kind in {"interface_name", "outbound_ip"}:
            valid = isinstance(value, str) and bool(value)
        if not valid:
            raise ValueError(f"{kind} 的值不符合公开目录")
        return self


class NextStep(EvidenceItem):
    code: StepCode
    params: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_params(self):
        schemas = {
            "verify_refund_if_overdue_unreceived": {"limit_workdays"},
            "adjust_project_whitelist": {"project", "outbound_ip"},
            "request_platform_enablement": {"project", "environment"},
            "record_diagnostics": {"fields"},
            "low_risk_checks": {"directions"},
        }
        required = schemas.get(self.code)
        if required is None:
            if self.params is not None:
                raise ValueError("此建议不接受额外参数")
            return self
        if self.params is None or set(self.params) != required:
            raise ValueError(f"{self.code} 需要参数 {sorted(required)}")
        if "limit_workdays" in required:
            value = self.params["limit_workdays"]
            if type(value) is not int or value not in {3, 5}:
                raise ValueError("时效参数必须为 3 或 5")
        for key in required & {"project", "outbound_ip", "environment"}:
            if not isinstance(self.params[key], str) or not self.params[key]:
                raise ValueError("范围参数必须为非空字符串")
        if "environment" in required and self.params["environment"] not in {"test", "production"}:
            raise ValueError("环境必须为 test 或 production")
        for key, choices in (("fields", MISSING_FIELDS),
                             ("directions", {"network", "recent_config_changes"})):
            if key in required:
                values = self.params[key]
                if not isinstance(values, list) or not values or any(
                    not isinstance(v, str) or v not in choices for v in values
                ):
                    raise ValueError(f"{key} 不符合公开目录")
        return self


class MissingInfo(EvidenceItem):
    field: Literal[tuple(sorted(MISSING_FIELDS))]
    purpose: Literal["decision", "handoff"]
    condition: Literal["if_approved"] | None = None

    @model_validator(mode="after")
    def validate_condition(self):
        if self.condition is not None and self.field != "elapsed_workdays":
            raise ValueError("if_approved 仅用于审核未知时的条件性计时")
        return self


class TerminalResult(ProtocolModel):
    disposition: Literal["answer", "request_info", "escalate"]
    reasons: list[Reason]
    facts: list[Fact]
    next_steps: list[NextStep]
    missing_info: list[MissingInfo]


class VisibleMaterial(ProtocolModel):
    reference_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


ROLE_TEXT = {
    "billing": "你是账单客服。依据用户提供的信息和实际可见政策作条件性判断，不能声称已核验后台或已执行退款、补偿、账单调整。疑似重复扣款需人工或财务核实，材料未齐不延迟升级。只索取必要定位材料，不索取完整银行卡号、密码或验证码。",
    "technical": "你是技术客服。依据用户提供的信息和实际可见政策给指导，不能编造根因、声称已修改配置或恢复服务。生产大面积不可用、支付链路异常、数据丢失或错乱需人工或二线处理，日志缺失不延迟升级。不索取完整 Token、密钥、密码或验证码，不建议无限重试、删除数据或关闭安全校验。",
}


class PublicTask(ProtocolModel):
    task_id: str = Field(min_length=1)
    role: Literal["billing", "technical"]
    user_text: str = Field(min_length=1)
    records: list[VisibleMaterial] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self):
        refs = [r.reference_id for r in self.records]
        if len(refs) != len(set(refs)) or any(
            not r.startswith("user:") or r == "user:1" for r in refs
        ):
            raise ValueError("附带记录使用不重复的 user: 引用，user:1 留给用户原文")
        return self

    def materials(self) -> list[VisibleMaterial]:
        role_text = ROLE_TEXT[self.role]
        return [VisibleMaterial(reference_id="role:1", content=role_text),
                VisibleMaterial(reference_id="user:1", content=self.user_text),
                *self.records]

    def messages(self, max_searches: int = 3) -> list[dict[str, str]]:
        materials = self.materials()
        search_limit = {1: "一次", 3: "三次"}[max_searches]
        return [
            {"role": "system", "content": "role:1\n" + materials[0].content
             + f"\n可调用 search(query)，每条轨迹最多{search_limit}检索；证据充分可提前终止。"
             + "最后一次观察后仍须提交终止结果。各条目引用实际已见材料，未知不当作否定。"
             + "target 必须原样使用题面中的记录或请求标识，如 R1、R2、Q1；整体咨询用 case。"
             + "不要用字段名、描述句或自行创建的名称作为 target。"
             + "evidence 是 reference_id 字符串数组，如 user:1、user:2、role:1、search:1:1；"
             + "不能写原文摘录、规则名称或未返回的编号。"
             + "某项判断若结合用户条件与政策，例如能力匹配、政策覆盖或时效比较，"
             + "该项 evidence 同时引用对应用户材料与实际政策片段。规则定义本身可只引用政策。"
             + "直接复述材料给出的完整范围状态或明确未知，只需实际材料；复合字段不额外要求政策。"
             + "字段语义直接蕴含的事实引用其实际前提，例如本次凭据有效蕴含本次未过期；"
             + "未过期不等于有效，401或403本身不证明过期或有效。跨记录适用范围判断仍需关系材料及政策。"
             + "引用需支持本条完整判断，其他条目的引用通常不代替本条证据。"
             + "仅两类政策常量允许组合表达：同一对象的payment_channel已独立正确且有据时，"
             + "limit_workdays可只引该渠道时限政策；同一对象的required_capability已独立正确且有据时，"
             + "platform_enablement_required事实可只引该能力的开通要求政策。"
             + "常量必须匹配该前提，未知或未覆盖渠道不能选定具体时限，不能借用另一对象。"
             + "这不代替政策覆盖、超时、当前升级及其他综合判断各自所需的完整事实与政策引用，"
             + "也不豁免已提交条目的错误值或错误引用。"
             + "事实项表达实际条件、适用规则及有据的比较或范围匹配判断，只提交与当前问题相关的内容。"
             + "不要猜测用户未提供的事实。终止 JSON 不加 Markdown 代码块、说明句或工具标签。"
             + "interface_name 和项目、出口地址原样使用题面值，不改为翻译、缩写或自创标识。"
             + "用户布尔状态可为 true、false 或 unknown；规则布尔判断须有据且为 true 或 false。"
             + "true、false 和数值使用 JSON 原生类型，不能写成带引号的布尔或数字字符串。"
             + "平台开通按项目及环境匹配，白名单按项目及出口地址匹配。"
             + "审核未知时先补审核状态，计时待补项写 condition=if_approved；已通过时不写此条件。"
             + "missing_info 的 decision 用于补足判断，handoff 用于人工定位；"
             + "已有升级条件时，其他对象的 decision 待补项不延迟升级。"
             + "有参数建议添加 params 对象，键和值按 step_params；无参数建议省略 params。\n公开值域及参数目录：\n"
             + json.dumps(public_value_catalog(), ensure_ascii=False)
             + "\n填写时使用 fact_labels 的键作为 facts.type、reason_codes 的键作为 reasons.code、"
             + "step_codes 的键作为 next_steps.code，不互换字段。"
             + "diagnostic_fields 仅用于 missing_info.field 或 record_diagnostics 的 fields；"
             + "不属于 fact_labels 的定位字段不能作为 facts.type。"
             + "scoped_facts 必须用所列完整对象，不能把带项目或环境的状态写成裸布尔值。"
             + "platform_enablement_required 和 whitelist_required 表示能力的规则要求，"
             + "不是当前是否仍待补办；当前满足状态分别写 feature_enabled、outbound_ip_allowed。"
             + "每项用户条件应引用确实包含它的用户材料，不能把 user:1 与附带记录混用。"
             + "适用于当前对象的条件性建议也须引用对象的用户条件与对应规则；"
             + "reason 的依据须覆盖所用的每条能力、计时或处理主体规则。"
             + "终止前检查处置、理由、比较与下一步相互一致；仅输出相关事实。"
             + "必要判断应包含已知条件、适用规则、比较或范围匹配；必要后续应包含政策要求的条件性处理。"
             + "时效解释分别体现计时起点、适用上限、用户条件及比较；"
             + "能力解释分别体现接口、请求范围、适用能力要求与已知配置状态。不适用项省略。"
             + "若政策同时给出当前指导与将来条件成立时的处理，输出应保留对应条件性建议。"
             + "尚缺决定性状态时先补该状态，不把条件尚未成立的等待或核实建议当作当前下一步。"
             + "终止 JSON 包含以下五个字段，各列表条目的基本结构如下（值为格式占位说明）：\n"
             + json.dumps({
                 "disposition": "answer、request_info 或 escalate",
                 "reasons": [{"target": "题面标识", "code": "reason_codes 中的键",
                              "evidence": ["实际可见材料的 reference_id"]}],
                 "facts": [{"target": "题面标识", "type": "fact_labels 中的键",
                            "value": "该 type 在目录中规定的值", "evidence": ["实际可见材料的 reference_id"]}],
                 "next_steps": [{"target": "题面标识", "code": "step_codes 中的键",
                                 "evidence": ["实际可见材料的 reference_id"]}],
                 "missing_info": [{"target": "题面标识", "field": "diagnostic_fields 中的字段",
                                   "purpose": "decision 或 handoff", "evidence": ["实际可见材料的 reference_id"]}],
             }, ensure_ascii=False)
             + "\n这五个顶层字段都必须保留；reasons、facts、next_steps、missing_info 没有条目时填 []。"
             + "不适用项省略仅指列表里的条目，不是省略顶层字段。"
             + "\nnext_steps 有参数时还须添加 params，完整结构为 {target, code, params, evidence}；"
             + "params 的键和值遵循 step_params，fields 和 directions 使用字符串数组，不能写成逗号分隔字符串。"
             + "missing_info 可带 condition，完整结构为 {target, field, purpose, condition, evidence}；"
             + "只有审核未知对象的 elapsed_workdays 待补项使用 condition=if_approved；其他条目省略 condition。"
             + "这些可选键是公开协议的一部分，不是额外字段。"
             + "\n结果字段说明：disposition 按当前主要下一步填写。需人工核实或平台处理时用 escalate；"
             + "缺决定性用户状态且尚无已成立人工条件时用 request_info；可直接解释或指导用户操作时用 answer。"
             + "给出人工处理建议不等于 answer，也不表示操作已经执行。"
             + "reasons 只表达当前处置理由，不罗列尚未成立的可能理由；能力的恒定要求写 facts，"
             + "不能因导出能力要求平台开通，就用 platform_enablement_required 理由断言当前未开通。"
             + "字段目录不是填写清单，无关字段不输出；unknown 仅用于本题相关但确实未知的用户状态。"
             + "时效解释需明确审核通过起算的依据；可用 timer_start 直接表达，"
             + "也可用同一对象有充分引用的字段组合表达同一含义，不必重复信息。"
             + "已提交的每个条目仍须正确且按上述规则有据。已成立的当前退款核实写 verify_refund，"
             + "将来超过上限且仍未到账才核实写 verify_refund_if_overdue_unreceived，并保留 limit_workdays 参数。"
             + "feature_enabled、outbound_ip_allowed、user_can_configure 必须保留 scoped_facts 所列对象结构，"
             + "不能省略 project、environment 或 outbound_ip 等规定键。"
             + "对象包含题面订单号时也原样引用订单号，不将已知对象写成 case。"
             + "每个判断和建议单独给足引用；不会因别的条目引用过政策或用户材料而自动继承证据。"},
            {"role": "user", "content": "\n\n".join(
                f"[{m.reference_id}]\n{m.content}" for m in materials[1:])},
        ]


class SearchAction(ProtocolModel):
    query: str = Field(min_length=1)


SEARCH_SCHEMA = {
    "name": "search", "description": "检索客服政策，查询由策略生成。",
    "parameters": SearchAction.model_json_schema(),
}


class KnowledgeSnippet(VisibleMaterial):
    source_id: str = Field(min_length=1)
    title: str
    section_path: list[str]


class ToolStatus(ProtocolModel):
    reference_id: str
    state: Literal["candidates", "empty", "unavailable", "invalid_arguments"]
    degraded: bool = False
    detail: str = ""


class SearchObservation(ProtocolModel):
    request_number: int = Field(ge=1)
    status: ToolStatus
    snippets: list[KnowledgeSnippet]

    @model_validator(mode="after")
    def validate_references(self):
        prefix = f"search:{self.request_number}:"
        refs = [s.reference_id for s in self.snippets]
        if self.status.reference_id != prefix + "status":
            raise ValueError("工具状态引用与请求编号不一致")
        if len(refs) != len(set(refs)) or any(
            not r.startswith(prefix) or not r[len(prefix):].isdigit() for r in refs
        ):
            raise ValueError("知识片段引用与请求编号不一致或重复")
        if (self.status.state == "candidates") != bool(self.snippets):
            raise ValueError("只有取得候选状态可以携带知识正文")
        return self

    def tool_content(self) -> str:
        return self.model_dump_json()


class SearchTurn(ProtocolModel):
    action: SearchAction | dict
    observation: SearchObservation

    @model_validator(mode="after")
    def validate_action(self):
        if self.observation.status.state != "invalid_arguments":
            self.action = SearchAction.model_validate(self.action)
        return self


class VisibleTrajectory(ProtocolModel):
    task_id: str
    materials: list[VisibleMaterial]
    searches: list[SearchTurn]

    @model_validator(mode="after")
    def validate_order(self):
        refs = [m.reference_id for m in self.materials]
        if len(refs) != len(set(refs)) or any(
            not r.startswith(("user:", "role:")) for r in refs
        ):
            raise ValueError("初始材料只能包含不重复的用户与角色引用")
        if [s.observation.request_number for s in self.searches] != list(
            range(1, len(self.searches) + 1)
        ):
            raise ValueError("检索观察必须按实际请求顺序编号")
        return self


def parse_result(value: str | dict | TerminalResult) -> TerminalResult:
    if isinstance(value, TerminalResult):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return TerminalResult.model_validate(value)
