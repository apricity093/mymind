"""Deterministic, role-scoped tools used by the Agent runtime."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler

    def llm_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def validate_tool_input(spec: AgentToolSpec, arguments: Any) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    schema = spec.input_schema
    properties = schema.get("properties", {})
    for field_name in schema.get("required", []):
        if field_name not in arguments:
            raise ValueError(f"缺少必需参数: {field_name}")
    unknown = set(arguments) - set(properties)
    if unknown and schema.get("additionalProperties") is False:
        raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        expected_type = type_map.get(expected)
        numeric_boolean = expected in {"number", "integer"} and isinstance(value, bool)
        if expected_type and (numeric_boolean or not isinstance(value, expected_type)):
            raise ValueError(f"参数 {key} 类型错误，期望 {expected}")


_SENSITIVE_KEYS = {
    "password", "passwd", "token", "access_token", "refresh_token", "api_key",
    "secret", "verification_code", "验证码", "支付凭证", "credential",
}

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(api[_ -]?key|password|passwd|token|secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\b(验证码|支付凭证)\s*[:：=]\s*[^\s,;，；]+"),
)


def _redact_text(value: str, limit: int = 160) -> str:
    safe = value
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        safe = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", safe)
    return safe[:limit]


def redact_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Keep traces useful without retaining credentials or unbounded payloads."""
    redacted: Dict[str, Any] = {}
    for key, value in arguments.items():
        lowered = str(key).lower()
        if lowered in _SENSITIVE_KEYS or any(part in lowered for part in ("password", "token", "secret", "key")):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, str):
            redacted[key] = _redact_text(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            redacted[key] = value
        elif isinstance(value, list):
            redacted[key] = [_redact_text(str(item), 80) for item in value[:10]]
        else:
            redacted[key] = _redact_text(str(value))
    return redacted


def inspect_request_context(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(arguments.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {"intent": intent, "required_fields": fields, "known_entities": req.entities or {}}


def lookup_error_code(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    code = str(arguments.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(code, ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]))
    return {"error_code": code, "meaning": meaning, "next_steps": steps, "server_log_checked": False}


def build_diagnostic_plan(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    reproduced = bool(arguments.get("reproduced", False))
    steps = ["复现并记录完整错误信息", "确认网络、DNS、代理和证书", "确认版本、配置和权限"]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": str(arguments.get("environment", "unknown"))[:80],
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(arguments.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    first = float(arguments["amount_a"])
    second = float(arguments["amount_b"])
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": req.request_id,
        "reason": str(arguments.get("reason", "需要人工客服继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    async def search_knowledge_base(req: Request, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = str(arguments.get("query") or req.message or "").strip()
        top_k = max(1, min(int(arguments.get("top_k", 5) or 5), 20))
        cache_key = json.dumps({"query": query, "top_k": top_k}, ensure_ascii=True, sort_keys=True)
        if cache_key in req.tool_result_cache:
            cached = dict(req.tool_result_cache[cache_key])
            cached["request_cached"] = True
            return cached
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}
        result = await tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
        payload = {
            "success": bool(getattr(result, "success", False)),
            "query": query,
            "top_k": top_k,
            "results": getattr(result, "data", []) if getattr(result, "success", False) else [],
            "cached": bool(getattr(result, "cached", False)),
            "reranked": bool(getattr(result, "reranked", False)),
            "degraded": bool(getattr(result, "degraded", False)),
            "error": getattr(result, "error", None),
        }
        if payload["degraded"]:
            payload["knowledge_status"] = "degraded"
        elif payload["success"]:
            payload["knowledge_status"] = "used" if payload["results"] else "empty"
        elif payload["error"] == "所有子查询均无结果":
            payload["knowledge_status"] = "empty"
        else:
            payload["knowledge_status"] = "error"
        req.tool_result_cache[cache_key] = dict(payload)
        return payload

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回文档片段；备用索引结果带 degraded 标记，工具故障提示不作为知识证据。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context", "查看当前请求的脱敏意图、紧急度、实体和上下文可用性。",
            {"focus": {"type": "string"}}, inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields", "根据当前意图给出下一轮需要补充的最少字段。",
            {}, suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code", "解释常见 HTTP 错误码，不声称读取服务端日志。",
            {"error_code": {"type": "string"}}, lookup_error_code, required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan", "根据环境和复现情况生成低风险排障顺序。",
            {"environment": {"type": "string"}, "reproduced": {"type": "boolean"}},
            build_diagnostic_plan, required=["environment", "reproduced"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields", "检查账单核验字段，不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string"}}, check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts", "计算两笔明确金额的差值，不判断重复扣款或退款结论。",
            {"amount_a": {"type": "number"}, "amount_b": {"type": "number"}},
            compare_amounts, required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary", "生成脱敏的人工交接摘要，不创建真实工单。",
            {"reason": {"type": "string"}}, create_handoff_summary,
        )
    }
