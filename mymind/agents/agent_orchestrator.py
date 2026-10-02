"""Multi-Agent routing, role contracts, tool execution and request tracing."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    billing_tools,
    build_shared_rag_tools,
    escalation_tools,
    general_tools,
    redact_arguments,
    technical_tools,
    validate_tool_input,
)
from agents.trace_store import InMemoryTraceStore, TraceStore
from core.cache_metrics import CacheMetricsCollector
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_gateway import CacheUsage, LLMGateway, LLMRequest, LLMResult, build_gateway
from core.llm_utils import extract_text_content
from core.prompt_cache import PromptCachePolicy

logger = logging.getLogger(__name__)


def _tool_protocol_unsupported(ex: Exception) -> bool:
    message = str(ex).lower()
    mentions_tooling = "tool" in message or "function call" in message
    mentions_support = any(term in message for term in ("not support", "unsupported", "not available", "unknown field"))
    return mentions_tooling and mentions_support


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


class AgentType(Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


@dataclass(frozen=True)
class AgentFeatureConfig:
    profile_enabled: bool = False
    tool_use_enabled: bool = False
    escalation_enabled: bool = False
    composer_enabled: bool = False
    trace_enabled: bool = False
    trace_api_enabled: bool = False
    knowledge_tool_mode: str = "disabled"
    max_tool_rounds: int = 3

    def __post_init__(self) -> None:
        if self.knowledge_tool_mode not in {"disabled", "supplemental", "tool_only"}:
            raise ValueError("knowledge_tool_mode 必须是 disabled、supplemental 或 tool_only")
        if not 1 <= int(self.max_tool_rounds) <= 10:
            raise ValueError("max_tool_rounds 必须在 1 到 10 之间")

    @classmethod
    def from_env(cls) -> "AgentFeatureConfig":
        return cls(
            profile_enabled=_env_bool("AGENT_PROFILE_ENABLED", False),
            tool_use_enabled=_env_bool("AGENT_TOOL_USE_ENABLED", False),
            escalation_enabled=_env_bool("AGENT_ESCALATION_ENABLED", False),
            composer_enabled=_env_bool("AGENT_COMPOSER_ENABLED", False),
            trace_enabled=_env_bool("AGENT_TRACE_ENABLED", False),
            trace_api_enabled=_env_bool("TRACE_API_ENABLED", False),
            knowledge_tool_mode=os.getenv("KNOWLEDGE_TOOL_MODE", "disabled").strip().lower(),
            max_tool_rounds=_env_int("AGENT_MAX_TOOL_ROUNDS", 3),
        )

    @classmethod
    def for_variant(cls, variant: str) -> "AgentFeatureConfig":
        name = variant.strip().upper()
        levels = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5}
        if name not in levels:
            raise ValueError(f"未知 Agent 实验变体: {variant}")
        level = levels[name]
        return cls(
            profile_enabled=level >= 1,
            tool_use_enabled=level >= 2,
            escalation_enabled=level >= 3,
            composer_enabled=level >= 4,
            trace_enabled=level >= 5,
            trace_api_enabled=False,
            knowledge_tool_mode="disabled",
        )


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class AgentStats:
    total: int = 0
    success: int = 0
    total_ms: float = 0.0
    monitor_penalty: float = 0.0
    cache_status: str = "unknown"

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    success: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    escalate: bool = False
    cache_metadata: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Request:
    message: str
    user_id: str
    conv_id: str
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None
    intent: Optional[IntentCategory] = None
    intent_group: str = ""
    urgency: Optional[UrgencyLevel] = None
    intent_confidence: float = 0.0
    entities: Dict[str, List[str]] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_already_loaded: bool = False
    tool_result_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)


@dataclass
class OrchestratorResult:
    request_id: str
    response: str
    agent_type: AgentType
    intent: Optional[IntentCategory]
    escalated: bool = False
    latency_ms: float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent, *self.supporting_agents]

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


def _aggregate_usage(usages: List[CacheUsage]) -> Optional[CacheUsage]:
    if not usages:
        return None
    provider = usages[-1].provider
    inputs = [usage.input_tokens for usage in usages]
    misses = [usage.cache_miss_tokens for usage in usages]
    outputs = [usage.output_tokens for usage in usages]
    statuses = [usage.status for usage in usages]
    return CacheUsage(
        provider=provider,
        input_tokens=sum(value for value in inputs if value is not None) if any(value is not None for value in inputs) else None,
        cache_read_tokens=sum(usage.cache_read_tokens for usage in usages),
        cache_write_tokens=sum(usage.cache_write_tokens for usage in usages),
        cache_miss_tokens=sum(value for value in misses if value is not None) if any(value is not None for value in misses) else None,
        eligible=any(usage.eligible is True for usage in usages) or None,
        status="hit" if "hit" in statuses else (statuses[-1] if statuses else "unknown"),
        raw={"rounds": len(usages)},
        output_tokens=sum(value for value in outputs if value is not None) if any(value is not None for value in outputs) else None,
    )


class BaseAgent:
    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        prompt_cache_policy: Optional[PromptCachePolicy] = None,
        gateway: Optional[LLMGateway] = None,
        metrics: Optional[CacheMetricsCollector] = None,
        features: Optional[AgentFeatureConfig] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self._client = client
        self._gateway = gateway
        self._metrics = metrics
        self._features = features or AgentFeatureConfig.from_env()
        self.profile = profile or self.profile
        self._model = self.profile.model or model
        self._skill_manager = skill_manager
        self._prompt_cache_policy = prompt_cache_policy or PromptCachePolicy()
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self.stats = AgentStats()

    def local_tools(self) -> Dict[str, AgentToolSpec]:
        return {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        if not self._features.tool_use_enabled:
            return {}
        tools = dict(self._shared_tools)
        tools.update(self.local_tools())
        return tools

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        started = time.monotonic()
        self.stats.total += 1
        try:
            content, cache_metadata, tools_used, tool_traces, usage = await self._call_llm(req)
            latency_ms = (time.monotonic() - started) * 1000
            if self._metrics is not None and usage is not None:
                self._metrics.record_provider(usage.provider, self._model, usage, latency_ms)
            if usage is not None:
                self.stats.cache_status = usage.status
            self.stats.success += 1
            self.stats.total_ms += latency_ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=latency_ms,
                escalate=self._needs_escalation(content),
                cache_metadata=cache_metadata,
                tools_used=tools_used,
                tool_traces=tool_traces,
            )
        except Exception as ex:
            latency_ms = (time.monotonic() - started) * 1000
            self.stats.total_ms += latency_ms
            logger.exception("%s Agent 处理失败", self.agent_type.value)
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=latency_ms,
                tool_traces=[{
                    "agent_type": self.agent_type.value,
                    "tool_name": "",
                    "success": False,
                    "error_code": type(ex).__name__,
                }],
            )

    async def _call_llm(
        self, req: Request
    ) -> tuple[str, Dict[str, Any], List[str], List[Dict[str, Any]], Optional[CacheUsage]]:
        messages = self._messages(req)
        stable_prompt = self._stable_prompt()
        dynamic_prompt = self._dynamic_prompt(req)
        tools = self.get_tools()
        usages: List[CacheUsage] = []
        tools_used: List[str] = []
        traces: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}

        if self._gateway is None:
            system, metadata = self._prompt_cache_policy.build_system(stable_prompt, dynamic_prompt)
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": self.profile.max_tokens if self._features.profile_enabled else 1024,
                "system": system,
                "messages": messages,
            }
            if self._features.profile_enabled:
                kwargs["temperature"] = self.profile.temperature
            response = await self._client.messages.create(
                **kwargs,
            )
            usage = getattr(response, "usage", None)
            metadata.update({
                "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
                "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            })
            return extract_text_content(response.content), metadata, [], [], None

        for round_index in range(self._features.max_tool_rounds):
            gateway_request = LLMRequest(
                model=self._model,
                stable_prompt=stable_prompt,
                dynamic_prompt=dynamic_prompt,
                messages=messages,
                tools=[spec.llm_schema() for spec in tools.values()] if tools else None,
                cache_identity=f"{self._model}:{self.agent_type.value}:prompt-v2",
                cache_mode="automatic",
                max_tokens=self.profile.max_tokens if self._features.profile_enabled else 1024,
                temperature=self.profile.temperature if self._features.profile_enabled else None,
            )
            try:
                result: LLMResult = await self._gateway.complete(gateway_request)
            except Exception as ex:
                if not tools or not _tool_protocol_unsupported(ex):
                    raise
                logger.warning("Provider 不支持工具协议，当前 Agent 回退为无工具生成: %s", ex)
                tools = {}
                gateway_request.tools = None
                result = await self._gateway.complete(gateway_request)
                traces.append({
                    "agent_type": self.agent_type.value,
                    "tool_name": "",
                    "tool_call_id": "",
                    "input": {},
                    "success": False,
                    "result_success": None,
                    "latency_ms": 0.0,
                    "cached": False,
                    "reranked": False,
                    "degraded": True,
                    "error_code": "provider_tool_unsupported",
                })
            usages.append(result.usage)
            metadata = dict(result.metadata)
            calls = list(getattr(result, "tool_calls", []) or [])
            if not calls:
                metadata["tool_rounds"] = round_index
                return result.text, metadata, list(dict.fromkeys(tools_used)), traces, _aggregate_usage(usages)

            assistant_message = getattr(result, "assistant_message", None)
            if assistant_message:
                messages.append(assistant_message)
            tool_results: List[Dict[str, Any]] = []
            for call in calls:
                spec = tools.get(call.name)
                tool_started = time.monotonic()
                execution_success = True
                result_success: Optional[bool] = None
                error_code = ""
                output: Any
                if spec is None:
                    execution_success = False
                    error_code = "unauthorized_tool"
                    output = {"success": False, "error": "工具不在当前 Agent 白名单中"}
                else:
                    try:
                        validate_tool_input(spec, call.arguments)
                        output = spec.handler(req, call.arguments)
                        if inspect.isawaitable(output):
                            output = await output
                        tools_used.append(call.name)
                        if isinstance(output, dict) and "success" in output:
                            result_success = bool(output.get("success"))
                    except ValueError as ex:
                        execution_success = False
                        error_code = "invalid_tool_arguments"
                        output = {"success": False, "error": str(ex)[:200]}
                    except Exception as ex:
                        execution_success = False
                        error_code = "tool_execution_error"
                        output = {"success": False, "error": str(ex)[:200]}
                latency_ms = (time.monotonic() - tool_started) * 1000
                traces.append({
                    "agent_type": self.agent_type.value,
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "input": redact_arguments(call.arguments),
                    "success": execution_success,
                    "result_success": result_success,
                    "latency_ms": round(latency_ms, 3),
                    "cached": bool(output.get("cached") or output.get("request_cached")) if isinstance(output, dict) else False,
                    "reranked": bool(output.get("reranked")) if isinstance(output, dict) else False,
                    "degraded": bool(output.get("degraded")) if isinstance(output, dict) else False,
                    "error_code": error_code or ("tool_result_error" if result_success is False else ""),
                })
                tool_results.append({"id": call.id, "output": output})
            self._append_tool_results(messages, tool_results)

        metadata["tool_rounds"] = self._features.max_tool_rounds
        metadata["tool_loop_exhausted"] = True
        return (
            "工具调用已达到安全轮数上限，暂未能完成自动处理。请补充必要信息或转人工客服。",
            metadata,
            list(dict.fromkeys(tools_used)),
            traces,
            _aggregate_usage(usages),
        )

    def _append_tool_results(self, messages: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> None:
        protocol = getattr(self._gateway, "tool_protocol", "openai")
        if protocol == "anthropic":
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": item["id"],
                        "content": json.dumps(item["output"], ensure_ascii=False),
                    }
                    for item in results
                ],
            })
            return
        for item in results:
            messages.append({
                "role": "tool",
                "tool_call_id": item["id"],
                "content": json.dumps(item["output"], ensure_ascii=False),
            })

    def _messages(self, req: Request) -> List[Dict[str, Any]]:
        clean = lambda value: str(value).encode("utf-8", errors="ignore").decode("utf-8")
        messages: List[Dict[str, Any]] = []
        if req.context:
            messages.extend([
                {"role": "user", "content": f"[背景信息]\n{clean(req.context)}"},
                {"role": "assistant", "content": "好的，我已了解背景信息。"},
            ])
        if req.entities:
            messages.extend([
                {"role": "user", "content": f"[结构化实体]\n{clean(json.dumps(req.entities, ensure_ascii=False))}"},
                {"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"},
            ])
        messages.append({"role": "user", "content": clean(req.message)})
        return messages

    def _stable_prompt(self) -> str:
        if not self._features.profile_enabled:
            return self.system_prompt
        profile = self.profile
        return (
            f"{self.system_prompt}\n\n[角色契约]\n"
            f"角色：{profile.role}\n职责：{profile.mission}\n"
            f"处理流程：{' -> '.join(profile.workflow)}\n"
            f"可用输入：{'；'.join(profile.input_contract)}\n"
            f"输出要求：{'；'.join(profile.output_contract)}\n"
            f"升级条件：{'；'.join(profile.handoff_conditions) or '按通用规则处理'}\n"
            f"工具范围：{'、'.join(profile.tool_scope) or '仅当前请求上下文'}\n"
            "不得声称完成未提供的查询、修改、退款或工单操作；证据不足时明确需要核验。"
        )

    def _dynamic_prompt(self, req: Request) -> str:
        parts: List[str] = []
        if self._features.profile_enabled:
            parts.append("[角色输入包]\n" + self._build_role_packet(req))
        if self._skill_manager is not None:
            skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
            if skill_prompt:
                parts.append("[动态 Skills]\n" + skill_prompt)
        return "\n\n".join(parts)

    def _build_role_packet(self, req: Request) -> str:
        return json.dumps({
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }, ensure_ascii=False)

    @staticmethod
    def _needs_escalation(content: str) -> bool:
        return any(keyword in content for keyword in ("转人工", "人工客服", "escalate", "specialist", "无法处理"))


class GeneralAgent(BaseAgent):
    agent_type = AgentType.GENERAL
    profile = AgentProfile(
        role="通用客服分诊与首轮接待",
        mission="回答基础问题、澄清需求，并识别专业 Agent 或人工升级需要。",
        workflow=("复述诉求", "判断业务范围", "回答或补充信息", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及资金、权限、隐私或复杂投诉", "用户明确要求人工"),
        tool_scope=("inspect_request_context", "suggest_required_fields", "search_knowledge_base"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = "你是 mymind 智能客服。友好、简洁地回答；超出能力时明确建议转接专业客服。"

    def local_tools(self) -> Dict[str, AgentToolSpec]:
        return general_tools()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet.update({"triage_targets": ["technical", "billing", "escalation"], "response_mode": "answer_or_clarify"})
        return json.dumps(packet, ensure_ascii=False)


class TechnicalAgent(BaseAgent):
    agent_type = AgentType.TECHNICAL
    profile = AgentProfile(
        role="技术故障诊断与排障",
        mission="基于错误码、环境和复现信息缩小根因范围并给出低风险排查步骤。",
        workflow=("确认现象", "判断影响范围", "排查网络/权限/配置/依赖", "验证", "判断升级"),
        input_contract=("错误码", "发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
        output_contract=("现象复述", "可能原因", "编号步骤", "验证方式", "待补信息"),
        handoff_conditions=("生产大面积不可用", "数据或权限异常", "需要后台日志或人工操作"),
        tool_scope=("lookup_error_code", "build_diagnostic_plan", "search_knowledge_base"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = "你是技术支持专家，提供清晰、低风险、可验证的排障步骤。"

    def local_tools(self) -> Dict[str, AgentToolSpec]:
        return technical_tools()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["diagnostic_fields"] = {
            "error_codes": req.entities.get("error_code", []),
            "risk_boundary": "不得索要密码、验证码或完整密钥；不得建议破坏性操作",
        }
        return json.dumps(packet, ensure_ascii=False)


class BillingAgent(BaseAgent):
    agent_type = AgentType.BILLING
    profile = AgentProfile(
        role="账单核验与售后处理",
        mission="区分扣款、退款、发票和订阅场景，说明事实与人工审核边界。",
        workflow=("确认场景", "收集核验字段", "区分金额", "说明路径与时效", "判断升级"),
        input_contract=("订单号", "金额与币种", "支付时间", "支付渠道", "用户期望", "知识库上下文"),
        output_contract=("核验信息", "当前可判断内容", "下一步路径", "时效边界"),
        handoff_conditions=("实际退款或补偿", "重复扣款", "发票作废重开", "企业合同或大额订单"),
        tool_scope=("check_billing_fields", "compare_amounts", "search_knowledge_base"),
        temperature=0.0,
        max_tokens=1100,
    )
    system_prompt = "你是账单服务专家。财务问题必须准确，实际退款或账单修改需要人工审核。"

    def local_tools(self) -> Dict[str, AgentToolSpec]:
        return billing_tools()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["verification_fields"] = {
            "order_id": req.entities.get("order_id", []),
            "amount": req.entities.get("amount", []),
            "date": req.entities.get("date", []),
            "risk_boundary": "不得承诺退款成功、立即到账或直接修改账单",
        }
        return json.dumps(packet, ensure_ascii=False)


class EscalationAgent(BaseAgent):
    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="人工升级与交接",
        mission="整理已知上下文、标记优先级并告知后续，不执行未经授权的业务操作。",
        workflow=("确认升级原因", "整理信息", "标记优先级", "生成交接摘要"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("升级原因", "已知信息摘要", "待补信息", "保守后续说明"),
        handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
        tool_scope=("create_handoff_summary",),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责人工升级交接，不要模拟已完成后台操作。"

    def local_tools(self) -> Dict[str, AgentToolSpec]:
        return escalation_tools()

    async def handle(self, req: Request) -> AgentResponse:
        started = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        urgency = req.urgency.name if req.urgency else "UNKNOWN"
        safe_entities = {key: [str(item)[:80] for item in value[:5]] for key, value in (req.entities or {}).items()}
        content = (
            "我已将这个问题标记为人工升级处理。\n\n"
            f"升级原因：意图={intent}，紧急度={urgency}\n"
            f"已记录信息：{json.dumps(safe_entities, ensure_ascii=False)}\n"
            "请勿发送密码、短信验证码或完整支付凭证；人工客服会根据会话记录继续核验。"
        )
        latency_ms = (time.monotonic() - started) * 1000
        self.stats.success += 1
        self.stats.total_ms += latency_ms
        return AgentResponse(self.agent_type, content, True, latency_ms=latency_ms, escalate=True)


class ResponseComposer:
    def __init__(
        self,
        gateway: LLMGateway,
        model: str,
        skill_manager: Optional[Any] = None,
        metrics: Optional[CacheMetricsCollector] = None,
    ):
        self._gateway = gateway
        self._model = model
        self._skill_manager = skill_manager
        self._metrics = metrics

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [item for item in responses if item.success and item.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content
        evidence = "\n\n".join(f"[{item.agent_type.value}]\n{item.content}" for item in successful)
        dynamic = ""
        if self._skill_manager is not None:
            dynamic = self._skill_manager.prompt_for(req.message, "general") or ""
        started = time.monotonic()
        try:
            result = await self._gateway.complete(LLMRequest(
                model=self._model,
                stable_prompt=(
                    "你是客服 Response Composer。以主 Agent 为主，合并专业结论、去重并处理冲突；"
                    "不得补造订单、退款、后台查询或工单结果；冲突时明确需要核验。只输出中文用户回复。"
                ),
                dynamic_prompt=dynamic,
                messages=[{"role": "user", "content": f"用户问题：{req.message}\n候选结果：\n{evidence}"}],
                cache_identity=f"{self._model}:composer:v1",
                max_tokens=1000,
                temperature=0.1,
            ))
            if self._metrics is not None:
                self._metrics.record_provider(result.usage.provider, self._model, result.usage, (time.monotonic() - started) * 1000)
            if result.text.strip():
                return result.text.strip()
        except Exception:
            logger.exception("Response Composer 失败，使用确定性合并")
        return "\n\n".join(item.content if index == 0 else f"补充说明：\n{item.content}" for index, item in enumerate(successful))


class AgentOrchestrator:
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
        IntentCategory.BILLING: AgentType.BILLING,
        IntentCategory.REFUND: AgentType.BILLING,
        IntentCategory.INVOICE: AgentType.BILLING,
        IntentCategory.PAYMENT_ISSUE: AgentType.BILLING,
        IntentCategory.ACCOUNT: AgentType.BILLING,
        IntentCategory.ACCOUNT_SECURITY: AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        prompt_cache_enabled: bool = False,
        prompt_cache_min_chars: int = 4096,
        provider: str = "anthropic",
        metrics: Optional[CacheMetricsCollector] = None,
        gateway: Optional[LLMGateway] = None,
        features: Optional[AgentFeatureConfig] = None,
        trace_store: Optional[TraceStore] = None,
        rag_tool_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)
        self.features = features or AgentFeatureConfig.from_env()
        self._skill_manager = skill_manager
        self.metrics = metrics or CacheMetricsCollector()
        cache_policy = PromptCachePolicy(prompt_cache_enabled, prompt_cache_min_chars)
        self.gateway = gateway or build_gateway(provider, api_key, model, base_url, cache_enabled=prompt_cache_enabled or provider != "anthropic")
        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model, gateway=self.gateway)
        self._trace_store = trace_store or (InMemoryTraceStore() if self.features.trace_enabled else None)

        def configured(agent_cls: type[BaseAgent]) -> BaseAgent:
            profile = agent_cls.profile
            env_name = f"MYMIND_{agent_cls.agent_type.value.upper()}_MODEL"
            legacy_name = f"ECHOMIND_{agent_cls.agent_type.value.upper()}_MODEL"
            override = os.getenv(env_name, "").strip() or os.getenv(legacy_name, "").strip()
            selected = replace(profile, model=override) if override else profile
            return agent_cls(client, model, skill_manager, cache_policy, self.gateway, self.metrics, self.features, selected)

        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [configured(GeneralAgent)],
            AgentType.TECHNICAL: [configured(TechnicalAgent)],
            AgentType.BILLING: [configured(BillingAgent)],
        }
        if self.features.escalation_enabled:
            self._pool[AgentType.ESCALATION] = [configured(EscalationAgent)]
        composer_model = os.getenv("MYMIND_COMPOSER_MODEL", "").strip() or os.getenv("ECHOMIND_COMPOSER_MODEL", "").strip() or model
        self._composer = ResponseComposer(self.gateway, composer_model, skill_manager, self.metrics)
        self.set_rag_tool_manager(rag_tool_manager)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_trace_store(self, trace_store: Optional[TraceStore]) -> None:
        self._trace_store = trace_store

    def set_rag_tool_manager(self, tool_manager: Optional[Any]) -> None:
        shared = {}
        if self.features.tool_use_enabled and self.features.knowledge_tool_mode in {"supplemental", "tool_only"}:
            shared = build_shared_rag_tools(tool_manager)
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(shared)

    async def recognize_intent(self, message: str, history: Optional[List[Dict[str, str]]] = None):
        return await self._intent_recognizer.recognize(message, history=history)

    async def run(self, req: Request) -> OrchestratorResult:
        started = time.monotonic()
        if req.intent is None:
            recognized = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent = recognized.intent
            req.intent_group = recognized.intent_group
            req.urgency = recognized.urgency
            req.intent_confidence = recognized.confidence
            req.entities = recognized.entities

        if self._needs_clarification(req):
            return self._finish(OrchestratorResult(
                request_id=req.request_id,
                response="我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？",
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                latency_ms=(time.monotonic() - started) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            ))

        decision = self._route_decision(req)
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        response = await self._execute(req, decision.primary_agent)
        escalated = response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in {
            IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF,
        }
        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - started) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
        )
        return self._finish(result)

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        started = time.monotonic()
        responses = await asyncio.gather(*[self._execute(req, item) for item in decision.agent_types], return_exceptions=True)
        valid = [item for item in responses if isinstance(item, AgentResponse)]
        if self.features.composer_enabled:
            combined = await self._composer.compose(req, valid)
        else:
            successful = [item for item in valid if item.success]
            combined = "\n\n".join(
                f"[{item.agent_type.value} - {'主处理' if item.agent_type == decision.primary_agent else '辅助处理'}]\n{item.content}"
                for item in successful
            ) or "抱歉，所有 Agent 均处理失败。"
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=any(item.escalate for item in valid),
            latency_ms=(time.monotonic() - started) * 1000,
            agent_types=[item.agent_type for item in valid if item.success] or decision.agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            tools_used=list(dict.fromkeys(name for item in valid for name in item.tools_used)),
            tool_traces=[trace for item in valid for trace in item.tool_traces],
        )
        return self._finish(result)

    def _finish(self, result: OrchestratorResult) -> OrchestratorResult:
        if not self.features.trace_enabled or self._trace_store is None:
            return result
        now = time.time()
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "timestamp_epoch": now,
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else result.agent_type.value,
            "supporting_agents": [item.value for item in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "escalated": result.escalated,
            "latency_ms": round(result.latency_ms, 3),
        }
        try:
            self._trace_store.save(trace)
        except Exception:
            logger.exception("保存请求 Trace 失败")
        return result

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self._trace_store.get(request_id) if self._trace_store is not None else None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._trace_store.recent(limit) if self._trace_store is not None else []

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        if urgency == UrgencyLevel.CRITICAL and self._pool.get(AgentType.ESCALATION):
            return AgentType.ESCALATION
        target = self._INTENT_ROUTING.get(intent) if intent else None
        return target if target and self._pool.get(target) else AgentType.GENERAL

    def _route_decision(self, req: Request) -> RoutingDecision:
        if req.urgency == UrgencyLevel.CRITICAL or req.intent in {IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF}:
            target = AgentType.ESCALATION if self._pool.get(AgentType.ESCALATION) else AgentType.GENERAL
            return RoutingDecision(target, reason="紧急度或意图触发人工升级节点", confidence=max(req.intent_confidence, 0.8))
        scores = self._domain_scores(req)
        available = {kind: score for kind, score in scores.items() if kind == AgentType.GENERAL or self._pool.get(kind)}
        ordered = sorted(available.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return RoutingDecision(AgentType.GENERAL, reason="无可用专属 Agent，降级到 GeneralAgent", confidence=0.1)
        primary, primary_score = ordered[0]
        supporting = [kind for kind, score in ordered[1:] if kind != AgentType.GENERAL and score >= 0.45]
        for kind in self._collaboration_targets(req):
            if kind != primary and kind not in supporting:
                supporting.append(kind)
        return RoutingDecision(primary, supporting, self._routing_reason(req, available, primary, supporting), round(min(primary_score, 1.0), 3))

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        message = req.message.lower()
        technical_terms = ("崩溃", "报错", "error", "crash", "无法登录", "登录失败", "重启", "超时", "401", "403", "404", "500")
        billing_terms = ("退款", "扣款", "扣了", "账单", "支付", "发票", "订阅", "金额", "refund", "invoice")
        targets: List[AgentType] = []
        if any(term in message for term in technical_terms) and self._pool.get(AgentType.TECHNICAL):
            targets.append(AgentType.TECHNICAL)
        if any(term in message for term in billing_terms) and self._pool.get(AgentType.BILLING):
            targets.append(AgentType.BILLING)
        return targets

    @staticmethod
    def _domain_scores(req: Request) -> Dict[AgentType, float]:
        message = req.message.lower()
        scores = {AgentType.GENERAL: 0.1, AgentType.TECHNICAL: 0.0, AgentType.BILLING: 0.0}
        if req.intent in {IntentCategory.QUERY, IntentCategory.ORDER_STATUS, IntentCategory.LOGISTICS, IntentCategory.REQUEST, IntentCategory.COMPLAINT, IntentCategory.GREETING, IntentCategory.FEEDBACK, IntentCategory.OTHER}:
            scores[AgentType.GENERAL] += 0.55
        if req.intent in {IntentCategory.TECHNICAL, IntentCategory.TECHNICAL_LOGIN, IntentCategory.TECHNICAL_CRASH}:
            scores[AgentType.TECHNICAL] += 0.75
        if req.intent in {IntentCategory.BILLING, IntentCategory.ACCOUNT, IntentCategory.ACCOUNT_SECURITY, IntentCategory.REFUND, IntentCategory.INVOICE, IntentCategory.PAYMENT_ISSUE}:
            scores[AgentType.BILLING] += 0.75
        technical = ["崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500", "401", "验证码"]
        billing = ["退款", "退货", "扣款", "发票", "账单", "支付", "订阅", "refund", "invoice", "多扣"]
        general = ["订单", "物流", "快递", "配送", "会员", "积分", "咨询", "帮助"]
        scores[AgentType.TECHNICAL] += min(0.45, sum(term in message for term in technical) * 0.18)
        scores[AgentType.BILLING] += min(0.45, sum(term in message for term in billing) * 0.18)
        scores[AgentType.GENERAL] += min(0.35, sum(term in message for term in general) * 0.12)
        if req.entities.get("error_code"):
            scores[AgentType.TECHNICAL] += 0.2
        if req.entities.get("amount"):
            scores[AgentType.BILLING] += 0.15
        if req.entities.get("order_id"):
            scores[AgentType.GENERAL] += 0.1
        return {kind: round(score, 3) for kind, score in scores.items()}

    @staticmethod
    def _routing_reason(req: Request, scores: Dict[AgentType, float], primary: AgentType, supporting: List[AgentType]) -> str:
        score_text = ", ".join(f"{kind.value}={score:.2f}" for kind, score in sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return (
            f"intent={req.intent.value if req.intent else 'unknown'}, group={req.intent_group or 'unknown'}, "
            f"primary={primary.value}, supporting={','.join(item.value for item in supporting) or 'none'}, scores=[{score_text}]"
        )

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        return req.intent == IntentCategory.OTHER and len((req.message or "").strip()) > 2 and req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        agents = self._pool.get(agent_type, [])
        return max(agents, key=lambda item: item.stats.routing_score()) if agents else None

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        agent = self._best_agent(agent_type) or self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(AgentType.GENERAL, "服务暂时不可用，请稍后重试。", False)
        response = await agent.handle(req)
        if not response.success and agent_type not in {AgentType.GENERAL, AgentType.ESCALATION}:
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)
        return response

    def get_stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for kind, agents in self._pool.items():
            for index, agent in enumerate(agents):
                result[f"{kind.value}_{index}"] = {
                    "total": agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms": round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                    "prompt_cache": {"provider": getattr(agent._gateway, "provider", "anthropic"), "status": agent.stats.cache_status},
                }
        result["cache_metrics"] = self.metrics.snapshot()
        result["features"] = {
            "profile_enabled": self.features.profile_enabled,
            "tool_use_enabled": self.features.tool_use_enabled,
            "escalation_enabled": self.features.escalation_enabled,
            "composer_enabled": self.features.composer_enabled,
            "trace_enabled": self.features.trace_enabled,
            "trace_api_enabled": self.features.trace_api_enabled,
            "knowledge_tool_mode": self.features.knowledge_tool_mode,
        }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        for kind, agents in self._pool.items():
            for index, agent in enumerate(agents):
                agent.stats.monitor_penalty = min(max(penalties.get(f"{kind.value}_{index}", 0.0), 0.0), 0.9)
