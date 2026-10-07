"""Provider-neutral LLM seam with normalized prompt-cache telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import logging
from typing import Any, Dict, List, Optional, Protocol

from anthropic import AsyncAnthropic

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - dependency is optional for Anthropic-only installs
    AsyncOpenAI = None  # type: ignore[assignment]

from core.llm_utils import extract_text_content


logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    for method in ("model_dump", "to_dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                result = fn()
                if isinstance(result, dict):
                    extra = getattr(value, "model_extra", None)
                    if isinstance(extra, dict):
                        result.update(extra)
                    return result
            except Exception:
                pass
    return dict(getattr(value, "__dict__", {}) or {})


def _first_int(data: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = data.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class CacheUsage:
    provider: str
    input_tokens: Optional[int] = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_miss_tokens: Optional[int] = None
    eligible: Optional[bool] = None
    status: str = "unknown"
    raw: Dict[str, Any] = field(default_factory=dict)
    output_tokens: Optional[int] = None

    @property
    def total_input_tokens(self) -> Optional[int]:
        if self.provider == "anthropic":
            values = [self.input_tokens, self.cache_read_tokens, self.cache_write_tokens]
            return sum(v for v in values if v is not None) if any(v is not None for v in values) else None
        if self.input_tokens is not None:
            return self.input_tokens
        if self.cache_miss_tokens is not None:
            return self.cache_read_tokens + self.cache_miss_tokens
        return None


@dataclass
class LLMRequest:
    model: str
    stable_prompt: str
    dynamic_prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    cache_identity: str = ""
    cache_mode: str = "automatic"
    max_tokens: int = 1024
    temperature: Optional[float] = None
    retry_incomplete: bool = True

    @property
    def stable_hash(self) -> str:
        return sha256(self.stable_prompt.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ToolCall:
    """Provider-neutral tool request emitted by a model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    usage: CacheUsage
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)
    assistant_message: Optional[Dict[str, Any]] = None


class LLMGateway(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResult: ...


class ProviderGateway:
    provider = "unknown"
    tool_protocol = "unknown"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @staticmethod
    def _metadata(request: LLMRequest, usage: CacheUsage) -> Dict[str, Any]:
        return {
            "provider": usage.provider,
            "model": request.model,
            "cache_identity": request.cache_identity,
            "stable_prefix_hash": request.stable_hash,
            "stable_prefix_chars": len(request.stable_prompt),
            "dynamic_prompt_chars": len(request.dynamic_prompt),
            "cache_mode": request.cache_mode,
            "cache_status": usage.status,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cache_miss_tokens": usage.cache_miss_tokens,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_input_tokens": usage.total_input_tokens,
        }


class AnthropicGateway(ProviderGateway):
    provider = "anthropic"
    tool_protocol = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None,
                 cache_enabled: bool = True, min_stable_chars: int = 4096):
        super().__init__(api_key, model, base_url)
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)
        self.cache_enabled = cache_enabled
        self.min_stable_chars = max(0, int(min_stable_chars))

    async def complete(self, request: LLMRequest) -> LLMResult:
        system: Any = request.stable_prompt
        eligible = (self.cache_enabled and request.cache_mode != "disabled" and
                    len(request.stable_prompt) >= self.min_stable_chars)
        if eligible:
            cache_control = {"type": "ephemeral"}
            if request.cache_mode == "1h":
                cache_control["ttl"] = "1h"
            system = [{"type": "text", "text": request.stable_prompt,
                       "cache_control": cache_control}]
            if request.dynamic_prompt:
                system.append({"type": "text", "text": request.dynamic_prompt})
        elif request.dynamic_prompt:
            system = f"{request.stable_prompt}\n\n{request.dynamic_prompt}"
        kwargs: Dict[str, Any] = {
            "model": request.model, "max_tokens": request.max_tokens,
            "system": system, "messages": request.messages,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "input_schema": item.get("input_schema", {"type": "object", "properties": {}}),
                }
                for item in request.tools
            ]
        response = await self.client.messages.create(**kwargs)
        initial_usage = _as_dict(getattr(response, "usage", None))
        text = extract_text_content(response.content)
        tool_calls = self._tool_calls(response.content)
        retried_response = False
        initial_text_empty = not text.strip()
        stop_reason = getattr(response, "stop_reason", None)
        if request.retry_incomplete and not tool_calls and (initial_text_empty or stop_reason == "max_tokens"):
            retried_response = True
            retry_tokens = max(2048, request.max_tokens * 2)
            logger.warning(
                "Provider response incomplete (text_empty=%s, stop_reason=%s); retrying with max_tokens=%s",
                initial_text_empty,
                stop_reason,
                retry_tokens,
            )
            kwargs["max_tokens"] = retry_tokens
            response = await self.client.messages.create(**kwargs)
            text = extract_text_content(response.content)
            tool_calls = self._tool_calls(response.content)
        raw = _as_dict(getattr(response, "usage", None))
        if retried_response:
            for name in (
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
            ):
                values = [_first_int(initial_usage, name), _first_int(raw, name)]
                if any(value is not None for value in values):
                    raw[name] = sum(value for value in values if value is not None)
        read = _first_int(raw, "cache_read_input_tokens") or 0
        write = _first_int(raw, "cache_creation_input_tokens") or 0
        input_tokens = _first_int(raw, "input_tokens")
        output_tokens = _first_int(raw, "output_tokens")
        status = "hit" if read > 0 else ("miss" if write > 0 else (
            "ineligible" if self.cache_enabled and not eligible else "unknown"
        ))
        usage = CacheUsage("anthropic", input_tokens, read, write, None,
                           True if eligible else (False if self.cache_enabled else None), status, raw,
                           output_tokens)
        metadata = self._metadata(request, usage)
        metadata.update({
            "stop_reason": getattr(response, "stop_reason", None),
            "response_retry": retried_response,
            "empty_response_retry": retried_response and initial_text_empty,
        })
        return LLMResult(
            text,
            usage,
            metadata,
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": response.content},
        )

    @staticmethod
    def _tool_calls(content: Any) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for block in content or []:
            data = _as_dict(block)
            block_type = data.get("type") or getattr(block, "type", None)
            if block_type != "tool_use":
                continue
            name = data.get("name") or getattr(block, "name", "")
            call_id = data.get("id") or getattr(block, "id", "")
            arguments = data.get("input")
            if arguments is None:
                arguments = getattr(block, "input", {})
            calls.append(ToolCall(str(call_id), str(name), dict(arguments or {})))
        return calls


class OpenAIGateway(ProviderGateway):
    provider = "openai"
    tool_protocol = "openai"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, cache_enabled: bool = True):
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is required for OpenAI-compatible providers")
        super().__init__(api_key, model, base_url)
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self.cache_enabled = cache_enabled

    async def complete(self, request: LLMRequest) -> LLMResult:
        system_content: Any = request.stable_prompt
        if request.cache_mode == "explicit" and self.provider == "openai":
            system_content = [{
                "type": "text", "text": request.stable_prompt,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }]
        messages = [{"role": "system", "content": system_content}]
        if request.dynamic_prompt:
            if isinstance(messages[0]["content"], list):
                messages[0]["content"].append({"type": "text", "text": request.dynamic_prompt})
            else:
                messages[0]["content"] += f"\n\n{request.dynamic_prompt}"
        messages.extend(request.messages)
        kwargs: Dict[str, Any] = {"model": request.model, "messages": messages,
                                  "max_tokens": request.max_tokens}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [self._openai_tool(item) for item in request.tools]
        if request.cache_identity and self.cache_enabled and self.provider == "openai":
            kwargs["prompt_cache_key"] = request.cache_identity
        if request.cache_mode == "explicit" and self.provider == "openai":
            kwargs["extra_body"] = {"prompt_cache_options": {"mode": "explicit"}}
        response = await self.client.chat.completions.create(**kwargs)
        raw_response = _as_dict(response)
        raw = _as_dict(raw_response.get("usage") or getattr(response, "usage", None))
        details = _as_dict(raw.get("prompt_tokens_details") or raw.get("input_tokens_details"))
        read = _first_int(details, "cached_tokens") or _first_int(raw, "cached_tokens") or 0
        write = _first_int(details, "cache_write_tokens") or _first_int(raw, "cache_write_tokens") or 0
        total = _first_int(raw, "prompt_tokens", "input_tokens")
        output_tokens = _first_int(raw, "completion_tokens", "output_tokens")
        eligible = ((total >= 1024) if total is not None else None) if self.provider == "openai" else None
        status = "hit" if read > 0 else ("ineligible" if eligible is False else (
            "miss" if total else "unknown"
        ))
        usage = CacheUsage("openai", total, read, write, max(0, (total or 0) - read) if total is not None else None,
                           eligible, status, raw, output_tokens)
        message = getattr(response.choices[0], "message", {})
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        text = extract_text_content(content)
        tool_calls = self._tool_calls(message)
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in tool_calls
            ]
        return LLMResult(
            text,
            usage,
            self._metadata(request, usage),
            tool_calls=tool_calls,
            assistant_message=assistant_message,
        )

    @staticmethod
    def _openai_tool(item: Dict[str, Any]) -> Dict[str, Any]:
        if item.get("type") == "function":
            return item
        return {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item.get("description", ""),
                "parameters": item.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _tool_calls(message: Any) -> List[ToolCall]:
        raw_calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
        calls: List[ToolCall] = []
        for raw_call in raw_calls or []:
            data = _as_dict(raw_call)
            function = _as_dict(data.get("function") or getattr(raw_call, "function", None))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_invalid_json": arguments}
            calls.append(ToolCall(
                str(data.get("id") or getattr(raw_call, "id", "")),
                str(function.get("name", "")),
                dict(arguments or {}),
            ))
        return calls


class DeepSeekGateway(OpenAIGateway):
    """DeepSeek native endpoint; automatic caching needs no request marker."""

    provider = "deepseek"

    async def complete(self, request: LLMRequest) -> LLMResult:
        result = await super().complete(request)
        result.usage.provider = "deepseek"
        raw = result.usage.raw
        details = _as_dict(raw.get("prompt_tokens_details") or raw.get("input_tokens_details"))
        read = _first_int(details, "prompt_cache_hit_tokens") or _first_int(raw, "prompt_cache_hit_tokens")
        miss = _first_int(details, "prompt_cache_miss_tokens") or _first_int(raw, "prompt_cache_miss_tokens")
        if read is not None:
            result.usage.cache_read_tokens = read
            result.usage.cache_miss_tokens = miss
            result.usage.status = "hit" if read > 0 else "miss"
            result.metadata.update({"provider": "deepseek", "cache_read_tokens": read,
                                    "cache_miss_tokens": miss})
        else:
            result.metadata["provider"] = "deepseek"
        return result


class DeepSeekAnthropicGateway(AnthropicGateway):
    """DeepSeek's Anthropic-compatible endpoint with automatic cache telemetry."""

    provider = "deepseek"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, cache_enabled: bool = True):
        super().__init__(api_key, model, base_url, cache_enabled=False)

    async def complete(self, request: LLMRequest) -> LLMResult:
        result = await super().complete(request)
        result.usage.provider = "deepseek"
        raw = result.usage.raw
        read = _first_int(raw, "prompt_cache_hit_tokens")
        miss = _first_int(raw, "prompt_cache_miss_tokens")
        if read is not None:
            result.usage.cache_read_tokens = read
            result.usage.cache_miss_tokens = miss
            result.usage.cache_write_tokens = 0
            result.usage.status = "hit" if read > 0 else "miss"
        else:
            uncached = result.usage.input_tokens
            if uncached is not None:
                result.usage.input_tokens = uncached + result.usage.cache_read_tokens + result.usage.cache_write_tokens
                result.usage.cache_miss_tokens = uncached + result.usage.cache_write_tokens
            result.usage.cache_write_tokens = 0
            result.usage.status = "hit" if result.usage.cache_read_tokens > 0 else (
                "miss" if result.usage.input_tokens is not None else "unknown"
            )
        result.metadata.update({
            "provider": "deepseek",
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_input_tokens": result.usage.total_input_tokens,
            "cache_read_tokens": result.usage.cache_read_tokens,
            "cache_write_tokens": result.usage.cache_write_tokens,
            "cache_miss_tokens": result.usage.cache_miss_tokens,
            "cache_status": result.usage.status,
        })
        return result


def build_gateway(provider: str, api_key: str, model: str, base_url: Optional[str] = None,
                  cache_enabled: bool = True) -> ProviderGateway:
    name = (provider or "anthropic").strip().lower()
    if name == "anthropic":
        return AnthropicGateway(api_key, model, base_url, cache_enabled)
    if name == "openai":
        return OpenAIGateway(api_key, model, base_url, cache_enabled)
    if name == "deepseek":
        endpoint = base_url or "https://api.deepseek.com"
        if endpoint.rstrip("/").endswith("/anthropic"):
            return DeepSeekAnthropicGateway(api_key, model, endpoint, cache_enabled)
        return DeepSeekGateway(api_key, model, endpoint, cache_enabled)
    raise ValueError(f"Unsupported LLM provider: {provider}")
