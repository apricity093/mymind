"""独立于聊天网关的 Qwen/Gemini 文本向量接入。"""
from __future__ import annotations

import logging
import asyncio
import math
import os
import threading
import time
from collections import Counter, deque, OrderedDict
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str = ""
    dimensions: int = 768
    api_key: str = field(default="", repr=False)
    base_url: str = ""

    @classmethod
    def from_env(cls):
        settings = cls(os.getenv("EMBEDDING_MODEL", "").strip(),
                       int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
                       os.getenv("EMBEDDING_API_KEY", "").strip(),
                       os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/"))
        if settings.model and not settings.api_key:
            raise ValueError("配置 EMBEDDING_MODEL 时必须设置 EMBEDDING_API_KEY（当前不存在）")
        if settings.model not in {"", "gemini-embedding-001", "qwen3.7-text-embedding-flash"}:
            raise ValueError("本版本支持 gemini-embedding-001 与 qwen3.7-text-embedding-flash")
        if not 128 <= settings.dimensions <= 3072:
            raise ValueError("EMBEDDING_DIMENSIONS 必须介于 128 和 3072")
        if settings.model == "qwen3.7-text-embedding-flash" and settings.dimensions not in {256, 512, 768, 1024}:
            raise ValueError("Qwen Flash 支持的维度为 256、512、768、1024")
        return settings


class EmbeddingError(RuntimeError):
    """仅保存异常类别/状态码，不转存 HTTP 正文、凭证或请求文本。"""


class EmbeddingTelemetry:
    def __init__(self, model="gemini-embedding-001"):
        self.model = model
        self.counts = Counter()
        self.events = deque(maxlen=100)
        self._lock = threading.Lock()

    def record(self, operation, identity, outcome, reason=""):
        event = dict(operation=operation, identity=identity, outcome=outcome, reason=reason,
                     model=self.model)
        with self._lock:
            self.counts[f"{operation}:{outcome}"] += 1
            self.events.append(event)
        log = logger.warning if outcome in {"google_failed", "provider_failed", "degraded"} else logger.info
        if outcome == "failed":
            log = logger.error
        log("embedding model=%s operation=%s identity=%s outcome=%s reason=%s",
            self.model, operation, identity, outcome, reason)

    def snapshot(self):
        with self._lock:
            return {"counts": dict(self.counts), "events": list(self.events)}


class TextEmbedding:
    def __init__(self, settings, telemetry=None, transport=None, min_interval=0):
        self.settings = settings
        self.telemetry = telemetry or EmbeddingTelemetry(settings.model or "disabled")
        self._client = httpx.Client(timeout=30, transport=transport)
        self._transport = transport
        self._count_supported = None
        self.usage = Counter()
        self.min_interval = min_interval
        self._next_request = 0
        self._deadline = threading.local()

    @contextmanager
    def query_budget(self, seconds):
        """能力探测、计数、限速等待与编码共用一个在线查询截止时间。"""
        previous = getattr(self._deadline, "end", None)
        self._deadline.end = time.monotonic() + seconds
        try:
            yield
        finally:
            self._deadline.end = previous

    def _remaining(self):
        end = getattr(self._deadline, "end", None)
        remaining = end - time.monotonic() if end is not None else 30.0
        if remaining <= 0:
            raise EmbeddingError("query_budget_exhausted")
        return remaining

    @property
    def enabled(self):
        return bool(self.settings.model)

    @property
    def identity(self):
        return f"{self.settings.model}:{self.settings.dimensions}"

    def _send(self, method, url, headers, payload=None, units=1):
        if self.min_interval:
            delay = self._next_request - time.monotonic()
            if delay > 0:
                before = time.monotonic()
                remaining = self._remaining()
                time.sleep(min(delay, remaining))
                self.usage["throttle_sleep_ms"] += (time.monotonic() - before) * 1000
                self._remaining()
            self._next_request = time.monotonic() + self.min_interval * units
        self.usage["requests"] += 1
        remaining = self._remaining()
        try:
            kwargs = dict(headers=headers, json=payload,
                          timeout=min(30.0, remaining))
            if getattr(self._deadline, "end", None) is not None:
                # 在线截止时间取消实际网络读写；同步调用者已经在工作线程。
                async def bounded_request():
                    async with httpx.AsyncClient(transport=self._transport) as client:
                        return await asyncio.wait_for(client.request(method, url, **kwargs), self._remaining())
                loop = asyncio.new_event_loop()
                try:
                    response = loop.run_until_complete(bounded_request())
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    # 已取消网络任务；关闭循环不等待系统DNS工作线程结束。
                    loop.close()
            else:
                response = self._client.request(method, url, **kwargs)
        except asyncio.TimeoutError:
            raise EmbeddingError("query_budget_exhausted") from None
        except httpx.HTTPError as ex:
            raise EmbeddingError(type(ex).__name__) from None
        self._remaining()
        return response

    def close(self):
        self._client.close()


class GeminiEmbedding(TextEmbedding):
    INPUT_LIMIT = 2048
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings, **kwargs):
        if settings.model not in {"", "gemini-embedding-001"}:
            raise ValueError("Gemini 实验仅支持 gemini-embedding-001，请使用匹配模型的接入")
        super().__init__(settings, **kwargs)

    @property
    def counting_mode(self):
        return "google_countTokens" if self._count_supported else "utf8_byte_upper_bound"

    def _request(self, method, suffix, payload=None, units=1):
        response = self._send(method, f"{self.ENDPOINT}/models/{self.settings.model}{suffix}",
                              {"x-goog-api-key": self.settings.api_key}, payload, units)
        if response.is_error:
            reason = f"HTTP_{response.status_code}"
            if response.status_code == 429:
                # 只读取 Google 配额字段，不转存可能包含请求文本的错误 message。
                try:
                    details = response.json().get("error", {}).get("details", [])
                    quotas = [v.get("quotaId", "") for d in details for v in d.get("violations", [])]
                    delay = next((d["retryDelay"] for d in details if "retryDelay" in d), "")
                    reason += f" quota={','.join(quotas)} retry={delay}"
                except (ValueError, TypeError):
                    pass
            raise EmbeddingError(reason)
        return response.json()

    @lru_cache(maxsize=4096)
    def count_tokens(self, text):
        """探测模型公开能力；不支持计数时使用保守字节上界，不声称是实际 token。"""
        if not text:
            return 0
        if self._count_supported is None:
            model = self._request("GET", "")
            self._count_supported = "countTokens" in model.get("supportedGenerationMethods", [])
        if self._count_supported:
            result = self._request("POST", ":countTokens", {"contents": [{"parts": [{"text": text}]}]})
            self.usage["count_requests"] += 1
            # Google 的 JSON protobuf 响应会省略值为 0 的 totalTokens。
            return int(result.get("totalTokens", 0))
        # 001 没有公开的本地 tokenizer 契约。UTF-8 字节预算比文本 token 更保守。
        return len(text.encode("utf-8"))

    def embed(self, texts, task, titles=None):
        if task not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"}:
            raise ValueError("不支持的 embedding 任务")
        vectors, requests = [], []
        for i, text in enumerate(texts):
            title = titles[i] if titles else ""
            submitted = f"{title}\n{text}" if title else text
            if self.count_tokens(submitted) + 16 > self.INPUT_LIMIT:
                raise EmbeddingError("input_budget_exceeded")
            payload = {"content": {"parts": [{"text": text}]}, "taskType": task,
                       "outputDimensionality": self.settings.dimensions}
            if title and task == "RETRIEVAL_DOCUMENT":
                payload["title"] = title
            requests.append({"model": f"models/{self.settings.model}", **payload})
        for offset in range(0, len(requests), 16):
            batch = requests[offset:offset + 16]
            if len(batch) == 1:
                result = self._request("POST", ":embedContent", batch[0])
                embeddings = [result["embedding"]]
            else:
                result = self._request("POST", ":batchEmbedContents", {"requests": batch}, units=len(batch))
                embeddings = result["embeddings"]
            if len(embeddings) != len(batch):
                raise EmbeddingError("unexpected_embedding_count")
            for embedding in embeddings:
                values = embedding["values"]
                if len(values) != self.settings.dimensions:
                    raise EmbeddingError("unexpected_dimensions")
                norm = math.sqrt(sum(v * v for v in values))
                if not norm or not math.isfinite(norm):
                    raise EmbeddingError("invalid_vector")
                vectors.append([v / norm for v in values])
                self.usage[f"{task}:texts"] += 1
            # API 没有返回用量时不伪造实际 token 数。
            usage = result.get("usageMetadata", {})
            if "promptTokenCount" in usage:
                self.usage["reported_tokens"] += usage["promptTokenCount"]
        return vectors


class QwenEmbedding(TextEmbedding):
    INPUT_LIMIT = 128000
    ENDPOINT = "https://maas.qianwenaiapi.com/compatible-mode/v1"

    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        self._token_cache = OrderedDict()

    @property
    def base_url(self):
        return self.settings.base_url or self.ENDPOINT

    @property
    def identity(self):
        return f"qwen:{self.settings.model}:{self.settings.dimensions}:{self.base_url}"

    @property
    def counting_mode(self):
        return "qwen_single_input_usage_tokens"

    def count_tokens(self, text):
        """单文本请求的真实 tokenizer 用量；计数产生的编码也计入实际成本。"""
        if not text:
            return 0
        cached = self._token_cache.get(text)
        if cached is not None:
            self.usage["token_count_cache_hits"] += 1
            return cached[0]
        if len(text.encode("utf-8")) + 16 > self.INPUT_LIMIT:
            raise EmbeddingError("input_budget_exceeded")
        self.usage["count_requests"] += 1
        vectors, tokens = self._encode_batch([text], "TOKEN_COUNT")
        if tokens is None:
            raise EmbeddingError("token_usage_missing")
        self._token_cache[text] = (int(tokens), vectors[0])
        if len(self._token_cache) > 4096:
            self._token_cache.popitem(last=False)
        return int(tokens)

    def _encode_batch(self, batch, task):
        response = self._send("POST", f"{self.base_url}/embeddings",
            {"Authorization": f"Bearer {self.settings.api_key}"},
            {"model": self.settings.model, "input": batch, "dimensions": self.settings.dimensions,
             "encoding_format": "float"}, units=len(batch))
        if response.is_error:
            raise EmbeddingError(f"HTTP_{response.status_code}")
        try:
            result = response.json()
            items = sorted(result["data"], key=lambda item: item["index"])
            if [item["index"] for item in items] != list(range(len(batch))):
                raise EmbeddingError("unexpected_embedding_count")
            vectors = []
            for item in items:
                values = item["embedding"]
                if len(values) != self.settings.dimensions:
                    raise EmbeddingError("unexpected_dimensions")
                norm = math.sqrt(sum(v*v for v in values))
                if not norm or not math.isfinite(norm):
                    raise EmbeddingError("invalid_vector")
                vectors.append([v/norm for v in values])
                self.usage[f"{task}:texts"] += 1
            tokens = result.get("usage", {}).get("total_tokens")
            if tokens is not None:
                self.usage["reported_tokens"] += tokens
            return vectors, tokens
        except (KeyError, TypeError, ValueError):
            raise EmbeddingError("invalid_embedding_response") from None

    def embed(self, texts, task, titles=None):
        if task not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"}:
            raise ValueError("不支持的 embedding 任务")
        inputs = [f"{titles[i]}\n{text}" if titles and titles[i] and task == "RETRIEVAL_DOCUMENT" else text
                  for i, text in enumerate(texts)]
        vectors = [None] * len(inputs)
        pending = []
        for i,text in enumerate(inputs):
            cached = self._token_cache.get(text) if task == "RETRIEVAL_DOCUMENT" else None
            if cached is not None:
                vectors[i] = list(cached[1])
                self.usage["document_count_vector_reuse"] += 1
            else:
                pending.append((i,text))
        batches, batch, size = [], [], 0
        for item in pending:
            count = len(item[1].encode("utf-8")) + 16
            if count > self.INPUT_LIMIT:
                raise EmbeddingError("input_budget_exceeded")
            if batch and (len(batch) == 20 or size + count > self.INPUT_LIMIT):
                batches.append(batch)
                batch, size = [], 0
            batch.append(item)
            size += count
        if batch:
            batches.append(batch)
        for batch in batches:
            encoded,_ = self._encode_batch([text for _,text in batch],task)
            for (i,_),vector in zip(batch,encoded):
                vectors[i] = vector
        return vectors


def build_embedding(settings=None, **kwargs):
    settings = settings or EmbeddingSettings.from_env()
    implementation = QwenEmbedding if settings.model == "qwen3.7-text-embedding-flash" else GeminiEmbedding
    return implementation(settings, **kwargs)
