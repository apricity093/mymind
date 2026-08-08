"""Thread-safe application and provider cache metrics."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any, Dict, Optional

try:
    from prometheus_client import Counter

    CACHE_REQUESTS = Counter(
        "mymind_cache_requests_total", "Cache requests by layer and outcome",
        ("layer", "provider", "model", "namespace", "outcome"),
    )
    CACHE_TOKENS = Counter(
        "mymind_prompt_cache_tokens_total", "Provider prompt-cache token telemetry",
        ("provider", "model", "kind"),
    )
except Exception:  # pragma: no cover
    CACHE_REQUESTS = CACHE_TOKENS = None


class CacheMetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latency: Dict[str, list[float]] = defaultdict(list)

    def record_application(self, namespace: str, outcome: str, latency_ms: Optional[float] = None) -> None:
        with self._lock:
            self._counters[f"application.{namespace}.{outcome}"] += 1
            if latency_ms is not None:
                self._latency[f"application.{namespace}"].append(float(latency_ms))
        if CACHE_REQUESTS is not None:
            CACHE_REQUESTS.labels("application", "", "", namespace, outcome).inc()

    def record_provider(self, provider: str, model: str, usage: Any, latency_ms: Optional[float] = None) -> None:
        prefix = f"provider.{provider}.{model}"
        with self._lock:
            self._counters[f"{prefix}.requests"] += 1
            status = getattr(usage, "status", "unknown")
            self._counters[f"{prefix}.{status}"] += 1
            self._counters[f"{prefix}.read_tokens"] += int(getattr(usage, "cache_read_tokens", 0) or 0)
            self._counters[f"{prefix}.write_tokens"] += int(getattr(usage, "cache_write_tokens", 0) or 0)
            miss = getattr(usage, "cache_miss_tokens", None)
            if miss is not None:
                self._counters[f"{prefix}.miss_tokens"] += int(miss)
            total = getattr(usage, "total_input_tokens", None)
            if total is not None:
                self._counters[f"{prefix}.input_tokens"] += int(total)
            if latency_ms is not None:
                self._latency[prefix].append(float(latency_ms))
        if CACHE_REQUESTS is not None:
            CACHE_REQUESTS.labels("provider", provider, model, "", getattr(usage, "status", "unknown")).inc()
        if CACHE_TOKENS is not None:
            CACHE_TOKENS.labels(provider, model, "read").inc(int(getattr(usage, "cache_read_tokens", 0) or 0))
            CACHE_TOKENS.labels(provider, model, "write").inc(int(getattr(usage, "cache_write_tokens", 0) or 0))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            result: Dict[str, Any] = {"counters": counters, "providers": {}}
            for key, value in counters.items():
                if not key.startswith("provider."):
                    continue
                parts = key.split(".")
                if len(parts) < 3:
                    continue
                provider_model = ".".join(parts[1:3])
                result["providers"].setdefault(provider_model, {})[".".join(parts[3:])] = value
            for key, values in self._latency.items():
                if values:
                    result.setdefault("latency_ms", {})[key] = {
                        "count": len(values), "p50": sorted(values)[len(values) // 2],
                        "max": max(values),
                    }
            return result


class RedisCacheMetricsCollector(CacheMetricsCollector):
    """Cross-process counter aggregation; latency samples remain process-local."""

    def __init__(self, client: Any, key: str = "mymind:cache_metrics") -> None:
        super().__init__()
        self.client = client
        self.key = key

    def _increment(self, field: str, value: int = 1) -> None:
        try:
            self.client.hincrby(self.key, field, int(value))
        except Exception:
            with self._lock:
                self._counters[field] += int(value)

    def record_application(self, namespace: str, outcome: str, latency_ms: Optional[float] = None) -> None:
        self._increment(f"application.{namespace}.{outcome}")
        if latency_ms is not None:
            with self._lock:
                self._latency[f"application.{namespace}"].append(float(latency_ms))

    def record_provider(self, provider: str, model: str, usage: Any, latency_ms: Optional[float] = None) -> None:
        prefix = f"provider.{provider}.{model}"
        self._increment(f"{prefix}.requests")
        self._increment(f"{prefix}.{getattr(usage, 'status', 'unknown')}")
        self._increment(f"{prefix}.read_tokens", int(getattr(usage, "cache_read_tokens", 0) or 0))
        self._increment(f"{prefix}.write_tokens", int(getattr(usage, "cache_write_tokens", 0) or 0))
        miss = getattr(usage, "cache_miss_tokens", None)
        if miss is not None:
            self._increment(f"{prefix}.miss_tokens", int(miss))
        total = getattr(usage, "total_input_tokens", None)
        if total is not None:
            self._increment(f"{prefix}.input_tokens", int(total))
        if latency_ms is not None:
            with self._lock:
                self._latency[prefix].append(float(latency_ms))

    def snapshot(self) -> Dict[str, Any]:
        try:
            raw = self.client.hgetall(self.key)
            counters = {str(k): int(v) for k, v in raw.items()}
        except Exception:
            return super().snapshot()
        result: Dict[str, Any] = {"counters": counters, "providers": {}}
        for key, value in counters.items():
            if key.startswith("provider."):
                parts = key.split(".")
                provider_model = ".".join(parts[1:3])
                result["providers"].setdefault(provider_model, {})[".".join(parts[3:])] = value
        return result


class ObservedCacheStore:
    """CacheStore decorator that preserves fail-open behavior and records outcomes."""

    def __init__(self, store: Any, metrics: CacheMetricsCollector):
        self.store = store
        self.metrics = metrics

    def get(self, namespace: str, key: str) -> Any:
        try:
            value = self.store.get(namespace, key)
        except Exception:
            self.metrics.record_application(namespace, "error")
            raise
        self.metrics.record_application(namespace, "hit" if value is not None else "miss")
        return value

    def set(self, namespace: str, key: str, value: Any, ttl: float) -> None:
        try:
            return self.store.set(namespace, key, value, ttl)
        except Exception:
            self.metrics.record_application(namespace, "error")
            raise

    def invalidate_namespace(self, namespace: str) -> int:
        try:
            result = self.store.invalidate_namespace(namespace)
            self.metrics.record_application(namespace, "invalidated")
            return result
        except Exception:
            self.metrics.record_application(namespace, "error")
            raise
