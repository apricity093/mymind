"""Request-level tool trace persistence with in-memory and Redis adapters."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Protocol


class TraceStore(Protocol):
    def save(self, trace: Dict[str, Any]) -> None: ...

    def get(self, request_id: str) -> Optional[Dict[str, Any]]: ...

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]: ...


class InMemoryTraceStore:
    def __init__(
        self,
        ttl_s: float = 86400,
        max_entries: int = 200,
        clock: Callable[[], float] = time.time,
    ):
        self.ttl_s = max(1.0, float(ttl_s))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._items: "OrderedDict[str, tuple[Dict[str, Any], float]]" = OrderedDict()
        self._lock = RLock()

    def _prune(self) -> None:
        now = self._clock()
        stale = [key for key, (_, expires_at) in self._items.items() if expires_at <= now]
        for key in stale:
            self._items.pop(key, None)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def save(self, trace: Dict[str, Any]) -> None:
        request_id = str(trace.get("request_id", ""))
        if not request_id:
            raise ValueError("trace request_id 不能为空")
        with self._lock:
            self._items[request_id] = (deepcopy(trace), self._clock() + self.ttl_s)
            self._items.move_to_end(request_id)
            self._prune()

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._prune()
            item = self._items.get(request_id)
            return deepcopy(item[0]) if item else None

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            self._prune()
            safe_limit = max(1, min(int(limit or 20), 100, len(self._items)))
            values = [item[0] for item in self._items.values()]
            return deepcopy(list(reversed(values[-safe_limit:])))


class RedisTraceStore:
    """Cross-worker trace storage backed by one key per request and a sorted index."""

    def __init__(self, client: Any, prefix: str = "mymind:trace", ttl_s: int = 86400, max_entries: int = 200):
        self.client = client
        self.prefix = prefix.rstrip(":")
        self.ttl_s = max(1, int(ttl_s))
        self.max_entries = max(1, int(max_entries))
        self.index_key = f"{self.prefix}:recent"

    def _key(self, request_id: str) -> str:
        return f"{self.prefix}:tool:{request_id}"

    def _prune(self, now: Optional[float] = None) -> None:
        timestamp = float(now if now is not None else time.time())
        self.client.zremrangebyscore(self.index_key, "-inf", timestamp - self.ttl_s)
        size = int(self.client.zcard(self.index_key) or 0)
        if size > self.max_entries:
            self.client.zremrangebyrank(self.index_key, 0, size - self.max_entries - 1)

    def save(self, trace: Dict[str, Any]) -> None:
        request_id = str(trace.get("request_id", ""))
        if not request_id:
            raise ValueError("trace request_id 不能为空")
        timestamp = float(trace.get("timestamp_epoch") or time.time())
        payload = json.dumps(trace, ensure_ascii=False, sort_keys=True)
        pipeline = self.client.pipeline(transaction=True)
        pipeline.set(self._key(request_id), payload, ex=self.ttl_s)
        pipeline.zadd(self.index_key, {request_id: timestamp})
        pipeline.expire(self.index_key, self.ttl_s)
        pipeline.execute()
        self._prune(timestamp)

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._key(request_id))
        if raw is None:
            self.client.zrem(self.index_key, request_id)
            return None
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        self._prune()
        safe_limit = max(1, min(int(limit or 20), 100))
        request_ids = self.client.zrevrange(self.index_key, 0, safe_limit - 1)
        traces: List[Dict[str, Any]] = []
        for raw_id in request_ids:
            request_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            trace = self.get(request_id)
            if trace is not None:
                traces.append(trace)
        return traces
