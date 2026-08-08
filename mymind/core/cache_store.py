"""Small cache seam shared by runtime code and deterministic experiments."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, Optional, Protocol


class CacheStore(Protocol):
    def get(self, namespace: str, key: str) -> Optional[Any]: ...

    def set(self, namespace: str, key: str, value: Any, ttl: float) -> None: ...

    def invalidate_namespace(self, namespace: str) -> int: ...


class InMemoryCacheStore:
    """Bounded TTL/LRU adapter with an injectable monotonic clock."""

    def __init__(self, max_entries: int = 5000, clock: Callable[[], float] = time.monotonic):
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._items: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._generations: Dict[str, int] = {}
        self._lock = RLock()

    def _full_key(self, namespace: str, key: str) -> str:
        generation = self._generations.get(namespace, 0)
        return f"{namespace}:{generation}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        with self._lock:
            full_key = self._full_key(namespace, key)
            item = self._items.get(full_key)
            if item is None:
                return None
            value, expires_at = item
            if self._clock() >= expires_at:
                self._items.pop(full_key, None)
                return None
            self._items.move_to_end(full_key)
            return value

    def set(self, namespace: str, key: str, value: Any, ttl: float) -> None:
        if ttl <= 0:
            return
        with self._lock:
            full_key = self._full_key(namespace, key)
            self._items[full_key] = (value, self._clock() + ttl)
            self._items.move_to_end(full_key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def invalidate_namespace(self, namespace: str) -> int:
        with self._lock:
            self._generations[namespace] = self._generations.get(namespace, 0) + 1
            prefix = f"{namespace}:"
            stale = [key for key in self._items if key.startswith(prefix)]
            for key in stale:
                self._items.pop(key, None)
            return self._generations[namespace]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


class RedisCacheStore:
    """JSON Redis adapter; namespace generations make invalidation atomic and cheap."""

    def __init__(self, client: Any, prefix: str = "mymind:cache"):
        self.client = client
        self.prefix = prefix.rstrip(":")

    def _generation_key(self, namespace: str) -> str:
        return f"{self.prefix}:generation:{namespace}"

    def _generation(self, namespace: str) -> int:
        raw = self.client.get(self._generation_key(namespace))
        return int(raw or 0)

    def _key(self, namespace: str, key: str) -> str:
        return f"{self.prefix}:{namespace}:{self._generation(namespace)}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        raw = self.client.get(self._key(namespace, key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, key: str, value: Any, ttl: float) -> None:
        if ttl <= 0:
            return
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True)
        self.client.set(self._key(namespace, key), payload, ex=max(1, int(ttl)))

    def invalidate_namespace(self, namespace: str) -> int:
        return int(self.client.incr(self._generation_key(namespace)))
