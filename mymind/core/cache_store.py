"""Small cache seam shared by runtime code and deterministic experiments."""

from __future__ import annotations

import json
from hashlib import sha256
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, Hashable, Optional, Protocol


def freeze_cache_key(value: Any) -> Hashable:
    """Represent JSON parameters as immutable values without serialization."""
    if isinstance(value, dict):
        return "object", tuple((key, freeze_cache_key(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return "array", tuple(freeze_cache_key(item) for item in value)
    return type(value).__name__, value


class CacheStore(Protocol):
    def get_generation(self, namespace: str) -> int: ...

    def get(self, namespace: str, key: Hashable) -> Optional[Any]: ...

    def set(
        self, namespace: str, key: Hashable, value: Any, ttl: float,
        *, generation: Optional[int] = None,
    ) -> None: ...

    def invalidate_namespace(self, namespace: str) -> int: ...


class InMemoryCacheStore:
    """Bounded TTL/LRU adapter with an injectable monotonic clock."""

    def __init__(self, max_entries: int = 5000, clock: Callable[[], float] = time.monotonic):
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._items: "OrderedDict[tuple, tuple[Any, float]]" = OrderedDict()
        self._generations: Dict[str, int] = {}
        self._lock = RLock()

    def _full_key(self, namespace: str, key: Hashable) -> tuple:
        generation = self._generations.get(namespace, 0)
        return namespace, generation, key

    def get_generation(self, namespace: str) -> int:
        with self._lock:
            return self._generations.get(namespace, 0)

    def get(self, namespace: str, key: Hashable) -> Optional[Any]:
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

    def set(
        self, namespace: str, key: Hashable, value: Any, ttl: float,
        *, generation: Optional[int] = None,
    ) -> None:
        if ttl <= 0:
            return
        with self._lock:
            if generation is not None and generation != self._generations.get(namespace, 0):
                return
            full_key = self._full_key(namespace, key)
            self._items[full_key] = (value, self._clock() + ttl)
            self._items.move_to_end(full_key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def invalidate_namespace(self, namespace: str) -> int:
        with self._lock:
            self._generations[namespace] = self._generations.get(namespace, 0) + 1
            stale = [key for key in self._items if key[0] == namespace]
            for key in stale:
                self._items.pop(key, None)
            return self._generations[namespace]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


class RedisCacheStore:
    """JSON Redis adapter with generation-checked writes and stale-key deletion."""

    _SET_IF_CURRENT = """
        local generation = tonumber(redis.call('GET', KEYS[1]) or '0')
        if generation ~= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
        return 1
    """

    def __init__(self, client: Any, prefix: str = "mymind:cache"):
        self.client = client
        self.prefix = prefix.rstrip(":")

    def _generation_key(self, namespace: str) -> str:
        return f"{self.prefix}:generation:{namespace}"

    def get_generation(self, namespace: str) -> int:
        raw = self.client.get(self._generation_key(namespace))
        return int(raw or 0)

    def _key(self, namespace: str, key: Hashable, generation: int) -> str:
        if not isinstance(key, str):
            key = sha256(json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        return f"{self.prefix}:{namespace}:{generation}:{key}"

    def get(self, namespace: str, key: Hashable) -> Optional[Any]:
        raw = self.client.get(self._key(namespace, key, self.get_generation(namespace)))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

    def set(
        self, namespace: str, key: Hashable, value: Any, ttl: float,
        *, generation: Optional[int] = None,
    ) -> None:
        if ttl <= 0:
            return
        if generation is None:
            generation = self.get_generation(namespace)
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True)
        self.client.eval(
            self._SET_IF_CURRENT, 2,
            self._generation_key(namespace), self._key(namespace, key, generation),
            generation, payload, max(1, int(ttl)),
        )

    def invalidate_namespace(self, namespace: str) -> int:
        generation = int(self.client.incr(self._generation_key(namespace)))
        prefix = f"{self.prefix}:{namespace}:"
        stale = []
        for key in self.client.scan_iter(match=f"{prefix}*", count=500):
            text_key = key.decode("utf-8") if isinstance(key, bytes) else key
            if int(text_key[len(prefix):].split(":", 1)[0]) < generation:
                stale.append(key)
        if stale:
            self.client.delete(*stale)
        return generation
