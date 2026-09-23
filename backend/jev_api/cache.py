"""Cache + rate-limit backend: Redis when JEV_REDIS_URL is set and reachable, else in-process memory.

The in-memory fallback keeps the app runnable on any laptop without Redis; it is per-process,
so multi-worker deployments should use Redis.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


class MemoryCache:
    backend = "memory"

    def __init__(self, max_items: int = 5000) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._max = max_items

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            if item[0] < time.time():
                self._data.pop(key, None)
                return None
            return json.loads(item[1])

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                now = time.time()
                for k in [k for k, (exp, _) in self._data.items() if exp < now] or list(self._data)[
                    : self._max // 10
                ]:
                    self._data.pop(k, None)
            self._data[key] = (time.time() + ttl, json.dumps(value, default=str))

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for k in [k for k in self._data if k.startswith(prefix)]:
                self._data.pop(k, None)

    def incr_window(self, key: str, window: int) -> int:
        with self._lock:
            exp, raw = self._data.get(key, (0.0, "0"))
            now = time.time()
            count = int(raw) + 1 if exp >= now else 1
            self._data[key] = (exp if exp >= now else now + window, str(count))
            return count

    def ping(self) -> bool:
        return True


class RedisCache:
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(
            url, socket_timeout=0.5, socket_connect_timeout=1.0, decode_responses=True
        )
        self._r.ping()

    def get(self, key: str) -> Any | None:
        raw = self._r.get(key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._r.set(key, json.dumps(value, default=str), ex=ttl)

    def delete_prefix(self, prefix: str) -> None:
        for k in self._r.scan_iter(match=f"{prefix}*", count=500):
            self._r.delete(k)

    def incr_window(self, key: str, window: int) -> int:
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window, nx=True)
        count, _ = pipe.execute()
        return int(count)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            log.warning("redis ping failed", exc_info=True)
            return False


Cache = MemoryCache | RedisCache


def build_cache(redis_url: str | None) -> Cache:
    if redis_url:
        try:
            cache = RedisCache(redis_url)
            log.info("cache backend: redis")
            return cache
        except Exception:
            log.warning("redis unavailable at startup, falling back to in-memory cache", exc_info=True)
    return MemoryCache()
