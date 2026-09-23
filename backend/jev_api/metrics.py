"""In-process metrics registry: thread-safe counters and bounded histogram summaries.

Counts are per process and reset on restart (served by GET /admin/metrics). A multi-worker
deployment reports one worker's view per request; export to Prometheus if that matters.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np


class Summary:
    """count/sum/min/max over all observations plus p50/p95 over the most recent `window`."""

    def __init__(self, window: int = 1000) -> None:
        self.count = 0
        self.total = 0.0
        self.min: float | None = None
        self.max: float | None = None
        self.last: float | None = None
        self._recent: deque[float] = deque(maxlen=window)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)
        self.last = value
        self._recent.append(value)

    def to_dict(self) -> dict[str, Any]:
        recent = np.asarray(self._recent, dtype=float)
        return {
            "count": self.count,
            "mean": round(self.total / self.count, 2) if self.count else None,
            "p50": round(float(np.percentile(recent, 50)), 2) if len(recent) else None,
            "p95": round(float(np.percentile(recent, 95)), 2) if len(recent) else None,
            "min": None if self.min is None else round(self.min, 2),
            "max": None if self.max is None else round(self.max, 2),
            "last": None if self.last is None else round(self.last, 2),
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self._counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._summaries: dict[str, dict[str, Summary]] = defaultdict(dict)
        self._gauges: dict[str, dict[str, Any]] = defaultdict(dict)

    def inc(self, group: str, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[group][name] += n

    def observe(self, group: str, name: str, value: float) -> None:
        with self._lock:
            s = self._summaries[group].get(name)
            if s is None:
                s = self._summaries[group][name] = Summary()
            s.observe(float(value))

    def set(self, group: str, name: str, value: Any) -> None:
        with self._lock:
            self._gauges[group][name] = value

    def record_request(self, method: str, route: str, status: int, ms: float) -> None:
        # route is the matched template (/movies/{movie_id}), never the raw path, so the number of
        # label values stays bounded
        with self._lock:
            self._counters["http_requests_by_status"][f"{status // 100}xx"] += 1
            self._counters["http_requests_by_route"][f"{method} {route}"] += 1
            if status >= 500:
                self._counters["http_errors"][f"{method} {route}"] += 1
            s = self._summaries["http_latency_ms"].get("all")
            if s is None:
                s = self._summaries["http_latency_ms"]["all"] = Summary()
            s.observe(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {"uptime_s": round(time.time() - self.started_at, 1)}
            for group, counters in self._counters.items():
                out[group] = dict(sorted(counters.items()))
            for group, sums in self._summaries.items():
                out[group] = {k: s.to_dict() for k, s in sorted(sums.items())}
            for group, gauges in self._gauges.items():
                out.setdefault(group, {}).update(gauges)
            return out

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._summaries.clear()
            self._gauges.clear()
            self.started_at = time.time()


metrics = MetricsRegistry()
