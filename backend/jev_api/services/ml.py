"""Holds the loaded RecommendationEngine and swaps it atomically when the active model changes."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from jev_api.metrics import metrics
from jev_ml.engine import RecommendationEngine
from jev_ml.registry import active_version, set_active

log = logging.getLogger(__name__)


class EngineHolder:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._engine: RecommendationEngine | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    def load(self, version: str | None = None) -> RecommendationEngine | None:
        version = version or active_version(self.models_dir)
        if version is None:
            self.last_error = "no trained model registered (run scripts/train_models.py)"
            log.warning(self.last_error)
            return None
        try:
            engine = RecommendationEngine(self.models_dir / version)
        except (OSError, ValueError, KeyError) as exc:
            self.last_error = f"failed to load model {version}: {exc}"
            metrics.inc("model", "load_errors")
            log.exception("model load failed")
            return None
        with self._lock:
            self._engine = engine
        self.last_error = None
        return engine

    def activate(self, version: str) -> RecommendationEngine:
        engine = RecommendationEngine(self.models_dir / version)  # load first: never swap to a broken model
        set_active(version, self.models_dir)
        with self._lock:
            self._engine = engine
        log.info("activated model %s", version)
        return engine

    @property
    def engine(self) -> RecommendationEngine | None:
        return self._engine


# a calibration file may carry its fitted mapping (thousands of points); the API shows the metrics
_MAX_CALIBRATION_LIST = 20


def engine_calibration(engine: RecommendationEngine | None) -> dict[str, Any] | None:
    """The serving model's recommendation-confidence calibration metrics (ECE, Brier, n, method,
    fitted_on; docs/intelligence.md, section 9.2), or None when the model has no calibration.

    Read defensively (`engine.calibration`, else `engine.health()["calibration"]`), so the API works
    with engines built before calibration existed."""
    if engine is None:
        return None
    cal: Any = getattr(engine, "calibration", None)
    if cal is None:
        try:
            cal = engine.health().get("calibration")
        except Exception:
            log.exception("engine health failed while reading calibration")
            return None
    if cal is not None and not isinstance(cal, dict):
        cal = getattr(cal, "metrics", None) if not hasattr(cal, "to_dict") else cal.to_dict()
    if not isinstance(cal, dict) or not cal:
        return None
    pruned = _prune(cal)
    return pruned if isinstance(pruned, dict) else None


def _prune(value: Any) -> Any:
    """Drop long lists (the fitted mapping) at any depth; keep metrics and small bin tables."""
    if isinstance(value, dict):
        return {
            str(k): _prune(v)
            for k, v in value.items()
            if not (isinstance(v, list | tuple) and len(v) > _MAX_CALIBRATION_LIST)
        }
    if isinstance(value, list | tuple):
        return [_prune(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
