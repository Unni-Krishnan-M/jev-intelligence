"""Holds the loaded RecommendationEngine and swaps it atomically when the active model changes."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from jev_api.metrics import metrics
from jev_ml.engine import RecommendationEngine
from jev_ml.registry import REGISTRY_FILE, active_version, set_active

log = logging.getLogger(__name__)


class EngineHolder:
    """The serving engine of this process.

    Swaps are load-first: a version that fails to load never replaces the serving one. Promotions made
    by another process (another API worker, scripts/retrain.py, the scheduler) are followed by polling
    models/registry.json: at most every ``JEV_GOVERNANCE_ENGINE_POLL_SECONDS`` a request stats the file,
    and when its mtime changed and the active pointer names another version, that version is loaded in
    a background thread while the current one keeps serving (docs/RETRAINING_AND_MODEL_GOVERNANCE.md).
    """

    def __init__(self, models_dir: Path, poll_seconds: float | None = None) -> None:
        self.models_dir = models_dir
        self._engine: RecommendationEngine | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        if poll_seconds is None:
            from jev_api.config import get_settings

            poll_seconds = get_settings().governance_engine_poll_seconds
        self.poll_seconds = float(poll_seconds)
        self._registry_mtime: float | None = self._mtime()
        self._next_poll = 0.0
        self._follower: threading.Thread | None = None

    def _mtime(self) -> float | None:
        try:
            return (self.models_dir / REGISTRY_FILE).stat().st_mtime
        except OSError:
            return None

    def load(self, version: str | None = None) -> RecommendationEngine | None:
        self._registry_mtime = self._mtime()
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

    def activate(
        self, version: str, action: str = "activate", actor: str | None = None
    ) -> RecommendationEngine:
        """Load ``version`` (raises if it is broken: nothing changes), point the registry at it, swap.
        ``action`` (activate | promote | rollback) is recorded in the registry history."""
        engine = RecommendationEngine(self.models_dir / version)  # load first: never swap to a broken model
        set_active(version, self.models_dir, action=action, actor=actor)
        with self._lock:
            self._engine = engine
            self._registry_mtime = self._mtime()
        self.last_error = None
        metrics.inc("model", f"swaps_{action}")
        log.info("%s model %s", action, version)
        return engine

    def follow_registry(self, wait: bool = False) -> None:
        """Load the registry's active version if another process changed it (see the class doc)."""
        mtime = self._mtime()
        if mtime is None or mtime == self._registry_mtime:
            return
        self._registry_mtime = mtime
        active = active_version(self.models_dir)
        current = self._engine.version if self._engine is not None else None
        if active is None or active == current:
            return
        if self._follower is not None and self._follower.is_alive():
            return
        log.info("registry names %s (serving %s): loading it", active, current)
        metrics.inc("model", "registry_follows")
        self._follower = threading.Thread(target=self.load, args=(active,), name="engine-follow", daemon=True)
        self._follower.start()
        if wait:
            self._follower.join()

    @property
    def engine(self) -> RecommendationEngine | None:
        if self.poll_seconds > 0:
            now = time.monotonic()
            if now >= self._next_poll:
                self._next_poll = now + self.poll_seconds
                try:
                    self.follow_registry()
                except Exception:  # never fail a request because of the follower
                    log.exception("registry follow failed")
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
