"""Holds the loaded RecommendationEngine and swaps it atomically when the active model changes."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

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
