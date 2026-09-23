"""Filesystem layout. Everything is relative to the project root (override with JEV_HOME)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("JEV_HOME")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("JEV_MODELS_DIR", ROOT / "models"))
EXPERIMENTS_DIR = Path(os.environ.get("JEV_EXPERIMENTS_DIR", ROOT / "experiments"))
CONFIGS_DIR = ROOT / "configs"
