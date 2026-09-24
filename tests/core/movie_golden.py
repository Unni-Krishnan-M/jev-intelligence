"""Golden essentials of the movie pipeline (regression guard for the platform-core migration).

The golden files were written by the pre-migration ``jev_ml.intel.run_pipeline`` (v1.1.0) and are
compared against the refactored movie adapter by ``test_movie_golden.py``. Comparison is a
*subset* check: every key in the golden must exist in the new output with an equal value (floats
within a tolerance); fields added by the platform contract are allowed. Timings and ``now`` are
excluded, and series keep only their last points.

Regenerate (only for an intentional, documented behaviour change):
    uv run python tests/core/movie_golden.py            # synthetic variants (CI)
    uv run python tests/core/movie_golden.py --real     # real MovieLens data (local only)
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "intel"))

from intel_synth import GENRES, make_inputs

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SERIES_TAIL = 12
REAL_NOW = datetime(2026, 9, 24, tzinfo=UTC)
REAL_VARIANTS = {"default": None, "2017-07-01": "2017-07-01"}
REL_TOL = 1e-6
ABS_TOL = 1e-9


def _feedback(now_ts: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feedback": list(rng.choice(["like", "clicked", "dislike"], 400, p=[0.5, 0.4, 0.1]))
            + list(rng.choice(["like", "clicked", "dislike"], 200, p=[0.3, 0.3, 0.4])),
            "timestamp": np.r_[
                now_ts - rng.uniform(8, 35, 400) * 86400, now_ts - rng.uniform(0, 7, 200) * 86400
            ],
        }
    )


def _full_inputs() -> Any:
    """Seed 0 plus a model manifest with per-user NDCG, live feedback and a Horror burst in the last
    complete month (exercises governance, live feedback and series-anomaly warnings)."""
    inp = make_inputs(0)
    r = inp.interactions
    movies = inp.movies
    horror = movies.loc[movies["genres"] == "Horror", "movie_id"].to_numpy()
    last = pd.Timestamp(float(r["timestamp"].max()), unit="s", tz="UTC")
    month0 = (last.tz_convert(None).to_period("M") - 1).start_time.tz_localize("UTC").timestamp()
    rng = np.random.default_rng(7)
    users = rng.choice(r["user_id"].unique(), 300)
    burst = pd.DataFrame(
        {
            "user_id": users + 10_000,
            "movie_id": rng.choice(horror, 300),
            "rating": 4.0,
            "timestamp": rng.uniform(month0, month0 + 25 * 86400, 300).astype(np.int64),
        }
    )
    inter = pd.concat([r, burst], ignore_index=True).sort_values("timestamp", kind="mergesort")
    rng2 = np.random.default_rng(0)
    per_user = {
        "hybrid": list(rng2.normal(0.3, 0.1, 200)),
        "als": list(rng2.normal(0.2, 0.1, 200)),
        "random": list(rng2.normal(0.0, 0.01, 200)),
    }
    manifest = {
        "version": "m1",
        "model_type": "hybrid",
        "dataset_version": "synthetic-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "trained_on_rows": 1000,
        "data_cutoff_ts": float(r["timestamp"].max()),
    }
    assert inp.now is not None
    return replace(
        inp,
        interactions=inter.reset_index(drop=True),
        model_manifest=manifest,
        per_user_ndcg=per_user,
        app_feedback=_feedback(inp.now.timestamp()),
    )


def synthetic_variants() -> dict[str, Any]:
    return {
        "default": make_inputs(0),
        "as_of_2013-06-15": make_inputs(0, as_of=datetime(2013, 6, 15, tzinfo=UTC)),
        "seed1": make_inputs(1),
        "full": _full_inputs(),
    }


def essentials(d: dict[str, Any]) -> dict[str, Any]:
    """Drop timings/now, keep only the tail of each series (everything else is compared)."""
    d = copy.deepcopy(d)
    d["run"].pop("stage_ms", None)
    d["run"].pop("now", None)
    for s in d["series"]:
        s["n_points"] = len(s["points"])
        s["points"] = s["points"][-SERIES_TAIL:]
    return d


def diff(golden: Any, new: Any, path: str = "$", out: list[str] | None = None) -> list[str]:
    """Subset comparison; returns human-readable differences (empty = match)."""
    out = [] if out is None else out
    if len(out) > 50:
        return out
    if isinstance(golden, dict):
        if not isinstance(new, dict):
            out.append(f"{path}: expected object, got {type(new).__name__}")
            return out
        for k, v in golden.items():
            if k not in new:
                out.append(f"{path}.{k}: missing")
            else:
                diff(v, new[k], f"{path}.{k}", out)
    elif isinstance(golden, list):
        if not isinstance(new, list) or len(new) != len(golden):
            n = len(new) if isinstance(new, list) else type(new).__name__
            out.append(f"{path}: expected list of {len(golden)}, got {n}")
            return out
        for i, (g, n) in enumerate(zip(golden, new, strict=True)):
            diff(g, n, f"{path}[{i}]", out)
    elif isinstance(golden, bool) or golden is None or isinstance(golden, str):
        if golden != new:
            out.append(f"{path}: {golden!r} != {new!r}")
    elif isinstance(golden, int | float):
        if (
            isinstance(new, bool)
            or not isinstance(new, int | float)
            or not math.isclose(float(golden), float(new), rel_tol=REL_TOL, abs_tol=ABS_TOL)
        ):
            out.append(f"{path}: {golden!r} != {new!r}")
    elif golden != new:
        out.append(f"{path}: {golden!r} != {new!r}")
    return out


def _write(name: str, d: dict[str, Any]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(essentials(d), sort_keys=True, separators=(",", ":")) + "\n"
    if name.endswith(".gz"):
        # mtime=0: byte-identical files for identical content
        (GOLDEN_DIR / name).write_bytes(gzip.compress(raw.encode(), mtime=0))
    else:
        (GOLDEN_DIR / name).write_text(raw)


def load(name: str) -> dict[str, Any]:
    p = GOLDEN_DIR / name
    raw = gzip.decompress(p.read_bytes()).decode() if name.endswith(".gz") else p.read_text()
    out: dict[str, Any] = json.loads(raw)
    return out


if __name__ == "__main__":
    from jev_ml.intel import load_default_inputs, run_pipeline

    assert GENRES
    if "--real" in sys.argv:
        for tag, as_of in REAL_VARIANTS.items():
            _write(
                f"movie_real_{tag}.json.gz",
                run_pipeline(load_default_inputs(as_of=as_of, now=REAL_NOW)).to_dict(),
            )
    else:
        for tag, inp in synthetic_variants().items():
            _write(f"movie_synth_{tag}.json", run_pipeline(inp).to_dict())
    for p in sorted(GOLDEN_DIR.glob("*.json*")):
        print(p.name, p.stat().st_size)
