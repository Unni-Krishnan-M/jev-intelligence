"""Golden snapshots of the generic adapter's series, anomaly and forecast output (the stages the
seasonal work touches), on a synthetic frame shaped like the us-unemployment dataset and run with
the real ``configs/domains/us-unemployment.yaml``. The goldens were written by the code before
daily/weekly and seasonal support was added; monthly behaviour must stay byte-identical."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from jev_ml.core import run_domain
from jev_ml.domains.generic import GenericAdapter, parse_config

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"
NOW = datetime(2026, 9, 24, tzinfo=UTC)
VARIANTS = {"default": None, "as_of_2008-06-01": "2008-06-01", "as_of_2020-05-01": "2020-05-01"}


def unemployment_frame(seed: int = 7) -> pd.DataFrame:
    raw = yaml.safe_load((ROOT / "configs" / "domains" / "us-unemployment.yaml").read_text())
    dl = raw["download"]
    regions = dl["groups"]["census_region"]
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-01", "2026-07-01", freq="MS")
    t = np.arange(len(months))
    rows = []
    for ent in dl["series"]:
        base = 4.5 + rng.normal(0, 0.8)
        cyc = 1.5 * np.sin(2 * np.pi * t / 110 + rng.uniform(0, 1))
        shock = np.where((months >= "2020-04-01") & (months < "2021-01-01"), 7.0 * np.exp(-(t - t[243]) / 4), 0)
        walk = np.cumsum(rng.normal(0, 0.05, len(t)))
        v = np.round(np.clip(base + cyc + walk + shock, 1, 25), 1)
        et = "country" if ent == "US" else "state"
        for m, x in zip(months, v, strict=True):
            rows.append((m.strftime("%Y-%m-%d"), ent, et, float(x), regions.get(ent)))
    return pd.DataFrame(rows, columns=["date", "entity", "entity_type", "value", "census_region"])


def adapter(frame: pd.DataFrame) -> GenericAdapter:
    raw = yaml.safe_load((ROOT / "configs" / "domains" / "us-unemployment.yaml").read_text())
    return GenericAdapter(parse_config(raw, ROOT), frame=frame)


def snapshot(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "series": [(s["id"], s["points"]) for s in d["series"]],
        "anomalies": [
            (a["id"], a["score"], a["suppressed"], a["value"], a["baseline"], a["severity"])
            for a in d["anomalies"]
        ],
        "forecasts": [
            {
                k: f[k]
                for k in (
                    "id",
                    "series_id",
                    "model",
                    "model_version",
                    "model_params",
                    "points",
                    "backtest",
                    "backtest_all_models",
                    "horizon_months",
                    "horizon_unit",
                    "features_used",
                )
            }
            for f in d["predictions"]["forecasts"]
        ],
    }


def run_variant(tag: str) -> dict[str, Any]:
    d = run_domain(adapter(unemployment_frame()), as_of=VARIANTS[tag], now=NOW).to_dict()
    return json.loads(json.dumps(snapshot(d), allow_nan=False))


def load(tag: str) -> dict[str, Any]:
    with gzip.open(GOLDEN / f"unemployment_synth_{tag}.json.gz", "rt") as f:
        out: dict[str, Any] = json.load(f)
    return out


def write_goldens() -> None:  # pragma: no cover (run once, by hand, before a contract change)
    GOLDEN.mkdir(exist_ok=True)
    for tag in VARIANTS:
        with gzip.open(GOLDEN / f"unemployment_synth_{tag}.json.gz", "wt") as f:
            json.dump(run_variant(tag), f, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover
    write_goldens()
