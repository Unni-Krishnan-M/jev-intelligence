"""The generic structured-dataset adapter: any long-format CSV + a YAML domain config.

``load`` reads the CSV, maps its columns onto the observation schema (``timestamp`` in UNIX seconds,
``entity_id = "<entity_type>:<entity>"``, ``value``, group columns, plus the raw ``entity`` name used
by ``group_by: entity`` specs), applies the *availability* rule (a row is usable only once
``timestamp + availability_lag_days <= as_of``: published statistics appear after a release delay,
so a replay never sees a value before it was published) and hands the rows to the core, which
validates them (schema, keys, values, timestamp range, duplicates, rows after as_of).

Adapter checks: values inside ``value_range``; every entity has data; gaps (missing periods between
an entity's first and last observation). Source freshness: fresh when as_of - last usable
observation <= ``max_lag_days``. Nothing domain-specific is added through ``extra``: the core's
generic risk families (adverse trend / forecast / anomaly) and the early-warning decision do the
work. Default ``as_of`` is the wall clock, so an out-of-date file shows up as a stale source.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.adapter import CoreContext, DomainData, DomainExtras, DomainInfo
from jev_ml.core.common import iso_from_epoch, to_utc
from jev_ml.core.config import CoreConfig
from jev_ml.core.quality import source_row, to_epoch
from jev_ml.core.series import season_classes
from jev_ml.domains.generic.config import GenericDomainConfig, load_config

DAY = 86400.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GenericAdapter:
    pipeline_version = "jev-core-1.0.0"

    def __init__(self, config: GenericDomainConfig | str | Path, frame: pd.DataFrame | None = None) -> None:
        self.cfg = config if isinstance(config, GenericDomainConfig) else load_config(Path(config))
        self.frame = frame  # optional in-memory data (tests); otherwise the configured file
        self.key = f"generic:{self.cfg.key}"
        src = self.cfg.source
        self.info = DomainInfo(
            key=self.key,
            name=self.cfg.name,
            description=self.cfg.description,
            entity_types=sorted({s.entity_type for s in self.cfg.specs}),
            frequency=self.cfg.frequency,
            sources=[
                {
                    "source": src.get("name", self.cfg.key),
                    "kind": src.get("kind", "dataset"),
                    "license": src.get("license"),
                    "url": src.get("url"),
                }
            ],
            capabilities={
                "recommendation": False,
                "user_intelligence": False,
                "lapse": False,
                "raters": False,
                "model_governance": False,
                "scenarios": True,
            },
        )

    # ---------------------------------------------------------------------------------------------

    def available(self) -> tuple[bool, str | None]:
        if self.frame is not None or self.cfg.data_file.exists():
            return True, None
        return False, (
            f"dataset file {self.cfg.source['file']} not found "
            f"(run: uv run python scripts/download_domain_data.py {self.cfg.key})"
        )

    def default_config(self) -> CoreConfig:
        return self.cfg.core_config()

    def _raw(self) -> pd.DataFrame:
        if self.frame is not None:
            return self.frame.copy()
        ok, reason = self.available()
        if not ok:
            raise FileNotFoundError(reason)
        return pd.read_csv(self.cfg.data_file)

    def _lag_days(self, entity: pd.Series, entity_type: pd.Series) -> np.ndarray:
        """Publication delay per row: an ``availability_lag_days`` map is looked up by entity name,
        then entity type, then ``default``."""
        lag = self.cfg.source.get("availability_lag_days", 0) or 0
        if isinstance(lag, dict):
            default = float(lag.get("default", 0))
            vals = [
                float(lag[e]) if e in lag else float(lag.get(t, default))
                for e, t in zip(entity.to_numpy(), entity_type.to_numpy(), strict=True)
            ]
            return np.asarray(vals, dtype=float)
        return np.full(len(entity), float(lag))

    def load(self, as_of: datetime | None, now: datetime | None, config: CoreConfig) -> DomainData:
        c = self.cfg.columns
        now_dt = to_utc(now) or datetime.now(UTC)
        as_of_dt = to_utc(as_of) or now_dt
        as_of_ts = as_of_dt.timestamp()
        raw = self._raw()
        n_raw = len(raw)
        missing = [c[k] for k in ("timestamp", "entity", "value") if c[k] not in raw.columns]
        if missing:
            raise ValueError(f"{self.key}: data file lacks columns {missing}")
        et_col = c.get("entity_type")
        etype = (
            raw[et_col].astype(str)
            if et_col and et_col in raw.columns
            else pd.Series([c.get("entity_type_default") or "entity"] * n_raw)
        )
        ts = to_epoch(raw[c["timestamp"]])
        obs = pd.DataFrame(
            {
                "timestamp": ts.to_numpy(dtype=float),
                "entity": raw[c["entity"]].astype(str).to_numpy(),
                "entity_type": etype.to_numpy(),
                "event_type": raw[c["event_type"]].astype(str).to_numpy()
                if c.get("event_type") and c["event_type"] in raw.columns
                else "observation",
                "value": pd.to_numeric(raw[c["value"]], errors="coerce").to_numpy(dtype=float),
                "source": self.cfg.source.get("name", self.cfg.key),
            }
        )
        obs["entity_id"] = obs["entity_type"] + ":" + obs["entity"]
        cc_col = self.cfg.calendar_check.get("column")
        if cc_col:
            if cc_col not in raw.columns:
                raise ValueError(f"{self.key}: calendar_check column {cc_col!r} not in the data file")
            obs["_calendar_label"] = raw[cc_col].astype(str).to_numpy()
        for g in c.get("groups") or []:
            if g not in raw.columns:
                raise ValueError(f"{self.key}: group column {g!r} not in the data file")
            obs[g] = raw[g].where(raw[g].notna(), None).to_numpy()
        # availability (publication delay): usable once timestamp + lag <= as_of
        avail = obs["timestamp"].to_numpy() + self._lag_days(obs["entity"], obs["entity_type"]) * DAY
        unpublished = (avail > as_of_ts) & (obs["timestamp"].to_numpy() <= as_of_ts)
        n_unpub = int(unpublished.sum())
        obs = obs.loc[~unpublished].reset_index(drop=True)
        usable = obs[obs["timestamp"] <= as_of_ts]
        first = float(usable["timestamp"].min()) if len(usable) else None
        last = float(usable["timestamp"].max()) if len(usable) else None
        src = self.cfg.source
        expected = src.get("expected_update")
        max_lag = src.get("max_lag_days")
        lag_days = (as_of_ts - last) / DAY if last is not None else None
        fresh = (
            None if (expected is None or max_lag is None or lag_days is None) else bool(lag_days <= max_lag)
        )
        prov = self._provenance()
        row = source_row(
            src.get("name", self.cfg.key),
            src.get("kind", "dataset"),
            len(usable),
            first,
            last,
            now_dt.timestamp(),
            as_of_ts,
            expected,
            fresh,
            f"{self.cfg.source['file']}; {n_unpub} rows not yet published at as_of "
            f"(availability lag {src.get('availability_lag_days', 0)} days)"
            + (f"; last usable observation {iso_from_epoch(last)}" if last is not None else ""),
        )
        row.update(
            {
                "license": src.get("license"),
                "url": src.get("url"),
                "checksum": prov.get("checksum"),
                "sla_days": float(max_lag) if max_lag is not None else None,
                "staleness_days": round(lag_days, 2) if lag_days is not None else None,
                "stale_response": "Re-download the dataset (scripts/download_domain_data.py) and re-run.",
            }
        )
        if prov.get("detail"):
            row["provenance"] = prov["detail"]
        ents = (
            obs.drop_duplicates("entity_id")[["entity_id", "entity_type", "entity", *(c.get("groups") or [])]]
            .sort_values("entity_id")
            .to_dict("records")
        )
        entities = [
            {
                "entity_id": e["entity_id"],
                "entity_type": e["entity_type"],
                "name": e["entity"],
                "attributes": {g: e[g] for g in c.get("groups") or []},
            }
            for e in ents
        ]
        return DomainData(
            observations=obs,
            specs=list(self.cfg.specs),
            as_of=as_of_dt,
            now=now_dt,
            sources=[row],
            data_version=f"{self.cfg.key}-{(prov.get('checksum') or 'unhashed').split(':')[-1][:12]}",
            entities=entities,
            quality_checks=[
                {
                    "name": f"{row['source']}_unpublished_at_as_of",
                    "passed": True,
                    "value": float(n_unpub),
                    "threshold": 0.0,
                    "severity": "info",
                    "detail": f"{n_unpub} rows dated <= as_of but not yet published (availability lag); "
                    "excluded (leakage guard)",
                    "source": row["source"],
                }
            ],
            core_checks=True,
            excluded_after_as_of=n_unpub,
            frequency={"month": "M", "week": "W", "day": "D"}[self.cfg.frequency],
            grid_end=self.cfg.grid_end,
            context=self.cfg,
        )

    def _provenance(self) -> dict[str, Any]:
        """Checksum of the data file (recorded provenance when present, else computed)."""
        if self.frame is not None:
            h = hashlib.sha256(pd.util.hash_pandas_object(self.frame, index=False).to_numpy().tobytes())
            return {"checksum": f"sha256:{h.hexdigest()}", "detail": "in-memory frame"}
        out: dict[str, Any] = {"checksum": f"sha256:{_sha256(self.cfg.data_file)}"}
        pf = self.cfg.provenance_file
        if pf is not None and pf.exists():
            p = json.loads(pf.read_text())
            recorded = (p.get("output") or {}).get("sha256")
            out["detail"] = {
                "fetched_at": p.get("fetched_at"),
                "files": len(p.get("files", [])),
                "recorded_sha256_matches": recorded == out["checksum"].split(":")[1] if recorded else None,
            }
        return out

    def quality_checks(self, data: DomainData) -> list[dict[str, Any]]:
        obs = data.observations
        src = data.sources[0]["source"] if data.sources else self.key
        out: list[dict[str, Any]] = []
        if self.cfg.value_range is not None:
            lo, hi = self.cfg.value_range
            bad = int(((obs["value"] < lo) | (obs["value"] > hi)).sum())
            out.append(
                {
                    "name": f"{src}_value_range",
                    "passed": bad == 0,
                    "value": float(bad),
                    "threshold": 0.0,
                    "severity": "error",
                    "detail": f"{bad} values outside [{lo}, {hi}]",
                    "source": src,
                }
            )
        if len(obs):
            freq = data.frequency
            per = pd.PeriodIndex(pd.to_datetime(obs["timestamp"], unit="s"), freq=freq)
            g = pd.DataFrame({"e": obs["entity_id"].to_numpy(), "p": np.asarray(per.asi8)})  # type: ignore[attr-defined]
            span = g.groupby("e")["p"].agg(lambda s: int(s.max() - s.min() + 1))
            have = g.drop_duplicates().groupby("e").size()
            gaps = int((span - have).clip(lower=0).sum())
            out.append(
                {
                    "name": f"{src}_gaps",
                    "passed": gaps == 0,
                    "value": float(gaps),
                    "threshold": 0.0,
                    "severity": "warning",
                    "detail": f"{gaps} missing periods between entities' first and last observations "
                    f"({len(span)} entities)",
                    "source": src,
                }
            )
        if self.cfg.calendar_check and len(obs) and "_calendar_label" in obs.columns:
            out.append(self._calendar_check(obs, src))
        return out

    def _calendar_check(self, obs: pd.DataFrame, src: str) -> dict[str, Any]:
        """Share of observed days whose rule-based calendar class (``season_classes``) matches the
        data's own day-type label: evidence that the calendar used for *future* days is right."""
        cc = self.cfg.calendar_check
        days = pd.to_datetime(obs["timestamp"], unit="s").dt.normalize()
        uniq = pd.DataFrame({"d": days, "l": obs["_calendar_label"]}).drop_duplicates("d")
        per = pd.PeriodIndex(uniq["d"], freq="D")
        cls = season_classes(per, 7, cc["calendar"])
        names = np.where(cls <= 4, "weekday", np.where(cls == 5, "saturday", "sunday_holiday"))
        want = np.array([cc["labels"].get(str(x)) for x in names], dtype=object)
        agree = float(np.mean(want == uniq["l"].to_numpy())) if len(uniq) else 1.0
        return {
            "name": f"{src}_calendar",
            "passed": agree >= 0.99,
            "value": round(agree, 5),
            "threshold": 0.99,
            "severity": "info" if agree >= 0.99 else "warning",
            "detail": f"calendar {cc['calendar']} matches column {cc['column']!r} on {agree:.2%} of "
            f"{len(uniq)} days ({round((1 - agree) * len(uniq))} disagree)",
            "source": src,
        }

    def extra(self, ctx: CoreContext) -> DomainExtras:
        return DomainExtras()


def adapter_for(key: str, root: Path | None = None) -> GenericAdapter:
    """The generic adapter of ``configs/domains/<key>.yaml`` (key with or without "generic:")."""
    from jev_ml.paths import ROOT

    name = key.split(":", 1)[1] if key.startswith("generic:") else key
    base = root or ROOT
    return GenericAdapter(load_config(base / "configs" / "domains" / f"{name}.yaml", base))
