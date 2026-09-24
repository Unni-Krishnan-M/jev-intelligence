"""Domain config of the generic adapter: ``configs/domains/<name>.yaml`` (platform contract section 6).

Schema (keys not listed are rejected, so a typo never silently changes a run)::

    key: us-unemployment                 # adapter key becomes "generic:<key>"
    name: US unemployment
    description: ...
    frequency: month                     # month (required support) | week
    grid_end: last_observation           # last_observation (published statistics) | as_of (event streams)
    source:
      name: fred-bls                     # DataSource.source
      file: data/raw/domains/<key>/data.csv   # long-format CSV, relative to the repo root
      provenance: data/raw/domains/<key>/provenance.json   # optional: url/licence/sha256 per file
      license: ...                       # DataSource.license
      url: ...
      expected_update: monthly           # null = no freshness SLA
      max_lag_days: 75                   # fresh when as_of - last observation <= this
      availability_lag_days: 38          # int, or {entity or entity_type: int, default: int}; usable only
                                         # once timestamp + lag <= as_of (publication delay)
    columns:
      timestamp: date                    # ISO dates/datetimes or UNIX seconds
      entity: entity
      value: value
      entity_type: entity_type           # optional column; else entity_type_default
      entity_type_default: entity
      event_type: null                   # optional column
      groups: [region]                   # optional group columns (scalar per row)
    value_range: [0, 100]                # optional validation range (error check)
    series:                              # SeriesSpec fields; group_by may be "entity"
      - {id: "rate:{group}", metric: level, group_by: entity, unit: "%", adverse_direction: up}
    thresholds: {}                       # CoreConfig field overrides (e.g. anomaly_z_threshold: 4.0)
    impact_weights: {}                   # declared impact weight per entity (0..1), labelled "declared"
    download: {...}                      # used by scripts/download_domain_data.py only
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from jev_ml.core.config import CoreConfig
from jev_ml.core.series import SeriesSpec

TOP_KEYS = {
    "key",
    "name",
    "description",
    "frequency",
    "grid_end",
    "source",
    "columns",
    "value_range",
    "series",
    "thresholds",
    "impact_weights",
    "download",
}
SOURCE_KEYS = {
    "name",
    "file",
    "provenance",
    "license",
    "url",
    "kind",
    "expected_update",
    "max_lag_days",
    "availability_lag_days",
}
COLUMN_KEYS = {"timestamp", "entity", "value", "entity_type", "entity_type_default", "event_type", "groups"}
SPEC_KEYS = {f.name for f in fields(SeriesSpec)} - {"groups"}


@dataclass
class GenericDomainConfig:
    key: str
    name: str
    description: str
    source: dict[str, Any]
    columns: dict[str, Any]
    specs: list[SeriesSpec]
    frequency: str = "month"
    grid_end: str = "last_observation"
    value_range: tuple[float, float] | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)
    impact_weights: dict[str, float] = field(default_factory=dict)
    download: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=Path)
    path: Path | None = None

    @property
    def data_file(self) -> Path:
        return self.root / str(self.source["file"])

    @property
    def provenance_file(self) -> Path | None:
        p = self.source.get("provenance")
        return self.root / str(p) if p else None

    def core_config(self) -> CoreConfig:
        """``CoreConfig`` with the domain's threshold overrides and declared impact weights."""
        known = {f.name for f in fields(CoreConfig)}
        bad = sorted(set(self.thresholds) - known)
        if bad:
            raise ValueError(f"{self.key}: unknown thresholds {bad}")
        kw: dict[str, Any] = {}
        for k, v in self.thresholds.items():
            kw[k] = tuple(v) if isinstance(v, list) else v
        return CoreConfig(**kw, impact_weights={str(k): float(v) for k, v in self.impact_weights.items()})


def _check_keys(d: dict[str, Any], allowed: set[str], where: str) -> None:
    bad = sorted(set(d) - allowed)
    if bad:
        raise ValueError(f"{where}: unknown keys {bad}")


def parse_config(raw: dict[str, Any], root: Path, path: Path | None = None) -> GenericDomainConfig:
    _check_keys(raw, TOP_KEYS, "domain config")
    for k in ("key", "name", "source", "columns", "series"):
        if k not in raw:
            raise ValueError(f"domain config: missing {k!r}")
    src = dict(raw["source"])
    _check_keys(src, SOURCE_KEYS, "source")
    if "file" not in src:
        raise ValueError("source: missing 'file'")
    cols = dict(raw["columns"])
    _check_keys(cols, COLUMN_KEYS, "columns")
    for k in ("timestamp", "entity", "value"):
        if not cols.get(k):
            raise ValueError(f"columns: missing {k!r}")
    cols.setdefault("groups", [])
    specs = []
    for i, sp in enumerate(raw["series"]):
        sp = dict(sp)
        _check_keys(sp, SPEC_KEYS, f"series[{i}]")
        metric = sp.get("metric")
        # counts back meaningful volume guards; a level or a mean of few entities does not
        sp.setdefault("volume_guard", metric in ("count", "share", "nunique"))
        sp.setdefault("source", src.get("name"))
        if sp.get("group_by") == cols["entity"]:
            sp["group_by"] = "entity"
        specs.append(SeriesSpec(**sp))
    freq = str(raw.get("frequency", "month"))
    if freq not in ("month", "week"):
        raise ValueError("frequency must be 'month' or 'week'")
    vr = raw.get("value_range")
    cfg = GenericDomainConfig(
        key=str(raw["key"]),
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        source=src,
        columns=cols,
        specs=specs,
        frequency=freq,
        grid_end=str(raw.get("grid_end", "last_observation")),
        value_range=(float(vr[0]), float(vr[1])) if vr else None,
        thresholds=dict(raw.get("thresholds") or {}),
        impact_weights=dict(raw.get("impact_weights") or {}),
        download=dict(raw.get("download") or {}),
        root=root,
        path=path,
    )
    cfg.core_config()  # validates the thresholds early
    return cfg


def load_config(path: Path, root: Path | None = None) -> GenericDomainConfig:
    from jev_ml.paths import ROOT

    raw = yaml.safe_load(Path(path).read_text())
    return parse_config(raw, root or ROOT, Path(path))
