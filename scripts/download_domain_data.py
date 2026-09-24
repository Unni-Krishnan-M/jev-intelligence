"""Download a generic domain's dataset as declared in configs/domains/<name>.yaml (``download``).

    uv run python scripts/download_domain_data.py us-unemployment
    uv run python scripts/download_domain_data.py cta-ridership

Kind ``fred``: one CSV per series from ``https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>``
(no API key). Raw files are kept next to the long-format CSV under data/raw/domains/<name>/ (gitignored,
never committed). ``provenance.json`` records each source URL, the licence note from the config, the
SHA-256 of every raw file and of the long CSV, row counts and the fetch time. Missing values ('.')
are dropped and counted.

Kind ``socrata`` (City of Chicago data portal, no key): each dataset is paged with ``$limit`` /
``$offset`` and a total ``$order`` (so pages never overlap or skip), every page is kept as a raw file
with its SHA-256, and the row count is checked against the server's ``count(*)``. The station
dataset is first ranked server-side (``$select=station_id, sum(rides) ... $group`` over the
declared ``rank_year``) and only the top ``top`` stations are fetched (tractable: ~9k rows each).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from jev_ml.domains.generic.config import load_config
from jev_ml.paths import ROOT


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            # default client headers: FRED stalls requests with some custom User-Agent strings
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def download_fred(name: str) -> Path:
    cfg = load_config(ROOT / "configs" / "domains" / f"{name}.yaml")
    dl = cfg.download
    if dl.get("kind") != "fred":
        raise SystemExit(f"{name}: download kind {dl.get('kind')!r} is not supported")
    out_csv = cfg.data_file
    raw_dir = out_csv.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    etypes = dl.get("entity_type") or {}
    groups = dl.get("groups") or {}
    frames, files = [], []
    for entity, sid in dl["series"].items():
        url = dl["url_template"].format(id=sid)
        body = fetch(url)
        (raw_dir / f"{sid}.csv").write_bytes(body)
        df = pd.read_csv(io.BytesIO(body))
        date_col = df.columns[0]
        val = pd.to_numeric(df[sid], errors="coerce")
        missing = int(val.isna().sum())
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
                "entity": entity,
                "entity_type": etypes.get(entity, etypes.get("default", "entity")),
                "value": val,
                "series_id": sid,
            }
        )
        for g, mapping in groups.items():
            part[g] = mapping.get(entity)
        part = part.dropna(subset=["value"])
        frames.append(part)
        files.append(
            {
                "entity": entity,
                "series_id": sid,
                "url": url,
                "file": f"raw/{sid}.csv",
                "sha256": sha256(body),
                "rows": len(part),
                "missing_dropped": missing,
                "first": part["date"].min(),
                "last": part["date"].max(),
            }
        )
        print(f"{entity:3s} {sid:7s} {len(part):4d} rows {part['date'].min()}..{part['date'].max()}")
    long = pd.concat(frames, ignore_index=True).sort_values(["date", "entity"], kind="mergesort")
    data = long.to_csv(index=False).encode()
    out_csv.write_bytes(data)
    prov = {
        "domain": cfg.key,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": cfg.source.get("url"),
        "license": cfg.source.get("license"),
        "files": files,
        "output": {"file": out_csv.name, "rows": len(long), "sha256": sha256(data)},
    }
    assert cfg.provenance_file is not None
    cfg.provenance_file.write_text(json.dumps(prov, indent=2))
    print(f"wrote {out_csv} ({len(long)} rows, sha256 {prov['output']['sha256'][:12]}...)")
    return out_csv


def _socrata_pages(
    url: str, params: dict[str, str], page: int, raw_dir: Path, tag: str
) -> tuple[pd.DataFrame, list[dict]]:
    frames, files, offset = [], [], 0
    while True:
        q = {**params, "$limit": str(page), "$offset": str(offset)}
        body = fetch(requests.Request("GET", url, params=q).prepare().url or url)
        name = f"{tag}_{offset:08d}.csv"
        (raw_dir / name).write_bytes(body)
        df = pd.read_csv(io.BytesIO(body))
        files.append(
            {"file": f"raw/{name}", "url": url, "params": q, "rows": len(df), "sha256": sha256(body)}
        )
        frames.append(df)
        if len(df) < page:
            break
        offset += page
    return pd.concat(frames, ignore_index=True), files


def download_socrata(name: str) -> Path:
    cfg = load_config(ROOT / "configs" / "domains" / f"{name}.yaml")
    dl = cfg.download
    out_csv = cfg.data_file
    raw_dir = out_csv.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    page = int(dl.get("page_size", 50000))
    parts, files = [], []
    d = dl["daily"]
    where = {"$where": d["where"]} if d.get("where") else {}
    df, f = _socrata_pages(d["url"], {"$order": d["order"], **where}, page, raw_dir, "daily")
    n_server = int(
        pd.read_csv(
            io.BytesIO(
                fetch(
                    requests.Request("GET", d["url"], params={"$select": "count(*)", **where}).prepare().url
                    or d["url"]
                )
            )
        ).iloc[0, 0]
    )
    if n_server != len(df):
        raise SystemExit(f"daily: fetched {len(df)} rows, server reports {n_server}")
    files += f
    for ent, col in d["values"].items():
        parts.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(df[d["date"]]).dt.strftime("%Y-%m-%d"),
                    "entity": ent,
                    "entity_type": d["entity_type"].get(ent, d["entity_type"].get("default", "entity")),
                    "value": pd.to_numeric(df[col], errors="coerce"),
                    "day_type": df[d["day_type"]],
                }
            )
        )
    st = dl.get("stations")
    ranking = None
    if st:
        y = int(st["rank_year"])
        rank_q = {
            "$select": "station_id, stationname, sum(rides) as total",
            "$where": f"date between '{y}-01-01T00:00:00' and '{y}-12-31T00:00:00'",
            "$group": "station_id, stationname",
            "$order": "total DESC",
            "$limit": str(st["top"]),
        }
        body = fetch(requests.Request("GET", st["url"], params=rank_q).prepare().url or st["url"])
        (raw_dir / "stations_rank.csv").write_bytes(body)
        ranking = pd.read_csv(io.BytesIO(body))
        files.append(
            {
                "file": "raw/stations_rank.csv",
                "url": st["url"],
                "params": rank_q,
                "rows": len(ranking),
                "sha256": sha256(body),
            }
        )
        ids = ", ".join(f"'{int(i)}'" for i in ranking["station_id"])
        sdf, f = _socrata_pages(
            st["url"],
            {"$order": "date, station_id", "$where": f"station_id in ({ids})"},
            page,
            raw_dir,
            "stations",
        )
        files += f
        names = {int(r.station_id): str(r.stationname) for r in ranking.itertuples()}
        parts.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(sdf["date"]).dt.strftime("%Y-%m-%d"),
                    "entity": [names[int(i)] for i in sdf["station_id"]],
                    "entity_type": "station",
                    "value": pd.to_numeric(sdf["rides"], errors="coerce"),
                    "day_type": sdf["daytype"],
                }
            )
        )
    long = pd.concat(parts, ignore_index=True)
    missing = int(long["value"].isna().sum())
    long = long.dropna(subset=["value"]).sort_values(["date", "entity"], kind="mergesort")
    data = long.to_csv(index=False).encode()
    out_csv.write_bytes(data)
    prov = {
        "domain": cfg.key,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": cfg.source.get("url"),
        "license": cfg.source.get("license"),
        "terms_of_use": dl.get("terms_of_use"),
        "datasets": {"daily": d["url"], **({"stations": st["url"]} if st else {})},
        "station_ranking": ranking.to_dict("records") if ranking is not None else None,
        "files": files,
        "output": {
            "file": out_csv.name,
            "rows": len(long),
            "missing_dropped": missing,
            "sha256": sha256(data),
            "first": long["date"].min(),
            "last": long["date"].max(),
        },
    }
    assert cfg.provenance_file is not None
    cfg.provenance_file.write_text(json.dumps(prov, indent=2, default=str))
    print(
        f"wrote {out_csv} ({len(long)} rows, {long['date'].min()}..{long['date'].max()}, "
        f"sha256 {prov['output']['sha256'][:12]}...)"
    )
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("domain", help="config name under configs/domains/ (e.g. us-unemployment)")
    args = p.parse_args()
    kind = load_config(ROOT / "configs" / "domains" / f"{args.domain}.yaml").download.get("kind")
    try:
        if kind == "socrata":
            download_socrata(args.domain)
        else:
            download_fred(args.domain)
    except requests.RequestException as exc:
        sys.exit(f"download failed: {exc}")


if __name__ == "__main__":
    main()
