"""Download a generic domain's dataset as declared in configs/domains/<name>.yaml (``download``).

    uv run python scripts/download_domain_data.py us-unemployment

Kind ``fred``: one CSV per series from ``https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>``
(no API key). Raw files are kept next to the long-format CSV under data/raw/domains/<name>/ (gitignored,
never committed). ``provenance.json`` records each source URL, the licence note from the config, the
SHA-256 of every raw file and of the long CSV, row counts and the fetch time. Missing values ('.')
are dropped and counted.
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("domain", help="config name under configs/domains/ (e.g. us-unemployment)")
    args = p.parse_args()
    try:
        download_fred(args.domain)
    except requests.RequestException as exc:
        sys.exit(f"download failed: {exc}")


if __name__ == "__main__":
    main()
