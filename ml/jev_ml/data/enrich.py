"""Optional metadata enrichment from Wikidata (CC0) keyed by the IMDb ids in MovieLens `links.csv`.

MovieLens ships only titles and genres. For content-based features we additionally want directors,
cast, keywords ("main subject"), fine-grained genres, an overview-like description, runtime and
country. Wikidata offers these without an API key.

The query runs one SPARQL request per property per batch, so it never builds a join cross-product.
Batches are cached on disk, which means an interrupted run resumes. Missing values are left missing;
nothing is guessed. The one exception is marked explicitly: when a film has no director statement
but its English description reads "... directed by X" or "... film by X", X is recorded under
`directors_from_description`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from jev_ml.paths import RAW_DIR

log = logging.getLogger(__name__)

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "JEV-recsys/1.0 (open-source research recommender; https://grouplens.org/)"
BATCH_SIZE = 250

# field name -> Wikidata property
LABEL_PROPERTIES: dict[str, str] = {
    "directors": "P57",
    "cast": "P161",
    "keywords": "P921",
    "wd_genres": "P136",
    "countries": "P495",
}

_DESCRIPTION_DIRECTOR = re.compile(r"(?:directed by|film by)\s+([^,;()]+?)(?:\s+and\s+([^,;()]+))?$")


def _sparql(query: str, retries: int = 5) -> list[dict[str, Any]]:
    for attempt in range(retries):
        try:
            resp = requests.post(
                ENDPOINT,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
                timeout=90,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                log.warning("wikidata %s, retrying in %.0fs", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            bindings: list[dict[str, Any]] = resp.json()["results"]["bindings"]
            return bindings
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            # ValueError covers truncated JSON bodies (requests.JSONDecodeError subclasses it)
            wait = 2 ** (attempt + 1)
            log.warning("wikidata request failed (%s), retrying in %ss", exc, wait)
            time.sleep(wait)
    raise RuntimeError("Wikidata query failed after retries")


def _values(imdb_ids: list[str]) -> str:
    return " ".join(f'"{i}"' for i in imdb_ids)


def _fetch_batch(imdb_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {i: {} for i in imdb_ids}
    values = _values(imdb_ids)

    scalar_q = f"""SELECT ?imdb ?film ?desc ?dur WHERE {{
      VALUES ?imdb {{ {values} }}
      ?film wdt:P345 ?imdb .
      OPTIONAL {{ ?film schema:description ?desc FILTER(LANG(?desc) = "en") }}
      OPTIONAL {{ ?film wdt:P2047 ?dur }}
    }}"""
    for b in _sparql(scalar_q):
        rec = out[b["imdb"]["value"]]
        rec["wikidata_id"] = b["film"]["value"].rsplit("/", 1)[-1]
        if "desc" in b:
            rec["description"] = b["desc"]["value"]
        if "dur" in b:
            try:
                rec["runtime_min"] = round(float(b["dur"]["value"]))
            except ValueError:
                log.debug("non-numeric duration for %s", b["imdb"]["value"])
    time.sleep(0.3)

    for field, prop in LABEL_PROPERTIES.items():
        q = f"""SELECT ?imdb ?label WHERE {{
          VALUES ?imdb {{ {values} }}
          ?film wdt:P345 ?imdb . ?film wdt:{prop} ?v .
          ?v rdfs:label ?label FILTER(LANG(?label) = "en")
        }}"""
        for b in _sparql(q):
            rec = out[b["imdb"]["value"]]
            values_list = rec.setdefault(field, [])
            label = b["label"]["value"]
            if label not in values_list:
                values_list.append(label)
        time.sleep(0.3)

    for rec in out.values():
        for field in LABEL_PROPERTIES:
            if field in rec:
                rec[field] = sorted(rec[field])
        if not rec.get("directors") and rec.get("description"):
            m = _DESCRIPTION_DIRECTOR.search(rec["description"])
            if m:
                rec["directors_from_description"] = [g.strip() for g in m.groups() if g]
    return out


def enrich_from_wikidata(movielens_dir: Path, dest: Path = RAW_DIR, force: bool = False) -> Path:
    links = pd.read_csv(movielens_dir / "links.csv", dtype={"imdbId": str})
    imdb_ids = sorted({f"tt{i}" for i in links["imdbId"].dropna()})
    cache_dir = dest / "wikidata_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest / "wikidata_enrichment.json"

    merged: dict[str, dict[str, Any]] = {}
    n_batches = (len(imdb_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch = imdb_ids[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        cache_file = cache_dir / f"batch_{bi:04d}.json"
        if cache_file.exists() and not force:
            data = json.loads(cache_file.read_text())
        else:
            log.info("wikidata batch %d/%d", bi + 1, n_batches)
            data = _fetch_batch(batch)
            cache_file.write_text(json.dumps(data, ensure_ascii=False))
        merged.update(data)

    found = sum(1 for v in merged.values() if v.get("wikidata_id"))
    log.info("wikidata matched %d/%d IMDb ids", found, len(imdb_ids))
    out_path.write_text(json.dumps(merged, ensure_ascii=False, sort_keys=True))
    return out_path
