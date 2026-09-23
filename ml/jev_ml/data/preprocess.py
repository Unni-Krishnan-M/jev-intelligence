"""Validate, clean and transform raw MovieLens (+ optional Wikidata) into processed tables.

Outputs (data/processed/):
  movies.csv         one row per movie with cleaned metadata (list fields are '|'-joined)
  interactions.csv   user_id, movie_id, rating, timestamp (explicit ratings)
  dataset_meta.json  dataset version hash, sizes, field lists, validation report
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.paths import PROCESSED_DIR, RAW_DIR

log = logging.getLogger(__name__)

LIST_SEP = "|"
_YEAR = re.compile(r"\((\d{4})\)\s*$")
_ARTICLE = re.compile(r"^(?P<main>.+), (?P<article>The|A|An|Les|La|Le|L'|Il|El|Die|Der|Das)$")
_WS = re.compile(r"\s+")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def check(self, cond: object, msg: str, fatal: bool = True) -> None:
        if not cond:
            (self.errors if fatal else self.warnings).append(msg)


def clean_title(raw: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999)."""
    title = raw.strip()
    year: int | None = None
    m = _YEAR.search(title)
    if m:
        year = int(m.group(1))
        title = title[: m.start()].strip()
    # alternate titles in parentheses, e.g. "City of Lost Children, The (Cité des enfants perdus, La)"
    alt = re.match(r"^(.*?)\s+\((.+)\)$", title)
    if alt and not alt.group(1).endswith(","):
        title = alt.group(1).strip()
    m2 = _ARTICLE.match(title)
    if m2:
        article = m2.group("article")
        sep = "" if article.endswith("'") else " "
        title = f"{article}{sep}{m2.group('main')}"
    return title, year


def normalize_tag(tag: str) -> str:
    return _WS.sub(" ", str(tag).strip().lower())


def validate_raw(
    movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame, links: pd.DataFrame
) -> ValidationReport:
    rep = ValidationReport()
    rep.check(movies["movieId"].is_unique, "movies.movieId not unique")
    rep.check(movies[["movieId", "title", "genres"]].notna().all().all(), "movies has nulls")
    rep.check(ratings[["userId", "movieId", "rating", "timestamp"]].notna().all().all(), "ratings has nulls")
    rep.check(ratings["rating"].between(0.5, 5.0).all(), "rating out of [0.5, 5]")
    rep.check(((ratings["rating"] * 2) % 1 == 0).all(), "rating not on 0.5 grid")
    rep.check((ratings["timestamp"] > 0).all(), "non-positive timestamps")
    unknown = ~ratings["movieId"].isin(movies["movieId"])
    rep.check(not unknown.any(), f"{int(unknown.sum())} ratings reference unknown movies")
    dup = ratings.duplicated(["userId", "movieId"])
    rep.check(not dup.any(), f"{int(dup.sum())} duplicate (user, movie) ratings", fatal=False)
    rep.check(links["movieId"].is_unique, "links.movieId not unique")
    missing_links = ~movies["movieId"].isin(links["movieId"])
    rep.check(not missing_links.any(), f"{int(missing_links.sum())} movies lack links", fatal=False)
    rep.check(tags["movieId"].isin(movies["movieId"]).all(), "tags reference unknown movies", fatal=False)
    rep.stats = {
        "n_movies": len(movies),
        "n_ratings": len(ratings),
        "n_users": int(ratings["userId"].nunique()),
        "n_tags": len(tags),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "first_ts": int(ratings["timestamp"].min()),
        "last_ts": int(ratings["timestamp"].max()),
        "density": float(len(ratings) / (ratings["userId"].nunique() * len(movies))),
    }
    return rep


def _join(values: Any) -> str:
    if not isinstance(values, list | tuple):
        return ""
    return LIST_SEP.join(str(v).replace(LIST_SEP, "/") for v in values)


def preprocess(
    raw_dir: Path = RAW_DIR / "ml-latest-small",
    out_dir: Path = PROCESSED_DIR,
    enrichment_path: Path | None = RAW_DIR / "wikidata_enrichment.json",
) -> dict[str, Any]:
    movies = pd.read_csv(raw_dir / "movies.csv")
    ratings = pd.read_csv(raw_dir / "ratings.csv")
    tags = pd.read_csv(raw_dir / "tags.csv")
    link_dtypes: dict[str, Any] = {"imdbId": str, "tmdbId": "Int64"}
    links = pd.read_csv(raw_dir / "links.csv", dtype=link_dtypes)

    report = validate_raw(movies, ratings, tags, links)
    for w in report.warnings:
        log.warning("validation: %s", w)
    if report.errors:
        raise ValueError(f"dataset validation failed: {report.errors}")

    # --- clean movies -------------------------------------------------------------------------
    parsed = movies["title"].map(clean_title)
    movies["clean_title"] = parsed.map(lambda t: t[0])
    movies["year"] = parsed.map(lambda t: t[1]).astype("Int64")
    movies["genres"] = movies["genres"].map(
        lambda g: "" if g == "(no genres listed)" else LIST_SEP.join(sorted(g.split("|")))
    )
    movies = movies.merge(links, on="movieId", how="left")
    movies["imdb_id"] = movies["imdbId"].map(lambda x: f"tt{x}" if isinstance(x, str) else "")

    # --- tags: normalized, aggregated per movie, ordered by frequency ---------------------------
    tags["tag_norm"] = tags["tag"].map(normalize_tag)
    tags = tags[tags["tag_norm"].str.len() > 1]
    tag_counts = (
        tags.groupby(["movieId", "tag_norm"]).size().reset_index(name="n")
        .sort_values(["movieId", "n", "tag_norm"], ascending=[True, False, True])
    )
    tag_agg = tag_counts.groupby("movieId")["tag_norm"].agg(lambda s: LIST_SEP.join(s)).rename("tags")
    movies = movies.merge(tag_agg, on="movieId", how="left")
    movies["tags"] = movies["tags"].fillna("")

    # --- Wikidata enrichment (optional) -------------------------------------------------------
    enriched = 0
    enrichment_hash = "none"
    for col in ("directors", "cast", "keywords", "wd_genres", "countries", "description"):
        movies[col] = ""
    movies["runtime_min"] = pd.array([pd.NA] * len(movies), dtype="Int64")
    movies["director_source"] = ""
    if enrichment_path is not None and enrichment_path.exists():
        raw_bytes = enrichment_path.read_bytes()
        enrichment_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
        wd: dict[str, dict[str, Any]] = json.loads(raw_bytes)
        rows = []
        for imdb in movies["imdb_id"]:
            rec = wd.get(imdb, {})
            directors = rec.get("directors") or rec.get("directors_from_description") or []
            source = "wikidata" if rec.get("directors") else (
                "wikidata_description" if rec.get("directors_from_description") else ""
            )
            rows.append({
                "directors": _join(directors),
                "director_source": source,
                "cast": _join(rec.get("cast")),
                "keywords": _join(rec.get("keywords")),
                "wd_genres": _join(rec.get("wd_genres")),
                "countries": _join(rec.get("countries")),
                "description": rec.get("description", ""),
                "runtime_min": rec.get("runtime_min"),
            })
        enr = pd.DataFrame(rows, index=movies.index)
        for col in enr.columns:
            movies[col] = enr[col]
        movies["runtime_min"] = pd.to_numeric(movies["runtime_min"], errors="coerce").astype("Int64")
        enriched = int((movies["wd_genres"] != "").sum() + 0)
        log.info("merged Wikidata enrichment for %d movies", enriched)
    else:
        log.warning("no Wikidata enrichment found; content features use MovieLens fields only")

    # --- interactions ---------------------------------------------------------------------------
    ratings = (
        ratings.sort_values(["userId", "movieId", "timestamp"])
        .drop_duplicates(["userId", "movieId"], keep="last")
        .rename(columns={"userId": "user_id", "movieId": "movie_id"})
    )
    stats = ratings.groupby("movie_id")["rating"].agg(n_ratings="size", mean_rating="mean")
    movies = movies.merge(stats, left_on="movieId", right_index=True, how="left")
    movies["n_ratings"] = movies["n_ratings"].fillna(0).astype(int)
    movies["mean_rating"] = movies["mean_rating"].round(3)

    movies_out = movies.rename(columns={"movieId": "movie_id", "title": "raw_title", "tmdbId": "tmdb_id"})[
        [
            "movie_id", "clean_title", "raw_title", "year", "genres", "tags", "directors",
            "director_source", "cast", "keywords", "wd_genres", "countries", "description",
            "runtime_min", "imdb_id", "tmdb_id", "n_ratings", "mean_rating",
        ]
    ].rename(columns={"clean_title": "title"})

    out_dir.mkdir(parents=True, exist_ok=True)
    movies_out.to_csv(out_dir / "movies.csv", index=False)
    ratings[["user_id", "movie_id", "rating", "timestamp"]].to_csv(out_dir / "interactions.csv", index=False)

    source_md5 = (raw_dir / "SOURCE_MD5").read_text().strip() if (raw_dir / "SOURCE_MD5").exists() else "unknown"
    dataset_version = f"ml-latest-small-{source_md5[:8]}-wd-{enrichment_hash}"
    coverage = {
        col: float(np.mean(movies_out[col].fillna("").astype(str) != ""))
        for col in ("genres", "tags", "directors", "cast", "keywords", "wd_genres", "description")
    }
    meta = {
        "dataset_version": dataset_version,
        "source": "MovieLens ml-latest-small (GroupLens) + Wikidata (CC0)",
        "source_md5": source_md5,
        "enrichment_sha256_12": enrichment_hash,
        "validation": {"errors": report.errors, "warnings": report.warnings, **report.stats},
        "movies_fields": list(movies_out.columns),
        "interactions_fields": ["user_id", "movie_id", "rating", "timestamp"],
        "n_movies": len(movies_out),
        "n_interactions": len(ratings),
        "n_users": int(ratings["user_id"].nunique()),
        "metadata_coverage": coverage,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("processed dataset %s: %s", dataset_version, {k: meta[k] for k in ("n_movies", "n_interactions", "n_users")})
    return meta
