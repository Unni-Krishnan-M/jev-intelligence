"""Content feature engineering: per-field TF-IDF with field weights → one L2-normalized item vector.

Every field is vectorized on its own (so a director token never competes with a plot word for IDF
mass). Each field matrix is scaled by sqrt(field_weight) before stacking, which makes the cosine
of two items approximately the field-weighted sum of per-field cosines.

Tokens carry a field prefix (`d:` director, `c:` cast, ...). The prefix lets explanations map a
contributing feature back to a human-readable reason, such as "your interest in Christopher Nolan".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from jev_ml.data.dataset import split_list

FIELD_PREFIX = {
    "genres": "g:",
    "directors": "d:",
    "cast": "c:",
    "keywords": "k:",
    "tags": "t:",
    "title": "w:",
    "description": "o:",
    "decade": "y:",
}

DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "genres": 1.0,
    "directors": 1.0,
    "cast": 0.6,
    "keywords": 0.8,
    "tags": 0.8,
    "title": 0.2,
    "description": 0.3,
    "decade": 0.3,
}

_EXTRA_STOP = {
    "film",
    "films",
    "movie",
    "directed",
    "director",
    "american",
    "british",
    "french",
    "starring",
    "part",
    "episode",
    "series",
    "animated",
    "short",
    "feature",
    "documentary",
    "english",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS | _EXTRA_STOP)
_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]+")
_WD_GENRE_SUFFIX = re.compile(r"\s+(film|movie)$")

# MovieLens genre names -> the same normalized vocabulary as Wikidata genres
_ML_GENRE_ALIASES = {"sci-fi": "science fiction", "children": "children's", "film-noir": "film noir"}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value.lower()).strip()


def normalize_genre(value: str) -> str:
    g = normalize_text(value)
    g = _ML_GENRE_ALIASES.get(g, g)
    g = _WD_GENRE_SUFFIX.sub("", g)
    return g.replace(" film", "").strip()


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(normalize_text(text)) if w not in STOP_WORDS and not w.isdigit()]


def movie_field_tokens(row: pd.Series | dict) -> dict[str, list[str]]:
    """Field → prefixed tokens for one movie row (processed movies.csv schema)."""
    genres = {normalize_genre(g) for g in split_list(row.get("genres"))}
    genres |= {normalize_genre(g) for g in split_list(row.get("wd_genres"))}
    year = row.get("year")
    decade: list[str] = []
    if year is not None and not pd.isna(year):
        decade = [f"{int(year) // 10 * 10}s"]
    tokens = {
        "genres": sorted(g for g in genres if g),
        "directors": [normalize_text(d) for d in split_list(row.get("directors"))],
        "cast": [normalize_text(c) for c in split_list(row.get("cast"))],
        "keywords": [normalize_text(k) for k in split_list(row.get("keywords"))],
        "tags": [normalize_text(t) for t in split_list(row.get("tags"))[:40]],
        "title": words(str(row.get("title") or "")),
        "description": words(str(row.get("description") or "")),
        "decade": decade,
    }
    return {f: [FIELD_PREFIX[f] + t for t in toks] for f, toks in tokens.items()}


def display_labels(row: pd.Series | dict) -> dict[str, str]:
    """Prefixed token → original-case label (for explanations)."""
    labels: dict[str, str] = {}
    for fname, prefix in (("directors", "d:"), ("cast", "c:"), ("keywords", "k:"), ("tags", "t:")):
        for v in split_list(row.get(fname)):
            labels.setdefault(prefix + normalize_text(v), v)
    for g in split_list(row.get("genres")) + split_list(row.get("wd_genres")):
        labels.setdefault("g:" + normalize_genre(g), normalize_genre(g).title().replace("'S", "'s"))
    return labels


def _identity(tokens: list[str]) -> list[str]:
    return tokens


@dataclass
class ContentFeaturizer:
    field_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS))
    min_df: int = 2
    vectorizers: dict[str, TfidfVectorizer] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def _make(self, fname: str) -> TfidfVectorizer:
        analyzer: Callable[[list[str]], list[str]] = _identity
        return TfidfVectorizer(
            analyzer=analyzer,
            sublinear_tf=True,
            min_df=1 if fname == "decade" else self.min_df,
            dtype=np.float32,
        )

    def fit_transform(self, movies: pd.DataFrame) -> sp.csr_matrix:
        docs = [movie_field_tokens(r) for r in movies.to_dict("records")]
        for r in movies.to_dict("records"):
            for k, v in display_labels(r).items():
                self.labels.setdefault(k, v)
        blocks = []
        for fname, w in self.field_weights.items():
            if w <= 0:
                continue
            vec = self._make(fname)
            field_docs = [d[fname] for d in docs]
            if not any(field_docs):
                continue
            try:
                x = vec.fit_transform(field_docs)
            except ValueError:  # empty vocabulary after min_df (e.g. field absent without enrichment)
                continue
            self.vectorizers[fname] = vec
            blocks.append(x * np.float32(np.sqrt(w)))
        mat = sp.hstack(blocks, format="csr", dtype=np.float32)
        vocab = set(self.feature_names())
        self.labels = {k: v for k, v in self.labels.items() if k in vocab}
        return normalize(mat, norm="l2", axis=1).astype(np.float32).tocsr()

    def transform(self, movies: pd.DataFrame | list[dict]) -> sp.csr_matrix:
        records = movies.to_dict("records") if isinstance(movies, pd.DataFrame) else movies
        docs = [movie_field_tokens(r) for r in records]
        blocks = []
        for fname, vec in self.vectorizers.items():
            x = vec.transform([d[fname] for d in docs])
            blocks.append(x * np.float32(np.sqrt(self.field_weights[fname])))
        mat = sp.hstack(blocks, format="csr", dtype=np.float32)
        return normalize(mat, norm="l2", axis=1).astype(np.float32).tocsr()

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for vec in self.vectorizers.values():
            names.extend(vec.get_feature_names_out().tolist())
        return names

    # --- portable (pickle-free) state ------------------------------------------------------
    def state(self) -> dict:
        return {
            "field_weights": self.field_weights,
            "min_df": self.min_df,
            "fields": {
                f: {
                    "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
                    "idf": vec.idf_.astype(float).tolist(),
                }
                for f, vec in self.vectorizers.items()
            },
            "labels": self.labels,
        }

    @classmethod
    def from_state(cls, state: dict) -> ContentFeaturizer:
        feat = cls(field_weights=state["field_weights"], min_df=state["min_df"], labels=state["labels"])
        for fname, fs in state["fields"].items():
            vec = TfidfVectorizer(
                analyzer=_identity, sublinear_tf=True, dtype=np.float32, vocabulary=fs["vocabulary"]
            )
            vec.idf_ = np.asarray(fs["idf"], dtype=np.float64)
            feat.vectorizers[fname] = vec
        return feat


def item_genre_matrix(
    movies: pd.DataFrame, genres: list[str] | None = None
) -> tuple[sp.csr_matrix, list[str]]:
    """Binary items × MovieLens-genres matrix (the 19 top-level genres users pick at onboarding)."""
    lists = [split_list(g) for g in movies["genres"]]
    if genres is None:
        genres = sorted({g for gl in lists for g in gl})
    gpos = {g: i for i, g in enumerate(genres)}
    rows, cols = [], []
    for i, gl in enumerate(lists):
        for g in gl:
            if g in gpos:
                rows.append(i)
                cols.append(gpos[g])
    mat = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(movies), len(genres))
    )
    return mat, genres
