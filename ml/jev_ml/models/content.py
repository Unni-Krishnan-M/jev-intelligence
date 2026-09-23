"""Content-based recommender: TF-IDF item vectors, signed user taste vector, cosine scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from jev_ml.features import DEFAULT_FIELD_WEIGHTS, ContentFeaturizer
from jev_ml.models.base import Contribution, Recommender, TrainContext, sparse_topk_rows
from jev_ml.signals import UserProfile

# prefix -> explanation "kind" for feature contributions
FEATURE_KIND = {"d:": "director", "c:": "cast", "k:": "keyword", "t:": "tag", "g:": "genre",
                "w:": "title", "o:": "description", "y:": "decade"}


class ContentRecommender(Recommender):
    name = "content"

    def __init__(self, field_weights: dict[str, float] | None = None, min_df: int = 2,
                 n_neighbors: int = 50) -> None:
        self.field_weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)
        self.min_df = min_df
        self.n_neighbors = n_neighbors
        self.featurizer = ContentFeaturizer(self.field_weights, min_df)
        self.item_vectors: sp.csr_matrix = sp.csr_matrix((0, 0), dtype=np.float32)
        self.neighbors: sp.csr_matrix = sp.csr_matrix((0, 0), dtype=np.float32)
        self._feature_names: list[str] = []

    # --- training -------------------------------------------------------------------------------
    def fit(self, ctx: TrainContext) -> ContentRecommender:
        self.item_vectors = self.featurizer.fit_transform(ctx.movies)
        self._feature_names = self.featurizer.feature_names()
        self.neighbors = self._compute_neighbors(self.item_vectors, self.n_neighbors)
        return self

    @staticmethod
    def _compute_neighbors(x: sp.csr_matrix, k: int, block: int = 1024) -> sp.csr_matrix:
        n = x.shape[0]
        xt = x.T.tocsc()
        rows, cols, vals = [], [], []
        for start in range(0, n, block):
            sims = (x[start : start + block] @ xt).toarray()
            r, c, v = sparse_topk_rows(sims, k, start)
            rows.append(r)
            cols.append(c)
            vals.append(v)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n), dtype=np.float32)

    # --- inference ------------------------------------------------------------------------------
    def user_vector(self, profile: UserProfile) -> np.ndarray | None:
        if profile.n_interactions == 0:
            return None
        w = profile.pref_weights
        if not np.any(w > 0):
            return None
        vec = np.asarray(self.item_vectors[profile.items].T @ w).ravel()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def score(self, profile: UserProfile) -> np.ndarray:
        u = self.user_vector(profile)
        if u is None:
            return np.zeros(self.item_vectors.shape[0])
        return np.asarray(self.item_vectors @ u).ravel().astype(np.float64)

    def similar(self, item: int, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        row = self.neighbors.getrow(item)
        order = np.lexsort((row.indices, -row.data))[:k]
        return row.indices[order].astype(np.int64), row.data[order].astype(np.float64)

    def similarity_to(self, item: int, others: np.ndarray) -> np.ndarray:
        if len(others) == 0:
            return np.zeros(0)
        return np.asarray((self.item_vectors[others] @ self.item_vectors[item].T).toarray()).ravel()

    def pairwise(self, items: np.ndarray) -> np.ndarray:
        x = self.item_vectors[items]
        return np.asarray((x @ x.T).toarray())

    def vectorize_new(self, movies: pd.DataFrame | list[dict]) -> sp.csr_matrix:
        """Vectorize movies unseen at training time (item cold start)."""
        return self.featurizer.transform(movies)

    def explain(self, profile: UserProfile, item: int) -> list[Contribution]:
        u = self.user_vector(profile)
        if u is None:
            return []
        contribs: list[Contribution] = []
        # 1) which liked item is most similar (anchor for "Because you liked X")
        liked = profile.items[profile.pref_weights > 0]
        if len(liked):
            sims = self.similarity_to(item, liked)
            best = int(np.argmax(sims))
            if sims[best] > 0:
                contribs.append(Contribution(kind="item", item=int(liked[best]), value=float(sims[best])))
        # 2) which shared features carry the dot product
        row = self.item_vectors.getrow(item)
        feat_contrib = row.data * u[row.indices]
        order = np.argsort(-feat_contrib)[:5]
        for j in order:
            if feat_contrib[j] <= 0:
                break
            fname = self._feature_names[row.indices[j]]
            kind = FEATURE_KIND.get(fname[:2], "feature")
            label = self.featurizer.labels.get(fname, fname[2:])
            contribs.append(Contribution(kind=kind, feature=fname, label=label, value=float(feat_contrib[j])))
        contribs.sort(key=lambda c: -c.value)
        return contribs

    # --- persistence ----------------------------------------------------------------------------
    def params(self) -> dict[str, Any]:
        return {"field_weights": self.field_weights, "min_df": self.min_df,
                "n_neighbors": self.n_neighbors}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "item_vectors.npz", self.item_vectors)
        sp.save_npz(path / "neighbors.npz", self.neighbors)
        (path / "featurizer.json").write_text(json.dumps(self.featurizer.state(), ensure_ascii=False))
        self._write_json(path / "content.json", self.params())

    @classmethod
    def load(cls, path: Path) -> ContentRecommender:
        p = cls._read_json(path / "content.json")
        obj = cls(p["field_weights"], p["min_df"], p["n_neighbors"])
        obj.item_vectors = sp.load_npz(path / "item_vectors.npz").tocsr().astype(np.float32)
        obj.neighbors = sp.load_npz(path / "neighbors.npz").tocsr()
        obj.featurizer = ContentFeaturizer.from_state(json.loads((path / "featurizer.json").read_text()))
        obj._feature_names = obj.featurizer.feature_names()
        return obj
