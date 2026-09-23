"""Item-item collaborative filtering over the sparse implicit user × item matrix.

sim(i, j) = <x_i, x_j> / (||x_i|| ||x_j|| + shrinkage), keeping the top-k neighbours per item.
The shrinkage term damps similarities that rest on only a few co-occurrences.
score(u, j) = sum over i in history(u) of w_ui * sim(i, j), where sparse vector-matrix products
make this cheap even for users the model has never seen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from jev_ml.models.base import Contribution, Recommender, TrainContext, sparse_topk_rows
from jev_ml.signals import UserProfile


class ItemKNNRecommender(Recommender):
    name = "itemknn"

    def __init__(self, k: int = 100, shrinkage: float = 10.0, binary: bool = False,
                 block_size: int = 1024) -> None:
        self.k = k
        self.shrinkage = shrinkage
        self.binary = binary
        self.block_size = block_size
        self.sim: sp.csr_matrix = sp.csr_matrix((0, 0), dtype=np.float32)

    def fit(self, ctx: TrainContext) -> ItemKNNRecommender:
        x = ctx.user_item.astype(np.float32).tocsc()
        if self.binary:
            x.data[:] = 1.0
        n = x.shape[1]
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel()).astype(np.float32)
        xt = x.T.tocsr()
        rows, cols, vals = [], [], []
        for start in range(0, n, self.block_size):
            stop = min(start + self.block_size, n)
            dots = (xt[start:stop] @ x).toarray()
            denom = np.outer(norms[start:stop], norms) + self.shrinkage
            sims = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
            r, c, v = sparse_topk_rows(sims, self.k, start)
            rows.append(r)
            cols.append(c)
            vals.append(v)
        self.sim = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(n, n), dtype=np.float32)
        return self

    def score(self, profile: UserProfile) -> np.ndarray:
        n = self.sim.shape[0]
        if profile.n_interactions == 0:
            return np.zeros(n)
        u = sp.csr_matrix((profile.weights.astype(np.float32),
                           (np.zeros(len(profile.items), dtype=np.int64), profile.items)), shape=(1, n))
        return np.asarray((u @ self.sim).toarray()).ravel().astype(np.float64)

    def explain(self, profile: UserProfile, item: int) -> list[Contribution]:
        if profile.n_interactions == 0:
            return []
        col = np.asarray(self.sim[profile.items, item].toarray()).ravel()
        contrib = col * profile.weights
        order = np.argsort(-contrib)[:3]
        return [Contribution(kind="item", item=int(profile.items[j]), value=float(contrib[j]))
                for j in order if contrib[j] > 0]

    def params(self) -> dict[str, Any]:
        return {"k": self.k, "shrinkage": self.shrinkage, "binary": self.binary}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "item_similarity.npz", self.sim)
        self._write_json(path / "itemknn.json", self.params())

    @classmethod
    def load(cls, path: Path) -> ItemKNNRecommender:
        p = cls._read_json(path / "itemknn.json")
        obj = cls(p["k"], p["shrinkage"], p["binary"])
        obj.sim = sp.load_npz(path / "item_similarity.npz").tocsr()
        return obj
