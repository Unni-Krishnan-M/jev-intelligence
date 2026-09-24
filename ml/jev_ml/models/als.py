"""Implicit-feedback matrix factorization with Alternating Least Squares.

Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit Feedback Datasets".
  preference p_ui = 1 if user u interacted with item i, else 0
  confidence c_ui = 1 + alpha * w_ui          (w_ui from signals.rating_weight)
  minimize  sum_ui c_ui (p_ui - x_u.y_i)^2 + reg * (||X||^2 + ||Y||^2)

Every half-step is an exact least-squares solve that uses the YtY precomputation trick:
    x_u = (YtY + Y_u^T (C_u - I) Y_u + reg*I)^-1  Y_u^T c_u
The fold-in for users unseen in training is the same solve against the fixed item factors, so any
JEV user can be scored as soon as they have one interaction, with no retraining.

The paper (§5) also gives an exact explanation of a score:
    s_ui = sum_j (y_i^T W_u y_j) c_uj,   where W_u = (Y^T C_u Y + reg*I)^-1
We use it to attribute a recommendation to the past items that drive it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from jev_ml.models.base import Contribution, Recommender, TrainContext
from jev_ml.signals import UserProfile

log = logging.getLogger(__name__)


class ALSRecommender(Recommender):
    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 10.0,
        iterations: int = 15,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors = np.zeros((0, factors))
        self.item_factors = np.zeros((0, factors))
        self.loss_history: list[float] = []
        self.train_seconds = 0.0

    # --- core solver ----------------------------------------------------------------------------
    def _solve_side(self, conf: sp.csr_matrix, fixed: np.ndarray, out: np.ndarray) -> None:
        f = fixed.shape[1]
        gram = fixed.T @ fixed
        reg = self.regularization * np.eye(f)
        indptr, indices, data = conf.indptr, conf.indices, conf.data
        for row in range(conf.shape[0]):
            s, e = indptr[row], indptr[row + 1]
            if s == e:
                out[row] = 0.0
                continue
            idx = indices[s:e]
            c = data[s:e]  # confidence c_ui = 1 + alpha * w
            y = fixed[idx]
            a = gram + (y.T * (c - 1.0)) @ y + reg
            b = y.T @ c
            out[row] = np.linalg.solve(a, b)

    def _loss(self, conf: sp.csr_matrix) -> float:
        x, y = self.user_factors, self.item_factors
        coo = conf.tocoo()
        pred_nnz = np.einsum("ij,ij->i", x[coo.row], y[coo.col])
        all_sq = float(np.sum((x.T @ x) * (y.T @ y)))
        nnz_term = float(np.sum(coo.data * (1.0 - pred_nnz) ** 2))
        zero_term = all_sq - float(np.sum(pred_nnz**2))
        reg = self.regularization * (float(np.sum(x**2)) + float(np.sum(y**2)))
        return (nnz_term + zero_term + reg) / conf.nnz

    def fit(self, ctx: TrainContext) -> ALSRecommender:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        w = ctx.user_item.tocsr().astype(np.float64)
        conf_ui = w.copy()
        conf_ui.data = 1.0 + self.alpha * conf_ui.data
        conf_iu = conf_ui.T.tocsr()
        n_users, n_items = w.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))
        self.loss_history = []
        for it in range(self.iterations):
            self._solve_side(conf_ui, self.item_factors, self.user_factors)
            self._solve_side(conf_iu, self.user_factors, self.item_factors)
            loss = self._loss(conf_ui)
            self.loss_history.append(loss)
            log.debug("als iter %d loss %.5f", it + 1, loss)
        self.train_seconds = time.perf_counter() - t0
        return self

    # --- inference ------------------------------------------------------------------------------
    def _fold_in_system(self, profile: UserProfile) -> tuple[np.ndarray, np.ndarray]:
        y = self.item_factors[profile.items]
        c = 1.0 + self.alpha * profile.weights
        a = self.item_factors.T @ self.item_factors + (y.T * (c - 1.0)) @ y
        a += self.regularization * np.eye(self.factors)
        return a, c

    def user_vector(self, profile: UserProfile) -> np.ndarray | None:
        if profile.train_user_index is not None and profile.train_user_index < len(self.user_factors):
            return self.user_factors[profile.train_user_index]
        if profile.n_interactions == 0:
            return None
        a, c = self._fold_in_system(profile)
        vec: np.ndarray = np.linalg.solve(a, self.item_factors[profile.items].T @ c)
        return vec

    def score(self, profile: UserProfile) -> np.ndarray:
        u = self.user_vector(profile)
        if u is None:
            return np.zeros(len(self.item_factors))
        return self.item_factors @ u

    def explain(self, profile: UserProfile, item: int) -> list[Contribution]:
        if profile.n_interactions == 0:
            return []
        a, c = self._fold_in_system(profile)
        wy = np.linalg.solve(a, self.item_factors[item])  # W_u y_i  (W_u symmetric)
        contrib = (self.item_factors[profile.items] @ wy) * c
        order = np.argsort(-contrib)[:3]
        return [
            Contribution(kind="item", item=int(profile.items[j]), value=float(contrib[j]))
            for j in order
            if contrib[j] > 0
        ]

    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "als_factors.npz",
            user_factors=self.user_factors.astype(np.float32),
            item_factors=self.item_factors.astype(np.float32),
        )
        self._write_json(
            path / "als.json",
            {**self.params(), "loss_history": self.loss_history, "train_seconds": self.train_seconds},
        )

    @classmethod
    def load(cls, path: Path) -> ALSRecommender:
        p = cls._read_json(path / "als.json")
        obj = cls(p["factors"], p["regularization"], p["alpha"], p["iterations"], p["seed"])
        f = np.load(path / "als_factors.npz")
        obj.user_factors = f["user_factors"].astype(np.float64)
        obj.item_factors = f["item_factors"].astype(np.float64)
        obj.loss_history = p.get("loss_history", [])
        obj.train_seconds = p.get("train_seconds", 0.0)
        return obj
