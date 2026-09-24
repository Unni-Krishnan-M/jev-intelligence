"""Recommender interface shared by all component models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import scipy.sparse as sp

from jev_ml.data.dataset import ItemIndex
from jev_ml.signals import UserProfile


@dataclass
class TrainContext:
    """Training inputs. `user_item` is users × items implicit confidence (see signals.py)."""

    movies: pd.DataFrame  # aligned with item_index (sorted by movie_id)
    item_index: ItemIndex
    interactions: pd.DataFrame  # the rows used to build user_item (train split only)
    user_item: sp.csr_matrix
    user_ids: np.ndarray
    seed: int = 42

    @property
    def n_items(self) -> int:
        return len(self.item_index)


@dataclass
class Contribution:
    """One piece of evidence behind a component score, used to build explanations."""

    kind: str  # "item" | "feature" | "genre" | "popularity" | "recency"
    value: float
    item: int | None = None
    feature: str | None = None
    label: str | None = None


class Recommender(ABC):
    name: ClassVar[str]

    @abstractmethod
    def fit(self, ctx: TrainContext) -> Recommender: ...

    @abstractmethod
    def score(self, profile: UserProfile) -> np.ndarray:
        """Unmasked scores for every item (shape [n_items]); higher is better."""

    def explain(self, profile: UserProfile, item: int) -> list[Contribution]:
        """Evidence for `item`'s score, strongest first. Default: none."""
        return []

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Recommender: ...

    def params(self) -> dict[str, Any]:
        return {}

    # helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def topk_indices(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Deterministic top-k: descending score, ties broken by ascending item index."""
    s = scores.astype(np.float64, copy=True)
    if mask is not None and len(mask):
        s[mask] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    part = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
    # include every item tied with the k-th score so that tie-breaking is by index, not partition
    kth = s[part].min()
    cand = np.flatnonzero(s >= kth)
    order = np.lexsort((cand, -s[cand]))
    return cand[order][:k].astype(np.int64)


def sparse_topk_rows(
    sim_block: np.ndarray, k: int, row_offset: int, exclude_self: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the k largest positive entries per row of a dense block → COO triplets."""
    rows, cols, vals = [], [], []
    n_rows, n_cols = sim_block.shape
    if exclude_self:
        r = np.arange(n_rows)
        c = r + row_offset
        valid = c < n_cols
        sim_block[r[valid], c[valid]] = 0.0
    kk = min(k, n_cols)
    idx = np.argpartition(-sim_block, kk - 1, axis=1)[:, :kk]
    top = np.take_along_axis(sim_block, idx, axis=1)
    for i in range(n_rows):
        keep = top[i] > 0
        rows.append(np.full(int(keep.sum()), i + row_offset, dtype=np.int64))
        cols.append(idx[i][keep].astype(np.int64))
        vals.append(top[i][keep].astype(np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
