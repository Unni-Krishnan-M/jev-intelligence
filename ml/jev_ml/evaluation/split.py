"""Train / validation / test splitting.

Default strategy — `user_temporal` (leave-latest-out per user):
  Each user's ratings are sorted by (timestamp, movie_id). The latest `test_frac` form the test set,
  the `val_frac` before those form validation, and everything earlier is training data.
  → no user's future ratings are ever used to predict their past (no per-user temporal leakage),
  → every evaluated user has history in training (the protocol measures personalization, not
    cold start; cold start is evaluated separately with truncated profiles).
  Caveat: other users' interactions that occur after a user's test timestamp may still be in
  training. `global_temporal` removes that leakage at the cost of fewer evaluable users.

Alternative — `global_temporal`: two global timestamp cut points at the (1-test-val) and
  (1-test) quantiles. Items consumed after cut 2 are test, between cuts are validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: str

    def summary(self) -> dict[str, int | str]:
        return {
            "strategy": self.strategy,
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "test_rows": len(self.test),
            "train_users": int(self.train["user_id"].nunique()),
            "val_users": int(self.val["user_id"].nunique()),
            "test_users": int(self.test["user_id"].nunique()),
        }


def user_temporal_split(
    df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2, min_train: int = 3
) -> Split:
    df = df.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort").reset_index(drop=True)
    pos = df.groupby("user_id").cumcount().to_numpy()
    n = df.groupby("user_id")["movie_id"].transform("size").to_numpy()
    n_test = np.maximum(1, np.round(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.round(n * val_frac)).astype(int)
    # users too small to split keep everything in train
    small = n < (min_train + 2)
    n_test = np.where(small, 0, n_test)
    n_val = np.where(small, 0, n_val)
    test_start = n - n_test
    val_start = test_start - n_val
    is_test = pos >= test_start
    is_val = (pos >= val_start) & ~is_test
    return Split(
        train=df[~is_test & ~is_val].reset_index(drop=True),
        val=df[is_val].reset_index(drop=True),
        test=df[is_test].reset_index(drop=True),
        strategy="user_temporal",
    )


def global_temporal_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2) -> Split:
    t1, t2 = np.quantile(df["timestamp"], [1 - test_frac - val_frac, 1 - test_frac])
    train = df[df["timestamp"] < t1]
    val = df[(df["timestamp"] >= t1) & (df["timestamp"] < t2)]
    test = df[df["timestamp"] >= t2]
    return Split(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        "global_temporal",
    )


def make_split(
    df: pd.DataFrame, strategy: str = "user_temporal", val_frac: float = 0.1, test_frac: float = 0.2
) -> Split:
    if strategy == "user_temporal":
        return user_temporal_split(df, val_frac, test_frac)
    if strategy == "global_temporal":
        return global_temporal_split(df, val_frac, test_frac)
    raise ValueError(f"unknown split strategy {strategy!r}")
