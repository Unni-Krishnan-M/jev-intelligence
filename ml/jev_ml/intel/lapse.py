"""PREDICT (lapse): P(an active user rates nothing in the next `horizon_days`).

Samples: at cutoffs every `lapse_cutoff_step_months` (1 Jan / 1 Jul), every user with >= 1 rating
in the `lapse_lookback_days` before the cutoff. Features use only ratings *before* the cutoff; the
label is "no rating in [cutoff, cutoff + horizon)". Only cutoffs whose label window closes <= as_of
are used, so labels never peek past as_of.

Temporal holdout: the latest `lapse_test_frac` of cutoffs are the test set; training cutoffs whose
label windows would overlap the first test cutoff are dropped. Within training, the latest
`lapse_val_frac` of cutoffs is a validation slice used only to choose between raw logistic
probabilities and Platt scaling (by validation ECE). The evaluated model is exactly the one used to
score users at as_of.

Metrics on the test cutoffs: AUC, Brier, ECE (10 equal-width bins), base rate, and the AUC of the
recency rule (days since last rating) as a baseline; reliability bins for the calibration plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from jev_ml.intel.common import fnum, short_hash
from jev_ml.intel.config import LAPSE_VERSION, IntelConfig

FEATURES = (
    "log_days_since_last",
    "log_tenure_days",
    "log_n_ratings",
    "log_n_lookback",
    "log_n_90d",
    "log_active_days_lookback",
    "share_last_30d",
    "mean_rating",
    "rating_std",
)
DAY = 86400.0


def features_at(ratings: pd.DataFrame, cutoff_ts: float, cfg: IntelConfig) -> pd.DataFrame:
    """Features per user active in the lookback window before `cutoff_ts` (ratings < cutoff)."""
    past = ratings[ratings["timestamp"] < cutoff_ts]
    look = past[past["timestamp"] >= cutoff_ts - cfg.lapse_lookback_days * DAY]
    if not len(look):
        return pd.DataFrame(columns=list(FEATURES))
    users = np.unique(look["user_id"].to_numpy())
    past = past[past["user_id"].isin(users)]
    g = past.groupby("user_id")
    first = g["timestamp"].min()
    last = g["timestamp"].max()
    n = g.size()
    gl = look.groupby("user_id")
    n_look = gl.size()
    days_look = look.assign(d=(look["timestamp"] // DAY)).groupby("user_id")["d"].nunique()
    r90 = look[look["timestamp"] >= cutoff_ts - 90 * DAY].groupby("user_id").size()
    r30 = look[look["timestamp"] >= cutoff_ts - 30 * DAY].groupby("user_id").size()
    f = pd.DataFrame(index=pd.Index(users, name="user_id"))
    f["days_since_last"] = (cutoff_ts - last.reindex(users)) / DAY
    f["log_days_since_last"] = np.log1p(f["days_since_last"])
    f["log_tenure_days"] = np.log1p((cutoff_ts - first.reindex(users)) / DAY)
    f["log_n_ratings"] = np.log1p(n.reindex(users))
    f["log_n_lookback"] = np.log1p(n_look.reindex(users))
    f["log_n_90d"] = np.log1p(r90.reindex(users).fillna(0))
    f["log_active_days_lookback"] = np.log1p(days_look.reindex(users))
    f["share_last_30d"] = (r30.reindex(users).fillna(0) / n_look.reindex(users)).astype(float)
    f["mean_rating"] = g["rating"].mean().reindex(users)
    f["rating_std"] = g["rating"].std(ddof=0).reindex(users).fillna(0.0)
    return f


def label_at(ratings: pd.DataFrame, users: np.ndarray, cutoff_ts: float, horizon_days: int) -> np.ndarray:
    win = ratings[
        (ratings["timestamp"] >= cutoff_ts) & (ratings["timestamp"] < cutoff_ts + horizon_days * DAY)
    ]
    active = set(win["user_id"].unique().tolist())
    return np.array([int(int(u) not in active) for u in users], dtype=np.int64)  # 1 = lapsed


def cutoffs(first_ts: float, as_of_ts: float, cfg: IntelConfig) -> list[pd.Timestamp]:
    """Cutoffs from 1 Jan of max(first data year, lapse_first_cutoff_year), every step months,
    whose label window closes <= as_of."""
    first_year = pd.Timestamp(first_ts, unit="s").year
    c = pd.Timestamp(year=max(first_year, cfg.lapse_first_cutoff_year), month=1, day=1)
    horizon = pd.Timedelta(days=cfg.lapse_horizon_days)
    as_of = pd.Timestamp(as_of_ts, unit="s")
    out = []
    while c + horizon <= as_of:
        out.append(c)
        c = c + pd.DateOffset(months=cfg.lapse_cutoff_step_months)
    return out


def _ts(c: pd.Timestamp) -> float:
    return float(c.tz_localize("UTC").timestamp()) if c.tzinfo is None else float(c.timestamp())


def ece_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(p) * abs(pred - obs)
        bins.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "predicted": fnum(pred, 4),
                "observed": fnum(obs, 4),
                "n": int(m.sum()),
            }
        )
    return float(ece), bins


class LapseModel:
    """Standardise -> logistic regression -> optional Platt scaling."""

    def __init__(self, cfg: IntelConfig) -> None:
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=cfg.lapse_c, max_iter=1000)
        self.platt: LogisticRegression | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> LapseModel:
        self.lr.fit(self.scaler.fit_transform(x), y)
        return self

    def raw_logit(self, x: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.lr.decision_function(self.scaler.transform(x))
        return out

    def fit_platt(self, x: np.ndarray, y: np.ndarray) -> None:
        self.platt = LogisticRegression(C=1e6, max_iter=1000).fit(self.raw_logit(x)[:, None], y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.raw_logit(x)
        if self.platt is not None:
            p: np.ndarray = self.platt.predict_proba(z[:, None])[:, 1]
            return p
        return 1.0 / (1.0 + np.exp(-z))


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def build_samples(ratings: pd.DataFrame, as_of_ts: float, cfg: IntelConfig) -> pd.DataFrame:
    rows = []
    for c in cutoffs(float(ratings["timestamp"].min()), as_of_ts, cfg):
        cts = _ts(c)
        f = features_at(ratings, cts, cfg)
        if not len(f):
            continue
        f = f.assign(y=label_at(ratings, f.index.to_numpy(), cts, cfg.lapse_horizon_days), cutoff=c)
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_lapse(
    ratings: pd.DataFrame, as_of_ts: float, cfg: IntelConfig, data_version: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Returns (contract `lapse` object, per-user scores at as_of)."""
    params = [cfg.lapse_horizon_days, cfg.lapse_lookback_days, cfg.lapse_c, data_version, as_of_ts]
    version = f"{LAPSE_VERSION}-{short_hash(params)}"
    base: dict[str, Any] = {
        "model_version": version,
        "horizon_days": cfg.lapse_horizon_days,
        "features": list(FEATURES),
        "metrics": None,
        "calibration": [],
        "population": None,
        "top": [],
        "status": "insufficient_data",
        "detail": None,
    }
    empty = pd.DataFrame()
    samples = build_samples(ratings, as_of_ts, cfg)
    if len(samples) < cfg.lapse_min_samples:
        base["detail"] = f"{len(samples)} labelled samples before as_of (need >= {cfg.lapse_min_samples})"
        return base, empty
    cuts = sorted(samples["cutoff"].unique())
    n_test = max(1, round(cfg.lapse_test_frac * len(cuts)))
    test_cuts = cuts[-n_test:]
    first_test = pd.Timestamp(test_cuts[0])
    horizon = pd.Timedelta(days=cfg.lapse_horizon_days)
    train_cuts = [c for c in cuts[:-n_test] if pd.Timestamp(c) + horizon <= first_test]
    n_val = max(1, round(cfg.lapse_val_frac * len(train_cuts)))
    fit_cuts, val_cuts = train_cuts[:-n_val], train_cuts[-n_val:]
    train = samples[samples["cutoff"].isin(train_cuts)]
    fit = samples[samples["cutoff"].isin(fit_cuts)]
    val = samples[samples["cutoff"].isin(val_cuts)]
    test = samples[samples["cutoff"].isin(test_cuts)]
    if (
        len(fit) < cfg.lapse_min_samples // 2
        or len(test) < cfg.lapse_min_test
        or train["y"].nunique() < 2
        or test["y"].nunique() < 2
    ):
        base["detail"] = (
            f"not enough labelled samples per split: fit {len(fit)}, val {len(val)}, test {len(test)} "
            f"(need >= {cfg.lapse_min_samples // 2} fit and {cfg.lapse_min_test} test, both classes)"
        )
        return base, empty
    cols = list(FEATURES)
    # calibration choice on validation: raw LR (fit on fit-cutoffs) vs Platt (fit on val)
    raw_model = LapseModel(cfg).fit(fit[cols].to_numpy(), fit["y"].to_numpy())
    calib = "none"
    if val["y"].nunique() == 2 and len(val) >= 30:
        p_raw_val = raw_model.predict(val[cols].to_numpy())
        ece_raw, _ = ece_bins(val["y"].to_numpy(), p_raw_val)
        # Platt judged on the same slice it is fit on would be optimistic -> 2-fold within val
        vv = val.sort_values(["cutoff", "user_id"]).reset_index(drop=True)
        half = np.arange(len(vv)) % 2 == 0
        eces: list[float] = []
        for a, b in ((half, ~half), (~half, half)):
            if vv.loc[a, "y"].nunique() < 2:
                eces = []
                break
            m = LapseModel(cfg).fit(fit[cols].to_numpy(), fit["y"].to_numpy())
            m.fit_platt(vv.loc[a, cols].to_numpy(), vv.loc[a, "y"].to_numpy())
            e, _ = ece_bins(vv.loc[b, "y"].to_numpy(), m.predict(vv.loc[b, cols].to_numpy()))
            eces.append(e)
        if eces and float(np.mean(eces)) < ece_raw:
            calib = "platt"
    if calib == "platt":
        model = raw_model
        model.fit_platt(val[cols].to_numpy(), val["y"].to_numpy())
    else:
        model = LapseModel(cfg).fit(train[cols].to_numpy(), train["y"].to_numpy())
    y_test = test["y"].to_numpy()
    p_test = model.predict(test[cols].to_numpy())
    ece, bins = ece_bins(y_test, p_test)
    # recency-rule baseline: days since last rating as the score; its probability is the training
    # lapse rate of the same recency decile (so it has a Brier score too)
    rec_tr = train["days_since_last"].to_numpy()
    edges = np.unique(np.quantile(rec_tr, np.linspace(0, 1, 11)))
    bin_tr = np.clip(np.digitize(rec_tr, edges[1:-1]), 0, len(edges) - 2)
    rates = np.array(
        [
            train["y"].to_numpy()[bin_tr == b].mean() if (bin_tr == b).any() else train["y"].mean()
            for b in range(len(edges) - 1)
        ]
    )
    rec_te = test["days_since_last"].to_numpy()
    p_rule = rates[np.clip(np.digitize(rec_te, edges[1:-1]), 0, len(edges) - 2)]
    base_rate_train = float(train["y"].mean())
    metrics = {
        "auc": fnum(_safe_auc(y_test, p_test), 4),
        "brier": fnum(brier_score_loss(y_test, p_test), 4),
        "ece": fnum(ece, 4),
        "base_rate": fnum(y_test.mean(), 4),
        "train_base_rate": fnum(base_rate_train, 4),
        "baseline_auc": fnum(_safe_auc(y_test, rec_te), 4),
        "baseline_brier": fnum(brier_score_loss(y_test, p_rule), 4),
        "base_rate_brier": fnum(brier_score_loss(y_test, np.full(len(y_test), base_rate_train)), 4),
        "calibration_method": calib,
        "n_train": len(train),
        "n_test": len(test),
        "train_cutoffs": [pd.Timestamp(c).strftime("%Y-%m-%d") for c in train_cuts],
        "test_cutoffs": [pd.Timestamp(c).strftime("%Y-%m-%d") for c in test_cuts],
    }
    # score at as_of
    now_f = features_at(ratings, as_of_ts + 1e-3, cfg)
    base.update({"metrics": metrics, "calibration": bins, "status": "ok"})
    if not len(now_f):
        base["population"] = {
            "n_scored": 0,
            "expected_lapses": 0.0,
            "high_risk": 0,
            "threshold": cfg.lapse_high_risk,
        }
        base["detail"] = f"no users active in the {cfg.lapse_lookback_days} days before as_of"
        return base, empty
    p_now = model.predict(now_f[cols].to_numpy())
    scored = now_f.assign(p=p_now)
    scored["n_lookback"] = np.expm1(scored["log_n_lookback"]).round().astype(int)
    base["population"] = {
        "n_scored": len(scored),
        "expected_lapses": fnum(p_now.sum(), 3),
        "mean_p": fnum(p_now.mean(), 4),
        "high_risk": int((p_now >= cfg.lapse_high_risk).sum()),
        "threshold": cfg.lapse_high_risk,
    }
    top = (
        scored.reset_index()
        .sort_values(["p", "user_id"], ascending=[False, True], kind="mergesort")
        .head(cfg.top_n)
    )
    base["top"] = [
        {
            "user_id": int(r["user_id"]),
            "p": fnum(r["p"], 4),
            "features": {
                "days_since_last": fnum(r["days_since_last"], 1),
                "ratings_lookback": int(r["n_lookback"]),
                "ratings_90d": round(float(np.expm1(r["log_n_90d"]))),
                "tenure_days": fnum(np.expm1(r["log_tenure_days"]), 1),
                "mean_rating": fnum(r["mean_rating"], 3),
            },
        }
        for _, r in top.iterrows()
    ]
    return base, scored
