"""Calibrated recommendation confidence: P(the user rates a recommended film >= 4).

Why not calibrate the served model directly
-------------------------------------------
The served model version is trained on *all* interactions, so its scores on any held-out rows would
be scores on its own training data (leakage). Instead this module repeats the experiment protocol of
``jev_ml.training`` with the served model's own hyper-parameters (``manifest.training_config``):

* **fit** (validation): component models fitted on the ``train`` part of the recorded split, every
  user's top-``k`` list scored against that user's ``validation`` rows. Isotonic regression maps the
  hybrid score (or the rank) to the observed relevance rate.
* **evaluate** (test): component models refitted on ``train + validation`` (as the experiment's test
  protocol does), top-``k`` lists scored against the user's ``test`` rows. The calibrator fitted
  above is applied unchanged, so the test metrics measure exactly the assumption the served model
  relies on: the score -> relevance mapping learnt from a model fitted on less data transfers to a
  model fitted on more data.

Label definition (important for reading the probability)
-------------------------------------------------------
A recommended film is *relevant* when the user rated it >= ``relevance_threshold`` within the
held-out window. The validation window holds ~``val_frac`` of each user's ratings; the test window
holds ~``test_frac``. A longer window contains more ratings and therefore more hits, so the primary
test evaluation is **window-matched**: each user's test target is truncated to their first ``n``
test ratings, ``n`` = that user's number of validation ratings. The full test window is reported as
a secondary number. Films the user never rated in the window count as not relevant (implicit
negatives, missing-not-at-random), so the probability is a *lower bound* on "would like it".

Method selection (without the test split)
-----------------------------------------
Two one-dimensional isotonic calibrators are considered: on the hybrid ``score`` and on the
``rank``. The one with the lower mean Brier score under a 2-fold cross-fit *within the validation
users* (folds by user id parity) is kept, then refitted on all validation candidates. The test split
is only ever used for the reported evaluation.

At serving time the engine evaluates the stored isotonic knots with ``numpy.interp`` (identical to
``IsotonicRegression.predict`` with ``out_of_bounds="clip"``), so no scikit-learn is needed and the
cost is a few microseconds per recommendation. Ranks beyond the fitted ``k`` get no confidence
(``None``): the calibrator never extrapolates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from jev_ml.data.dataset import (
    ItemIndex,
    load_dataset_meta,
    load_interactions,
    load_movies,
    profiles_from_interactions,
)
from jev_ml.evaluation.split import make_split
from jev_ml.models.content import ContentRecommender
from jev_ml.models.hybrid import HybridConfig, HybridRanker
from jev_ml.paths import MODELS_DIR, PROCESSED_DIR
from jev_ml.signals import LIKE_THRESHOLD

log = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"
CALIBRATION_VERSION = "cal-1.0.0"
FEATURES = ("score", "rank")
N_BINS = 10
DEFAULT_K = 50  # served pages are 20 long and paginate; 50 covers the first pages
STRATA: tuple[int | None, ...] = (0, 3, 10, None)  # profile truncations; None = full profile
DEFAULT_HORIZON = 5  # label horizon: the next 5 ratings (401 of 610 users have >= 5 validation rows)


# --------------------------------------------------------------------------------------------------
# the serving-side calibrator


@dataclass(frozen=True)
class Stratum:
    """One profile-size stratum: an isotonic map feature -> P(relevant) for profiles with
    ``min_profile <= n_interactions <= max_profile`` (``max_profile`` None = no upper bound)."""

    name: str
    min_profile: int
    max_profile: int | None
    feature: str  # "score" | "rank"
    x: np.ndarray
    y: np.ndarray

    def applies(self, n_profile: int) -> bool:
        return n_profile >= self.min_profile and (self.max_profile is None or n_profile <= self.max_profile)

    def value(self, score: np.ndarray, rank: np.ndarray) -> np.ndarray:
        v = np.asarray(score if self.feature == "score" else rank, dtype=np.float64)
        return np.interp(v, self.x, self.y)


@dataclass(frozen=True)
class Calibrator:
    """Serving-side calibrator loaded from calibration.json (profile-size strata of isotonic maps)."""

    strata: tuple[Stratum, ...]
    max_rank: int
    version: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibrator:
        strata = []
        for st in d["strata"]:
            m = st["mapping"]
            x = np.asarray(m["x"], dtype=np.float64)
            y = np.asarray(m["y"], dtype=np.float64)
            if m["feature"] not in FEATURES or len(x) == 0 or len(x) != len(y):
                raise ValueError(f"invalid calibration mapping in stratum {st.get('name')}")
            if np.any(np.diff(x) < 0) or y.min() < 0 or y.max() > 1:
                raise ValueError(f"calibration mapping of {st.get('name')} is not sorted within [0, 1]")
            dy = np.diff(y)
            if (m["increasing"] and np.any(dy < -1e-12)) or (not m["increasing"] and np.any(dy > 1e-12)):
                raise ValueError(f"calibration mapping of {st.get('name')} is not monotone")
            rng = st["applies_to"]
            strata.append(Stratum(st["name"], int(rng["min"]), rng["max"], m["feature"], x, y))
        if not strata:
            raise ValueError("calibration has no strata")
        return cls(tuple(strata), int(d["k"]), str(d["calibration_version"]))

    def stratum_for(self, n_profile: int) -> Stratum | None:
        return next((s for s in self.strata if s.applies(n_profile)), None)

    def predict(self, score: float, rank: int, n_profile: int) -> float | None:
        """Calibrated probability, or None outside the calibrated range (rank > k, no stratum)."""
        if rank < 1 or rank > self.max_rank or not np.isfinite(score):
            return None
        st = self.stratum_for(n_profile)
        if st is None:
            return None
        return float(st.value(np.array([score]), np.array([rank]))[0])


def load_calibration(model_dir: Path) -> dict[str, Any] | None:
    """The calibration.json of a model version, or None when absent/unreadable."""
    path = model_dir / CALIBRATION_FILE
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except (OSError, ValueError):
        log.warning("unreadable %s; recommendations will carry no confidence", path)
        return None
    return data


def calibration_summary(d: dict[str, Any]) -> dict[str, Any]:
    """The metrics part of calibration.json (what the API exposes): no knots, no reliability bins."""
    keys: tuple[str, ...] = ("n", "n_users", "observed_rate", "mean_predicted", "ece", "ece_equal_mass")
    keys += ("brier", "base_rate_brier", "brier_skill_vs_base_rate", "auc")
    strata = []
    for st in d.get("strata") or []:
        strata.append(
            {
                "name": st.get("name"),
                "profile_truncation": st.get("profile_truncation"),
                "applies_to": st.get("applies_to"),
                "feature": (st.get("mapping") or {}).get("feature"),
                "base_rate": st.get("base_rate"),
                "validation": {k: (st.get("validation") or {}).get(k) for k in keys},
                "test": {k: (st.get("test") or {}).get(k) for k in keys},
            }
        )
    return {
        "calibration_version": d.get("calibration_version"),
        "model_version": d.get("model_version"),
        "dataset_version": d.get("dataset_version"),
        "created_at": d.get("created_at"),
        "target": d.get("target"),
        "confidence_kind": d.get("confidence_kind"),
        "method": d.get("method"),
        "k": d.get("k"),
        "horizon_ratings": d.get("horizon_ratings"),
        "fitted_on": d.get("fitted_on"),
        "headline": d.get("headline"),
        "strata": strata,
        "assumptions": d.get("assumptions"),
    }


# --------------------------------------------------------------------------------------------------
# metrics


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        out.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "predicted": round(float(p[m].mean()), 5),
                "observed": round(float(y[m].mean()), 5),
                "n": int(m.sum()),
            }
        )
    return out


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error, equal-width bins, weighted by bin size."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    return _binned_gap(p, y, idx, n_bins)


def ece_quantile(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE with equal-mass bins (by sorted prediction). Equal-width bins put most of a low-probability
    predictor into one or two bins, where over- and under-confidence can cancel."""
    order = np.argsort(p, kind="mergesort")
    idx = np.empty(len(p), dtype=np.int64)
    idx[order] = np.minimum((np.arange(len(p)) * n_bins) // max(len(p), 1), n_bins - 1)
    return _binned_gap(p, y, idx, n_bins)


def _binned_gap(p: np.ndarray, y: np.ndarray, idx: np.ndarray, n_bins: int) -> float:
    n = max(len(p), 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.sum() / n * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(tot)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, p))


def metrics(
    p: np.ndarray, y: np.ndarray, base_rate: float, raw_score: np.ndarray | None = None
) -> dict[str, Any]:
    """Calibration quality of p vs the constant base-rate forecaster (fitted on validation)."""
    out: dict[str, Any] = {
        "n": len(y),
        "n_users": None,
        "observed_rate": round(float(y.mean()), 5),
        "mean_predicted": round(float(p.mean()), 5),
        "ece": round(ece(p, y), 5),
        "ece_equal_mass": round(ece_quantile(p, y), 5),
        "brier": round(brier(p, y), 5),
        "base_rate_brier": round(brier(np.full(len(y), base_rate), y), 5),
        "auc": None if (a := _auc(p, y)) is None else round(a, 4),
        "reliability": reliability_bins(p, y),
    }
    out["brier_skill_vs_base_rate"] = (
        round(1 - out["brier"] / out["base_rate_brier"], 4) if out["base_rate_brier"] > 0 else None
    )
    if raw_score is not None:
        raw = np.clip(raw_score, 0.0, 1.0)
        out["raw_score_as_probability"] = {"ece": round(ece(raw, y), 5), "brier": round(brier(raw, y), 5)}
    return out


# --------------------------------------------------------------------------------------------------
# candidate collection (offline)


def collect_candidates(
    ranker: HybridRanker,
    train: pd.DataFrame,
    target: pd.DataFrame,
    train_user_ids: np.ndarray,
    k: int,
    threshold: float,
    truncate_profile_to: int | None = None,
) -> pd.DataFrame:
    """Top-k list per user (profile from ``train``) labelled against the user's ``target`` rows.

    Every user with training history and at least one target row is included, also users whose
    target holds no relevant film (they contribute true negatives; excluding them would bias the
    rate upwards)."""
    idx = ItemIndex(ranker.movies["movie_id"].to_numpy())
    train_users = set(train["user_id"].unique().tolist())
    users = sorted(int(u) for u in target["user_id"].unique().tolist() if u in train_users)
    positions = {int(u): i for i, u in enumerate(train_user_ids)}
    src = train[train["user_id"].isin(users)]
    if truncate_profile_to is not None:
        src = (
            src.sort_values(["user_id", "timestamp", "movie_id"]).groupby("user_id").head(truncate_profile_to)
        )
        positions = {}  # truncated users are folded in, like brand-new app users
    profiles = profiles_from_interactions(src, idx, np.asarray(users), positions)
    rel = target[target["rating"] >= threshold]
    relevant = {
        int(g["user_id"].iloc[0]): set(idx.indices_of(g["movie_id"].to_numpy()).tolist())
        for _, g in rel.groupby("user_id")
    }
    rows: list[tuple[int, int, float, int, int]] = []
    for u in users:
        prof = profiles[u]
        rs = relevant.get(u, set())
        for pos, r in enumerate(ranker.rank(prof, k=k, explain=False), start=1):
            rows.append((u, pos, float(r.score), int(r.item in rs), int(prof.n_interactions)))
    return pd.DataFrame(rows, columns=["user_id", "rank", "score", "label", "n_profile"])


def next_ratings(target: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Each user's first ``horizon`` target rows in time order; users with fewer rows are dropped
    (their window would be shorter, which biases the hit rate down)."""
    t = target.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    n = t.groupby("user_id")["movie_id"].transform("size")
    t = t[n >= horizon]
    return t[t.groupby("user_id").cumcount() < horizon].reset_index(drop=True)


def _fit_isotonic(x: np.ndarray, y: np.ndarray, increasing: bool) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=increasing, out_of_bounds="clip")
    iso.fit(x, y)
    return iso


def _feature(df: pd.DataFrame, feature: str) -> tuple[np.ndarray, bool]:
    """Feature values and the monotone direction (a higher score / a lower rank = more relevant)."""
    if feature == "score":
        return df["score"].to_numpy(dtype=np.float64), True
    return df["rank"].to_numpy(dtype=np.float64), False


def _knots(iso: IsotonicRegression) -> tuple[list[float], list[float]]:
    return [float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_]


def select_feature(val: pd.DataFrame) -> dict[str, Any]:
    """2-fold cross-fit by user parity inside the validation candidates; lower mean Brier wins."""
    fold = (val["user_id"].to_numpy() % 2).astype(int)
    y = val["label"].to_numpy(dtype=np.float64)
    res: dict[str, Any] = {}
    for feat in FEATURES:
        x, inc = _feature(val, feat)
        pred = np.empty(len(y))
        for f in (0, 1):
            tr, te = fold != f, fold == f
            pred[te] = _fit_isotonic(x[tr], y[tr], inc).predict(x[te])
        res[feat] = {"brier": round(brier(pred, y), 6), "ece": round(ece(pred, y), 5)}
    best = min(FEATURES, key=lambda f: (res[f]["brier"], FEATURES.index(f)))
    return {
        "protocol": "2-fold cross-fit within validation users (user_id parity)",
        "candidates": res,
        "chosen": best,
    }


# --------------------------------------------------------------------------------------------------
# the offline job


def stratum_bounds(representative: list[float]) -> list[tuple[int, int | None]]:
    """Profile-size ranges per stratum: split at the geometric midpoint of consecutive
    representative sizes on the (n + 1) scale; the last stratum is open-ended."""
    bounds: list[tuple[int, int | None]] = []
    lo = 0
    for i, r in enumerate(representative):
        if i == len(representative) - 1:
            bounds.append((lo, None))
            break
        cut = int(np.floor(np.sqrt((r + 1.0) * (representative[i + 1] + 1.0)) - 1.0))
        cut = max(cut, lo)
        bounds.append((lo, cut))
        lo = cut + 1
    return bounds


def calibrate(
    movies: pd.DataFrame,
    interactions: pd.DataFrame,
    training_config: dict[str, Any],
    hybrid_config: dict[str, Any],
    seed: int,
    model_version: str,
    k: int = DEFAULT_K,
    horizon: int = DEFAULT_HORIZON,
    strata: tuple[int | None, ...] = STRATA,
) -> dict[str, Any]:
    """Pure computation of the calibration.json payload (no file IO).

    ``training_config`` is the model's recorded ``manifest.training_config`` (split, models,
    evaluation); ``hybrid_config`` its ``manifest.hybrid_config``. ``strata`` lists the profile
    truncations to calibrate (None = the full training profile)."""
    from jev_ml.training import build_context, fit_components

    t0 = time.perf_counter()
    if "split" not in training_config or "models" not in training_config:
        raise ValueError("training_config must record the split and model parameters")
    ecfg = training_config.get("evaluation") or {}
    threshold = float(ecfg.get("relevance_threshold", LIKE_THRESHOLD))
    split = make_split(interactions, **training_config["split"])
    mcfg = json.loads(json.dumps(training_config["models"]))
    mcfg["als"] = {kk: v for kk, v in mcfg["als"].items() if kk != "seed"}
    hcfg = HybridConfig.from_dict(hybrid_config)

    # content depends only on metadata; the training pipeline fits it once on the train context
    ctx_tr = build_context(movies, split.train, seed)
    content = ContentRecommender(**mcfg["content"]).fit(ctx_tr)
    # fit side: models fitted on the train split only, labels from the validation split
    ranker_tr = fit_components(ctx_tr, mcfg, seed, content=content).ranker(movies, hcfg)
    val_target = next_ratings(split.val, horizon)
    # evaluation side: models refitted on train+validation (the experiment's test protocol)
    trainval = pd.concat([split.train, split.val], ignore_index=True)
    ctx_tv = build_context(movies, trainval, seed)
    ranker_tv = fit_components(ctx_tv, mcfg, seed, content=content).ranker(movies, hcfg)
    test_target = next_ratings(split.test, horizon)

    fitted: list[dict[str, Any]] = []
    for trunc in strata:
        name = "profile_full" if trunc is None else f"profile_{trunc}"
        val = collect_candidates(ranker_tr, split.train, val_target, ctx_tr.user_ids, k, threshold, trunc)
        if val.empty or val["label"].nunique() < 2:
            raise ValueError(f"{name}: validation candidates carry a single label; nothing to calibrate")
        sel = select_feature(val)
        feat = sel["chosen"]
        xv, inc = _feature(val, feat)
        yv = val["label"].to_numpy(dtype=np.float64)
        iso = _fit_isotonic(xv, yv, inc)
        kx, ky = _knots(iso)
        base_rate = float(yv.mean())
        vm = metrics(iso.predict(xv), yv, base_rate, val["score"].to_numpy())
        vm["n_users"] = int(val["user_id"].nunique())
        vm["note"] = "in-sample: the calibrator was fitted on these candidates"
        test = collect_candidates(ranker_tv, trainval, test_target, ctx_tv.user_ids, k, threshold, trunc)
        stratum = Stratum(name, 0, None, feat, np.asarray(kx), np.asarray(ky))
        tm = _evaluate(stratum, test, base_rate)
        fitted.append(
            {
                "name": name,
                "profile_truncation": trunc,
                "representative_profile_size": float(np.median(val["n_profile"])),
                "feature": feat,
                "mapping": {
                    "feature": feat,
                    "increasing": inc,
                    "x": kx,
                    "y": ky,
                    "interpolation": "linear between knots, clipped at the ends",
                },
                "selection": sel,
                "base_rate": round(base_rate, 5),
                "fit_candidates": len(val),
                "validation": vm,
                "test": tm,
                "_stratum": stratum,
            }
        )
    for st, (lo, hi) in zip(
        fitted, stratum_bounds([f["representative_profile_size"] for f in fitted]), strict=True
    ):
        st["applies_to"] = {"min": lo, "max": hi, "unit": "profile interactions"}

    warm = fitted[-1]
    full_window = _evaluate(
        warm["_stratum"],
        collect_candidates(
            ranker_tv, trainval, split.test, ctx_tv.user_ids, k, threshold, warm["profile_truncation"]
        ),
        float(warm["base_rate"]),
    )
    for st in fitted:
        del st["_stratum"]
    mapping_payload = [(st["name"], st["applies_to"], st["mapping"]) for st in fitted]
    fitted_on = {
        "split": "validation",
        "strategy": split.strategy,
        "split_config": training_config["split"],
        "horizon_ratings": horizon,
        "target_rows": len(val_target),
        "users": int(val_target["user_id"].nunique()),
        "model_fitted_on": "train split only (same hyper-parameters as the served model)",
        "test_rows_used_for_fit": 0,
    }
    digest = hashlib.sha256(
        json.dumps(
            {"mapping": mapping_payload, "fitted_on": fitted_on, "model": model_version}, sort_keys=True
        ).encode()
    ).hexdigest()[:10]
    headline_keys: tuple[str, ...] = (
        "n",
        "observed_rate",
        "mean_predicted",
        "ece",
        "ece_equal_mass",
        "brier",
    )
    headline_keys += ("base_rate_brier", "brier_skill_vs_base_rate", "auc")
    return {
        "calibration_version": f"{CALIBRATION_VERSION}-{digest}",
        "model_version": model_version,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": (
            f"P(the user rates the recommended film >= {threshold:g} among their next {horizon} ratings)"
        ),
        "confidence_kind": "probability",
        "method": "isotonic regression per profile-size stratum (feature chosen per stratum by "
        "validation cross-fit)",
        "k": k,
        "horizon_ratings": horizon,
        "relevance_threshold": threshold,
        "seed": seed,
        "fitted_on": fitted_on,
        "strata": fitted,
        "headline": {
            "stratum": warm["name"],
            "split": "test",
            **{kk: warm["test"].get(kk) for kk in headline_keys},
            "cold_strata_test": {
                st["name"]: {
                    kk: st["test"].get(kk) for kk in ("observed_rate", "mean_predicted", "ece", "brier")
                }
                for st in fitted[:-1]
            },
        },
        "test_full_window": {
            **full_window,
            "note": "all test rows of each user (a longer label horizon than the calibrator's, so the "
            "observed rate is higher than predicted by construction)",
        },
        "test_protocol": (
            "components refitted on train+validation; top-k lists scored against each user's first "
            f"{horizon} test ratings (users with >= {horizon}); truncated strata fold the user in from "
            "their first n training interactions, like new app users. The calibrators are applied "
            "unchanged."
        ),
        "assumptions": [
            "The served model is trained on all rows, so it cannot be calibrated on held-out data without "
            "leakage. The calibrators are fitted on a model with the same hyper-parameters trained on the "
            "train split only and are assumed to transfer to the served model; the test evaluation (models "
            "refitted on train+validation, calibrators unchanged) measures that transfer.",
            f"Relevance = rated >= {threshold:g} among the user's next {horizon} ratings. Films the user did "
            "not rate in that window count as not relevant (implicit negatives), so the probability is a "
            "lower bound on whether the user would like the film if shown it.",
            f"Fitted on MovieLens users with >= {horizon} validation ratings and unfiltered lists. Short "
            "profiles are covered by truncated-profile strata; requests with filters (genre, year, "
            "discovery shelves) are outside the calibrated conditions.",
            f"Ranks beyond {k} get no confidence (the calibrator does not extrapolate).",
        ],
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _evaluate(stratum: Stratum, df: pd.DataFrame, base_rate: float) -> dict[str, Any]:
    if df.empty:
        return {"n": 0, "detail": "no evaluable users"}
    y = df["label"].to_numpy(dtype=np.float64)
    p = stratum.value(df["score"].to_numpy(), df["rank"].to_numpy())
    m = metrics(p, y, base_rate, df["score"].to_numpy())
    m["n_users"] = int(df["user_id"].nunique())
    return m


def calibrate_model(
    version: str,
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DIR,
    k: int = DEFAULT_K,
    horizon: int = DEFAULT_HORIZON,
    write: bool = True,
) -> dict[str, Any]:
    """Calibrate a saved model version on the processed data it was trained on; write the file."""
    model_dir = models_dir / version
    manifest: dict[str, Any] = json.loads((model_dir / "manifest.json").read_text())
    meta = load_dataset_meta(processed_dir / "dataset_meta.json")
    if meta.get("dataset_version") != manifest.get("dataset_version"):
        raise ValueError(
            f"processed data {meta.get('dataset_version')} != model dataset "
            f"{manifest.get('dataset_version')}: "
            "the recorded split cannot be reproduced"
        )
    tcfg = manifest.get("training_config") or {}
    seed = int(manifest.get("training_seed", tcfg.get("seed", 42)))
    out = calibrate(
        load_movies(processed_dir / "movies.csv"),
        load_interactions(processed_dir / "interactions.csv"),
        tcfg,
        manifest["hybrid_config"],
        seed,
        version,
        k=k,
        horizon=horizon,
    )
    out["dataset_version"] = manifest.get("dataset_version")
    if write:
        (model_dir / CALIBRATION_FILE).write_text(json.dumps(out, indent=2, allow_nan=False))
    return out
