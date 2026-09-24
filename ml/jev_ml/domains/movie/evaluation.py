"""Offline evaluation of the intelligence layer itself -> EvaluationReport (docs/intelligence.md).

Protocols (all on data <= as_of, all randomness seeded):

* forecast: rolling-origin backtest per forecast series. To avoid selection bias the model is
  *selected* on the first half of the origins and *scored* on the second half (nested); ``naive_mase``
  is the naive model on the same scored points.
* lapse: the temporal holdout of ``run_lapse`` (train on earlier cutoffs, test on later ones),
  plus the recency rule and the constant base rate as baselines.
* anomaly.injection (shilling): SYNTHETIC attack profiles (random / average / bandwagon push
  attacks, Lam & Riedl 2004; Mobasher et al. 2007) are added to a copy of the real ratings. Profile
  sizes are drawn from real users' profile sizes and timestamps are copied from a random real user,
  so the attackers' *temporal* behaviour is realistic and only rating content differs. The deployed
  detector (IsolationForest, deployed contamination) is compared with a single-feature baseline rule
  (flag the same number of users with the highest leave-one-out |rating - item mean|).
* anomaly.series: known spikes/drops of k robust sigmas (k in 3, 5, 8) injected at a real series'
  month (on the detector's scale); detection = the point is flagged with the right sign. The
  false-alarm rate is the flag rate on the same real points without injection; real data contains
  real anomalies, so it is an upper bound.
* change_point: synthetic AR(1) series of trend-window length whose noise level and lag-1
  autocorrelation are estimated from the real log volume series; half carry a mean shift of 1 or 2
  sigmas at a random admissible month.
* latency: repeated full pipeline runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jev_ml.core.anomalies import scan_series
from jev_ml.core.common import clean, fnum, iso
from jev_ml.core.forecast import MODELS, backtest_model, step_quantiles
from jev_ml.core.series import Series
from jev_ml.core.trends import change_point
from jev_ml.domains.movie.config import PIPELINE_VERSION, IntelConfig
from jev_ml.domains.movie.ingest import PipelineInputs, load_default_inputs, prepare
from jev_ml.domains.movie.lapse import run_lapse
from jev_ml.domains.movie.raters import isolation_scores, rater_features
from jev_ml.domains.movie.series import build_series

ATTACK_TYPES = ("random", "average", "bandwagon")


# --------------------------------------------------------------------------------------------------
# forecast


def _subset_metrics(bt: Any, rows: np.ndarray) -> dict[str, float | None]:
    sc = bt.scaled[rows]
    return {
        "mase": float(np.nanmean(sc)) if np.isfinite(sc).any() else None,
        "smape": float(np.nanmean(bt.smape[rows])) if np.isfinite(bt.smape[rows]).any() else None,
        "mae": float(np.nanmean(bt.abs_err[rows])) if np.isfinite(bt.abs_err[rows]).any() else None,
    }


def evaluate_forecasts(series: list[Series], cfg: IntelConfig) -> dict[str, Any]:
    horizon = cfg.forecast_horizon
    per: list[dict[str, Any]] = []
    for s in series:
        if not s.forecast:
            continue
        _, raw, _ = s.complete()
        raw = np.where(np.isfinite(raw), raw, 0.0)
        tr = "log1p"
        bts = {m: backtest_model(m, raw, tr, cfg, horizon, cfg.forecast_backtest_origins) for m in MODELS}
        if any(b is None for b in bts.values()):
            continue
        n_o = len(bts["naive"].origins)  # type: ignore[union-attr]
        sel_rows, score_rows = np.arange(n_o // 2), np.arange(n_o // 2, n_o)
        sel = {m: _subset_metrics(b, sel_rows)["mase"] for m, b in bts.items()}
        best = min(MODELS, key=lambda m: (sel[m] if sel[m] is not None else np.inf, MODELS.index(m)))
        bt = bts[best]
        met = _subset_metrics(bt, score_rows)
        cov = _coverage_rows(bt, cfg, horizon, score_rows)  # honest coverage on the scored origins
        per.append(
            {
                "series_id": s.id,
                "model": best,
                "mase": fnum(met["mase"], 4),
                "smape": fnum(met["smape"], 4),
                "mae": fnum(met["mae"], 3),
                "coverage80": fnum(cov, 4),
                "naive_mase": fnum(_subset_metrics(bts["naive"], score_rows)["mase"], 4),
                "origins": len(score_rows),
                "selection_origins": len(sel_rows),
            }
        )
    mases = [p["mase"] for p in per if p["mase"] is not None]
    beats = [p for p in per if p["mase"] is not None and p["naive_mase"] is not None]
    covs = [p["coverage80"] for p in per if p["coverage80"] is not None]
    return {
        "per_series": per,
        "summary": {
            "n_series": len(per),
            "median_mase": fnum(np.median(mases), 4) if mases else None,
            "share_beating_naive": fnum(np.mean([p["mase"] < p["naive_mase"] for p in beats]), 4)
            if beats
            else None,
            "mean_coverage80": fnum(np.mean(covs), 4) if covs else None,
            "median_naive_mase": fnum(np.median([p["naive_mase"] for p in beats]), 4) if beats else None,
        },
    }


def _coverage_rows(bt: Any, cfg: IntelConfig, horizon: int, rows: np.ndarray) -> float | None:
    """Honest coverage over a subset of origins: each origin's interval uses only residuals
    realised before it (including those of the earlier, unscored origins)."""
    hits = total = 0
    for j in rows:
        o = bt.origins[j]
        q = step_quantiles(np.where(bt.target < o, bt.resid, np.nan), cfg, horizon)
        if q is None:
            continue
        lo, hi = q
        for h in range(horizon):
            r = bt.resid[j, h]
            if np.isfinite(r) and np.isfinite(lo[h]):
                total += 1
                hits += int(lo[h] - 1e-12 <= r <= hi[h] + 1e-12)
    return hits / total if total else None


# --------------------------------------------------------------------------------------------------
# shilling injection


def _inject(
    ratings: pd.DataFrame, kind: str, n_attack: int, rng: np.random.Generator, cfg: IntelConfig
) -> tuple[pd.DataFrame, set[int]]:
    sizes = ratings.groupby("user_id").size().to_numpy()
    counts = ratings.groupby("movie_id").size()
    items = counts.index.to_numpy()
    item_mean = ratings.groupby("movie_id")["rating"].mean()
    popular = counts.sort_values(ascending=False, kind="mergesort").index.to_numpy()[:50]
    tail = counts[counts <= cfg.rater_longtail_max_count].index.to_numpy()
    target = int(rng.choice(tail if len(tail) else items))
    mu, sd = float(ratings["rating"].mean()), float(ratings["rating"].std())
    templates = ratings.groupby("user_id")["timestamp"].apply(lambda s: np.sort(s.to_numpy()))
    tmpl_ids = templates.index.to_numpy()
    next_uid = int(ratings["user_id"].max()) + 1
    rows, attackers = [], set()
    for a in range(n_attack):
        uid = next_uid + a
        attackers.add(uid)
        size = int(min(max(int(rng.choice(sizes)), 20), 400))
        n_pop = max(1, size // 10) if kind == "bandwagon" else 0
        pop = (
            rng.choice(popular[popular != target], size=n_pop, replace=False)
            if n_pop
            else np.array([], dtype=np.int64)
        )
        pool = items[(items != target) & ~np.isin(items, pop)]
        filler = rng.choice(pool, size=size - 1 - n_pop, replace=False)
        if kind == "random":
            fr = np.clip(np.round(rng.normal(mu, sd, len(filler)) * 2) / 2, 0.5, 5.0)
        elif kind == "average":
            fr = np.clip(np.round(item_mean.reindex(filler).to_numpy() * 2) / 2, 0.5, 5.0)
        else:  # bandwagon: popular items at 5, random filler
            fr = np.clip(np.round(rng.normal(mu, sd, len(filler)) * 2) / 2, 0.5, 5.0)
        mids = np.concatenate([[target], pop, filler]).astype(np.int64)
        rs = np.concatenate([[5.0], np.full(n_pop, 5.0), fr])
        tmpl = templates.loc[int(rng.choice(tmpl_ids))]
        ts = np.sort(rng.choice(tmpl, size=len(mids), replace=True)) + rng.integers(0, 60, len(mids))
        rows.append(
            pd.DataFrame({"user_id": uid, "movie_id": mids, "rating": rs, "timestamp": ts.astype(np.int64)})
        )
    return pd.concat([ratings, *rows], ignore_index=True), attackers


def _prf(flag: set[int], truth: set[int]) -> tuple[float, float, float]:
    tp = len(flag & truth)
    p = tp / len(flag) if flag else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_shilling(
    ratings: pd.DataFrame, cfg: IntelConfig, n_attack: int = 12, trials: int = 3
) -> dict[str, Any]:
    genuine = int((ratings.groupby("user_id").size() >= cfg.rater_min_ratings).sum())
    out = []
    for kind in ATTACK_TYPES:
        acc: dict[str, list[float]] = {k: [] for k in ("p", "r", "f", "auc", "bp", "br", "bf", "bauc")}
        for t in range(trials):
            rng = np.random.default_rng(cfg.seed + 1000 * t + ATTACK_TYPES.index(kind))
            data, attackers = _inject(ratings, kind, n_attack, rng, cfg)
            feats = rater_features(data, cfg)
            feats = feats[feats["n_ratings"] >= cfg.rater_min_ratings]
            s = isolation_scores(feats, cfg)
            y = feats.index.isin(list(attackers)).astype(int)
            k = max(1, int(np.ceil(cfg.rater_contamination * len(feats))))
            flag = set(s.sort_values(ascending=False, kind="mergesort").index[:k].tolist())
            base = feats["item_mean_abs_dev"]
            bflag = set(base.sort_values(ascending=False, kind="mergesort").index[:k].tolist())
            p, r, f = _prf(flag, attackers)
            bp, br, bf = _prf(bflag, attackers)
            for key, v in zip(
                acc,
                (p, r, f, roc_auc_score(y, s.to_numpy()), bp, br, bf, roc_auc_score(y, base.to_numpy())),
                strict=True,
            ):
                acc[key].append(float(v))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        out.append(
            {
                "type": kind,
                "n": n_attack,
                "trials": trials,
                "precision": fnum(m["p"], 4),
                "recall": fnum(m["r"], 4),
                "f1": fnum(m["f"], 4),
                "auc": fnum(m["auc"], 4),
                "baseline_precision": fnum(m["bp"], 4),
                "baseline_recall": fnum(m["br"], 4),
                "baseline_f1": fnum(m["bf"], 4),
                "baseline_auc": fnum(m["bauc"], 4),
            }
        )
    return {
        "protocol": (
            f"SYNTHETIC shilling profiles injected into a copy of the real ratings <= as_of: {n_attack} "
            "attackers "
            f"per trial, {trials} seeded trials per type; push attack on one long-tail target item "
            "(rated 5). "
            "random: filler ~ N(global mean, global sd); average: filler at the item mean; bandwagon: "
            "10 % of "
            "the profile = top-50 popular films at 5 + random filler. Profile sizes drawn from real "
            "users (20..400), "
            "timestamps copied from a random real user. Detector = deployed IsolationForest flagging the top "
            f"{100 * cfg.rater_contamination:.0f} % of raters; baseline = the same number of users with "
            "the highest "
            "leave-one-out mean |rating - item mean|. AUC is threshold-free."
        ),
        "n_genuine": genuine,
        "n_injected": n_attack * trials * len(ATTACK_TYPES),
        "baseline_rule": "top-k by leave-one-out mean |rating - item mean| (same k as the detector)",
        "attack_types": out,
    }


# --------------------------------------------------------------------------------------------------
# series anomalies


def evaluate_series_anomalies(series: list[Series], cfg: IntelConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed + 7)
    cands = [s for s in series if s.id.startswith(("volume:all", "active_users:all", "share:genre:"))]
    ks = (3.0, 5.0, 8.0)
    det = {k: [0, 0] for k in ks}
    fa_hits = fa_total = 0
    trials = 0
    for s in cands:
        months, values, counts = s.complete()
        n = len(values)
        for i in range(max(cfg.anomaly_min_baseline_months, n - cfg.anomaly_scan_months), n):
            lo = max(0, i - cfg.anomaly_baseline_months)
            base = values[lo:i]
            ok = np.isfinite(base)
            if ok.sum() < cfg.anomaly_min_baseline_months or not np.isfinite(values[i]):
                continue
            tb = np.log1p(base[ok]) if s.is_count else base[ok]
            med = float(np.median(tb))
            sigma = float(np.median(np.abs(tb - med))) / 0.6745
            if sigma <= 0:
                continue

            def trunc(
                v: np.ndarray,
                i: int = i,
                s: Series = s,
                months: pd.PeriodIndex = months,
                counts: np.ndarray = counts,
            ) -> Series:
                vv = v[: i + 1].copy()
                return replace(
                    s, months=months[: i + 1], values=vv, partial_last=False, counts=counts[: i + 1]
                )

            flagged = scan_series(trunc(values), cfg, "eval", {}, scan_months=1)
            fa_total += 1
            fa_hits += int(bool(flagged))
            for k in ks:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                target = med + sign * k * sigma
                v = values.copy()
                if s.is_count:
                    v[i] = max(0.0, float(np.expm1(target)))
                else:
                    v[i] = target if s.metric != "share" else float(np.clip(target, 0.0, 1.0))
                res = scan_series(trunc(v), cfg, "eval", {}, scan_months=1)
                want = "series_spike" if sign > 0 else "series_drop"
                det[k][0] += int(any(a["kind"] == want for a in res))
                det[k][1] += 1
                trials += 1
    by_k = {f"{k:g}_sigma": fnum(d[0] / d[1], 4) if d[1] else None for k, d in det.items()}
    hits = sum(d[0] for d in det.values())
    return {
        "protocol": (
            "Real series (platform volume/active users, genre shares); for each of the last 12 complete "
            "months a "
            "spike or drop (random sign) of k robust sigmas (k = 3, 5, 8, sigma = MAD/0.6745 of the trailing "
            "baseline, detector scale) replaces the observed value; detection = flagged with the "
            "correct sign "
            "(min-volume guard ignored: suppressed flags count). False-alarm rate = flag rate of the "
            "same real, "
            "uninjected points (an upper bound: real anomalies exist)."
        ),
        "detection_rate": fnum(hits / trials, 4) if trials else None,
        "detection_by_magnitude": by_k,
        "false_alarm_rate": fnum(fa_hits / fa_total, 4) if fa_total else None,
        "n_trials": trials,
        "n_points": fa_total,
    }


# --------------------------------------------------------------------------------------------------
# change points


def evaluate_change_points(series: list[Series], cfg: IntelConfig, n_trials: int = 400) -> dict[str, Any]:
    """Synthetic AR(1) study of the deployed change-point test (and, for comparison, of the pre-1.1
    in-window phi estimate). Each trial draws ``change_point_history_max`` periods of history plus
    one trend window from the same stationary AR(1) process; the history feeds the null's phi
    exactly as ``trend_of`` does."""
    vol = next(s for s in series if s.id == "volume:all")
    _, v, _ = vol.complete()
    y = np.log1p(v[-60:])
    d = y - y.mean()
    phi = float(np.clip(np.corrcoef(d[:-1], d[1:])[0, 1], 0.0, 0.9))
    innov = d[1:] - phi * d[:-1]
    sigma_e = float(np.median(np.abs(innov - np.median(innov))) / 0.6745)
    sigma = sigma_e / np.sqrt(1 - phi**2)
    n, m = cfg.trend_window_months, cfg.change_point_min_segment
    n_hist = cfg.change_point_history_max
    rng = np.random.default_rng(cfg.seed + 11)
    det = fa = n_shift = n_null = 0
    fa_window = det_window = 0
    loc_err: list[float] = []
    by: dict[str, list[int]] = {"1_sigma": [0, 0], "2_sigma": [0, 0]}
    for t in range(n_trials):
        e = np.zeros(n_hist + n)
        e[0] = rng.normal(0, sigma)
        for i in range(1, n_hist + n):
            e[i] = phi * e[i - 1] + rng.normal(0, sigma_e)
        full = e + float(np.median(y))
        hist, x = full[:n_hist], full[n_hist:].copy()
        shifted = t % 2 == 1
        k_true = int(rng.integers(m, n - m + 1))
        mag = 1.0 if t % 4 == 1 else 2.0
        if shifted:
            x[k_true:] += mag * sigma
        seed = cfg.seed + 100 + t
        cp = change_point(
            x, m, cfg.change_point_permutations, seed, history=hist, history_min=cfg.change_point_history_min
        )
        hit = cp is not None and cp["p_value"] < cfg.change_point_alpha
        cpw = change_point(x, m, cfg.change_point_permutations, seed)
        hit_w = cpw is not None and cpw["p_value"] < cfg.change_point_alpha
        if shifted:
            n_shift += 1
            det_window += int(hit_w)
            key = f"{mag:g}_sigma"
            by[key][1] += 1
            if hit:
                det += 1
                by[key][0] += 1
                loc_err.append(abs(cp["index"] - k_true))  # type: ignore[index]
        else:
            n_null += 1
            fa += int(hit)
            fa_window += int(hit_w)
    return {
        "protocol": (
            f"Synthetic AR(1) series: {n_hist} periods of history + a {n}-month window, phi = {phi:.2f} and "
            f"innovation sd = {sigma_e:.3f} estimated from the real log1p(volume:all) over the last 60 "
            "months; half the trials carry a mean shift of 1 or 2 marginal sigmas at a random window "
            f"month (>= {m} from each end). Detection = change-point p < {cfg.change_point_alpha} "
            "(permutation or AR(1) null with phi from the history, as deployed since core-1.1.0)."
        ),
        "detection_rate": fnum(det / n_shift, 4) if n_shift else None,
        "detection_by_magnitude": {k: fnum(a / b, 4) if b else None for k, (a, b) in by.items()},
        "false_alarm_rate": fnum(fa / n_null, 4) if n_null else None,
        "mean_abs_location_error_months": fnum(np.mean(loc_err), 3) if loc_err else None,
        "n_trials": n_trials,
        # the pre-1.1 null (phi estimated inside the 24-point window) on the same trials
        "window_phi": {
            "false_alarm_rate": fnum(fa_window / n_null, 4) if n_null else None,
            "detection_rate": fnum(det_window / n_shift, 4) if n_shift else None,
        },
    }


# --------------------------------------------------------------------------------------------------


def evaluate_latency(inputs: PipelineInputs, cfg: IntelConfig, n_runs: int = 5) -> dict[str, Any]:
    totals, stages = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        from jev_ml.core.pipeline import run_domain
        from jev_ml.domains.movie.adapter import MovieAdapter

        res = run_domain(MovieAdapter(inputs), inputs.as_of, inputs.now, inputs.suppressed_keys, cfg)
        totals.append(1000 * (time.perf_counter() - t0))
        stages.append(res.data["run"]["stage_ms"])
    keys = stages[0].keys()
    return {
        "pipeline_ms_mean": fnum(np.mean(totals), 2),
        "pipeline_ms_p95": fnum(np.quantile(totals, 0.95), 2),
        "n_runs": n_runs,
        "stage_ms": {k: fnum(np.mean([s[k] for s in stages]), 2) for k in keys},
    }


def evaluate(
    inputs: PipelineInputs | None = None, config: IntelConfig | None = None, n_latency_runs: int = 5
) -> dict[str, Any]:
    """Build the EvaluationReport dict (does not write files)."""
    cfg = config or IntelConfig()
    inputs = inputs or load_default_inputs()
    prep = prepare(inputs, cfg)
    series = build_series(prep, cfg)
    lapse, _ = run_lapse(prep.ratings, prep.as_of_ts, cfg, prep.data_version)
    lm = lapse.get("metrics") or {}
    notes: list[str] = []
    fc = evaluate_forecasts(series, cfg)
    shill = evaluate_shilling(prep.ratings, cfg)
    ser = evaluate_series_anomalies(series, cfg)
    cp = evaluate_change_points(series, cfg)
    lat = evaluate_latency(inputs, cfg, n_latency_runs)

    s = fc["summary"]
    if s["share_beating_naive"] is not None:
        notes.append(
            f"Forecasts: selected model beats naive on {100 * s['share_beating_naive']:.0f} % of "
            f"{s['n_series']} series "
            f"(median MASE {s['median_mase']} vs naive {s['median_naive_mase']}); MASE > 1 means worse "
            "than the in-sample "
            "one-step naive scale, expected for bursty monthly counts."
        )
    if s["mean_coverage80"] is not None and s["mean_coverage80"] < 0.75:
        notes.append(
            f"80 % intervals under-cover ({s['mean_coverage80']:.2f}): residual quantiles come from <= "
            "24 origins and "
            "volume is non-stationary; treat intervals as optimistic."
        )
    if s["mean_coverage80"] is not None and s["mean_coverage80"] > 0.9:
        notes.append(
            f"80 % intervals over-cover ({s['mean_coverage80']:.2f}) on the scored origins: "
            "the finite-sample order statistics are conservative with few residuals; "
            "intervals are wide rather than optimistic."
        )
    if lapse.get("status") == "ok":
        notes.append(
            f"Lapse: holdout AUC {lm['auc']} vs recency rule {lm['baseline_auc']}; Brier {lm['brier']} "
            "vs base rate "
            f"{lm['base_rate_brier']}; ECE {lm['ece']}. Test base rate {lm['base_rate']} (train "
            f"{lm['train_base_rate']})."
        )
    else:
        notes.append(f"Lapse model not evaluated: {lapse.get('detail')}")
    for a in shill["attack_types"]:
        if (a["f1"] or 0) < (a["baseline_f1"] or 0):
            notes.append(
                f"Shilling ({a['type']}): the single-feature baseline beats IsolationForest (F1 "
                f"{a['baseline_f1']} vs {a['f1']})."
            )
        if (a["auc"] or 0) >= 0.9 and (a["recall"] or 0) < 0.5:
            notes.append(
                f"Shilling ({a['type']}): ranking is good (AUC {a['auc']}) but at the deployed "
                f"{100 * cfg.rater_contamination:.0f} % flag budget recall is only {a['recall']}: "
                "genuine outlying raters compete for the same review slots."
            )
        if (a["baseline_auc"] or 0) < 0.5:
            notes.append(
                f"Shilling ({a['type']}): the deviation baseline is anti-correlated with attackers "
                f"(AUC {a['baseline_auc']}); attackers copying item means look *more* typical "
                "than real users."
            )
        if (a["auc"] or 0) < 0.7:
            notes.append(f"Shilling ({a['type']}): detector AUC {a['auc']} is weak for this attack model.")
    if (cp["false_alarm_rate"] or 0) > cfg.change_point_alpha * 2:
        notes.append(
            f"Change points: false-alarm rate {cp['false_alarm_rate']} exceeds the nominal "
            f"{cfg.change_point_alpha}: "
            "the AR(1) null plugs in a lag-1 coefficient estimated from only 24 months (bias-corrected, but "
            "noisy), so autocorrelated series still yield spurious shifts."
        )
    notes.append(
        "Series anomaly false-alarm rate is measured on real points that may contain real anomalies "
        "(upper bound)."
    )
    return clean(
        {
            "created_at": iso(datetime.now(UTC)),
            "pipeline_version": PIPELINE_VERSION,
            "data_version": prep.data_version,
            "as_of": iso(prep.as_of),
            "forecast": fc,
            "lapse": {
                "metrics": lm,
                "calibration": lapse.get("calibration", []),
                "baselines": [
                    {
                        "name": "recency_rule",
                        "auc": lm.get("baseline_auc"),
                        "brier": lm.get("baseline_brier"),
                    },
                    {"name": "base_rate", "auc": 0.5, "brier": lm.get("base_rate_brier")},
                ],
                "status": lapse.get("status"),
            },
            "anomaly": {"injection": shill, "series": ser},
            "change_point": cp,
            "latency": lat,
            "notes": notes,
            "config": cfg.to_dict(),
        }
    )


def _markdown(rep: dict[str, Any]) -> str:
    f, la, an, cp, lat = (
        rep["forecast"],
        rep["lapse"]["metrics"],
        rep["anomaly"],
        rep["change_point"],
        rep["latency"],
    )
    lines = [
        "# JEV intelligence layer: offline evaluation",
        "",
        f"- pipeline `{rep['pipeline_version']}`, data `{rep['data_version']}`, as of {rep['as_of']}",
        f"- created {rep['created_at']}",
        "",
        "## Forecasts (rolling origin; model selected on first half of origins, scored on second half)",
        "",
        "| series | model | MASE | naive MASE | sMAPE | MAE | coverage80 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in f["per_series"]:
        lines.append(
            f"| {p['series_id']} | {p['model']} | {p['mase']} | {p['naive_mase']} | {p['smape']} | "
            f"{p['mae']} | {p['coverage80']} |"
        )
    s = f["summary"]
    lines += [
        "",
        f"Median MASE {s['median_mase']} (naive {s['median_naive_mase']}), share beating naive "
        f"{s['share_beating_naive']}, "
        f"mean coverage80 {s['mean_coverage80']}.",
        "",
        "## Lapse model (temporal holdout)",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k in (
        "auc",
        "baseline_auc",
        "brier",
        "baseline_brier",
        "base_rate_brier",
        "ece",
        "base_rate",
        "n_train",
        "n_test",
    ):
        lines.append(f"| {k} | {la.get(k)} |")
    lines += ["", "Reliability bins:", "", "| bin | predicted | observed | n |", "|---|---:|---:|---:|"]
    for b in rep["lapse"]["calibration"]:
        lines.append(f"| {b['bin']} | {b['predicted']} | {b['observed']} | {b['n']} |")
    inj = an["injection"]
    lines += [
        "",
        "## Rater anomalies: synthetic shilling injection",
        "",
        inj["protocol"],
        "",
        "| attack | precision | recall | F1 | AUC | baseline P | baseline R | baseline F1 | baseline AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in inj["attack_types"]:
        lines.append(
            f"| {a['type']} | {a['precision']} | {a['recall']} | {a['f1']} | {a['auc']} | "
            f"{a['baseline_precision']} | "
            f"{a['baseline_recall']} | {a['baseline_f1']} | {a['baseline_auc']} |"
        )
    ser = an["series"]
    lines += [
        "",
        "## Series anomalies (injected spikes)",
        "",
        ser["protocol"],
        "",
        f"Detection rate {ser['detection_rate']} (by magnitude {ser['detection_by_magnitude']}), "
        f"false-alarm rate {ser['false_alarm_rate']}, {ser['n_trials']} trials.",
        "",
        "## Change points",
        "",
        cp["protocol"],
        "",
        f"Detection rate {cp['detection_rate']} (by magnitude {cp['detection_by_magnitude']}), "
        "false-alarm rate "
        f"{cp['false_alarm_rate']}, mean |location error| {cp['mean_abs_location_error_months']} "
        f"months, {cp['n_trials']} trials.",
        "",
        "## Latency",
        "",
        f"Mean {lat['pipeline_ms_mean']} ms, p95 {lat['pipeline_ms_p95']} ms over {lat['n_runs']} runs. "
        f"Stages: {lat['stage_ms']}",
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in rep["notes"]],
        "",
    ]
    return "\n".join(lines)


def write_report(rep: dict[str, Any], experiments_dir: Path | None = None) -> Path:
    from jev_ml import paths

    base = experiments_dir or paths.EXPERIMENTS_DIR
    out = base / f"intel-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, allow_nan=False))
    (out / "REPORT.md").write_text(_markdown(rep))
    return out


def run_evaluation(
    inputs: PipelineInputs | None = None,
    config: IntelConfig | None = None,
    experiments_dir: Path | None = None,
    n_latency_runs: int = 5,
) -> tuple[dict[str, Any], Path]:
    """Evaluate and write experiments/intel-eval-<UTC ts>/{report.json, REPORT.md}."""
    rep = evaluate(inputs, config, n_latency_runs)
    return rep, write_report(rep, experiments_dir)


__all__ = ["evaluate", "lapse_keep", "movie_outcome_fn", "movie_units", "run_evaluation", "write_report"]


# --------------------------------------------------------------------------------------------------
# platform evaluation: warning precision from leak-free monthly replays (docs/platform.md section 7)

# warning kinds whose outcome cannot be observed in a replay, with the reason (excluded from precision)
MOVIE_EXCLUDED_KINDS = {
    "anomaly:series_spike / series_drop": (
        "v1.1 series anomalies are direction-agnostic (a spike is not declared adverse), so there is no "
        "adverse condition to confirm"
    ),
    "anomaly:rater_behaviour, risk:rating_manipulation": "no ground-truth labels of real manipulation",
    "anomaly:live_feedback, risk:recommendation_rejection": "app events are live only, not replayable",
    "risk:model_staleness, risk:model_quality": "skipped in replays (the model was trained after as_of)",
    "risk:data_quality, risk:data_staleness": "input quality, not a future outcome",
}


def lapse_keep(res: Any) -> dict[str, Any]:
    """Users the lapse model flags (P >= lapse_high_risk) at the replay's as_of (same model as the run)."""
    prep = res.prepared
    cfg = res.config
    _, scored = run_lapse(prep.ratings, prep.as_of_ts, cfg, prep.data_version)
    high = (
        []
        if scored is None or not len(scored)
        else [int(u) for u in scored.index[scored["p"] >= cfg.lapse_high_risk]]
    )
    return {"high_risk": high, "as_of_ts": prep.as_of_ts}


def movie_units(replay: dict[str, Any], cfg: IntelConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or IntelConfig()
    d = replay["result"]
    warned = {w["key"] for w in d["warnings"]}
    units = [
        {"kind": "audience_lapse", "unit": "platform:all", "warned": "risk:audience_lapse:all" in warned}
    ]
    for t in d["trends"]:
        if t["metric"] == "share" and (t["recent_mean"] or 0.0) >= cfg.genre_min_share:
            units.append(
                {
                    "kind": "genre_demand_decline",
                    "unit": t["series_id"],
                    "warned": f"risk:genre_demand_decline:{t['entity']}" in warned,
                    "adverse_direction": "down",
                    "last_complete": d["run"]["last_complete_month"],
                }
            )
    return units


def movie_outcome_fn(
    full_ratings: pd.DataFrame, full_series: dict[str, Series], cfg: IntelConfig, h: int
) -> Any:
    """Lapse: confirmed when the realised share of flagged users with no rating in the next
    ``lapse_horizon_days`` is >= ``lapse_high_risk`` (the warning's "likely to lapse" claim held for
    the flagged group). Genre decline: an adverse (down) move of the share beyond noise within ``h``
    months (``core.evaluation.series_adverse_move``)."""
    from jev_ml.core.evaluation import series_outcome_fn

    last_ts = float(full_ratings["timestamp"].max())
    by_user = full_ratings.groupby("user_id")["timestamp"].apply(lambda s: np.sort(s.to_numpy()))
    series_fn = series_outcome_fn(full_series, h)

    def fn(replay: dict[str, Any], u: dict[str, Any]) -> tuple[bool | None, str]:
        if u["kind"] != "audience_lapse":
            return series_fn(replay, u)
        ex = replay["extra"]
        t0 = ex["as_of_ts"]
        t1 = t0 + cfg.lapse_horizon_days * 86400
        if t1 > last_ts:
            return None, f"the {cfg.lapse_horizon_days}-day label window runs past the data"
        users = ex["high_risk"]
        if not users:
            return None, "no user flagged at this replay"
        lapsed = sum(
            not ((ts >= t0) & (ts < t1)).any() for ts in (by_user.get(uid, np.array([])) for uid in users)
        )
        rate = lapsed / len(users)
        return rate >= cfg.lapse_high_risk, f"{lapsed}/{len(users)} flagged users lapsed ({rate:.2f})"

    return fn
