"""Offline evaluation of preference drift and of the recommendation strategies (docs/platform.md §7).

(a) Detector precision / recall on labelled synthetic drift. Label 1: a real user A's history
    followed by the last m events of another real user B (movies A already has are skipped), re-timed
    to continue in A's own session structure (A's inter-active-day gaps and events per day), so it changes
    *what* is watched and how it is rated, not how often. Label 0: the untouched real histories of
    the same A users. Real histories can contain genuine drift, so the false-positive rate is an
    upper bound. Reported per aspect, overall (``drift_detected``) and for the preference aspects the
    strategy policy uses, by splice size m.
(b) Adaptation effect on the recommender's test protocol (per-user temporal split recorded in the
    active model's manifest; history = train + validation, relevance = held-out test rating >= 4):
    for users whose history shows preference drift, NDCG@10 / recall@10 of standard vs
    adapt_to_recent vs explore, paired bootstrap 95 % CI of the per-user difference. Two model
    settings: ``refit`` = component models refitted on train + validation with the active model's
    recorded hyper-parameters (leak-free for the test window; the primary result) and ``active`` =
    the active artifacts, which were trained on all data including the test window (leaky; reported
    for comparison only). Profiles are always folded in from the history, as for app users.

Writes experiments/drift-eval-<UTC ts>/{report.json, REPORT.md}.
    --users N        cap the users of (a) (default: all eligible)
    --boot N         paired bootstrap samples (default 2000)
    --skip-refit     only the active-artifact setting in (b) (fast, leaky)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.drift import DriftConfig, summarize
from jev_ml.data.dataset import load_interactions
from jev_ml.domains.movie.user_intel import (
    StrategyConfig,
    adapt_half_life_days,
    decay_profile,
    drift_aspects,
    user_events,
)
from jev_ml.engine import Interaction, RecommendationEngine
from jev_ml.evaluation.metrics import ndcg_at_k, recall_at_k
from jev_ml.evaluation.split import make_split
from jev_ml.models.hybrid import HybridRanker
from jev_ml.paths import EXPERIMENTS_DIR, PROCESSED_DIR
from jev_ml.signals import LIKE_THRESHOLD

log = logging.getLogger("evaluate_drift")
DAY = 86400.0
PREF = StrategyConfig().preference_aspects
SPLICE_SIZES = (10, 20, 40)


def interactions_of(g: pd.DataFrame) -> list[Interaction]:
    return [
        Interaction(int(m), "rating", float(r), float(t))
        for m, r, t in zip(g["movie_id"], g["rating"], g["timestamp"], strict=True)
    ]


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def prf(pos: list[bool], neg: list[bool]) -> dict[str, Any]:
    tp, fp = int(sum(pos)), int(sum(neg))
    p_n, n_n = len(pos), len(neg)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / p_n if p_n else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {
        "precision": None if prec is None else round(prec, 4),
        "recall": None if rec is None else round(rec, 4),
        "recall_ci95": wilson(tp, p_n),
        "f1": round(f1, 4),
        "fpr_upper_bound": round(fp / n_n, 4) if n_n else None,
        "fpr_ci95": wilson(fp, n_n),
        "tp": tp,
        "fp": fp,
        "n_pos": p_n,
        "n_neg": n_n,
    }


# --- (a) detector -----------------------------------------------------------------------------------
def detect(
    engine: RecommendationEngine, its: list[Interaction], cfg: DriftConfig, sessions: bool
) -> dict[str, Any]:
    ev = user_events(engine, its)
    aspects, _ = drift_aspects(engine, ev, None, cfg, sessions)
    head = summarize(aspects, cfg)
    return {
        "status": head["status"],
        "drift": bool(head["drift_detected"]),
        "pref": any(a.significant for a in aspects if a.aspect in PREF),
        "aspects": {a.aspect: (a.status, a.significant) for a in aspects},
    }


def splice(a: pd.DataFrame, b: pd.DataFrame, m: int, rng: np.random.Generator) -> pd.DataFrame | None:
    """A's history + B's last m new-to-A events, laid out in A's own session structure: new active
    days follow A's inter-active-day gaps, and each day holds as many events as a random A day."""
    seen = set(a["movie_id"].tolist())
    tail = b[~b["movie_id"].isin(seen)].sort_values(["timestamp", "movie_id"]).tail(m)
    if len(tail) < m:
        return None
    days = np.floor(a["timestamp"].to_numpy(dtype=np.float64) / DAY).astype(np.int64)
    uniq, counts = np.unique(days, return_counts=True)
    gaps = np.diff(uniq) if len(uniq) > 1 else np.asarray([1])
    new_ts: list[float] = []
    day = int(uniq[-1])
    while len(new_ts) < m:
        day += int(rng.choice(gaps))
        c = int(rng.choice(counts))
        for j in range(min(c, m - len(new_ts))):
            new_ts.append(day * DAY + 12 * 3600 + 120 * j)
    add = pd.DataFrame(
        {
            "user_id": a["user_id"].iloc[0],
            "movie_id": tail["movie_id"].to_numpy(),
            "rating": tail["rating"].to_numpy(),
            "timestamp": np.asarray(new_ts, dtype=np.int64),
        }
    )
    return pd.concat([a, add], ignore_index=True)


def _scores(pos: list[dict[str, Any]], neg: list[dict[str, Any]], aspects: list[str]) -> dict[str, Any]:
    res: dict[str, Any] = {
        "overall_drift_detected": prf([p["drift"] for p in pos], [n["drift"] for n in neg]),
        "preference_drift": prf([p["pref"] for p in pos], [n["pref"] for n in neg]),
        "per_aspect": {},
    }
    for asp in aspects:
        pp = [p["aspects"][asp][1] for p in pos if p["aspects"][asp][0] == "ok"]
        nn = [n["aspects"][asp][1] for n in neg if n["aspects"][asp][0] == "ok"]
        res["per_aspect"][asp] = prf(pp, nn)
    return res


MODES = {"session_permutation": True, "event_permutation": False}


def eval_detector(
    engine: RecommendationEngine, inter: pd.DataFrame, cfg: DriftConfig, max_users: int | None, seed: int
) -> dict[str, Any]:
    known = set(engine.item_index.movie_ids.tolist())
    inter = inter[inter["movie_id"].isin(known)]
    groups = {int(u): g for u, g in inter.groupby("user_id")}
    span = inter.groupby("user_id")["timestamp"].agg(lambda s: (s.max() - s.min()) / DAY)
    size = inter.groupby("user_id").size()
    eligible = sorted(
        int(u)
        for u in size.index
        if size[u] >= cfg.min_historical + cfg.recent_n and span[u] >= cfg.min_span_days
    )
    rng = np.random.default_rng(seed)
    if max_users and len(eligible) > max_users:
        eligible = sorted(rng.choice(eligible, size=max_users, replace=False).tolist())
    all_users = sorted(groups)
    log.info("detector: %d eligible users", len(eligible))
    t0 = time.perf_counter()
    spliced: dict[int, list[pd.DataFrame]] = {m: [] for m in SPLICE_SIZES}
    for m in SPLICE_SIZES:
        for u in eligible:
            b_id = u
            while b_id == u:
                b_id = int(rng.choice(all_users))
            sp_df = splice(groups[u], groups[b_id], m, rng)
            if sp_df is not None:
                spliced[m].append(sp_df)
    out: dict[str, Any] = {"n_users": len(eligible), "modes": {}}
    for mode, sessions in MODES.items():
        neg = [detect(engine, interactions_of(groups[u]), cfg, sessions) for u in eligible]
        neg = [r for r in neg if r["status"] == "ok"]
        aspects = [a for a in neg[0]["aspects"] if a != "acceptance"] if neg else []
        res_m: dict[str, Any] = {"negatives_tested": len(neg), "by_splice_size": {}}
        for m in SPLICE_SIZES:
            pos = [detect(engine, interactions_of(df), cfg, sessions) for df in spliced[m]]
            pos = [r for r in pos if r["status"] == "ok"]
            res = _scores(pos, neg, aspects)
            res_m["by_splice_size"][str(m)] = res
            log.info(
                "%s m=%d: overall P=%s R=%s FPR=%s; preference P=%s R=%s FPR=%s",
                mode,
                m,
                res["overall_drift_detected"]["precision"],
                res["overall_drift_detected"]["recall"],
                res["overall_drift_detected"]["fpr_upper_bound"],
                res["preference_drift"]["precision"],
                res["preference_drift"]["recall"],
                res["preference_drift"]["fpr_upper_bound"],
            )
        out["modes"][mode] = res_m
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["notes"] = [
        "positives and negatives use the same A users (balanced 1:1); precision depends on this prevalence",
        "activity_rate is not changed by the splice (re-timed at A's own pace): its 'recall' is a "
        "false-alarm rate on spliced histories, not a detection rate",
        "acceptance needs app feedback, which MovieLens does not have: not evaluated",
        "false-positive rates are upper bounds: untouched real histories can contain genuine drift",
        "session_permutation (the served configuration) permutes whole active days; event_permutation "
        "permutes single events and is reported for comparison",
    ]
    return out


# --- (b) adaptation effect ------------------------------------------------------------------------
def refit_ranker(engine: RecommendationEngine, history: pd.DataFrame) -> HybridRanker:
    from jev_ml.training import build_context, fit_components

    manifest = engine.manifest
    tcfg = manifest["training_config"]
    seed = int(manifest.get("training_seed", tcfg.get("seed", 42)))
    mcfg = json.loads(json.dumps(tcfg["models"]))
    mcfg["als"] = {k: v for k, v in mcfg["als"].items() if k != "seed"}
    ctx = build_context(engine.movies, history, seed)
    comps = fit_components(ctx, mcfg, seed, content=engine.content)  # content = metadata only
    return comps.ranker(engine.movies, engine.config)


def paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    d = b - a
    if len(d) == 0:
        return {"delta": None, "ci95": [None, None], "p_better": None, "n_users": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": round(float(d.mean()), 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "p_better": round(float((boots > 0).mean()), 4),
        "n_users": len(d),
        "n_improved": int((d > 0).sum()),
        "n_worse": int((d < 0).sum()),
    }


def eval_adaptation(
    engine: RecommendationEngine,
    inter: pd.DataFrame,
    cfg: StrategyConfig,
    n_boot: int,
    seed: int,
    refit: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = make_split(inter, **engine.manifest["training_config"]["split"])
    history = pd.concat([split.train, split.val], ignore_index=True)
    rel = split.test[split.test["rating"] >= LIKE_THRESHOLD]
    hist_groups = {int(u): g for u, g in history.groupby("user_id")}
    rel_items = {
        int(u): set(engine.item_index.indices_of(g["movie_id"].to_numpy()).tolist())
        for u, g in rel.groupby("user_id")
    }
    users = sorted(u for u in rel_items if u in hist_groups)
    info: dict[str, dict[str, Any]] = {}
    for u in users:
        its = interactions_of(hist_groups[u])
        ev = user_events(engine, its)
        aspects, wsplit = drift_aspects(engine, ev, None, cfg.drift, cfg.session_clusters)
        head = summarize(aspects, cfg.drift)
        ev_aspects, _ = drift_aspects(engine, ev, None, cfg.drift, sessions=False)
        summarize(ev_aspects, cfg.drift)
        info[u] = {
            "its": its,
            "status": head["status"],
            "pref": any(a.significant for a in aspects if a.aspect in PREF),
            "pref_event_mode": any(a.significant for a in ev_aspects if a.aspect in PREF),
            "any": bool(head["drift_detected"]),
            "half_life": adapt_half_life_days(wsplit, cfg),
        }
    log.info("adaptation: %d users, drift analysis in %.1fs", len(users), time.perf_counter() - t0)
    settings: dict[str, HybridRanker] = {"active": engine.ranker}
    if refit:
        t1 = time.perf_counter()
        settings = {"refit": refit_ranker(engine, history), **settings}
        log.info("refit on train+validation in %.1fs", time.perf_counter() - t1)
    variants = {
        "standard": ({}, None),
        "adapt_to_recent": ({"floor": cfg.adapt_weight_floor}, None),
        "explore": ({}, cfg.explore_lambda),
        # exploratory variants (not used by the policy): stronger decay, milder exploration
        "adapt_strong_floor_0.1": ({"floor": 0.1}, None),
        "explore_lambda_0.7": ({}, 0.7),
    }
    groups = {
        "preference_drift": [u for u in users if info[u]["pref"]],
        "any_drift": [u for u in users if info[u]["any"]],
        "no_drift_tested": [u for u in users if info[u]["status"] == "ok" and not info[u]["any"]],
        "insufficient_history": [u for u in users if info[u]["status"] != "ok"],
        "preference_drift_event_mode": [u for u in users if info[u]["pref_event_mode"]],
        "all_users": list(users),
    }
    out: dict[str, Any] = {
        "protocol": "per-user temporal split from the active manifest; history = train + validation; "
        "relevance = test rating >= 4; profiles folded in from history; K = 10",
        "n_users": len(users),
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "settings": {},
    }
    for sname, ranker in settings.items():
        rankers = {
            lam: dataclasses.replace(ranker.config, diversity_lambda=lam)
            for lam in {v[1] for v in variants.values() if v[1] is not None}
        }
        lam_rankers: dict[float, HybridRanker] = {}
        for lam, conf in rankers.items():
            r2 = HybridRanker.__new__(HybridRanker)
            r2.__dict__.update(ranker.__dict__)
            r2.config = conf
            lam_rankers[lam] = r2
        per: dict[str, dict[str, dict[int, float]]] = {v: {"ndcg": {}, "recall": {}} for v in variants}
        needed = users
        for u in needed:
            base = engine.build_profile(info[u]["its"])
            for vname, (pk, lam) in variants.items():
                prof = base
                if "floor" in pk:
                    prof = decay_profile(base, info[u]["half_life"], pk["floor"])
                rk = lam_rankers[lam] if lam is not None else ranker
                recs = [r.item for r in rk.rank(prof, k=10, explain=False)]
                per[vname]["ndcg"][u] = ndcg_at_k(recs, rel_items[u], 10)
                per[vname]["recall"][u] = recall_at_k(recs, rel_items[u], 10)
        res: dict[str, Any] = {}
        for gname in (
            "preference_drift",
            "any_drift",
            "no_drift_tested",
            "preference_drift_event_mode",
            "all_users",
        ):
            us = groups[gname]
            g: dict[str, Any] = {
                "mean_ndcg10": {
                    v: round(float(np.mean([per[v]["ndcg"][u] for u in us])), 5) for v in variants
                }
                if us
                else {},
                "mean_recall10": {
                    v: round(float(np.mean([per[v]["recall"][u] for u in us])), 5) for v in variants
                }
                if us
                else {},
                "vs_standard": {},
            }
            for v in variants:
                if v == "standard":
                    continue
                a = np.asarray([per["standard"]["ndcg"][u] for u in us])
                b = np.asarray([per[v]["ndcg"][u] for u in us])
                ar = np.asarray([per["standard"]["recall"][u] for u in us])
                br = np.asarray([per[v]["recall"][u] for u in us])
                g["vs_standard"][v] = {
                    "ndcg10": paired(a, b, n_boot, seed),
                    "recall10": paired(ar, br, n_boot, seed + 1),
                }
            res[gname] = g
        out["settings"][sname] = res
        pd_ = res["preference_drift"]["vs_standard"]
        log.info(
            "%s: preference-drift users adapt ΔNDCG %s %s, explore ΔNDCG %s %s",
            sname,
            pd_["adapt_to_recent"]["ndcg10"]["delta"],
            pd_["adapt_to_recent"]["ndcg10"]["ci95"],
            pd_["explore"]["ndcg10"]["delta"],
            pd_["explore"]["ndcg10"]["ci95"],
        )
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


# --- (c) latency ------------------------------------------------------------------------------------
def eval_latency(engine: RecommendationEngine, inter: pd.DataFrame, cfg: StrategyConfig) -> dict[str, Any]:
    """strategy_for on every real user's full history (standard profile passed in, as the backend
    already builds it for the request), plus the full user_intelligence payload."""
    from jev_ml.domains.movie.user_intel import strategy_for, user_intelligence

    groups = {int(u): interactions_of(g) for u, g in inter.groupby("user_id")}
    sizes, t_fast, t_full = [], [], []
    for its in groups.values():
        prof = engine.build_profile(its)
        t = time.perf_counter()
        strategy_for(engine, its, config=cfg, profile=prof)
        t_fast.append(1000 * (time.perf_counter() - t))
        sizes.append(len(its))
    for its in list(groups.values())[::10]:
        t = time.perf_counter()
        user_intelligence(engine, its, config=cfg)
        t_full.append(1000 * (time.perf_counter() - t))
    tf, sz = np.asarray(t_fast), np.asarray(sizes)
    typical = tf[(sz >= np.percentile(sz, 25)) & (sz <= np.percentile(sz, 75))]
    return {
        "strategy_for_ms": {
            "p50": round(float(np.percentile(tf, 50)), 2),
            "p95": round(float(np.percentile(tf, 95)), 2),
            "max": round(float(tf.max()), 2),
            "n_users": len(tf),
            "typical_users_p95": round(float(np.percentile(typical, 95)), 2),
            "typical_users": "interquartile history size "
            f"{int(np.percentile(sz, 25))}-{int(np.percentile(sz, 75))} events",
        },
        "user_intelligence_ms": {
            "p50": round(float(np.percentile(t_full, 50)), 2),
            "p95": round(float(np.percentile(t_full, 95)), 2),
            "n_users": len(t_full),
        },
    }


# --- report ------------------------------------------------------------------------------------------
def _fmt_ci(ci: list[Any]) -> str:
    return "n/a" if ci[0] is None else f"{ci[0]:+.4f}..{ci[1]:+.4f}"


def markdown(rep: dict[str, Any]) -> str:
    lines = [
        f"# Drift evaluation {rep['run']}",
        "",
        f"Model `{rep['model_version']}`, data `{rep['dataset_version']}`.",
        "",
        f"Drift config: `{rep['drift_config']}`.",
        "",
        "## (a) Detector on labelled synthetic drift",
        "",
    ]
    det = rep["detector"]
    lines += [f"{det['n_users']} eligible real users (label 0 = untouched, label 1 = spliced).", ""]
    for mode, res_m in det["modes"].items():
        lines += [f"### {mode} ({res_m['negatives_tested']} untouched histories tested)", ""]
        lines += [
            "| splice m | target | precision | recall (95% CI) | F1 | FPR upper bound |",
            "|---|---|---|---|---|---|",
        ]
        for m, r in res_m["by_splice_size"].items():
            rows = [("overall", r["overall_drift_detected"]), ("preference aspects", r["preference_drift"])]
            rows += list(r["per_aspect"].items())
            for name, v in rows:
                ci = v["recall_ci95"]
                lines.append(
                    f"| {m} | {name} | {v['precision']} | {v['recall']} ({ci[0]}–{ci[1]}) "
                    f"| {v['f1']} | {v['fpr_upper_bound']} |"
                )
        lines.append("")
    lines += ["", *[f"- {n}" for n in det["notes"]], "", "## (b) Adaptation effect", ""]
    ad = rep["adaptation"]
    lines += [ad["protocol"], "", f"Group sizes: {ad['group_sizes']}", ""]
    for sname, res in ad["settings"].items():
        lines += [f"### setting `{sname}`", ""]
        lines += [
            "| group | variant | mean NDCG@10 | ΔNDCG@10 (95% CI) | Δrecall@10 (95% CI) |",
            "|---|---|---|---|---|",
        ]
        for gname, g in res.items():
            if not g["mean_ndcg10"]:
                continue
            lines.append(f"| {gname} | standard | {g['mean_ndcg10']['standard']} | – | – |")
            for v, d in g["vs_standard"].items():
                nd, rc = d["ndcg10"], d["recall10"]
                lines.append(
                    f"| {gname} | {v} | {g['mean_ndcg10'][v]} | {nd['delta']} ({_fmt_ci(nd['ci95'])}) "
                    f"| {rc['delta']} ({_fmt_ci(rc['ci95'])}) |"
                )
        lines.append("")
    lines += ["## (c) Latency", "", f"`{rep['latency']}`", ""]
    lines += ["## Notes", "", *[f"- {n}" for n in rep["notes"]], ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = RecommendationEngine.load_active()
    inter = load_interactions(PROCESSED_DIR / "interactions.csv")
    cfg = StrategyConfig()
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    t0 = time.perf_counter()
    det = eval_detector(engine, inter, cfg.drift, args.users, args.seed)
    ada = eval_adaptation(engine, inter, cfg, args.boot, args.seed, refit=not args.skip_refit)
    lat = eval_latency(engine, inter, cfg)
    log.info("latency: %s", lat)
    notes = [
        "the `active` setting is leaky: the active model was trained on all interactions, including "
        "every test rating; `refit` refits the components on train + validation only (content "
        "vectors come from metadata and are reused)",
        "adapt_to_recent half-life = recent-window span clamped to "
        f"[{cfg.adapt_half_life_min_days:g}, {cfg.adapt_half_life_max_days:g}] days, weight floor "
        f"{cfg.adapt_weight_floor}; explore = MMR λ {cfg.explore_lambda} "
        f"(standard λ {engine.config.diversity_lambda})",
        "variants named *_strong_* / *_lambda_0.7 are exploratory and not used by the policy",
    ]
    rep = {
        "run": run,
        "model_version": engine.version,
        "dataset_version": engine.manifest.get("dataset_version"),
        "drift_config": dataclasses.asdict(cfg.drift),
        "strategy_config": {
            k: v for k, v in dataclasses.asdict(cfg).items() if k not in ("drift", "evaluated_effects")
        },
        "detector": det,
        "adaptation": ada,
        "latency": lat,
        "notes": notes,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out = EXPERIMENTS_DIR / f"drift-eval-{run}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, allow_nan=False))
    (out / "REPORT.md").write_text(markdown(rep))
    print(json.dumps({"run_dir": str(Path(out))}, indent=2))


if __name__ == "__main__":
    main()
