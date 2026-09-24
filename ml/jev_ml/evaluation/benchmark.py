"""Reproducible recommender benchmark: both split protocols, CIs, paired tests, cold start,
calibration and baseline tuning on one budget. Entry point: ``scripts/benchmark_recommenders.py``.

Per protocol (``user_temporal``, ``global_temporal``):

1. split, ``leakage.check_split``; content features rebuilt with only the tags written before the
   stage's cut (``fit`` = before validation, ``test`` = before test); under the global cut the films
   released after it are excluded from every candidate set.
2. **incumbent** = the active model's recorded hyper-parameters (or the config defaults). Its
   components are fitted on train (validation stage) and on train+validation (test stage).
3. **baseline tuning** on validation with the same budget for each baseline (12 configurations for
   popularity, item-kNN and ALS; ``benchmark.tuning``), then the full hybrid tuning procedure of
   ``training.tune`` (the one that produced the served model), all on this protocol's validation.
4. **cold stages** tuned on validation (``cold_start.tune_cold_stages``) on top of the incumbent.
5. **test**: warm and cold (profile sizes 0-10, with and without simulated onboarding genres)
   for every model; per-user bootstrap 95 % CIs and paired comparisons (Holm-adjusted).
6. **calibration**: isotonic strata (current) vs the logistic calibrator, both fitted on
   validation only, compared on test with user-cluster bootstrap CIs.

Everything is written to ``experiments/rec-benchmark-<ts>/`` (results.json, per_user.json,
REPORT.md) with the dataset version, git commit and a config hash.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from jev_ml import __version__
from jev_ml.calibration import LOGISTIC_FEATURES, calibrate
from jev_ml.calibration import metrics as cal_metrics
from jev_ml.data.dataset import load_dataset_meta, load_interactions, load_movies
from jev_ml.evaluation import cold_start as cs
from jev_ml.evaluation.evaluator import EvalResult, Evaluator
from jev_ml.evaluation.leakage import check_split, future_items, movies_with_tags_before, tag_cutoffs
from jev_ml.evaluation.split import Split, make_split
from jev_ml.evaluation.stats import bootstrap_ci, cluster_bootstrap, holm, paired_comparison
from jev_ml.models.content import ContentRecommender
from jev_ml.models.hybrid import HybridConfig
from jev_ml.models.popularity import PopularityRecommender
from jev_ml.paths import CONFIGS_DIR, EXPERIMENTS_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR

log = logging.getLogger(__name__)

KEY_METRICS = ("ndcg@10", "recall@10")


# --------------------------------------------------------------------------------------------------
# helpers


def _git_commit() -> str | None:
    from jev_ml.training import _git_commit as gc

    return gc()


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def incumbent_params(models_dir: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """(component params, hybrid config, source) of the active model, or the config defaults."""
    from jev_ml.registry import active_version

    version = active_version(models_dir)
    if version and (models_dir / version / "manifest.json").exists():
        man = json.loads((models_dir / version / "manifest.json").read_text())
        mcfg = json.loads(json.dumps(man["training_config"]["models"]))
        mcfg["als"] = {k: v for k, v in mcfg["als"].items() if k != "seed"}
        return mcfg, dict(man["hybrid_config"]), str(version)
    mcfg = json.loads(json.dumps(cfg["models"]))
    return mcfg, dict(mcfg["hybrid"]), "configs/experiment.yaml"


def _summ(res: EvalResult, b: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_users": res.n_users,
        "metrics": {k: round(v, 6) for k, v in res.metrics.items()},
    }
    for m in KEY_METRICS:
        ci = bootstrap_ci(res.per_user[m], b=b, seed=seed)
        out[m] = {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]}
    return out


def _paired(a: EvalResult, b_res: EvalResult, cfg: dict[str, Any], seed: int, metric: str) -> dict[str, Any]:
    if a.user_ids != b_res.user_ids:
        raise AssertionError("paired comparison on different users")
    return paired_comparison(
        a.per_user[metric], b_res.per_user[metric], b=cfg["b"], permutations=cfg["perm"], seed=seed
    )


class _Stage:
    """Fitted components and evaluation inputs of one stage (validation or test) of a protocol."""

    def __init__(
        self,
        name: str,
        fit_rows: pd.DataFrame,
        target: pd.DataFrame,
        movies: pd.DataFrame,
        mcfg: dict[str, Any],
        seed: int,
        exclude: np.ndarray | None,
    ) -> None:
        from jev_ml.training import build_context, fit_components

        self.name = name
        self.fit_rows, self.target, self.movies, self.exclude = fit_rows, target, movies, exclude
        self.ctx = build_context(movies, fit_rows, seed)
        self.content = ContentRecommender(**mcfg["content"]).fit(self.ctx)
        self.comps = fit_components(self.ctx, mcfg, seed, content=self.content)


def _warm_evaluator(st: _Stage, ecfg: dict[str, Any], seed: int, max_users: int | None) -> Evaluator:
    return Evaluator(
        st.fit_rows,
        st.target,
        st.ctx.item_index,
        st.ctx.user_ids,
        st.comps.content.pairwise,
        st.comps.popularity.item_popularity_fraction(),
        ks=tuple(ecfg["ks"]),
        relevance_threshold=ecfg["relevance_threshold"],
        max_users=max_users,
        seed=seed,
        exclude_items=st.exclude,
    )


def _cold_evaluator(
    strategy: str, st: _Stage, ecfg: dict[str, Any], n: int, onb: bool, max_users: int | None, seed: int
) -> Evaluator:
    ev = cs.cold_evaluator(
        strategy,
        st.fit_rows,
        st.target,
        st.movies,
        st.ctx.item_index,
        st.ctx.user_ids,
        st.comps.content.pairwise,
        st.comps.popularity.item_popularity_fraction(),
        n,
        onb,
        ecfg,
        exclude_items=st.exclude,
    )
    if max_users and len(ev.users) > max_users:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(ev.users, size=max_users, replace=False).tolist())
        ev.users = keep
        ev.relevant = {u: ev.relevant[u] for u in keep}
        ev.profiles = {u: ev.profiles[u] for u in keep}
    return ev


# --------------------------------------------------------------------------------------------------
# baseline tuning (same budget for every baseline)


def tune_popularity(
    st: _Stage, ev: Evaluator, grid: dict[str, list[Any]], metric: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from jev_ml.training import component_fn

    trials = []
    best: tuple[float, dict[str, Any]] = (-1.0, {})
    base = {"trending_half_life_days": 365, "bayes_prior_votes": 10}
    for reach, hl in itertools.product(grid["reach_weight"], grid["score_half_life_days"]):
        params = {**base, "reach_weight": reach, "score_half_life_days": hl}
        model = PopularityRecommender(**params).fit(st.ctx)
        v = ev.evaluate(f"popularity[r={reach},hl={hl}]", component_fn(model)).metrics[metric]
        trials.append({"params": params, metric: v})
        if v > best[0]:
            best = (v, params)
    return best[1], trials


# --------------------------------------------------------------------------------------------------
# one protocol


def run_protocol(
    strategy: str,
    interactions: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None,
    cfg: dict[str, Any],
    bcfg: dict[str, Any],
    inc_models: dict[str, Any],
    inc_hybrid: dict[str, Any],
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from jev_ml.training import _CachedSignalsRanker, component_fn, hybrid_fn, random_fn, tune

    t0 = time.perf_counter()
    seed = int(cfg["seed"])
    ecfg = cfg["evaluation"]
    metric = ecfg["selection_metric"]
    max_users = bcfg["max_users"]
    stat_cfg = {"b": bcfg["bootstrap_b"], "perm": bcfg["permutations"]}
    split: Split = make_split(interactions, strategy, cfg["split"]["val_frac"], cfg["split"]["test_frac"])
    leak = {"split": check_split(split), "summary": split.summary()}

    # leakage controls: tags before each stage's cut; films released after a global cut excluded
    mv_fit = movies_with_tags_before(movies, tags, tag_cutoffs(split, "fit"))
    trainval = pd.concat([split.train, split.val], ignore_index=True)
    mv_test = movies_with_tags_before(movies, tags, tag_cutoffs(split, "test"))
    ex_fit = ex_test = None
    if strategy == "global_temporal":
        ex_fit = future_items(movies, split.train, float(split.val["timestamp"].min()))
        ex_test = future_items(movies, trainval, float(split.test["timestamp"].min()))
    leak["tags"] = {
        "raw_tags_available": tags is not None,
        "movies_with_tags_all": int((movies["tags"] != "").sum()),
        "movies_with_tags_fit_stage": int((mv_fit["tags"] != "").sum()),
        "movies_with_tags_test_stage": int((mv_test["tags"] != "").sum()),
    }
    leak["future_items_excluded"] = {
        "fit_stage": 0 if ex_fit is None else len(ex_fit),
        "test_stage": 0 if ex_test is None else len(ex_test),
    }

    inc_h = HybridConfig.from_dict(inc_hybrid)
    val = _Stage("validation", split.train, split.val, mv_fit, inc_models, seed, ex_fit)
    test = _Stage("test", trainval, split.test, mv_test, inc_models, seed, ex_test)
    ev_val = _warm_evaluator(val, ecfg, seed, max_users)

    # --- baseline tuning (validation) -----------------------------------------------------------
    tuning: dict[str, Any] = {}
    pop_params, pop_trials = tune_popularity(val, ev_val, bcfg["tuning"]["popularity"], metric)
    tuning["popularity"] = {"best": pop_params, "trials": pop_trials, "budget": len(pop_trials)}
    retuned: dict[str, Any] | None = None
    if not quick:
        cfg_t = {**cfg, "tuning": {**bcfg["tuning"], "enabled": True}}
        tuned = tune(split.train, split.val, mv_fit, cfg_t, val.content)
        retuned = {
            "itemknn": tuned["itemknn"],
            "als": {k: v for k, v in tuned["als"].items() if k != "seed"},
            "hybrid": tuned["hybrid"],
        }
        tuning["itemknn"] = {"best": tuned["itemknn"], "budget": len(tuned["trials"]["itemknn"])}
        tuning["als"] = {"best": retuned["als"], "budget": len(tuned["trials"]["als"])}
        tuning["hybrid"] = {"best": tuned["hybrid"], "budget": len(tuned["trials"]["hybrid"])}

    # --- cold stages (validation) ---------------------------------------------------------------
    sizes = [n for n in bcfg["cold_sizes"] if n <= 10]
    val_cold = {
        (n, onb): _cold_evaluator(strategy, val, ecfg, n, onb, max_users, seed)
        for n in sizes
        for onb in (False, True)
    }
    val_cold = {k: v for k, v in val_cold.items() if v.users}
    stage_report = cs.tune_cold_stages(
        lambda: _CachedSignalsRanker(val.comps.ranker(val.movies, inc_h)),
        val_cold,
        inc_h,
        seed,
        n_dirichlet=bcfg["stage_tuning"]["dirichlet"],
        n_profile=bcfg["stage_tuning"]["profile"],
        metric=metric,
    )
    stage_report["n_users"] = {f"n={k[0]},onboarding={k[1]}": len(v.users) for k, v in val_cold.items()}
    cold_h = HybridConfig.from_dict({**asdict(inc_h), "cold_stages": stage_report["cold_stages"]})

    # --- test ------------------------------------------------------------------------------------
    pop_t = PopularityRecommender(**pop_params).fit(test.ctx)
    comps = test.comps
    fns = {
        "random": random_fn(len(movies), seed),
        "popularity": component_fn(comps.popularity),
        "popularity_tuned": component_fn(pop_t),
        "content": component_fn(comps.content),
        "itemknn": component_fn(comps.itemknn),
        "als": component_fn(comps.als),
        "hybrid": hybrid_fn(comps.ranker(test.movies, inc_h)),
        "hybrid_cold": hybrid_fn(comps.ranker(test.movies, cold_h)),
    }
    if retuned is not None:
        from jev_ml.training import fit_components

        mcfg_rt = {**inc_models, "itemknn": retuned["itemknn"], "als": retuned["als"]}
        comps_rt = fit_components(test.ctx, mcfg_rt, seed, content=test.content)
        fns["itemknn_retuned"] = component_fn(comps_rt.itemknn)
        fns["als_retuned"] = component_fn(comps_rt.als)
        fns["hybrid_retuned"] = hybrid_fn(
            comps_rt.ranker(test.movies, HybridConfig.from_dict(retuned["hybrid"]))
        )
    # tag-leakage effect: the incumbent hybrid with content fitted on *all* tags
    from jev_ml.training import Components, build_context

    content_all = ContentRecommender(**inc_models["content"]).fit(build_context(movies, trainval, seed))
    comps_all = Components(comps.popularity, content_all, comps.itemknn, comps.als)
    fns["hybrid_all_tags"] = hybrid_fn(comps_all.ranker(movies, inc_h))

    ev_test = _warm_evaluator(test, ecfg, seed, max_users)
    warm = {name: ev_test.evaluate(name, fn) for name, fn in fns.items()}
    cold_models = ("popularity", "popularity_tuned", "content", "itemknn", "als", "hybrid", "hybrid_cold")
    cold: dict[tuple[int, bool], dict[str, EvalResult]] = {}
    for n in sizes:
        for onb in (False, True):
            ev = _cold_evaluator(strategy, test, ecfg, n, onb, max_users, seed)
            if ev.users:
                cold[(n, onb)] = {m: ev.evaluate(f"{m}[n={n},onb={onb}]", fns[m]) for m in cold_models}

    # --- statistics --------------------------------------------------------------------------------
    b, perm = stat_cfg["b"], stat_cfg["perm"]
    out: dict[str, Any] = {
        "strategy": strategy,
        "leakage": leak,
        "tuning": tuning,
        "cold_stage_tuning": stage_report,
    }
    out["warm"] = {m: _summ(r, b, seed) for m, r in warm.items()}
    comps_warm = {
        f"hybrid_vs_{o}": _paired(warm["hybrid"], warm[o], stat_cfg, seed, "ndcg@10")
        for o in warm
        if o not in ("hybrid", "hybrid_all_tags")
    }
    comps_warm["hybrid_cold_vs_hybrid"] = _paired(
        warm["hybrid_cold"], warm["hybrid"], stat_cfg, seed, "ndcg@10"
    )
    comps_warm["hybrid_vs_hybrid_all_tags"] = _paired(
        warm["hybrid"], warm["hybrid_all_tags"], stat_cfg, seed, "ndcg@10"
    )
    adj = holm({k: v["p_value"] for k, v in comps_warm.items() if v["p_value"] is not None})
    for k, v in comps_warm.items():
        v["p_holm"] = adj.get(k)
    out["warm_paired_ndcg10"] = comps_warm
    out["warm_paired_recall10"] = {
        f"hybrid_vs_{o}": _paired(warm["hybrid"], warm[o], stat_cfg, seed, "recall@10")
        for o in ("popularity", "popularity_tuned", "itemknn", "als")
    }
    out["cold"] = {
        f"n={n},onboarding={onb}": {m: _summ(r, b, seed) for m, r in res.items()}
        for (n, onb), res in cold.items()
    }
    out["cold_buckets"] = _bucket_stats(cold, stat_cfg, seed)
    out["calibration"] = _calibration_comparison(
        strategy, interactions, mv_fit, cfg, inc_models, inc_hybrid, bcfg, seed
    )
    out["seconds"] = round(time.perf_counter() - t0, 1)
    per_user = {
        "warm": {m: {"users": r.user_ids, "ndcg@10": r.per_user["ndcg@10"]} for m, r in warm.items()},
        "cold": {
            f"n={n},onboarding={onb}": {m: r.per_user["ndcg@10"] for m, r in res.items()}
            for (n, onb), res in cold.items()
        },
    }
    del b, perm
    return out, per_user


def _bucket_stats(
    cold: dict[tuple[int, bool], dict[str, EvalResult]], stat_cfg: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Per bucket: average each user's NDCG@10 over the bucket's cells (sizes x onboarding), then
    CIs and paired tests on those per-user means (the users are the same in every cell)."""
    out: dict[str, Any] = {}
    for b in cs.BUCKETS:
        keys = [k for k in cold if k[0] in b["reps"]]
        for label, sel in (
            ("all", keys),
            ("onboarding", [k for k in keys if k[1]]),
            ("no_onboarding", [k for k in keys if not k[1]]),
        ):
            if not sel:
                continue
            users = cold[sel[0]]["hybrid"].user_ids
            if any(cold[k]["hybrid"].user_ids != users for k in sel):
                continue
            per_model = {
                m: np.mean([cold[k][m].per_user["ndcg@10"] for k in sel], axis=0) for m in cold[sel[0]]
            }
            entry: dict[str, Any] = {
                "cells": [f"n={k[0]},onboarding={k[1]}" for k in sel],
                "n_users": len(users),
                "ndcg@10": {m: bootstrap_ci(v, b=stat_cfg["b"], seed=seed) for m, v in per_model.items()},
            }
            pairs = {
                "hybrid_cold_vs_hybrid": ("hybrid_cold", "hybrid"),
                "hybrid_cold_vs_popularity": ("hybrid_cold", "popularity"),
                "hybrid_cold_vs_popularity_tuned": ("hybrid_cold", "popularity_tuned"),
                "hybrid_vs_popularity": ("hybrid", "popularity"),
            }
            entry["paired"] = {
                name: paired_comparison(
                    per_model[a], per_model[c], b=stat_cfg["b"], permutations=stat_cfg["perm"], seed=seed
                )
                for name, (a, c) in pairs.items()
            }
            out[f"{b['name']}/{label}"] = entry
    return out


# --------------------------------------------------------------------------------------------------
# calibration: isotonic vs logistic


def _auc_stat(y: np.ndarray, p: np.ndarray) -> Any:
    from sklearn.metrics import roc_auc_score

    def f(idx: np.ndarray) -> float | None:
        yy = y[idx]
        if yy.min() == yy.max():
            return None
        return float(roc_auc_score(yy, p[idx]))

    return f


def _calibration_comparison(
    strategy: str,
    interactions: pd.DataFrame,
    movies_fit: pd.DataFrame,
    cfg: dict[str, Any],
    inc_models: dict[str, Any],
    inc_hybrid: dict[str, Any],
    bcfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    from jev_ml.calibration import Stratum

    tcfg = {
        "split": {**cfg["split"], "strategy": strategy},
        "models": inc_models,
        "evaluation": cfg["evaluation"],
    }
    try:
        cal = calibrate(
            movies_fit,
            interactions,
            tcfg,
            inc_hybrid,
            seed,
            "benchmark",
            k=bcfg["calibration"]["k"],
            horizon=bcfg["calibration"]["horizon"],
            method="logistic",
            return_frames=True,
        )
    except ValueError as exc:  # e.g. a stratum without positives under the global cut
        return {"error": str(exc)}
    frames = cal.pop("_frames")
    lmaps = cal.pop("_logistic_maps")
    b = min(500, int(bcfg["bootstrap_b"]))
    keys: tuple[str, ...] = ("auc", "brier", "base_rate_brier", "brier_skill_vs_base_rate", "ece")
    keys += ("ece_equal_mass",)
    keys += ("observed_rate", "mean_predicted", "reliability")
    strata = []
    for st, (_, te) in zip(cal["strata"], frames, strict=True):
        m = st["mapping"]
        iso = Stratum(st["name"], 0, None, m["feature"], np.asarray(m["x"]), np.asarray(m["y"]))
        y = te["label"].to_numpy(dtype=np.float64)
        users = te["user_id"].to_numpy()
        preds = {
            "isotonic": iso.value(te["score"].to_numpy(), te["rank"].to_numpy()),
            "logistic": lmaps[st["name"]].predict_frame(te),
        }
        preds["served"] = preds[st["serving_method"]]
        entry: dict[str, Any] = {
            "name": st["name"],
            "applies_to": st["applies_to"],
            "serving_method": st["serving_method"],
            "crossfit_log_loss": st["logistic_candidate"]["crossfit_log_loss"],
            "logistic_coef": dict(
                zip(st["logistic_candidate"]["features"], st["logistic_candidate"]["coef"], strict=True)
            ),
            "logistic_C": st["logistic_candidate"]["C"],
            "n_test_rows": len(te),
            "n_test_users": len(np.unique(users)),
        }
        for name, pr in preds.items():
            mm = cal_metrics(pr, y, float(st["base_rate"]))
            ci = cluster_bootstrap(users, _auc_stat(y, pr), b=b, seed=seed)
            entry[name] = {**{k: mm.get(k) for k in keys}, "auc_ci": [ci["lo"], ci["hi"]]}
        for name in ("logistic", "served"):
            fa, fi = _auc_stat(y, preds[name]), _auc_stat(y, preds["isotonic"])

            def diff(idx: np.ndarray, fa: Any = fa, fi: Any = fi) -> float | None:
                a, c = fa(idx), fi(idx)
                return None if a is None or c is None else a - c

            pa, pi = preds[name], preds["isotonic"]

            def bdiff(idx: np.ndarray, pa: np.ndarray = pa, pi: np.ndarray = pi, y: np.ndarray = y) -> float:
                return float(np.mean((pa[idx] - y[idx]) ** 2) - np.mean((pi[idx] - y[idx]) ** 2))

            entry[f"auc_diff_{name}_minus_isotonic"] = cluster_bootstrap(users, diff, b=b, seed=seed)
            entry[f"brier_diff_{name}_minus_isotonic"] = cluster_bootstrap(users, bdiff, b=b, seed=seed)
        strata.append(entry)
    return {
        "target": cal["target"],
        "fitted_on": cal["fitted_on"],
        "features": list(LOGISTIC_FEATURES),
        "strata": strata,
        "seconds": cal["seconds"],
    }


# --------------------------------------------------------------------------------------------------
# latency


def serving_latency(
    models_dir: Path, stages: list[dict[str, Any]] | None, n: int = 60
) -> dict[str, Any] | None:
    """Median ms per 20-item request of the active engine: as served, and with cold stages."""
    import copy

    from jev_ml.engine import Interaction, RecommendationEngine

    try:
        engine = RecommendationEngine.load_active(models_dir)
    except (FileNotFoundError, OSError, ValueError):
        return None
    ids = engine.movies.sort_values("n_ratings", ascending=False)["movie_id"].tolist()
    profiles = {
        "n=0": engine.build_profile([], genre_prefs=["Comedy", "Drama"]),
        "n=3": engine.build_profile([Interaction(int(m), "rating", 4.5) for m in ids[:3]]),
        "n=40": engine.build_profile([Interaction(int(m), "rating", 4.0) for m in ids[:40]]),
    }

    def timed(e: RecommendationEngine) -> dict[str, float]:
        out = {}
        for name, prof in profiles.items():
            e.recommend(prof, k=20)
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                e.recommend(prof, k=20)
                ts.append((time.perf_counter() - t) * 1000)
            out[name] = round(float(np.median(ts)), 2)
        return out

    res = {"model": engine.version, "served": timed(engine)}
    if stages:
        e2 = copy.copy(engine)
        e2.ranker = copy.copy(engine.ranker)
        e2.ranker.config = HybridConfig.from_dict({**asdict(engine.ranker.config), "cold_stages": stages})
        res["with_cold_stages"] = timed(e2)
    return res


# --------------------------------------------------------------------------------------------------
# orchestration


def load_benchmark_config(path: Path | None, quick: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg: dict[str, Any] = yaml.safe_load((path or CONFIGS_DIR / "experiment.yaml").read_text())
    bcfg = json.loads(json.dumps(cfg["benchmark"]))
    q = bcfg.pop("quick", {})
    bcfg["max_users"] = None
    if quick:
        bcfg.update({k: v for k, v in q.items() if k != "stage_tuning"})
        bcfg["stage_tuning"] = q.get("stage_tuning", bcfg["stage_tuning"])
    return cfg, bcfg


def run_benchmark(
    config_path: Path | None = None,
    quick: bool = False,
    protocols: list[str] | None = None,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    tags_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    t0 = time.perf_counter()
    cfg, bcfg = load_benchmark_config(config_path, quick)
    seed = int(cfg["seed"])
    np.random.seed(seed)
    movies = load_movies(processed_dir / "movies.csv")
    interactions = load_interactions(processed_dir / "interactions.csv")
    meta = load_dataset_meta(processed_dir / "dataset_meta.json")
    tags_path = tags_path or RAW_DIR / "ml-latest-small" / "tags.csv"
    tags = pd.read_csv(tags_path) if tags_path.exists() else None
    inc_models, inc_hybrid, inc_source = incumbent_params(models_dir, cfg)
    protocols = protocols or list(bcfg["protocols"])
    run_id = f"rec-benchmark-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}{'-quick' if quick else ''}"
    out_dir = experiments_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    results: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "quick": quick,
        "command": " ".join(["uv run python", *sys.argv]) if sys.argv else None,
        "dataset_version": meta.get("dataset_version"),
        "git_commit": _git_commit(),
        "jev_ml_version": __version__,
        "python": platform.python_version(),
        "seed": seed,
        "config_hash": _hash(
            {
                "benchmark": bcfg,
                "split": cfg["split"],
                "evaluation": cfg["evaluation"],
                "incumbent": [inc_models, inc_hybrid],
            }
        ),
        "benchmark_config": bcfg,
        "incumbent": {"source": inc_source, "models": inc_models, "hybrid": inc_hybrid},
        "protocols": {},
    }
    per_user: dict[str, Any] = {}
    for p in protocols:
        log.info("=== protocol %s ===", p)
        results["protocols"][p], per_user[p] = run_protocol(
            p, interactions, movies, tags, cfg, bcfg, inc_models, inc_hybrid, quick
        )
    results["decision"] = decide(results, float(bcfg["non_inferiority_margin"]))
    stages = results["protocols"][protocols[0]]["cold_stage_tuning"]["cold_stages"]
    results["latency_ms"] = serving_latency(models_dir, stages)
    results["seconds_total"] = round(time.perf_counter() - t0, 1)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default, allow_nan=False)
    )
    (out_dir / "per_user.json").write_text(json.dumps(per_user, default=_json_default))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    from jev_ml.evaluation.benchmark_report import write_markdown

    (out_dir / "REPORT.md").write_text(write_markdown(results))
    return results, out_dir


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def decide(results: dict[str, Any], margin: float) -> dict[str, Any]:
    """Serving decisions, from test numbers only (the choices themselves were made on validation).

    Cold stages: adopted when, in every protocol and every cold bucket, the paired NDCG@10
    difference vs the incumbent has a CI lower bound >= -margin (non-inferior) and at least one
    bucket is significantly better. Calibration: the per-stratum selected calibrator (isotonic or
    logistic, chosen on validation) is adopted when no stratum of any protocol has a test AUC or
    Brier significantly worse than isotonic and at least one stratum has a significantly higher
    AUC."""
    stage_ok, stage_better, notes = True, False, []
    cal_ok, cal_better = True, False
    for p, r in results["protocols"].items():
        for key, e in r["cold_buckets"].items():
            if not key.endswith("/all"):
                continue
            d = e["paired"]["hybrid_cold_vs_hybrid"]
            if d["lo"] is None:
                continue
            if d["lo"] < -margin:
                stage_ok = False
                notes.append(f"{p} {key}: hybrid_cold - hybrid CI lower bound {d['lo']:+.4f} < -{margin}")
            if d["significant"] and d["diff"] > 0:
                stage_better = True
        cal = r.get("calibration") or {}
        if not cal.get("strata"):
            cal_ok = False
            notes.append(f"{p}: no calibration comparison")
            continue
        for st in cal["strata"]:
            a = st["auc_diff_served_minus_isotonic"]
            bd = st["brier_diff_served_minus_isotonic"]
            if a["lo"] is not None and a["hi"] is not None and a["hi"] < 0:
                cal_ok = False
                notes.append(f"{p} {st['name']}: served calibrator AUC significantly below isotonic")
            if bd["lo"] is not None and bd["lo"] > 0:
                cal_ok = False
                notes.append(f"{p} {st['name']}: served calibrator Brier significantly above isotonic")
            if a["lo"] is not None and a["lo"] > 0:
                cal_better = True
    return {
        "adopt_cold_stages": bool(stage_ok and stage_better),
        "cold_stages_non_inferior": stage_ok,
        "cold_stages_significantly_better_somewhere": stage_better,
        "adopt_logistic_calibrator": bool(cal_ok and cal_better),
        "calibrator_not_worse_anywhere": cal_ok,
        "calibrator_auc_better_somewhere": cal_better,
        "margin": margin,
        "notes": notes,
    }
