"""End-to-end experiment + training pipeline.

  split → tune on validation (train only) → evaluate on test (train+val) → train production models
  on all interactions → save artifacts + manifest → register version → write experiment report.

Training (this module) and inference (engine.py) share only the model classes and signals.py.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from jev_ml import __version__
from jev_ml.data.dataset import (
    ItemIndex,
    build_user_item_matrix,
    load_dataset_meta,
    load_interactions,
    load_movies,
)
from jev_ml.evaluation.evaluator import EvalResult, Evaluator, RecommendFn
from jev_ml.evaluation.split import make_split
from jev_ml.models.als import ALSRecommender
from jev_ml.models.base import Recommender, TrainContext, topk_indices
from jev_ml.models.content import ContentRecommender
from jev_ml.models.hybrid import SIGNALS, HybridConfig, HybridRanker
from jev_ml.models.itemknn import ItemKNNRecommender
from jev_ml.models.popularity import PopularityRecommender
from jev_ml.paths import CONFIGS_DIR, EXPERIMENTS_DIR, MODELS_DIR, PROCESSED_DIR
from jev_ml.registry import register_version
from jev_ml.signals import UserProfile

log = logging.getLogger(__name__)

EVALUATED_MODELS = ("random", "popularity", "content", "itemknn", "als", "hybrid")


@dataclass
class Components:
    popularity: PopularityRecommender
    content: ContentRecommender
    itemknn: ItemKNNRecommender
    als: ALSRecommender

    def ranker(self, movies: pd.DataFrame, hybrid_cfg: HybridConfig) -> HybridRanker:
        return HybridRanker(movies, self.popularity, self.content, self.itemknn, self.als, hybrid_cfg)


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIGS_DIR / "experiment.yaml"
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    return cfg


def build_context(movies: pd.DataFrame, interactions: pd.DataFrame, seed: int) -> TrainContext:
    item_index = ItemIndex(movies["movie_id"].to_numpy())
    ui, user_ids = build_user_item_matrix(interactions, item_index)
    return TrainContext(
        movies=movies,
        item_index=item_index,
        interactions=interactions,
        user_item=ui,
        user_ids=user_ids,
        seed=seed,
    )


def fit_components(
    ctx: TrainContext, mcfg: dict[str, Any], seed: int, content: ContentRecommender | None = None
) -> Components:
    """Fit all component models. Content depends only on metadata, so it can be reused."""
    pop = PopularityRecommender(**mcfg["popularity"]).fit(ctx)
    if content is None:
        content = ContentRecommender(**mcfg["content"]).fit(ctx)
    knn = ItemKNNRecommender(**mcfg["itemknn"]).fit(ctx)
    als = ALSRecommender(**mcfg["als"], seed=seed).fit(ctx)
    return Components(pop, content, knn, als)


# --- recommend functions used by the evaluator --------------------------------------------------
def component_fn(model: Recommender) -> RecommendFn:
    def fn(profile: UserProfile, k: int) -> np.ndarray:
        return topk_indices(model.score(profile), k, profile.consumed)

    return fn


def random_fn(n_items: int, seed: int) -> RecommendFn:
    def fn(profile: UserProfile, k: int) -> np.ndarray:
        rng = np.random.default_rng(seed + int(profile.items.sum()) % 100_000)
        return topk_indices(rng.random(n_items), k, profile.consumed)

    return fn


def hybrid_fn(ranker: HybridRanker) -> RecommendFn:
    def fn(profile: UserProfile, k: int) -> np.ndarray:
        return np.asarray([r.item for r in ranker.rank(profile, k, explain=False)], dtype=np.int64)

    return fn


class _CachedSignalsRanker(HybridRanker):
    """Memoizes per-user raw signals across hybrid tuning trials (same component models)."""

    def __init__(self, base: HybridRanker) -> None:
        self.__dict__.update(base.__dict__)
        self._cache: dict[tuple[int, float], dict[str, np.ndarray]] = {}

    def raw_signals(self, profile: UserProfile) -> dict[str, np.ndarray]:
        # content scores depend on the damping exponent, so it is part of the key
        key = (id(profile), float(self.config.content_support_damping))
        if key not in self._cache:
            self._cache[key] = {k: v.astype(np.float32) for k, v in super().raw_signals(profile).items()}
        return self._cache[key]


def make_evaluator(
    train: pd.DataFrame,
    target: pd.DataFrame,
    ctx: TrainContext,
    comps: Components,
    ecfg: dict[str, Any],
    seed: int,
    max_users: int | None = None,
    truncate: int | None = None,
) -> Evaluator:
    return Evaluator(
        train,
        target,
        ctx.item_index,
        ctx.user_ids,
        comps.content.pairwise,
        comps.popularity.item_popularity_fraction(),
        ks=tuple(ecfg["ks"]),
        relevance_threshold=ecfg["relevance_threshold"],
        max_users=max_users,
        seed=seed,
        truncate_profile_to=truncate,
    )


def evaluate_all(
    ev: Evaluator,
    comps: Components,
    movies: pd.DataFrame,
    hcfg: HybridConfig,
    seed: int,
    models: tuple[str, ...] = EVALUATED_MODELS,
) -> dict[str, EvalResult]:
    fns: dict[str, RecommendFn] = {
        "random": random_fn(len(movies), seed),
        "popularity": component_fn(comps.popularity),
        "content": component_fn(comps.content),
        "itemknn": component_fn(comps.itemknn),
        "als": component_fn(comps.als),
        "hybrid": hybrid_fn(comps.ranker(movies, hcfg)),
    }
    return {name: ev.evaluate(name, fns[name]) for name in models}


# --- tuning --------------------------------------------------------------------------------------
def tune(
    split_train: pd.DataFrame,
    split_val: pd.DataFrame,
    movies: pd.DataFrame,
    cfg: dict[str, Any],
    content: ContentRecommender,
) -> dict[str, Any]:
    seed = cfg["seed"]
    ecfg = cfg["evaluation"]
    metric = ecfg["selection_metric"]
    tcfg = cfg["tuning"]
    mcfg = json.loads(json.dumps(cfg["models"]))
    ctx = build_context(movies, split_train, seed)
    pop = PopularityRecommender(**mcfg["popularity"]).fit(ctx)
    base = Components(
        pop, content, ItemKNNRecommender(**mcfg["itemknn"]), ALSRecommender(**mcfg["als"], seed=seed)
    )
    ev = make_evaluator(split_train, split_val, ctx, base, ecfg, seed)
    trials: dict[str, list[dict[str, Any]]] = {"itemknn": [], "als": [], "hybrid": []}

    # item-kNN grid
    best_knn: tuple[float, dict[str, Any], ItemKNNRecommender | None] = (-1.0, mcfg["itemknn"], None)
    for k, shrink in itertools.product(tcfg["itemknn"]["k"], tcfg["itemknn"]["shrinkage"]):
        params = {**mcfg["itemknn"], "k": k, "shrinkage": shrink}
        model = ItemKNNRecommender(**params).fit(ctx)
        res = ev.evaluate(f"itemknn[k={k},s={shrink}]", component_fn(model))
        trials["itemknn"].append({"params": params, metric: res.metrics[metric]})
        if res.metrics[metric] > best_knn[0]:
            best_knn = (res.metrics[metric], params, model)

    # ALS grid
    best_als: tuple[float, dict[str, Any], ALSRecommender | None] = (-1.0, mcfg["als"], None)
    grid = itertools.product(tcfg["als"]["factors"], tcfg["als"]["regularization"], tcfg["als"]["alpha"])
    for f, reg, alpha in grid:
        params = {**mcfg["als"], "factors": f, "regularization": reg, "alpha": alpha}
        model_als = ALSRecommender(**params, seed=seed).fit(ctx)
        res = ev.evaluate(f"als[f={f},r={reg},a={alpha}]", component_fn(model_als))
        trials["als"].append(
            {"params": params, metric: res.metrics[metric], "train_seconds": model_als.train_seconds}
        )
        if res.metrics[metric] > best_als[0]:
            best_als = (res.metrics[metric], params, model_als)

    assert best_knn[2] is not None and best_als[2] is not None
    comps = Components(pop, content, best_knn[2], best_als[2])

    # hybrid: default weights first, then seeded Dirichlet random search × diversity lambdas
    rng = np.random.default_rng(seed)
    hbase = HybridConfig.from_dict(mcfg["hybrid"])
    candidates: list[HybridConfig] = [hbase]
    for _ in range(tcfg["hybrid"]["random_trials"]):
        w = rng.dirichlet(np.ones(len(SIGNALS)))
        candidates.append(
            HybridConfig.from_dict(
                {
                    **asdict(hbase),
                    "weights": dict(zip(SIGNALS, w.round(4).tolist(), strict=True)),
                    "diversity_lambda": 1.0,
                }
            )
        )
    ranker = _CachedSignalsRanker(comps.ranker(movies, hbase))
    best_h: tuple[float, HybridConfig] = (-1.0, hbase)
    for hc in candidates:
        ranker.config = hc
        res = ev.evaluate("hybrid-trial", hybrid_fn(ranker))
        trials["hybrid"].append(
            {"params": asdict(hc), metric: res.metrics[metric], "diversity@10": res.metrics["diversity@10"]}
        )
        if res.metrics[metric] > best_h[0]:
            best_h = (res.metrics[metric], hc)
    # cold-start behaviour: ramp length, boost and content damping, tuned on validation users whose
    # profiles are truncated to their first N interactions (fold-in path, like brand-new app users)
    cold_n = ecfg.get("cold_start_profile_size")
    if cold_n:
        ev_cold = make_evaluator(split_train, split_val, ctx, base, ecfg, seed, truncate=cold_n)
        ranker_cold = _CachedSignalsRanker(comps.ranker(movies, best_h[1]))
        best_c: tuple[float, HybridConfig] = (-1.0, best_h[1])
        for ramp, boost, damp in itertools.product(
            tcfg["hybrid"].get("behavioral_ramp", [10]),
            tcfg["hybrid"].get("cold_start_boost", [2.0]),
            tcfg["hybrid"].get("content_support_damping", [1.0]),
        ):
            hc = HybridConfig.from_dict(
                {
                    **asdict(best_h[1]),
                    "behavioral_ramp": ramp,
                    "cold_start_boost": boost,
                    "content_support_damping": damp,
                }
            )
            ranker_cold.config = hc
            r_cold = ev_cold.evaluate(f"hybrid-cold[r={ramp},b={boost},d={damp}]", hybrid_fn(ranker_cold))
            ranker.config = hc
            r_warm = ev.evaluate("hybrid-warm-check", hybrid_fn(ranker))
            # objective: mean of warm and cold NDCG — the product must serve both kinds of users
            obj = 0.5 * (r_cold.metrics[metric] + r_warm.metrics[metric])
            trials["hybrid"].append(
                {
                    "params": asdict(hc),
                    "cold_" + metric: r_cold.metrics[metric],
                    metric: r_warm.metrics[metric],
                    "objective": obj,
                    "sweep": "cold",
                }
            )
            if obj > best_c[0]:
                best_c = (obj, hc)
        best_h = (best_h[0], best_c[1])

    # diversity sweep on the best weights: keep the most diverse lambda within 2% of the best
    best_cfg = best_h[1]
    lam_results = []
    for lam in tcfg["hybrid"]["diversity_lambda"]:
        hc = HybridConfig.from_dict({**asdict(best_cfg), "diversity_lambda": lam})
        ranker.config = hc
        res = ev.evaluate(f"hybrid[lambda={lam}]", hybrid_fn(ranker))
        lam_results.append((lam, res.metrics[metric], res.metrics["diversity@10"]))
        trials["hybrid"].append(
            {
                "params": asdict(hc),
                metric: res.metrics[metric],
                "diversity@10": res.metrics["diversity@10"],
                "sweep": "lambda",
            }
        )
    top = max(m for _, m, _ in lam_results)
    ok = [t for t in lam_results if t[1] >= 0.98 * top]
    chosen_lambda = min(ok, key=lambda t: t[0])[0]  # smallest lambda = most diversity
    best_cfg = HybridConfig.from_dict({**asdict(best_cfg), "diversity_lambda": chosen_lambda})

    return {
        "itemknn": best_knn[1],
        "als": best_als[1],
        "hybrid": asdict(best_cfg),
        "trials": trials,
        "selection_metric": metric,
    }


# --- persistence -------------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - git on PATH is optional metadata
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_model_version(
    comps: Components,
    movies: pd.DataFrame,
    hcfg: HybridConfig,
    manifest_extra: dict[str, Any],
    models_dir: Path = MODELS_DIR,
) -> Path:
    created = datetime.now(UTC)
    payload = json.dumps({"cfg": asdict(hcfg), **manifest_extra}, sort_keys=True, default=str)
    short = hashlib.sha256(payload.encode()).hexdigest()[:8]
    version = f"jev-{created.strftime('%Y%m%dT%H%M%SZ')}-{short}"
    out = models_dir / version
    out.mkdir(parents=True, exist_ok=False)
    comps.popularity.save(out / "popularity")
    comps.content.save(out / "content")
    comps.itemknn.save(out / "itemknn")
    comps.als.save(out / "als")
    movies.to_csv(out / "movies.csv", index=False)
    (out / "hybrid.json").write_text(json.dumps(asdict(hcfg), indent=2))
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    size = sum((out / f).stat().st_size for f in files)
    manifest = {
        "version": version,
        "created_at": created.isoformat(),
        "model_type": "hybrid",
        "components": {
            "popularity": comps.popularity.params(),
            "content": comps.content.params(),
            "itemknn": comps.itemknn.params(),
            "als": comps.als.params(),
        },
        "hybrid_config": asdict(hcfg),
        "artifact_path": str(out),
        "artifact_files": files,
        "artifact_bytes": size,
        "n_items": len(movies),
        "jev_ml_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "als_loss_history": comps.als.loss_history,
        **manifest_extra,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return out


def run_pipeline(
    config_path: Path | None = None,
    quick: bool = False,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    activate: bool = False,  # candidates are gated before promotion (docs/RETRAINING_AND_MODEL_GOVERNANCE.md)
    train_production: bool = True,
    fixed_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full pipeline. `train_production=False` → evaluation only (no model version is written).
    `fixed_models` → evaluate these component/hybrid params instead of tuning (e.g. a saved model's)."""
    from jev_ml.evaluation.report import write_report

    t_start = time.perf_counter()
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    np.random.seed(seed)
    ecfg = cfg["evaluation"]
    movies = load_movies(processed_dir / "movies.csv")
    interactions = load_interactions(processed_dir / "interactions.csv")
    meta = load_dataset_meta(processed_dir / "dataset_meta.json")
    split = make_split(interactions, **cfg["split"])
    log.info("split: %s", split.summary())

    run_id = f"{cfg['experiment_name']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    exp_dir = experiments_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=False)

    # content model depends only on metadata → fit once, reuse everywhere
    ctx_all_items = build_context(movies, split.train, seed)
    content = ContentRecommender(**cfg["models"]["content"]).fit(ctx_all_items)

    tuned: dict[str, Any] = {}
    mcfg = json.loads(json.dumps(cfg["models"]))
    if fixed_models is not None:
        mcfg = json.loads(json.dumps(fixed_models))
    elif cfg["tuning"]["enabled"] and not quick:
        t0 = time.perf_counter()
        tuned = tune(split.train, split.val, movies, cfg, content)
        tuned["seconds"] = time.perf_counter() - t0
        mcfg["itemknn"] = tuned["itemknn"]
        mcfg["als"] = {k: v for k, v in tuned["als"].items() if k != "seed"}
        mcfg["hybrid"] = tuned["hybrid"]
        log.info("tuned in %.0fs: knn=%s als=%s", tuned["seconds"], tuned["itemknn"], tuned["als"])
    hcfg = HybridConfig.from_dict(mcfg["hybrid"])

    # test evaluation: fit on train+val, score on test
    trainval = pd.concat([split.train, split.val], ignore_index=True)
    ctx_tv = build_context(movies, trainval, seed)
    comps_tv = fit_components(ctx_tv, mcfg, seed, content=content)
    ev_test = make_evaluator(trainval, split.test, ctx_tv, comps_tv, ecfg, seed)
    test_results = evaluate_all(ev_test, comps_tv, movies, hcfg, seed)
    cold_n = ecfg.get("cold_start_profile_size")
    cold_results: dict[str, EvalResult] = {}
    if cold_n:
        ev_cold = make_evaluator(trainval, split.test, ctx_tv, comps_tv, ecfg, seed, truncate=cold_n)
        cold_results = evaluate_all(ev_cold, comps_tv, movies, hcfg, seed)

    # production models: all interactions
    comps_full = comps_tv
    if train_production:
        ctx_full = build_context(movies, interactions, seed)
        comps_full = fit_components(ctx_full, mcfg, seed, content=content)

    metrics_payload = {
        "test": {k: v.metrics for k, v in test_results.items()},
        "cold_start": {k: v.metrics for k, v in cold_results.items()},
        "n_eval_users": {
            "test": ev_test.users.__len__(),
            "cold_start": len(cold_results and next(iter(cold_results.values())).per_user_ndcg),
        },
    }
    version_name: str | None = None
    if train_production:
        version_name = _save_and_register(
            comps_full,
            movies,
            hcfg,
            cfg,
            mcfg,
            meta,
            split,
            run_id,
            metrics_payload,
            interactions,
            models_dir,
            activate,
            seed,
        )
    result = _result(
        run_id,
        cfg,
        mcfg,
        meta,
        seed,
        split,
        metrics_payload,
        tuned,
        version_name,
        comps_full,
        test_results,
        t_start,
        quick,
    )
    write_report(result, exp_dir)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(processed_dir / "dataset_meta.json", exp_dir / "dataset_meta.json")
    log.info(
        "pipeline finished in %.0fs → model %s, experiment %s", result["seconds_total"], version_name, run_id
    )
    return result


def _save_and_register(
    comps_full: Components,
    movies: pd.DataFrame,
    hcfg: HybridConfig,
    cfg: dict[str, Any],
    mcfg: dict[str, Any],
    meta: dict[str, Any],
    split: Any,
    run_id: str,
    metrics_payload: dict[str, Any],
    interactions: pd.DataFrame,
    models_dir: Path,
    activate: bool,
    seed: int,
) -> str:
    model_dir = save_model_version(
        comps_full,
        movies,
        hcfg,
        {
            "experiment_run": run_id,
            "experiment_name": cfg["experiment_name"],
            "dataset_version": meta["dataset_version"],
            "training_seed": seed,
            "training_config": {**cfg, "models": mcfg},
            "split": split.summary(),
            "metrics": metrics_payload["test"]["hybrid"],
            "trained_on_rows": len(interactions),
        },
        models_dir,
    )
    manifest = json.loads((model_dir / "manifest.json").read_text())
    register_version(manifest, models_dir, activate=activate)
    version: str = manifest["version"]
    return version


def _result(
    run_id: str,
    cfg: dict[str, Any],
    mcfg: dict[str, Any],
    meta: dict[str, Any],
    seed: int,
    split: Any,
    metrics_payload: dict[str, Any],
    tuned: dict[str, Any],
    version: str | None,
    comps_full: Components,
    test_results: dict[str, EvalResult],
    t_start: float,
    quick: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": cfg["experiment_name"],
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": meta["dataset_version"],
        "training_seed": seed,
        "config": {**cfg, "models_tuned": mcfg},
        "split": split.summary(),
        "metrics": metrics_payload,
        "tuning": tuned,
        "model_version": version,
        "als_loss_history": comps_full.als.loss_history,
        "per_user_ndcg10": {k: v.per_user_ndcg for k, v in test_results.items()},
        "seconds_total": time.perf_counter() - t_start,
        "quick": quick,
    }
