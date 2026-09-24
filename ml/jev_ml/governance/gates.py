"""The promotion gate: a candidate against the active model on the same frozen split.

What is compared. Both production artifacts were trained on all of their data, so scoring them on any
held-out split of the candidate's snapshot would leak (the candidate saw that split). The gate
therefore compares the two *training recipes* (component hyper-parameters + hybrid config, read from
each manifest) refit with the same seed on the train+validation part of the candidate snapshot's split
and scored on its test part: the same users, the same profiles, the same relevant items. The split is
the candidate's configured split of its snapshot, recomputed deterministically and fingerprinted.
The artifacts themselves are then checked for what only an artifact can show: it loads, passes the
self-check, serves within the latency budget and carries a calibration no worse than the incumbent's.

Gates (all thresholds in GateConfig, set from ``JEV_GOVERNANCE_GATE_*``):

0. ``split``        the frozen split passes jev_ml.evaluation.leakage.check_split (else nothing is compared).
1. ``artifact``     the candidate loads (RecommendationEngine validates shapes, finite factors) and a
                    self-check serves a cold and a warm list: k unique films, finite scores, nothing
                    the profile already holds. A failure skips everything else.
2. ``ndcg@10``      non-inferiority: the lower bound of the paired bootstrap CI of the per-user
                    difference (candidate - incumbent) >= -ndcg_margin.
3. ``recall@10``    the same with recall_margin.
4. ``cold_start``   the same NDCG@10 test on profiles truncated to the configured cold-start size,
                    with cold_start_margin.
5. ``coverage@10``  catalogue coverage >= (1 - coverage_max_relative_drop) x the incumbent's.
6. ``calibration``  when the incumbent is calibrated: the candidate is too, headline test ECE <=
                    incumbent + ece_margin and AUC >= incumbent - auc_margin. Skipped when neither is.
7. ``latency``      p95 of build_profile + recommend(k=10) over sampled test users <= latency_p95_ms.

Without an incumbent (bootstrap), gates 2-6 are ``skipped`` and the decision rests on 1 and 7.

Power rule for the margins. A non-inferiority gate passes when ``diff - z * SE >= -margin`` (z the
two-sided (1 - alpha) quantile, 1.645 at alpha 0.10). For a candidate that is exactly as good as the
incumbent (true diff 0) the pass probability is ``Phi(margin / SE - z)``; it reaches ``power`` when
``margin >= (z + z_power) * SE`` (``power_margin``). The defaults apply that rule with power 0.80
(z + z_power = 1.645 + 0.842 = 2.49) to the paired SEs observed on the MovieLens split (594 users,
``models/jev-20260924T174707Z-0298b516/gate.json``): NDCG@10 SE 0.0039 -> 0.010, Recall@10 SE
0.0044 -> 0.011, cold-start NDCG@10 SE 0.0022 -> 0.006. The earlier 0.005 NDCG margin passed an
equivalent candidate only ~36 % of the time. Every non-inferiority gate records its observed SE, the
pass probability an equivalent candidate would have had (``power_if_equivalent``) and the margin the
rule would require (``margin_for_80pct``), so an underpowered comparison is visible in gate.json.
The three gates are conjunctive, so the joint pass rate of an equivalent candidate is lower
(about 0.8 x 0.8 x 0.9 if they were independent; they are positively correlated, so higher).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.calibration import load_calibration
from jev_ml.data.dataset import load_interactions, load_movies
from jev_ml.evaluation.evaluator import EvalResult
from jev_ml.evaluation.leakage import LeakageError, check_split, movies_with_tags_before, tag_cutoffs
from jev_ml.evaluation.split import make_split
from jev_ml.evaluation.stats import paired_comparison
from jev_ml.models.hybrid import HybridConfig
from jev_ml.paths import MODELS_DIR, RAW_DIR
from jev_ml.registry import load_manifest

log = logging.getLogger(__name__)

GATE_VERSION = "gate-1.1.0"
PASS, FAIL, SKIP = "pass", "fail", "skipped"


@dataclass(frozen=True)
class GateConfig:
    # margins from the power rule in the module docstring (80 % pass rate for an equivalent model)
    ndcg_margin: float = 0.010
    recall_margin: float = 0.011
    cold_start: bool = True
    cold_start_margin: float = 0.006
    coverage_max_relative_drop: float = 0.2
    ece_margin: float = 0.01
    auc_margin: float = 0.01
    latency_p95_ms: float = 250.0
    latency_requests: int = 50
    bootstrap_b: int = 2000
    permutations: int = 2000
    # two-sided (1 - alpha) CI: alpha 0.10 gives the one-sided 95 % lower bound of non-inferiority
    alpha: float = 0.10
    max_users: int | None = None
    seed: int = 42
    # content features of both refits only see tags written before the test period (the leakage
    # control of jev_ml.evaluation.leakage); needs the raw tags.csv, else all tags are kept (noted)
    leak_free_tags: bool = True

    def quick(self) -> GateConfig:
        """Laptop/CI budget: fewer permutations and latency probes; same thresholds and the same
        bootstrap B (the CI lower bound decides the gate, and the bootstrap is cheap next to the
        refits, so it is not reduced)."""
        return replace(
            self,
            permutations=min(self.permutations, 1000),
            latency_requests=min(self.latency_requests, 20),
        )


def power_margin(se: float, alpha: float = 0.10, power: float = 0.80) -> float:
    """Smallest non-inferiority margin at which a candidate with true difference 0 passes with
    probability ``power``, given the paired standard error ``se`` and a two-sided (1 - alpha) CI."""
    nd = NormalDist()
    return (nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) * se


def pass_probability_if_equivalent(se: float, margin: float, alpha: float = 0.10) -> float | None:
    """P(gate passes | true difference 0) = Phi(margin / se - z_(1 - alpha/2))."""
    if se <= 0:
        return 1.0 if margin >= 0 else None
    nd = NormalDist()
    return nd.cdf(margin / se - nd.inv_cdf(1 - alpha / 2))


def _raw_tags(snapshot_dir: Path) -> Path | None:
    """The raw MovieLens tags.csv matching the snapshot's base dataset, or None.

    A snapshot may carry its own tags.csv; otherwise data/raw's file is used only when the snapshot
    was built from the same processed dataset (same dataset_version), never for other data."""
    own = snapshot_dir / "tags.csv"
    if own.exists():
        return own
    from jev_ml import paths

    raw = RAW_DIR / "ml-latest-small" / "tags.csv"
    meta_path = paths.PROCESSED_DIR / "dataset_meta.json"
    if not raw.exists() or not meta_path.exists():
        return None
    processed_version = json.loads(meta_path.read_text()).get("dataset_version")
    base = None
    for name, key in (
        ("manifest.json", "base_dataset_version"),
        ("dataset_meta.json", "base_dataset_version"),
    ):
        f = snapshot_dir / name
        if f.exists():
            base = json.loads(f.read_text()).get(key)
            if base:
                break
    if base is None and (snapshot_dir / "dataset_meta.json").exists():  # the processed dir itself
        base = json.loads((snapshot_dir / "dataset_meta.json").read_text()).get("dataset_version")
    return raw if base is not None and base == processed_version else None


def _recipe(manifest: dict[str, Any]) -> tuple[dict[str, Any], HybridConfig]:
    tcfg = manifest.get("training_config") or {}
    mcfg = json.loads(json.dumps(tcfg["models"]))
    mcfg["als"] = {k: v for k, v in mcfg["als"].items() if k != "seed"}
    return mcfg, HybridConfig.from_dict(manifest["hybrid_config"])


def _fingerprint(df: pd.DataFrame) -> str:
    cols = df[["user_id", "movie_id", "timestamp"]].sort_values(["user_id", "movie_id", "timestamp"])
    return hashlib.sha256(cols.to_csv(index=False).encode()).hexdigest()[:16]


def self_check(engine: Any, k: int = 10) -> list[str]:
    """Problems found when serving a cold and a warm list (empty = healthy)."""
    from jev_ml.engine import Interaction

    problems: list[str] = []
    n = len(engine.movies)
    want = min(k, n)
    cold = engine.recommend(engine.build_profile([]), k=k)
    popular = engine.movies.sort_values(["n_ratings", "movie_id"], ascending=[False, True])["movie_id"]
    held = [int(m) for m in popular.head(5)]
    warm_profile = engine.build_profile(
        [Interaction(movie_id=m, kind="rating", value=5.0, timestamp=0.0) for m in held]
    )
    warm = engine.recommend(warm_profile, k=k)
    for name, recs in (("cold", cold), ("warm", warm)):
        ids = [r.movie_id for r in recs]
        if len(ids) < want:
            problems.append(f"{name} list has {len(ids)} items, expected {want}")
        if len(set(ids)) != len(ids):
            problems.append(f"{name} list repeats a film")
        if not all(np.isfinite(r.score) for r in recs):
            problems.append(f"{name} list has non-finite scores")
    if set(held) & {r.movie_id for r in warm}:
        problems.append("warm list recommends films the profile already rated")
    return problems


def _latency(engine: Any, rows: pd.DataFrame, users: list[int], n: int, seed: int) -> dict[str, Any]:
    from jev_ml.engine import Interaction

    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(users, size=min(n, len(users)), replace=False).tolist()) if users else []
    grouped = rows[rows["user_id"].isin(pick)].groupby("user_id")
    engine.recommend(engine.build_profile([]), k=10)  # warm-up (lazy caches)
    times: list[float] = []
    for u in pick:
        g = grouped.get_group(u)
        events = [
            Interaction(movie_id=int(m), kind="rating", value=float(r), timestamp=float(t))
            for m, r, t in zip(g["movie_id"], g["rating"], g["timestamp"], strict=True)
        ]
        t0 = time.perf_counter()
        engine.recommend(engine.build_profile(events), k=10)
        times.append((time.perf_counter() - t0) * 1000)
    if not times:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    arr = np.asarray(times)
    return {
        "n": len(times),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def _evaluate_recipe(
    name: str,
    manifest: dict[str, Any],
    movies: pd.DataFrame,
    trainval: pd.DataFrame,
    test: pd.DataFrame,
    ecfg: dict[str, Any],
    seed: int,
    max_users: int | None,
    cold_n: int | None,
) -> tuple[EvalResult, EvalResult | None]:
    from jev_ml.training import build_context, fit_components, hybrid_fn, make_evaluator

    mcfg, hcfg = _recipe(manifest)
    ctx = build_context(movies, trainval, seed)
    comps = fit_components(ctx, mcfg, seed)
    ranker = comps.ranker(movies, hcfg)
    ev = make_evaluator(trainval, test, ctx, comps, ecfg, seed, max_users=max_users)
    warm = ev.evaluate(name, hybrid_fn(ranker))
    cold = None
    if cold_n:
        ev_cold = make_evaluator(trainval, test, ctx, comps, ecfg, seed, max_users=max_users, truncate=cold_n)
        cold = ev_cold.evaluate(f"{name}-cold", hybrid_fn(ranker))
    return warm, cold


def _non_inferior(
    cand: list[float], inc: list[float], margin: float, cfg: GateConfig, metric: str
) -> dict[str, Any]:
    cmp = paired_comparison(
        cand, inc, b=cfg.bootstrap_b, permutations=cfg.permutations, alpha=cfg.alpha, seed=cfg.seed
    )
    lo = cmp["lo"]
    passed = lo is not None and lo >= -margin
    d = np.asarray(cand, dtype=np.float64) - np.asarray(inc, dtype=np.float64)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    return {
        "status": PASS if passed else FAIL,
        "metric": metric,
        "candidate": float(np.mean(cand)) if cand else None,
        "incumbent": float(np.mean(inc)) if inc else None,
        "diff": cmp["diff"],
        "ci_lo": lo,
        "ci_hi": cmp["hi"],
        "ci_level": 1 - cfg.alpha,
        "margin": margin,
        "se": se,
        "power_if_equivalent": None if se is None else pass_probability_if_equivalent(se, margin, cfg.alpha),
        "margin_for_80pct": None if se is None else power_margin(se, cfg.alpha, 0.80),
        "p_value": cmp.get("p_value"),
        "wins": cmp.get("wins"),
        "losses": cmp.get("losses"),
        "n_users": cmp["n"],
        "reason": None
        if passed
        else f"{metric}: CI lower bound {lo:.4f} of candidate - incumbent is below -{margin}",
    }


def _calibration_gate(
    cand_dir: Path, cand: str, inc_dir: Path | None, inc: str | None, cfg: GateConfig
) -> dict[str, Any]:
    def headline(d: Path, v: str) -> dict[str, Any] | None:
        data = load_calibration(d)
        if data is None or data.get("model_version") != v:
            return None
        h = data.get("headline")
        return h if isinstance(h, dict) else None

    ch = headline(cand_dir, cand)
    ih = headline(inc_dir, inc) if inc_dir is not None and inc is not None else None
    if ih is None and ch is None:
        return {"status": SKIP, "reason": None, "note": "neither model is calibrated"}
    if ih is None:
        return {
            "status": PASS,
            "reason": None,
            "candidate_ece": ch.get("ece") if ch else None,
            "note": "the incumbent has no calibration",
        }
    if ch is None:
        return {
            "status": FAIL,
            "incumbent_ece": ih.get("ece"),
            "reason": "calibration: the incumbent serves calibrated confidence and the candidate has none",
        }
    reasons = []
    ece_c, ece_i = ch.get("ece"), ih.get("ece")
    if ece_c is None or ece_i is None or ece_c > ece_i + cfg.ece_margin:
        reasons.append(f"calibration: ECE {ece_c} > incumbent {ece_i} + {cfg.ece_margin}")
    auc_c, auc_i = ch.get("auc"), ih.get("auc")
    if auc_c is not None and auc_i is not None and auc_c < auc_i - cfg.auc_margin:
        reasons.append(f"calibration: AUC {auc_c} < incumbent {auc_i} - {cfg.auc_margin}")
    return {
        "status": FAIL if reasons else PASS,
        "candidate_ece": ece_c,
        "incumbent_ece": ece_i,
        "ece_margin": cfg.ece_margin,
        "candidate_auc": auc_c,
        "incumbent_auc": auc_i,
        "auc_margin": cfg.auc_margin,
        "reason": "; ".join(reasons) or None,
    }


def evaluate_candidate(
    candidate: str,
    incumbent: str | None,
    snapshot_dir: Path,
    cfg: GateConfig | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Run every gate. Returns ``{"passed", "reasons", "gates": {...}, ...}`` (JSON-safe)."""
    from jev_ml.engine import RecommendationEngine

    cfg = cfg or GateConfig()
    t0 = time.perf_counter()
    gates: dict[str, dict[str, Any]] = {}
    out: dict[str, Any] = {
        "gate_version": GATE_VERSION,
        "candidate": candidate,
        "incumbent": incumbent,
        "snapshot_dir": str(snapshot_dir),
        "config": asdict(cfg),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "gates": gates,
        "method": "paired bootstrap on per-user metrics of both training recipes refit on the "
        "candidate snapshot's frozen split (train+val -> test)",
    }

    # 1. artifact loads and passes the self-check
    try:
        engine = RecommendationEngine(models_dir / candidate)
        problems = self_check(engine)
    except Exception as exc:  # any load failure means "never serve this"
        engine, problems = None, [f"artifact failed to load: {type(exc).__name__}: {exc}"]
    gates["artifact"] = {
        "status": FAIL if problems else PASS,
        "problems": problems,
        "load_seconds": round(engine.load_seconds, 3) if engine is not None else None,
        "reason": "; ".join(problems) or None,
    }
    if problems or engine is None:
        return _finish(out, t0)

    cand_man = load_manifest(candidate, models_dir)
    inc_man = load_manifest(incumbent, models_dir) if incumbent and incumbent != candidate else None
    tcfg = cand_man.get("training_config") or {}
    seed = int(cand_man.get("training_seed", tcfg.get("seed", 42)))
    ecfg = tcfg["evaluation"]
    movies = load_movies(snapshot_dir / "movies.csv")
    inter = load_interactions(snapshot_dir / "interactions.csv")
    split = make_split(inter, **tcfg["split"])
    trainval = pd.concat([split.train, split.val], ignore_index=True)
    out["split"] = {**split.summary(), "config": tcfg["split"], "test_fingerprint": _fingerprint(split.test)}
    # the frozen split must honour its own temporal contract (jev_ml.evaluation.leakage)
    try:
        out["split"]["leakage_check"] = check_split(split)
        gates["split"] = {"status": PASS, "reason": None}
    except LeakageError as exc:
        gates["split"] = {"status": FAIL, "reason": f"split leakage: {exc}"}
        return _finish(out, t0)
    tags_path = _raw_tags(snapshot_dir)
    if cfg.leak_free_tags and tags_path is not None:
        movies = movies_with_tags_before(movies, pd.read_csv(tags_path), tag_cutoffs(split, "test"))
        out["split"]["tags"] = f"before the test period ({tags_path.name})"
    else:
        out["split"]["tags"] = "all tags (movies.csv; raw tags.csv unavailable or leak_free_tags off)"
    cold_n = ecfg.get("cold_start_profile_size") if cfg.cold_start else None

    # 7. latency (artifact), measured before the refits so the CPU is not busy
    rng_users = sorted(set(split.test["user_id"].tolist()) & set(trainval["user_id"].tolist()))
    lat = _latency(engine, trainval, rng_users, cfg.latency_requests, cfg.seed)
    lat_ok = lat["p95_ms"] is not None and lat["p95_ms"] <= cfg.latency_p95_ms
    gates["latency"] = {
        "status": PASS if lat_ok else FAIL,
        **lat,
        "budget_p95_ms": cfg.latency_p95_ms,
        "reason": None if lat_ok else f"latency: p95 {lat['p95_ms']} ms exceeds {cfg.latency_p95_ms} ms",
    }
    if inc_man is not None:
        try:
            gates["latency"]["incumbent"] = _latency(
                RecommendationEngine(models_dir / str(incumbent)),
                trainval,
                rng_users,
                cfg.latency_requests,
                cfg.seed,
            )
        except Exception:  # informational only
            log.exception("incumbent latency probe failed")

    if inc_man is None:
        note = "no incumbent: nothing to compare" if incumbent is None else "candidate is the incumbent"
        for name in ("ndcg@10", "recall@10", "cold_start", "coverage@10"):
            gates[name] = {"status": SKIP, "reason": None, "note": note}
        gates["calibration"] = {"status": SKIP, "reason": None, "note": note}
        return _finish(out, t0)

    c_warm, c_cold = _evaluate_recipe(
        "candidate", cand_man, movies, trainval, split.test, ecfg, seed, cfg.max_users, cold_n
    )
    i_warm, i_cold = _evaluate_recipe(
        "incumbent", inc_man, movies, trainval, split.test, ecfg, seed, cfg.max_users, cold_n
    )
    if c_warm.user_ids != i_warm.user_ids:
        raise AssertionError("gate compared different users")

    # 2-3. non-inferiority on NDCG@10 and Recall@10
    gates["ndcg@10"] = _non_inferior(
        c_warm.per_user["ndcg@10"], i_warm.per_user["ndcg@10"], cfg.ndcg_margin, cfg, "ndcg@10"
    )
    gates["recall@10"] = _non_inferior(
        c_warm.per_user["recall@10"], i_warm.per_user["recall@10"], cfg.recall_margin, cfg, "recall@10"
    )
    # 4. cold start
    if c_cold is not None and i_cold is not None:
        g = _non_inferior(
            c_cold.per_user["ndcg@10"],
            i_cold.per_user["ndcg@10"],
            cfg.cold_start_margin,
            cfg,
            "cold-start ndcg@10",
        )
        gates["cold_start"] = {**g, "profile_size": cold_n}
    else:
        gates["cold_start"] = {"status": SKIP, "reason": None, "note": "cold-start protocol disabled"}
    # 5. coverage must not collapse
    cc, ic = c_warm.metrics.get("coverage@10", 0.0), i_warm.metrics.get("coverage@10", 0.0)
    floor = (1 - cfg.coverage_max_relative_drop) * ic
    cov_ok = cc >= floor
    gates["coverage@10"] = {
        "status": PASS if cov_ok else FAIL,
        "candidate": cc,
        "incumbent": ic,
        "floor": floor,
        "max_relative_drop": cfg.coverage_max_relative_drop,
        "reason": None if cov_ok else f"coverage@10 {cc:.4f} collapsed below {floor:.4f}",
    }
    # 6. calibration
    gates["calibration"] = _calibration_gate(
        models_dir / candidate, candidate, models_dir / str(incumbent), str(incumbent), cfg
    )
    out["metrics"] = {
        "candidate": {k: c_warm.metrics[k] for k in ("ndcg@10", "recall@10", "coverage@10", "diversity@10")},
        "incumbent": {k: i_warm.metrics[k] for k in ("ndcg@10", "recall@10", "coverage@10", "diversity@10")},
        "n_users": c_warm.n_users,
    }
    return _finish(out, t0)


def _finish(out: dict[str, Any], t0: float) -> dict[str, Any]:
    reasons = [g["reason"] for g in out["gates"].values() if g.get("status") == FAIL and g.get("reason")]
    out["passed"] = not any(g.get("status") == FAIL for g in out["gates"].values())
    out["reasons"] = reasons
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return json.loads(json.dumps(out, default=_json_default, allow_nan=False))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)
