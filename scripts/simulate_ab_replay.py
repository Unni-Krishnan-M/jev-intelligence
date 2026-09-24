"""OFFLINE A/B REPLAY on MovieLens: offline replay using held-out ratings, not live traffic.

MovieLens has no online traffic, so this script demonstrates the online-experiment machinery
(docs/EXPERIMENTATION.md) on logged data, honestly labelled:

1. Split the ratings with the project's user_temporal protocol (per member: oldest 70 % train, next 10 %
   validation, newest 20 % test). History = train + validation; the future = the test ratings.
2. Train a model on the history only (the active model's component and hybrid config), so no held-out
   rating leaks into the model. Catalogue statistics (n_ratings, mean_rating) are recomputed from the
   history too.
3. Build a throwaway SQLite database (all migrations), load the catalogue, one member per MovieLens user
   and their history ratings (with their original timestamps).
4. Create and start an experiment through the same service code the admin API uses, then serve every
   test-split member once through the real serving path (``services.recommend.personalized``: sticky
   hash assignment, variant config, strategy decision, exposure logging).
5. Outcomes = each member's held-out future ratings, attributed to their exposure (source "replay";
   >= 4 is positive, <= 2 is negative, rank = the served rank or NULL when the list missed it).
6. The report is produced by the same metric and statistics code as GET /experiments/online/{key}/results.

What this is not: members did not see these lists, so "interaction" means "later rated an item the list
contained" and the effect of showing a list on behaviour is not measured. Treat the numbers as an offline
ranking comparison run through the online pipeline, not as an online result.

    uv run python scripts/simulate_ab_replay.py [--out experiments/ab-replay-<ts>] [--limit-users N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "offline replay using held-out ratings, not live traffic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit-users", type=int, default=0, help="replay only the first N test users (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-db", action="store_true", help="keep the throwaway database and model")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="jev-ab-replay-"))
    # the real model registry (read only: the champion's config) and report directory, resolved before
    # the environment points jev_ml / jev_api at the throwaway ones
    real_models = Path(os.environ.get("JEV_MODELS_DIR", ROOT / "models"))
    real_experiments = Path(os.environ.get("JEV_EXPERIMENTS_DIR", ROOT / "experiments"))
    # the throwaway database and model directory must be configured before jev_api is imported
    os.environ["JEV_DATABASE_URL"] = f"sqlite:///{work / 'replay.db'}"
    os.environ["JEV_MODELS_DIR"] = str(work / "models")
    os.environ["JEV_EXPERIMENTS_DIR"] = str(work / "experiments")
    os.environ.setdefault("JEV_JWT_SECRET", "offline-replay-only-not-a-deployment-secret-000")
    os.environ["JEV_INTEL_RUN_ON_STARTUP"] = "false"
    os.environ.pop("JEV_REDIS_URL", None)
    sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "ml")]

    import pandas as pd
    from sqlalchemy import insert

    from jev_ml.data.dataset import load_interactions, load_movies
    from jev_ml.evaluation.split import user_temporal_split
    from jev_ml.models.hybrid import HybridConfig
    from jev_ml.paths import PROCESSED_DIR
    from jev_ml.registry import active_version, load_manifest, register_version
    from jev_ml.training import build_context, fit_components, save_model_version

    t_all = time.perf_counter()
    movies = load_movies(PROCESSED_DIR / "movies.csv")
    inter = load_interactions(PROCESSED_DIR / "interactions.csv")
    split = user_temporal_split(inter, val_frac=0.1, test_frac=0.2)
    history = pd.concat([split.train, split.val], ignore_index=True)
    future = split.test
    print(f"split: {split.summary()}")

    # the active model's configuration, retrained on the history only
    champion = active_version(real_models)
    if champion is None:
        print("no active model: run scripts/train_models.py first", file=sys.stderr)
        return 2
    manifest = load_manifest(champion, real_models)
    mcfg = manifest["training_config"]["models"]
    hcfg = HybridConfig.from_dict(manifest["hybrid_config"])
    stats = history.groupby("movie_id")["rating"].agg(n_ratings="size", mean_rating="mean")
    movies_h = movies.drop(columns=["n_ratings", "mean_rating"]).merge(
        stats, left_on="movie_id", right_index=True, how="left"
    )
    movies_h["n_ratings"] = movies_h["n_ratings"].fillna(0).astype(int)
    t0 = time.perf_counter()
    comps = fit_components(build_context(movies_h, history, args.seed), mcfg, args.seed)
    models_dir = Path(os.environ["JEV_MODELS_DIR"])
    out_model = save_model_version(
        comps,
        movies_h,
        hcfg,
        {
            "dataset_version": manifest.get("dataset_version"),
            "training_seed": args.seed,
            "training_config": manifest["training_config"],
            "trained_on_rows": len(history),
            "split": {**split.summary(), "note": "offline A/B replay: trained on train + validation only"},
            "metrics": {},
        },
        models_dir,
    )
    register_version(json.loads((out_model / "manifest.json").read_text()), models_dir, activate=True)
    fit_s = time.perf_counter() - t0
    print(f"trained replay model {out_model.name} on {len(history)} history rows in {fit_s:.1f}s")

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", os.environ["JEV_DATABASE_URL"])
    command.upgrade(cfg, "head")

    from jev_api.cache import MemoryCache
    from jev_api.db import SessionLocal
    from jev_api.models import Rating, User
    from jev_api.models.experiments import AbExposure
    from jev_api.services import experiments as xp
    from jev_api.services.recommend import personalized
    from jev_api.services.sync import seed_catalog
    from jev_ml.engine import RecommendationEngine
    from jev_ml.models.hybrid import RecommendationFilters

    engine = RecommendationEngine(out_model)
    users = sorted(int(u) for u in future["user_id"].unique())
    if args.limit_users:
        users = users[: args.limit_users]
    with SessionLocal() as db:
        movies_csv = work / "movies_history.csv"
        movies_h.to_csv(movies_csv, index=False)
        seed_catalog(db, movies_csv)
        db.execute(
            insert(User),
            [
                {
                    "id": u,
                    "email": f"replay-{u}@replay.invalid",
                    "password_hash": "!",
                    "display_name": f"MovieLens user {u}",
                    "is_admin": False,
                    "onboarding_completed": True,
                    "profile_version": 0,
                    "recommendation_prefs": {},
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                }
                for u in users
            ],
        )
        hist = history[history["user_id"].isin(users)]
        ts = pd.to_datetime(hist["timestamp"], unit="s", utc=True)
        db.execute(
            insert(Rating),
            [
                {"user_id": int(u), "movie_id": int(m), "rating": float(r), "created_at": t, "updated_at": t}
                for u, m, r, t in zip(hist["user_id"], hist["movie_id"], hist["rating"], ts, strict=True)
            ],
        )
        db.commit()

        body = {
            "key": f"replay-{datetime.now(UTC).strftime('%Y%m%dt%H%M%S')}",
            "name": "Offline replay: MMR diversity and recency decay vs the champion config",
            "hypothesis": (
                "More MMR diversity (lambda 0.6 vs the tuned 0.8) or a 365-day recency half-life on the "
                "profile changes NDCG@10 on members' next ratings. " + LABEL + "."
            ),
            "primary_metric": "ndcg_at_10",
            "guardrails": [
                {"metric": "negative_rate", "max_increase": 0.02},
                {"metric": "latency_p95_ms", "max_ratio": 1.5, "min_absolute_ms": 5.0},
            ],
            "traffic_percent": 100.0,
            "attribution_window_hours": 24 * 365 * 30,  # replay: every held-out rating is "after" serving
            "analysis": {"alpha": 0.05, "power": 0.8, "mde_relative": 0.05, "min_users_per_variant": 100},
            "variants": [
                {"name": "control", "is_control": True, "description": "the champion config as served"},
                {
                    "name": "diverse",
                    "description": "MMR diversity lambda 0.6",
                    "config": {"hybrid_overrides": {"diversity_lambda": 0.6}},
                },
                {
                    "name": "recency",
                    "description": "profile recency half-life 365 days",
                    "config": {"recency_half_life_days": 365.0},
                },
            ],
        }
        exp = xp.create_experiment(db, body, None, data_source=xp.REPLAY)
        xp.transition(db, exp, "start", None)
        print(f"experiment {exp.key} running; replaying {len(users)} test-split members")

        cache = MemoryCache(max_items=50_000)
        filters = RecommendationFilters()
        t0 = time.perf_counter()
        for i, uid in enumerate(users, 1):
            user = db.get(User, uid)
            assert user is not None
            personalized(db, cache, engine, user, 10, 0, filters, "replay")
            if i % 100 == 0:
                print(f"  served {i}/{len(users)} ({time.perf_counter() - t0:.0f}s)")
        serve_s = time.perf_counter() - t0

        # outcomes: the held-out future ratings, attributed to the member's exposure
        exposures = {
            e.user_id: e
            for e in db.query(AbExposure).filter(AbExposure.experiment_id == exp.id).order_by(AbExposure.id)
        }
        fut = future[future["user_id"].isin(list(exposures))]
        rows = []
        for u, m, r in zip(fut["user_id"], fut["movie_id"], fut["rating"], strict=True):
            e = exposures[int(u)]
            ranks = {int(it["movie_id"]): int(it["rank"]) for it in e.items}
            rows.append(
                {
                    "exposure_id": e.id,
                    "experiment_id": exp.id,
                    "variant": e.variant,
                    "user_id": int(u),
                    "movie_id": int(m),
                    "kind": "rating",
                    "value": float(r),
                    "rank": ranks.get(int(m)),
                    "source": "replay",
                    "occurred_at": e.served_at + timedelta(seconds=1),
                }
            )
        xp.record_outcomes(db, rows)
        xp.transition(db, exp, "stop", None)
        xp.transition(db, exp, "conclude", None)
        result = dict(exp.result or {})

    result["replay"] = {
        "label": LABEL,
        "dataset": manifest.get("dataset_version"),
        "split": split.summary(),
        "history_rows": len(history),
        "future_rows_attributed": len(rows),
        "positive_threshold": 4.0,
        "negative_threshold": 2.0,
        "list_length": 10,
        "model": {
            "version": out_model.name,
            "trained_on": "train + validation",
            "fit_seconds": round(fit_s, 1),
        },
        "champion_config_from": champion,
        "members_replayed": len(users),
        "serve_seconds": round(serve_s, 1),
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "not_measured": "the causal effect of showing a list on behaviour (nobody saw these lists)",
    }
    out = args.out or real_experiments / f"ab-replay-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "REPORT.md").write_text(render(result))
    print(render(result))
    print(f"report: {out}")
    if not args.keep_db:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"kept work dir {work}")
    return 0


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.2f} %"


def _num(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _label(v: dict) -> str:
    return v["name"] + (" (control)" if v["is_control"] and v["name"] != "control" else "")


def render(r: dict) -> str:
    lines = [
        f"# A/B report: {r['experiment']}",
        "",
        f"**{r['label'].upper()}.**",
        "",
        f"Primary metric: `{r['primary_metric']}` (alpha {r['alpha']}, Bonferroni-adjusted "
        f"{_num(r['alpha_primary'], 4)} per treatment). Unit of analysis: {r['unit_of_analysis']}.",
        "",
        "| variant | users | exposures | NDCG@10 | positive rate | rating rate | negative rate | "
        "diversity | novelty (bits) | coverage | p95 latency ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in r["variants"]:
        lines.append(
            f"| {_label(v)} | {v['exposed_users']} | "
            f"{v['exposures']} | "
            f"{_num(v['means']['ndcg_at_10']['mean'])} | {_pct(v['rates']['positive_rate']['rate'])} | "
            f"{_pct(v['rates']['rating_rate']['rate'])} | {_pct(v['rates']['negative_rate']['rate'])} | "
            f"{_num(v['means']['diversity']['mean'])} | {_num(v['means']['novelty']['mean'], 2)} | "
            f"{_pct(v['coverage']['value'])} | {_num(v['latency_ms']['p95'], 1)} |"
        )
    lines += ["", "## Treatment vs control", ""]
    lines += ["| variant | metric | diff | 95 % CI | p | significant |", "|---|---|---|---|---|---|"]
    for name, comp in r["comparisons"].items():
        for metric in ("ndcg_at_10", "positive_rate", "rating_rate", "negative_rate", "diversity", "novelty"):
            c = comp[metric]
            ci = c["ci"]
            lines.append(
                f"| {name} | {metric} | {_num(c['diff'])} | "
                f"{_num(ci[0])} .. {_num(ci[1])} | {_num(c['p_value'], 4)} | "
                f"{'yes' if c['significant'] else 'no'} |"
            )
    srm = r["srm"]
    lines += [
        "",
        "## Checks",
        "",
        f"- SRM (assigned): observed {srm['assigned']['observed']}, expected {srm['assigned']['expected']}, "
        f"p = {srm['assigned']['p_value']} -> {'MISMATCH' if srm['detected'] else 'ok'}",
        f"- Sample size: {r['sample_size']['required_per_variant']} users per variant needed for a "
        f"{r['sample_size']['mde_relative']:.0%} relative change; smallest variant has "
        f"{r['sample_size']['smallest_variant_users']}",
    ]
    lines += [f"- Warning: {w}" for w in r["warnings"]]
    for g in r["guardrails"]:
        lines.append(
            f"- Guardrail {g['metric']} ({g['variant']}): control {g['control_value']}, treatment "
            f"{g['treatment_value']} -> {'BREACHED: ' + str(g['reason']) if g['breached'] else 'ok'}"
        )
    c = r["conclusion"]
    lines += ["", "## Conclusion", "", f"**{c['decision']}** (winner: {c['winner'] or 'none'})", ""]
    lines += [f"- {x}" for x in c["reasons"]]
    if "replay" in r:
        rp = r["replay"]
        lines += [
            "",
            "## Replay protocol",
            "",
            f"- {rp['label']}; {rp['members_replayed']} test-split members, {rp['future_rows_attributed']} "
            f"held-out ratings as outcomes (>= {rp['positive_threshold']} positive, <= "
            f"{rp['negative_threshold']} negative).",
            f"- Model {rp['model']['version']} trained on {rp['history_rows']} history rows "
            f"(train + validation, config of {rp['champion_config_from']}).",
            f"- Not measured: {rp['not_measured']}.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
