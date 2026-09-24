"""Evaluate the platform engine on every available domain (docs/platform.md section 7).

    uv run python scripts/evaluate_domains.py [--movie-months 24] \
        [--domains movie,us-unemployment,cta-ridership]

Per domain: forecast backtests (MAE, RMSE, MASE, 80 % coverage), warning precision / false-positive
rate from leak-free monthly replays, and the flip rate of early_warning_level between consecutive
replays. Movie: the last 24 months of MovieLens. US unemployment: 2006-01..2010-12 and
2019-01..2021-12, plus a rolling replay of the served forecast (v1.2 trio vs + drift / theta).
CTA ridership (docs/SECOND_DOMAIN_CASE_STUDY.md): the domain-aware config and its plain generic
variant on identical terms: forecasts replayed weekly (tuning window 2010-2014, evaluation window
2015-2024), warnings from monthly leak-free replays 2015-01..2024-12 against the seasonal
confirmation rule, flip rates, the COVID replay (daily as-of dates, operator and public lag), the
calm 2008 check and what-if scenarios. Every warning evaluation stores its per-unit rows
(``warnings.rows``) and a month-cluster bootstrap CI of the lift (``warnings.lift_uncertainty``).
Writes experiments/platform-eval-<UTC ts>/{report.json,
REPORT.md, runs/} (gitignored).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core import run_domain
from jev_ml.core.common import clean
from jev_ml.core.evaluation import (
    decision_consistency,
    evaluate_forecasts,
    generic_units,
    monthly_as_ofs,
    run_replays,
    series_outcome_fn,
    warning_replay,
)
from jev_ml.core.forecast import MODELS, TREND_MODELS, rolling_evaluation
from jev_ml.core.scenario import run_scenario
from jev_ml.domains import available, get_adapter
from jev_ml.domains.generic import GenericAdapter, parse_config
from jev_ml.domains.generic.evaluation import (
    anomaly_counts,
    entity_units,
    plain_config,
    seasonal_outcome_fn,
    warned_flip_rate,
)
from jev_ml.paths import ROOT

H = 6  # periods ahead in which a warned adverse condition must materialise


def eval_movie(months: int) -> dict[str, Any]:
    from jev_ml.domains.movie.evaluation import (
        MOVIE_EXCLUDED_KINDS,
        lapse_keep,
        movie_outcome_fn,
        movie_units,
    )

    ad = get_adapter("movie")
    now = datetime(2026, 9, 24, tzinfo=UTC)
    base = run_domain(ad, now=now)
    end = base.data["run"]["last_complete_month"]
    start = (np.datetime64(end[:7]) - np.timedelta64(months - 1, "M")).astype(str)
    as_ofs = monthly_as_ofs(start, end)
    reps = run_replays(lambda a: run_domain(ad, as_of=a, now=now), as_ofs, keep=lapse_keep)
    full = {s.id: s for s in base.series_objects}
    outcome = movie_outcome_fn(base.prepared.ratings, full, base.config, H)
    return {
        "domain": "movie",
        "as_of_default": base.data["run"]["as_of"],
        "forecast": evaluate_forecasts(base.series_objects, base.config),
        "warnings": warning_replay(reps, movie_units, outcome, MOVIE_EXCLUDED_KINDS, keep_rows=True),
        "consistency": decision_consistency(reps),
        "replays": {
            "as_of": [as_ofs[0], as_ofs[-1]],
            "n": len(reps),
            "ms_mean": float(np.mean([r["ms"] for r in reps])),
        },
        "warnings_per_replay": [len(r["result"]["warnings"]) for r in reps],
    }


def eval_generic(key: str, windows: list[tuple[str, str]]) -> dict[str, Any]:
    ad = get_adapter(key)
    now = datetime(2026, 9, 24, tzinfo=UTC)
    base = run_domain(ad, now=now)
    full = {s.id: s for s in base.series_objects}
    outcome = series_outcome_fn(full, H)
    all_reps, per_window = [], []
    for a, b in windows:
        reps = run_replays(lambda x: run_domain(ad, as_of=x, now=now), monthly_as_ofs(a, b))
        all_reps += reps
        per_window.append(
            {
                "window": [a, b],
                "warnings": warning_replay(reps, generic_units, outcome)["overall"],
                "consistency": decision_consistency(reps),
                "warnings_per_replay": [len(r["result"]["warnings"]) for r in reps],
            }
        )
    return {
        "domain": key,
        "as_of_default": base.data["run"]["as_of"],
        "forecast": evaluate_forecasts(base.series_objects, base.config),
        "warnings": warning_replay(
            all_reps,
            generic_units,
            outcome,
            {"entity situations (data_quality, data_staleness)": "input quality, not a future outcome"},
            keep_rows=True,
        ),
        "windows": per_window,
        "consistency_pooled_within_windows": {
            k: sum(w["consistency"][k] for w in per_window)
            for k in ("situation_pairs", "flips", "warning_boundary_flips")
        },
        "replays": {"n": len(all_reps), "ms_mean": float(np.mean([r["ms"] for r in all_reps]))},
    }


NOW = datetime(2026, 9, 24, tzinfo=UTC)
CTA_TUNE = ("2010-01-01", "2014-12-31")
CTA_EVAL = ("2015-01-01", "2024-12-31")


def _cta_raw() -> dict[str, Any]:
    import yaml

    raw: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "domains" / "cta-ridership.yaml").read_text())
    return raw


def _adapter(raw: dict[str, Any]) -> GenericAdapter:
    return GenericAdapter(parse_config(raw, ROOT))


def _forecast_block(
    series: list[Any], cfg: Any, window: tuple[str, str], step: int, **kw: Any
) -> dict[str, Any]:
    per = []
    for s in series:
        if not s.forecast:
            continue
        e = rolling_evaluation(s, cfg, start=window[0], end=window[1], step=step, **kw)
        if e is not None:
            per.append(e)

    def med(k: str) -> float | None:
        v = [p[k] for p in per if p.get(k) is not None]
        return float(np.median(v)) if v else None

    return {
        "window": list(window),
        "per_series": per,
        "summary": {
            "n_series": len(per),
            **{
                f"median_{k}": med(k)
                for k in (
                    "mase",
                    "random_walk_mase",
                    "seasonal_naive_mase",
                    "relative_mae_vs_naive",
                    "relative_mae_vs_seasonal_naive",
                    "coverage80",
                )
            },
            "beats_naive": sum((p["relative_mae_vs_naive"] or 9) < 1 for p in per),
            "beats_seasonal_naive": sum((p["relative_mae_vs_seasonal_naive"] or 9) < 1 for p in per),
        },
    }


def _first_warnings(reps: list[dict[str, Any]]) -> dict[str, str | None]:
    first: dict[str, str | None] = {}
    for r in reps:
        for u in entity_units(r):
            first.setdefault(u["entity"], None)
            if u["warned"] and first[u["entity"]] is None:
                first[u["entity"]] = r["as_of"]
    return first


def _brief(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "as_of": d["run"]["as_of"],
        "last_complete": d["run"]["last_complete_month"],
        "headline": d["summary"]["headline"],
        "counts": d["summary"]["counts"],
        "warnings": [
            {
                "key": w["key"],
                "severity": w["severity"],
                "level": w.get("early_warning_level"),
                "title": w["title"],
            }
            for w in d["warnings"]
        ],
        "trends": [
            {k: t[k] for k in ("series_id", "direction", "q_value", "recent_mean", "prior_mean", "slope")}
            for t in d["trends"]
            if t["direction"] != "flat"
        ],
        "anomalies": [
            {
                k: a[k]
                for k in (
                    "series_id",
                    "detected_at",
                    "kind",
                    "score",
                    "severity",
                    "value",
                    "baseline",
                    "suppressed",
                )
            }
            for a in d["anomalies"]
        ],
        "forecasts": [
            {
                "series_id": f["series_id"],
                "model": f["model"],
                "backtest": f["backtest"],
                "first": f["points"][0],
                "last": f["points"][-1],
            }
            for f in d["predictions"]["forecasts"]
        ],
        "risks": [
            {k: r[k] for k in ("kind", "entity", "level", "score", "title")}
            for r in d["risks"]
            if r["level"] != "low"
        ],
        "ms": d["run"]["stage_ms"]["total"],
    }


def eval_cta(out: Path) -> dict[str, Any]:
    raw = _cta_raw()
    variants = {"domain_aware": raw, "plain_generic": plain_config(raw)}
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"domain": "generic:cta-ridership", "variants": {}}
    aware_base = None
    for name, cfg_raw in variants.items():
        ad = _adapter(cfg_raw)
        base = run_domain(ad, now=NOW)
        if name == "domain_aware":
            aware_base = base
        daily = [s for s in base.series_objects if s.id.startswith("boardings:")]
        kw = {"scale_lag": 7, "benchmark_period": 7}
        fc_tune = _forecast_block(daily, base.config, CTA_TUNE, 7, **kw)
        fc_eval = _forecast_block(daily, base.config, CTA_EVAL, 7, **kw)
        as_ofs = monthly_as_ofs(CTA_EVAL[0], CTA_EVAL[1])
        reps = run_replays(lambda a, ad=ad: run_domain(ad, as_of=a, now=NOW), as_ofs)
        full = {s.id: s for s in base.series_objects}
        outcome = seasonal_outcome_fn(full)
        wr = warning_replay(reps, entity_units, outcome, keep_rows=True)
        by_year = {}
        for y in range(2015, 2025):
            sub = [r for r in reps if r["as_of"].startswith(str(y))]
            by_year[str(y)] = warning_replay(sub, entity_units, outcome)["overall"]
        covid = run_replays(
            lambda a, ad=ad: run_domain(ad, as_of=a, now=NOW),
            [str(d.date()) for d in pd.date_range("2020-02-20", "2020-03-25")],
        )
        calm = run_replays(
            lambda a, ad=ad: run_domain(ad, as_of=a, now=NOW),
            [str(d.date()) for d in pd.date_range("2008-09-01", "2008-10-31")],
        )
        res["variants"][name] = {
            "forecast_tuning_window": fc_tune,
            "forecast_evaluation_window": fc_eval,
            "warnings": wr,
            "warnings_by_year": by_year,
            "consistency": decision_consistency(reps),
            "unit_warned_flips": warned_flip_rate(reps, entity_units),
            "anomalies": anomaly_counts(reps),
            "covid_first_warning": _first_warnings(covid),
            "covid_daily": [
                {"as_of": r["as_of"], "warned": sorted(u["entity"] for u in entity_units(r) if u["warned"])}
                for r in covid
            ],
            "calm_2008": {
                "replays": len(calm),
                "warned_replays": sum(any(u["warned"] for u in entity_units(r)) for r in calm),
                "warnings": sum(len(r["result"]["warnings"]) for r in calm),
                "anomalies": anomaly_counts(calm),
                "warned_units": sorted(
                    {(r["as_of"], u["entity"]) for r in calm for u in entity_units(r) if u["warned"]}
                )[:40],
            },
            "replays": {"n": len(reps), "ms_mean": float(np.mean([r["ms"] for r in reps]))},
        }
    # real runs of the served (domain-aware) config
    assert aware_base is not None
    ad = _adapter(raw)
    case = {}
    for tag, a in (
        ("default", None),
        ("2020-03-20", "2020-03-20"),
        ("2008-10-01", "2008-10-01"),
        ("2021-07-01", "2021-07-01"),
    ):
        r = aware_base if a is None else run_domain(ad, as_of=a, now=NOW)
        d = r.to_dict()
        (runs_dir / f"cta_{tag}.json").write_text(json.dumps(d, indent=1, allow_nan=False))
        case[tag] = _brief(d)
        if tag in ("default", "2021-07-01"):
            spec = {
                "series_id": "boardings:total",
                "horizon_months": 14,
                "scenarios": [
                    {"name": "Recovery continues", "kind": "continue"},
                    {"name": "Recovery stalls", "kind": "custom", "trend_multiplier": 0.0},
                    {"name": "Recovery reverses", "kind": "reverse"},
                    {
                        "name": "Shock -15 % from day 8",
                        "kind": "shock",
                        "level_shift_pct": -15,
                        "shock_month": 8,
                    },
                ],
            }
            case[tag]["scenario"] = run_scenario(r, spec)
    # the COVID replay under the public portal's publication lag (~45 days)
    pub = json.loads(json.dumps(raw))
    pub["source"]["availability_lag_days"] = 45
    pad = _adapter(pub)
    pub_reps = run_replays(
        lambda a: run_domain(pad, as_of=a, now=NOW),
        [str(d.date()) for d in pd.date_range("2020-03-01", "2020-05-15", freq="7D")],
    )
    res["covid_public_lag_45d"] = {
        "first_warning": _first_warnings(pub_reps),
        "last_complete_by_as_of": {r["as_of"]: r["result"]["run"]["last_complete_month"] for r in pub_reps},
    }
    res["case_runs"] = case
    res["protocol"] = {
        "forecast": "rolling replay of the served procedure every 7th day; horizon 14 days; MASE "
        "scaled by the in-sample lag-7 naive MAE for both configs; relative MAE = model MAE / "
        "benchmark MAE on the same points",
        "warnings": "monthly as-of replays on the 1st, 2015-01..2024-12; unit = (replay, entity); "
        "confirmed = sustained adverse move of the weekday-aligned yoy of the 7-day total beyond "
        "2 sigma sqrt(k), k <= 4 weeks",
        "tuning_window": list(CTA_TUNE),
        "evaluation_window": list(CTA_EVAL),
    }
    return res


def eval_unemployment_forecasts() -> dict[str, Any]:
    ad = get_adapter("generic:us-unemployment")
    base = run_domain(ad, now=NOW)
    out = {}
    for name, models in (("v1.2 trio", MODELS), ("+ drift, theta", MODELS + TREND_MODELS)):
        out[name] = _forecast_block(
            base.series_objects, base.config, ("2005-01-01", "2026-12-31"), 1, models=models
        )
    return {"domain": "generic:us-unemployment", "rolling_forecast": out}


def markdown(rep: dict[str, Any]) -> str:
    lines = ["# JEV platform evaluation", "", f"created {rep['created_at']}; horizon h = {H} periods", ""]
    for d in rep.get("extra", []):
        lines += [
            f"## {d['domain']} (seasonal / variant evaluation)",
            "",
            "```json",
            json.dumps(_md_view(d), indent=1),
            "```",
            "",
        ]
    for d in rep["domains"]:
        f = d["forecast"]["summary"]
        w = d["warnings"]["overall"]
        c = d.get("consistency") or {}
        lines += [
            f"## {d['domain']}",
            "",
            f"- forecast ({f['n_series']} series): median MAE {f['median_mae']}, RMSE {f['median_rmse']}, "
            f"MASE {f['median_mase']} (naive {f['median_naive_mase']}), "
            f"beats naive {f['share_beating_naive']}, "
            f"coverage80 {f['mean_coverage80']}",
            f"- warnings (all kinds scored): {json.dumps(w)}",
        ]
        if d["warnings"].get("lift_uncertainty"):
            lines.append(f"- lift uncertainty: {json.dumps(d['warnings']['lift_uncertainty'])}")
        for k, v in d["warnings"]["per_kind"].items():
            lines.append(f"  - {k}: {json.dumps(v)}")
        for k, v in d["warnings"]["excluded_kinds"].items():
            lines.append(f"  - excluded {k}: {v}")
        if c:
            lines.append(f"- consistency: {json.dumps(c)}")
        for win in d.get("windows", []):
            lines.append(
                f"- window {win['window']}: warnings {json.dumps(win['warnings'])}; "
                f"consistency {json.dumps(win['consistency'])}"
            )
        lines.append("")
    return "\n".join(lines)


def _md_view(d: dict[str, Any]) -> dict[str, Any]:
    if "variants" not in d:
        return {k: v["summary"] for k, v in d["rolling_forecast"].items()}
    out: dict[str, Any] = {}
    for name, v in d["variants"].items():
        out[name] = {
            "forecast_tuning": v["forecast_tuning_window"]["summary"],
            "forecast_eval": v["forecast_evaluation_window"]["summary"],
            "warnings": v["warnings"]["overall"],
            "lift_uncertainty": v["warnings"].get("lift_uncertainty"),
            "by_year": {
                y: {k: w[k] for k in ("warned", "tp", "fp", "fn", "tn", "precision", "recall")}
                for y, w in v["warnings_by_year"].items()
            },
            "flip": v["consistency"],
            "unit_flips": v["unit_warned_flips"],
            "anomalies": v["anomalies"],
            "covid_first_warning": v["covid_first_warning"],
            "calm_2008": {
                k: v["calm_2008"][k] for k in ("replays", "warned_replays", "warnings", "anomalies")
            },
        }
    out["covid_public_lag_45d"] = d["covid_public_lag_45d"]["first_warning"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--movie-months", type=int, default=24)
    p.add_argument("--domains", default="movie,us-unemployment,cta-ridership")
    args = p.parse_args()
    want = set(args.domains.split(","))
    avail = {d["key"]: d for d in available()}
    doms, extra = [], []
    out = ROOT / "experiments" / f"platform-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    if "movie" in want and avail["movie"]["available"]:
        doms.append(eval_movie(args.movie_months))
    if "cta-ridership" in want and avail.get("generic:cta-ridership", {}).get("available"):
        extra.append(eval_cta(out))
    if "us-unemployment" in want and avail.get("generic:us-unemployment", {}).get("available"):
        extra.append(eval_unemployment_forecasts())
        doms.append(
            eval_generic(
                "generic:us-unemployment", [("2006-01-01", "2010-12-01"), ("2019-01-01", "2021-12-01")]
            )
        )
    rep = clean(
        {
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon": H,
            "domains": doms,
            "extra": extra,
        }
    )
    (out / "report.json").write_text(json.dumps(rep, indent=2, allow_nan=False))
    (out / "REPORT.md").write_text(markdown(rep))
    print(markdown(rep))
    print(f"written to {Path(out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
