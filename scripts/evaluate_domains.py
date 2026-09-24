"""Evaluate the platform engine on every available domain (docs/platform.md section 7).

    uv run python scripts/evaluate_domains.py [--movie-months 24] [--quick]

Per domain: forecast backtests (MAE, RMSE, MASE, 80 % coverage), warning precision / false-positive
rate from leak-free monthly replays, and the flip rate of early_warning_level between consecutive
replays. Movie: the last 24 months of MovieLens. US unemployment: 2006-01..2010-12 and
2019-01..2021-12. Writes experiments/platform-eval-<UTC ts>/{report.json, REPORT.md} (gitignored).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
from jev_ml.domains import available, get_adapter
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
        "warnings": warning_replay(reps, movie_units, outcome, MOVIE_EXCLUDED_KINDS),
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
        ),
        "windows": per_window,
        "consistency_pooled_within_windows": {
            k: sum(w["consistency"][k] for w in per_window)
            for k in ("situation_pairs", "flips", "warning_boundary_flips")
        },
        "replays": {"n": len(all_reps), "ms_mean": float(np.mean([r["ms"] for r in all_reps]))},
    }


def markdown(rep: dict[str, Any]) -> str:
    lines = ["# JEV platform evaluation", "", f"created {rep['created_at']}; horizon h = {H} periods", ""]
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--movie-months", type=int, default=24)
    args = p.parse_args()
    avail = {d["key"]: d for d in available()}
    doms = []
    if avail["movie"]["available"]:
        doms.append(eval_movie(args.movie_months))
    if avail.get("generic:us-unemployment", {}).get("available"):
        doms.append(
            eval_generic(
                "generic:us-unemployment", [("2006-01-01", "2010-12-01"), ("2019-01-01", "2021-12-01")]
            )
        )
    rep = clean(
        {"created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "horizon": H, "domains": doms}
    )
    out = ROOT / "experiments" / f"platform-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, allow_nan=False))
    (out / "REPORT.md").write_text(markdown(rep))
    print(markdown(rep))
    print(f"written to {Path(out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
