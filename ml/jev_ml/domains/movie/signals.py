"""EXPLAIN (movie domain): live-feedback and model signals (the core builds the others).

Strength: negative-feedback test -> 1 - p (confidence kind ``evidence``); new data since training ->
new-data share / retrain threshold, capped at 1 (exact counts: ``rule``); hybrid lead -> bootstrap
P(hybrid better) (``probability``).
"""

from __future__ import annotations

from typing import Any

from jev_ml.core.common import evidence, fnum, iso, iso_from_epoch
from jev_ml.core.signals import make_signal
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import Prepared


def movie_signals(
    prep: Prepared,
    live: dict[str, Any],
    stale: dict[str, Any] | None,
    boot: dict[str, Any] | None,
    cfg: IntelConfig,
    as_of_key: str,
) -> list[dict[str, Any]]:
    sigs: list[dict[str, Any]] = []
    if live.get("status") == "ok":
        d = live["recent_rate"] - live["prior_rate"]
        sigs.append(
            make_signal(
                "live",
                "live:negative_feedback",
                "platform",
                "app",
                "Negative recommendation feedback (7 d)",
                fnum(live["recent_rate"], 4),
                "share",
                1 - live["p_value"],
                "up" if d > 0 else "down" if d < 0 else "flat",
                "app",
                iso_from_epoch(prep.event_clock_ts) if prep.event_clock_ts is not None else iso(prep.now),
                f"last {cfg.live_recent_days} d vs prior {cfg.live_prior_days} d",
                0.0,
                [evidence("test", "two-proportion z", fnum(live["z"], 3), f"p {live['p_value']:.4f}")],
                as_of_key,
                baseline=fnum(live["prior_rate"], 4),
                change=fnum(d, 4),
                confidence=1 - live["p_value"],
                confidence_kind="evidence",
            )
        )
    if stale is not None:
        ratio = stale["new_frac"] / cfg.retrain_new_event_frac
        sigs.append(
            make_signal(
                "model",
                "model:new_data",
                "model",
                str(prep.model_version),
                f"{stale['new_events']['total']} ratings since model training",
                stale["new_events"]["total"],
                "ratings",
                ratio,
                "up" if stale["new_events"]["total"] else "flat",
                "model",
                iso(prep.as_of),
                None,
                0.0,
                [
                    evidence(
                        "metric",
                        "new-event share",
                        fnum(stale["new_frac"], 6),
                        f"threshold {cfg.retrain_new_event_frac}",
                    )
                ],
                as_of_key,
                baseline=stale["trained_on_rows"],
                change=stale["new_events"]["total"],
                confidence=1.0,
                confidence_kind="rule",
            )
        )
    if boot and not boot.get("insufficient") and boot.get("hybrid") and prep.model_replayable:
        h = boot["hybrid"]
        sigs.append(
            make_signal(
                "model",
                "model:hybrid_lead",
                "model",
                str(prep.model_version),
                f"Hybrid leads {h['best_single']} by {100 * h['lead']:.1f} % NDCG@10",
                fnum(h["lead"], 4),
                "relative NDCG@10",
                h["p_hybrid_better"],
                "up" if h["lead"] > 0 else "down",
                "model",
                iso(prep.as_of),
                None,
                0.0,
                [
                    evidence(
                        "test",
                        "bootstrap P(hybrid better)",
                        fnum(h["p_hybrid_better"], 4),
                        f"95 % CI of lead {[round(x, 4) for x in h['lead_ci95']]}",
                    )
                ],
                as_of_key,
                baseline=fnum(boot["mean_ndcg"].get(h["best_single"]), 5),
                change=fnum(h["lead"], 4),
                confidence=h["p_hybrid_better"],
                confidence_kind="probability",
            )
        )
    return sigs
