"""The movie domain adapter: MovieLens + app events + the recommender's artefacts.

``MovieAdapter(inputs=None)`` loads ``inputs`` (a ``PipelineInputs``) or, when None, the real files
(``load_default_inputs``). ``load`` validates the frames with the movie checks and returns the
ratings as generic observations with the movie series specs; ``extra`` adds the movie stages:
rater anomalies and live feedback, the lapse model, model governance (bootstrap over per-user NDCG),
the movie risks, the three movie decision batches (model governance, genre programming with slot
shares, audience), movie signals, and the decision -> action mapping. It wraps the existing
recommender code and intelligence modules; nothing is copied.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from jev_ml.core.adapter import CoreContext, DomainData, DomainExtras, DomainInfo
from jev_ml.core.config import CoreConfig
from jev_ml.domains.movie import risks as mrisk
from jev_ml.domains.movie.actions import decision_actions
from jev_ml.domains.movie.config import PIPELINE_VERSION, IntelConfig
from jev_ml.domains.movie.decisions import build_decision_batches, share_window_inputs
from jev_ml.domains.movie.ingest import (
    MOVIELENS_LICENSE,
    MOVIELENS_URL,
    PipelineInputs,
    Prepared,
    load_default_inputs,
    prepare,
)
from jev_ml.domains.movie.lapse import run_lapse
from jev_ml.domains.movie.modelstats import bootstrap_models
from jev_ml.domains.movie.raters import anomaly_warning_text, detect_raters, live_feedback
from jev_ml.domains.movie.series import movie_observations, movie_specs
from jev_ml.domains.movie.signals import movie_signals

INFO = DomainInfo(
    key="movie",
    name="Movies",
    description=(
        "MovieLens ratings, live app activity and the hybrid recommender's evaluation: genre demand, "
        "audience lapse, rater manipulation, model governance and editorial programming."
    ),
    entity_types=["platform", "genre", "user", "model", "source"],
    frequency="month",
    sources=[
        {
            "source": "movielens",
            "kind": "static_snapshot",
            "license": MOVIELENS_LICENSE,
            "url": MOVIELENS_URL,
        },
        {"source": "app", "kind": "live", "license": None, "url": None},
        {"source": "model", "kind": "artefact", "license": None, "url": None},
    ],
    capabilities={
        "recommendation": True,
        "user_intelligence": importlib.util.find_spec("jev_ml.domains.movie.user_intel") is not None,
        "lapse": True,
        "raters": True,
        "model_governance": True,
        "scenarios": True,
    },
)


class MovieAdapter:
    key = "movie"
    info = INFO
    pipeline_version = PIPELINE_VERSION

    def __init__(
        self,
        inputs: PipelineInputs | None = None,
        processed_dir: Path | None = None,
        models_dir: Path | None = None,
        experiments_dir: Path | None = None,
    ) -> None:
        self.inputs = inputs
        self.dirs = {
            "processed_dir": processed_dir,
            "models_dir": models_dir,
            "experiments_dir": experiments_dir,
        }

    def default_config(self) -> IntelConfig:
        return IntelConfig()

    def _inputs(self, as_of: datetime | None, now: datetime | None) -> PipelineInputs:
        inputs = (
            self.inputs
            if self.inputs is not None
            else load_default_inputs(
                processed_dir=self.dirs["processed_dir"],
                models_dir=self.dirs["models_dir"],
                experiments_dir=self.dirs["experiments_dir"],
            )
        )
        return replace(
            inputs,
            as_of=as_of if as_of is not None else inputs.as_of,
            now=now if now is not None else inputs.now,
        )

    def load(self, as_of: datetime | None, now: datetime | None, config: CoreConfig) -> DomainData:
        cfg = config if isinstance(config, IntelConfig) else IntelConfig()
        prep = prepare(self._inputs(as_of, now), cfg)
        return DomainData(
            observations=movie_observations(prep),
            specs=movie_specs(prep, cfg),
            as_of=prep.as_of,
            now=prep.now,
            sources=prep.sources,
            data_version=prep.data_version,
            quality_checks=list(prep.quality["checks"]),
            core_checks=False,
            core_checks_reason=(
                "the movie ingest validates the MovieLens and app frames (movielens_*/app_* checks)"
            ),
            excluded_after_as_of=prep.excluded_after_as_of,
            frequency="M",
            grid_end="as_of",
            model_version=prep.model_version,
            diagnostics=list(prep.diagnostics),
            context=prep,
            share_forecast_min_share=cfg.genre_min_share,
        )

    def quality_checks(self, data: DomainData) -> list[dict[str, Any]]:
        return []  # every movie check already ran in ``prepare`` (DomainData.quality_checks)

    def extra(self, ctx: CoreContext) -> DomainExtras:
        prep: Prepared = ctx.data.context
        cfg = ctx.config if isinstance(ctx.config, IntelConfig) else IntelConfig()
        key, sup, dq = ctx.as_of_key, ctx.suppressed, float(ctx.quality["score"])
        inputs = prep.inputs
        diags: list[dict[str, Any]] = []
        with ctx.timer.stage("anomalies"):
            rater_anoms, rater_feats = detect_raters(prep, cfg, key, sup)
            live_anoms, live = live_feedback(prep, cfg, key, sup)
            if live.get("status") != "ok":
                diags.append({"stage": "live_feedback", "status": "skipped", "reason": live.get("reason")})
        with ctx.timer.stage("lapse"):
            lapse, scored = run_lapse(prep.ratings, prep.as_of_ts, cfg, prep.data_version)
        if lapse["status"] != "ok":
            diags.append({"stage": "lapse", "status": "insufficient_data", "reason": lapse["detail"]})
        manifest = inputs.model_manifest if inputs is not None else None
        per_user = inputs.per_user_ndcg if inputs is not None else None
        with ctx.timer.stage("risk"):
            boot = bootstrap_models(per_user, cfg) if prep.model_replayable else None
            fc_by = {f["series_id"]: f for f in ctx.forecasts}
            risks = mrisk.genre_risks(ctx.trends, fc_by, dq, cfg, key)
            risks += mrisk.lapse_risk(lapse, scored, prep, dq, cfg, key)
            manip, infl = mrisk.manipulation_risk(rater_anoms, rater_feats, prep, dq, cfg, key)
            risks += manip
            mrisks, stale = mrisk.model_risks(prep, manifest, boot, dq, cfg, key)
            risks += mrisks
            risks += mrisk.rejection_risk(live, dq, cfg, key)
        with ctx.timer.stage("decide"):
            decisions, batches = build_decision_batches(
                manifest=manifest,
                stale=stale,
                boot=boot,
                data_version=prep.data_version,
                replayable=prep.model_replayable,
                trends=ctx.trends,
                forecasts_by_series=fc_by,
                share_windows=share_window_inputs(ctx.share_forecasts, ctx.forecast_states, cfg),
                rater_anoms=rater_anoms,
                influence=infl,
                lapse=lapse,
                lapse_scored=scored,
                cfg=cfg,
                as_of_key=key,
            )
        with ctx.timer.stage("explain"):
            signals = movie_signals(prep, live, stale, boot, cfg, key)

        def actions(plan: Any, ds: list[dict[str, Any]]) -> None:
            decision_actions(plan, ds, manifest)

        return DomainExtras(
            anomalies=rater_anoms[: cfg.top_n] + live_anoms,
            predictions={"lapse": lapse},
            risks=risks,
            decisions=decisions,
            decision_batches=batches,
            signals=signals,
            data={"live": dict(live)},
            diagnostics=diags,
            decision_actions=actions,
            anomaly_text=anomaly_warning_text,
        )


def available(processed_dir: Path | None = None) -> tuple[bool, str | None]:
    from jev_ml import paths

    p = (processed_dir or paths.PROCESSED_DIR) / "interactions.csv"
    if p.exists():
        return True, None
    return (
        False,
        f"no processed MovieLens data at {p} (run scripts/download_data.py and scripts/preprocess_data.py)",
    )
