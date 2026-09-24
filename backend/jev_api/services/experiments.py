"""Online A/B experiments on the /recommendations surface (docs/EXPERIMENTATION.md).

Everything lives in the application database; no experimentation service.

* Lifecycle: draft -> running <-> paused -> stopped -> concluded. Invalid transitions raise
  ``ExperimentConflictError`` (409). A started experiment's config is immutable; only the traffic share can
  be ramped. At most one running-or-paused experiment per surface (code check + partial unique index).
* Assignment: ``bucket = sha256(salt ":" user_id) mod 10000`` decides enrolment (bucket < traffic x 100)
  and an independent ``sha256(salt ":variant:" user_id)`` walks the variant weights. The assignment is
  stored at the member's first enrolled request and reused afterwards (sticky), so a ramp up only adds
  members and a ramp down only stops new enrolment. Admins and test accounts are never enrolled.
* Exposure log: one ``ab_exposures`` row per served list, cache hits included, inside or outside an
  experiment (gap P1 #6).
* Outcomes: interactions of enrolled members within ``attribution_window_hours`` of an exposure,
  attributed to their latest exposure containing the item (else their latest exposure in the window).
* Member decisions: every served recommendation_strategy decision is persisted (gap P1 #7).
* Results: per-variant metrics and statistics (``experiment_stats``); a winner only when the primary
  metric is significant and no guardrail is breached, else "inconclusive".
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import logging
import math
import secrets
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from jev_api.config import Settings, get_settings
from jev_api.metrics import metrics
from jev_api.models import Favorite, Movie, MovieGenre, Rating, RecommendationFeedback, User, WatchHistory
from jev_api.models.catalog import Genre
from jev_api.models.experiments import (
    AbAssignment,
    AbExperiment,
    AbExposure,
    AbOutcome,
    AbVariant,
    MemberDecision,
)
from jev_api.services import audit
from jev_api.services import experiment_stats as st
from jev_ml.engine import RecommendationEngine
from jev_ml.models.hybrid import HybridConfig
from jev_ml.registry import read_registry
from jev_ml.signals import LIKE_THRESHOLD

log = logging.getLogger(__name__)

SURFACE_RECOMMENDATIONS = "recommendations"
BUCKETS = 10_000
LIVE, REPLAY = "live", "offline_replay"
REPLAY_LABEL = "offline replay using held-out ratings, not live traffic"
LIVE_LABEL = "live traffic"
NEGATIVE_RATING = 2.0  # a rating at or below this is a negative outcome (guardrail)

# action -> (allowed from-states, to-state, audit action)
TRANSITIONS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "start": (("draft", "paused"), "running", "experiment.start"),
    "pause": (("running",), "paused", "experiment.pause"),
    "stop": (("running", "paused"), "stopped", "experiment.stop"),
    "conclude": (("stopped",), "concluded", "experiment.conclude"),
}
RAMP_STATES = ("draft", "running", "paused")

RATE_METRICS = ("interaction_rate", "positive_rate", "rating_rate", "feedback_rate", "negative_rate")
MEAN_METRICS = ("ndcg_at_10", "diversity", "novelty")
INTERACTION_KINDS = frozenset({"click", "like", "rating", "watch", "favorite"})
FEEDBACK_KINDS = frozenset({"like", "dislike", "not_interested"})
_HYBRID_FIELDS = {f.name for f in dataclasses.fields(HybridConfig)}
_UNIT_INTERVAL = {"diversity_lambda", "content_support_damping"}


class ExperimentError(Exception):
    """A request the experiment cannot honour (422)."""


class ExperimentConflictError(Exception):
    """An invalid lifecycle transition or a surface already taken (409)."""


class ExperimentNotFoundError(Exception):
    pass


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else _aware(dt).isoformat()


# --- assignment -------------------------------------------------------------------------------------
def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def bucket_of(salt: str, user_id: int) -> int:
    """Enrolment bucket 0..9999: sha256(salt ":" user_id). Stable across workers and restarts."""
    return _hash_int(f"{salt}:{user_id}") % BUCKETS


def variant_index(salt: str, user_id: int, weights: list[float]) -> int:
    """The variant a member falls into, by an independent hash (so enrolment and the variant split are
    uncorrelated: ramping the traffic share never moves anyone between variants)."""
    u = _hash_int(f"{salt}:variant:{user_id}") / 2.0**64
    total = float(sum(weights))
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w / total
        if u < acc:
            return i
    return len(weights) - 1


def enrolled(salt: str, user_id: int, traffic_percent: float) -> bool:
    return bucket_of(salt, user_id) < round(traffic_percent * BUCKETS / 100.0)


def excluded(user: User, settings: Settings | None = None) -> bool:
    """Admins and test accounts are never enrolled (contamination control, configurable)."""
    s = settings or get_settings()
    if s.experiments_exclude_admins and user.is_admin:
        return True
    email = (user.email or "").lower()
    patterns = [p.strip().lower() for p in s.experiments_excluded_email_patterns.split(",") if p.strip()]
    return any(fnmatch.fnmatchcase(email, p) for p in patterns)


@dataclass(frozen=True)
class Arm:
    """The variant a request is served under."""

    experiment_id: int
    key: str
    variant: str
    config: dict[str, Any]

    @property
    def tag(self) -> str:  # part of the recommendation cache key
        return f"x{self.experiment_id}.{self.variant}"

    @property
    def model_version(self) -> str | None:
        v = self.config.get("model_version")
        return str(v) if v else None

    @property
    def strategy_decision(self) -> bool:
        return bool(self.config.get("strategy_decision", True))

    @property
    def hybrid_overrides(self) -> dict[str, Any]:
        return dict(self.config.get("hybrid_overrides") or {})

    @property
    def recency_half_life_days(self) -> float | None:
        v = self.config.get("recency_half_life_days")
        return float(v) if v else None


def running_experiment(db: Session, surface: str) -> AbExperiment | None:
    return db.scalar(
        select(AbExperiment).where(AbExperiment.surface == surface, AbExperiment.status == "running")
    )


def arm_for(db: Session, user: User, surface: str = SURFACE_RECOMMENDATIONS) -> Arm | None:
    """The member's variant in the surface's running experiment, or None (not running, excluded, not
    enrolled). Writes the sticky assignment on first enrolment."""
    settings = get_settings()
    if not settings.experiments_enabled:
        return None
    exp = running_experiment(db, surface)
    if exp is None or excluded(user, settings):
        return None
    variants = list(exp.variants)
    by_name = {v.name: v for v in variants}
    row = db.scalar(
        select(AbAssignment).where(AbAssignment.experiment_id == exp.id, AbAssignment.user_id == user.id)
    )
    if row is None:
        if not enrolled(exp.salt, user.id, exp.traffic_percent):
            return None
        v = variants[variant_index(exp.salt, user.id, [x.weight for x in variants])]
        row = AbAssignment(
            experiment_id=exp.id,
            user_id=user.id,
            variant_id=v.id,
            variant=v.name,
            bucket=bucket_of(exp.salt, user.id),
        )
        db.add(row)
        try:
            db.commit()
            metrics.inc("experiments.assignments", f"{exp.key}:{v.name}")
        except IntegrityError:  # a concurrent request assigned first: use its row (the same variant)
            db.rollback()
            row = db.scalar(
                select(AbAssignment).where(
                    AbAssignment.experiment_id == exp.id, AbAssignment.user_id == user.id
                )
            )
            if row is None:
                return None
    variant = by_name.get(row.variant)
    if variant is None:
        return None
    return Arm(exp.id, exp.key, variant.name, dict(variant.config or {}))


# --- challenger engines -----------------------------------------------------------------------------
_challengers: OrderedDict[str, RecommendationEngine] = OrderedDict()
_challenger_lock = threading.Lock()


def serving_engine(arm: Arm | None, champion: RecommendationEngine) -> RecommendationEngine:
    """The engine a variant is served with: the champion, or a registered challenger version loaded
    lazily through the normal loader (``RecommendationEngine(models_dir / version)``) and kept in a
    small LRU (``experiments_max_challenger_models``). Each loaded challenger holds one full engine in
    memory (the MovieLens model is ~12 MB on disk, a few tens of MB resident). Raises on failure."""
    version = arm.model_version if arm is not None else None
    if not version or version == champion.version:
        return champion
    settings = get_settings()
    with _challenger_lock:
        eng = _challengers.get(version)
        if eng is not None:
            _challengers.move_to_end(version)
            return eng
        if settings.experiments_max_challenger_models <= 0:
            raise RuntimeError("challenger models are disabled (JEV_EXPERIMENTS_MAX_CHALLENGER_MODELS=0)")
        eng = RecommendationEngine(settings.models_dir / version)
        _challengers[version] = eng
        while len(_challengers) > settings.experiments_max_challenger_models:
            _challengers.popitem(last=False)
        metrics.inc("experiments.challenger_loads", version)
        log.info("loaded challenger model %s for an experiment", version)
        return eng


def loaded_challengers() -> list[str]:
    with _challenger_lock:
        return list(_challengers)


# --- exposure log -----------------------------------------------------------------------------------
def log_exposure(
    db: Session,
    *,
    user: User,
    request_id: str,
    items: list[dict[str, Any]],
    model_version: str,
    arm: Arm | None,
    cached: bool,
    source_request_id: str | None,
    decision_id: str | None,
    latency_ms: float | None,
    context: str,
    surface: str = SURFACE_RECOMMENDATIONS,
    served_at: datetime | None = None,
) -> AbExposure | None:
    """One row per served list. Commits. Never raises: a failure is logged and counted, the request is
    served regardless."""
    if not get_settings().experiments_log_exposures:
        return None
    row = AbExposure(
        request_id=request_id,
        source_request_id=source_request_id,
        user_id=user.id,
        surface=surface,
        context=context,
        experiment_id=arm.experiment_id if arm else None,
        variant=arm.variant if arm else None,
        cached=cached,
        model_version=model_version,
        decision_id=decision_id,
        items=[
            {
                "movie_id": int(i["movie_id"]),
                "rank": int(i["rank"]),
                "recommendation_id": i.get("recommendation_id"),
            }
            for i in items
        ],
        n_items=len(items),
        latency_ms=None if latency_ms is None else round(latency_ms, 3),
        served_at=served_at or datetime.now(UTC),
    )
    try:
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()
        metrics.inc("experiments.errors", "exposure")
        log.exception("exposure logging failed", extra={"extra_fields": {"user": user.id}})
        return None
    metrics.inc("experiments.exposures", "cache_hit" if cached else "generated")
    if arm is not None:
        metrics.inc("experiments.exposures_by_variant", f"{arm.key}:{arm.variant}")
    return row


# --- member decisions (P1 #7) -----------------------------------------------------------------------
def persist_decision(db: Session, user: User, dec: dict[str, Any], model_version: str, source: str) -> None:
    """Store a served strategy decision once per (decision_id, state_hash). Commits. Never raises."""
    try:
        state = dict(dec.get("state") or {})
        decision_id = str(dec["id"])
        state_hash = str(
            state.get("state_hash") or hashlib.sha256(repr(sorted(state.items())).encode()).hexdigest()[:16]
        )
        exists = db.scalar(
            select(MemberDecision.id).where(
                MemberDecision.decision_id == decision_id, MemberDecision.state_hash == state_hash
            )
        )
        if exists is not None:
            return
        db.add(
            MemberDecision(
                decision_id=decision_id,
                user_id=user.id,
                spec_id=str(dec.get("spec_id") or "recommendation_strategy"),
                policy_version=dec.get("policy_version"),
                answer=dec.get("answer"),
                served_strategy=str(state.get("served_strategy") or "standard"),
                confidence=dec.get("confidence"),
                confidence_kind=str(dec.get("confidence_kind") or "margin"),
                abstained=bool(dec.get("abstained")),
                state_hash=state_hash,
                state=state,
                rationale=list(dec.get("rationale") or []),
                evidence=[e for e in (dec.get("evidence") or []) if isinstance(e, dict)],
                model_version=model_version,
                source=source,
            )
        )
        db.commit()
        metrics.inc("member_decisions", "persisted")
    except IntegrityError:  # the same decision state written concurrently
        db.rollback()
    except Exception:
        db.rollback()
        metrics.inc("member_decisions", "errors")
        log.exception("persisting the member decision failed", extra={"extra_fields": {"user": user.id}})


def resolve_decision(db: Session, decision_id: str, user_id: int | None = None) -> MemberDecision | None:
    q = select(MemberDecision).where(MemberDecision.decision_id == decision_id)
    if user_id is not None:
        q = q.where(MemberDecision.user_id == user_id)
    return db.scalar(q.order_by(MemberDecision.created_at.desc(), MemberDecision.id.desc()).limit(1))


def decision_out(row: MemberDecision) -> dict[str, Any]:
    return {
        "decision_id": row.decision_id,
        "spec_id": row.spec_id,
        "policy_version": row.policy_version,
        "answer": row.answer,
        "served_strategy": row.served_strategy,
        "confidence": row.confidence,
        "confidence_kind": row.confidence_kind,
        "abstained": row.abstained,
        "state_hash": row.state_hash,
        "state": row.state,
        "rationale": row.rationale,
        "evidence": row.evidence,
        "model_version": row.model_version,
        "source": row.source,
        "created_at": _iso(row.created_at),
    }


# --- definition, validation and lifecycle ------------------------------------------------------------
def get_experiment(db: Session, key: str) -> AbExperiment:
    exp = db.scalar(select(AbExperiment).where(AbExperiment.key == key))
    if exp is None:
        raise ExperimentNotFoundError(key)
    return exp


def _validate_variants(variants: list[dict[str, Any]], settings: Settings) -> None:
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ExperimentError("variant names must be unique")
    if names and "list" in names:
        raise ExperimentError("'list' is a reserved name")
    if sum(1 for v in variants if v.get("is_control")) != 1:
        raise ExperimentError("exactly one variant must be the control (is_control: true)")
    registered: set[str] | None = None
    for v in variants:
        cfg = v.get("config") or {}
        bad = set(cfg.get("hybrid_overrides") or {}) - _HYBRID_FIELDS
        if bad:
            raise ExperimentError(
                f"variant {v['name']}: unknown hybrid_overrides {sorted(bad)} (HybridConfig fields: "
                f"{sorted(_HYBRID_FIELDS)})"
            )
        for key, value in (cfg.get("hybrid_overrides") or {}).items():
            if key in _UNIT_INTERVAL and (
                not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0
            ):
                raise ExperimentError(f"variant {v['name']}: {key} must be a number in [0, 1]")
            if key == "weights" and not (
                isinstance(value, dict) and all(isinstance(x, int | float) and x >= 0 for x in value.values())
            ):
                raise ExperimentError(f"variant {v['name']}: weights must map signal -> non-negative number")
        try:
            HybridConfig.from_dict(cfg.get("hybrid_overrides") or {})
        except (TypeError, ValueError) as exc:
            raise ExperimentError(f"variant {v['name']}: invalid hybrid_overrides ({exc})") from exc
        version = cfg.get("model_version")
        if version:
            if registered is None:
                registered = set(read_registry(settings.models_dir).get("versions") or [])
            if version not in registered or not (settings.models_dir / version / "manifest.json").exists():
                raise ExperimentError(f"variant {v['name']}: model_version {version!r} is not registered")
    if sum(1 for v in variants if (v.get("config") or {}).get("model_version")) and (
        settings.experiments_max_challenger_models <= 0
    ):
        raise ExperimentError("challenger models are disabled (JEV_EXPERIMENTS_MAX_CHALLENGER_MODELS=0)")


def _set_variants(exp: AbExperiment, variants: list[dict[str, Any]]) -> None:
    exp.variants.clear()
    for i, v in enumerate(variants):
        exp.variants.append(
            AbVariant(
                name=v["name"],
                position=i,
                is_control=bool(v.get("is_control")),
                weight=float(v.get("weight", 1.0)),
                description=str(v.get("description") or ""),
                config=dict(v.get("config") or {}),
            )
        )


def create_experiment(
    db: Session, body: dict[str, Any], actor: User | None, data_source: str = LIVE
) -> AbExperiment:
    settings = get_settings()
    if db.scalar(select(AbExperiment.id).where(AbExperiment.key == body["key"])) is not None:
        raise ExperimentConflictError(f"experiment {body['key']!r} already exists")
    _validate_variants(body["variants"], settings)
    exp = AbExperiment(
        key=body["key"],
        name=body["name"],
        surface=body.get("surface") or SURFACE_RECOMMENDATIONS,
        status="draft",
        hypothesis=body.get("hypothesis") or "",
        primary_metric=body.get("primary_metric") or "interaction_rate",
        guardrails=list(body.get("guardrails") or []),
        traffic_percent=float(body.get("traffic_percent", 100.0)),
        salt=secrets.token_hex(16),
        attribution_window_hours=float(
            body.get("attribution_window_hours") or settings.experiments_default_attribution_hours
        ),
        analysis={**dict(body.get("analysis") or {}), "data_source": data_source},
        created_by=actor.id if actor is not None else None,
    )
    _set_variants(exp, body["variants"])
    db.add(exp)
    db.flush()
    audit.record(db, "experiment.create", actor, "ab_experiment", exp.key, _audit_detail(exp))
    db.commit()
    return exp


def update_experiment(
    db: Session, exp: AbExperiment, body: dict[str, Any], actor: User | None
) -> AbExperiment:
    if exp.status != "draft":
        raise ExperimentConflictError(
            f"experiment {exp.key!r} is {exp.status}: a started experiment is immutable (create a new one)"
        )
    if body.get("variants") is not None:
        _validate_variants(body["variants"], get_settings())
        _set_variants(exp, body["variants"])
    for field in (
        "name",
        "hypothesis",
        "primary_metric",
        "guardrails",
        "traffic_percent",
        "attribution_window_hours",
    ):
        if body.get(field) is not None:
            setattr(exp, field, body[field])
    if body.get("analysis") is not None:
        exp.analysis = {**dict(body["analysis"]), "data_source": exp.analysis.get("data_source", LIVE)}
    db.commit()
    return exp


def delete_experiment(db: Session, exp: AbExperiment) -> None:
    if exp.status != "draft":
        raise ExperimentConflictError(f"only a draft can be deleted; {exp.key!r} is {exp.status}")
    db.delete(exp)
    db.commit()


def allowed_actions(exp: AbExperiment) -> list[str]:
    out = [a for a, (src, _, _) in TRANSITIONS.items() if exp.status in src]
    if exp.status in RAMP_STATES:
        out.append("ramp")
    if exp.status == "draft":
        out += ["update", "delete"]
    return out


def transition(
    db: Session, exp: AbExperiment, action: str, actor: User | None, now: datetime | None = None
) -> AbExperiment:
    if action not in TRANSITIONS:
        raise ExperimentError(f"unknown action {action!r}")
    sources, target, audit_action = TRANSITIONS[action]
    if exp.status not in sources:
        raise ExperimentConflictError(
            f"cannot {action} an experiment that is {exp.status} (allowed from: {', '.join(sources)})"
        )
    now = now or datetime.now(UTC)
    detail: dict[str, Any] = {"from": exp.status, "to": target}
    if target == "running":
        other = db.scalar(
            select(AbExperiment.key).where(
                AbExperiment.surface == exp.surface,
                AbExperiment.status.in_(("running", "paused")),
                AbExperiment.id != exp.id,
            )
        )
        if other is not None:
            raise ExperimentConflictError(
                f"surface {exp.surface!r} already has an active experiment ({other!r}); stop it first"
            )
        exp.started_at = exp.started_at or now
    elif target == "stopped":
        exp.stopped_at = now
    elif target == "concluded":
        result = compute_results(db, exp)
        exp.result = result
        exp.concluded_at = now
        detail["decision"] = result["conclusion"]["decision"]
        detail["winner"] = result["conclusion"]["winner"]
    exp.status = target
    audit.record(db, audit_action, actor, "ab_experiment", exp.key, detail)
    try:
        db.commit()
    except IntegrityError as exc:  # the partial unique index caught a concurrent start
        db.rollback()
        raise ExperimentConflictError(f"surface {exp.surface!r} already has an active experiment") from exc
    metrics.inc("experiments.transitions", action)
    return exp


def ramp(db: Session, exp: AbExperiment, traffic_percent: float, actor: User | None) -> AbExperiment:
    if exp.status not in RAMP_STATES:
        raise ExperimentConflictError(f"cannot ramp an experiment that is {exp.status}")
    if not 0.0 <= traffic_percent <= 100.0:
        raise ExperimentError("traffic_percent must be within 0..100")
    before = exp.traffic_percent
    exp.traffic_percent = float(traffic_percent)
    audit.record(
        db, "experiment.ramp", actor, "ab_experiment", exp.key, {"from": before, "to": exp.traffic_percent}
    )
    db.commit()
    return exp


def _audit_detail(exp: AbExperiment) -> dict[str, Any]:
    return {
        "surface": exp.surface,
        "primary_metric": exp.primary_metric,
        "traffic_percent": exp.traffic_percent,
        "variants": [{"name": v.name, "weight": v.weight, "control": v.is_control} for v in exp.variants],
    }


def experiment_out(db: Session, exp: AbExperiment) -> dict[str, Any]:
    counts = dict(
        db.execute(
            select(AbAssignment.variant, func.count())
            .where(AbAssignment.experiment_id == exp.id)
            .group_by(AbAssignment.variant)
        )
        .tuples()
        .all()
    )
    result = exp.result or None
    return {
        "id": exp.id,
        "key": exp.key,
        "name": exp.name,
        "surface": exp.surface,
        "status": exp.status,
        "hypothesis": exp.hypothesis,
        "primary_metric": exp.primary_metric,
        "guardrails": exp.guardrails,
        "traffic_percent": exp.traffic_percent,
        "attribution_window_hours": exp.attribution_window_hours,
        "analysis": exp.analysis,
        "data_source": exp.analysis.get("data_source", LIVE),
        "variants": [
            {
                "name": v.name,
                "is_control": v.is_control,
                "weight": v.weight,
                "description": v.description,
                "config": v.config,
                "assigned_users": int(counts.get(v.name, 0)),
            }
            for v in exp.variants
        ],
        "started_at": exp.started_at,
        "stopped_at": exp.stopped_at,
        "concluded_at": exp.concluded_at,
        "created_at": exp.created_at,
        "updated_at": exp.updated_at,
        "created_by": exp.created_by,
        "allowed_actions": allowed_actions(exp),
        "conclusion": result.get("conclusion") if result else None,
    }


# --- outcome attribution ----------------------------------------------------------------------------
@dataclass
class _Exp:
    id: int
    variant: str
    served_at: datetime
    ranks: dict[int, int]


def _events(
    db: Session, user_ids: list[int], lo: datetime, hi: datetime
) -> list[tuple[int, int, str, float | None, datetime]]:
    """(user_id, movie_id, kind, value, occurred_at) of the four interaction sources in [lo, hi]."""
    out: list[tuple[int, int, str, float | None, datetime]] = []
    for chunk in (user_ids[i : i + 500] for i in range(0, len(user_ids), 500)):
        fb = RecommendationFeedback
        for u, m, kind, at in db.execute(
            select(fb.user_id, fb.movie_id, fb.feedback, fb.created_at).where(
                fb.user_id.in_(chunk), fb.created_at >= lo, fb.created_at <= hi
            )
        ):
            out.append((u, m, "click" if kind == "clicked" else kind, None, _aware(at)))
        for u, m, r, at in db.execute(
            select(Rating.user_id, Rating.movie_id, Rating.rating, Rating.updated_at).where(
                Rating.user_id.in_(chunk), Rating.updated_at >= lo, Rating.updated_at <= hi
            )
        ):
            out.append((u, m, "rating", float(r), _aware(at)))
        for u, m, at in db.execute(
            select(WatchHistory.user_id, WatchHistory.movie_id, WatchHistory.watched_at).where(
                WatchHistory.user_id.in_(chunk), WatchHistory.watched_at >= lo, WatchHistory.watched_at <= hi
            )
        ):
            out.append((u, m, "watch", None, _aware(at)))
        for u, m, at in db.execute(
            select(Favorite.user_id, Favorite.movie_id, Favorite.created_at).where(
                Favorite.user_id.in_(chunk),
                Favorite.source == "user",
                Favorite.created_at >= lo,
                Favorite.created_at <= hi,
            )
        ):
            out.append((u, m, "favorite", None, _aware(at)))
    return out


def attribute_outcomes(db: Session, exp: AbExperiment) -> int:
    """Attribute enrolled members' interactions to exposures of ``exp`` (idempotent). An event at t is
    attributed when served_at <= t <= served_at + window, to the member's latest such exposure that
    listed the item, else to their latest such exposure (rank NULL: a relevant item the list missed).
    Returns the number of new outcome rows. Commits."""
    window = timedelta(hours=exp.attribution_window_hours)
    per_user: dict[int, list[_Exp]] = defaultdict(list)
    for eid, uid, variant, served_at, items in db.execute(
        select(AbExposure.id, AbExposure.user_id, AbExposure.variant, AbExposure.served_at, AbExposure.items)
        .where(AbExposure.experiment_id == exp.id)
        .order_by(AbExposure.served_at, AbExposure.id)
    ):
        ranks = {int(i["movie_id"]): int(i["rank"]) for i in items or []}
        per_user[uid].append(_Exp(eid, str(variant), _aware(served_at), ranks))
    if not per_user:
        return 0
    lo = min(e[0].served_at for e in per_user.values())
    hi = max(e[-1].served_at for e in per_user.values()) + window
    existing = set(
        db.execute(
            select(AbOutcome.exposure_id, AbOutcome.kind, AbOutcome.movie_id).where(
                AbOutcome.experiment_id == exp.id
            )
        )
        .tuples()
        .all()
    )
    new = 0
    for uid, movie_id, kind, value, at in _events(db, list(per_user), lo, hi):
        window_exps = [e for e in per_user[uid] if e.served_at <= at <= e.served_at + window]
        if not window_exps:
            continue
        listing = [e for e in window_exps if movie_id in e.ranks]
        chosen = listing[-1] if listing else window_exps[-1]
        k = (chosen.id, kind, movie_id)
        if k in existing:
            continue
        existing.add(k)
        db.add(
            AbOutcome(
                exposure_id=chosen.id,
                experiment_id=exp.id,
                variant=chosen.variant,
                user_id=uid,
                movie_id=movie_id,
                kind=kind,
                value=value,
                rank=chosen.ranks.get(movie_id),
                source="live",
                occurred_at=at,
            )
        )
        new += 1
    db.commit()
    if new:
        metrics.inc("experiments.outcomes", "attributed", new)
    return new


def record_outcomes(db: Session, rows: Iterable[dict[str, Any]]) -> int:
    """Bulk-insert pre-attributed outcomes (the offline replay: held-out ratings, source "replay")."""
    objs = [AbOutcome(**r) for r in rows]
    db.add_all(objs)
    db.commit()
    return len(objs)


# --- results ----------------------------------------------------------------------------------------
def _positive(kind: str, value: float | None) -> bool:
    return kind in ("like", "favorite", "watch") or (kind == "rating" and (value or 0.0) >= LIKE_THRESHOLD)


def _negative(kind: str, value: float | None) -> bool:
    return kind in ("dislike", "not_interested") or (
        kind == "rating" and value is not None and value <= NEGATIVE_RATING
    )


def _catalogue(db: Session, movie_ids: set[int]) -> tuple[dict[int, frozenset[str]], dict[int, float], int]:
    """genre sets, self-information (bits) and the catalogue size."""
    genres: dict[int, set[str]] = defaultdict(set)
    ids = sorted(movie_ids)
    for chunk in (ids[i : i + 900] for i in range(0, len(ids), 900)):
        for mid, name in db.execute(
            select(MovieGenre.movie_id, Genre.name)
            .join(Genre, Genre.id == MovieGenre.genre_id)
            .where(MovieGenre.movie_id.in_(chunk))
        ):
            genres[mid].add(name)
    n_catalogue, total = db.execute(
        select(func.count(Movie.id), func.coalesce(func.sum(Movie.n_ratings), 0))
    ).one()
    info: dict[int, float] = {}
    denom = float(total) + float(n_catalogue or 1)
    for chunk in (ids[i : i + 900] for i in range(0, len(ids), 900)):
        for mid, n in db.execute(select(Movie.id, Movie.n_ratings).where(Movie.id.in_(chunk))):
            info[mid] = -math.log2((float(n or 0) + 1.0) / denom)
    return {m: frozenset(g) for m, g in genres.items()}, info, int(n_catalogue or 0)


def _seed(salt: str, metric: str, variant: str) -> int:
    return _hash_int(f"{salt}:bootstrap:{metric}:{variant}") % (2**32)


def compute_results(db: Session, exp: AbExperiment, attribute: bool = True) -> dict[str, Any]:
    """Per-variant metrics, treatment-vs-control statistics, SRM, sample-size warnings, guardrails and
    the conclusion. The member is the unit of analysis; per-exposure rates are descriptive only."""
    settings = get_settings()
    data_source = exp.analysis.get("data_source", LIVE)
    if attribute and data_source == LIVE:
        attribute_outcomes(db, exp)
    analysis = {
        "alpha": 0.05,
        "power": 0.8,
        "mde_relative": 0.1,
        "min_users_per_variant": 100,
        **exp.analysis,
    }
    alpha = float(analysis["alpha"])
    variants = list(exp.variants)
    control = next(v for v in variants if v.is_control)
    treatments = [v for v in variants if not v.is_control]

    assigned = dict(
        db.execute(
            select(AbAssignment.variant, func.count())
            .where(AbAssignment.experiment_id == exp.id)
            .group_by(AbAssignment.variant)
        )
        .tuples()
        .all()
    )
    exposures = db.execute(
        select(
            AbExposure.id,
            AbExposure.user_id,
            AbExposure.variant,
            AbExposure.items,
            AbExposure.latency_ms,
            AbExposure.cached,
        ).where(AbExposure.experiment_id == exp.id)
    ).all()
    outcomes: dict[int, list[tuple[int, str, float | None, int | None]]] = defaultdict(list)
    n_outcomes = 0
    for eid, mid, kind, value, rank in db.execute(
        select(
            AbOutcome.exposure_id, AbOutcome.movie_id, AbOutcome.kind, AbOutcome.value, AbOutcome.rank
        ).where(AbOutcome.experiment_id == exp.id)
    ):
        outcomes[eid].append((mid, kind, value, rank))
        n_outcomes += 1
    exposed_items = {int(i["movie_id"]) for e in exposures for i in (e.items or [])}
    genres, info, n_catalogue = _catalogue(db, exposed_items)

    # per variant -> per user -> exposure-level measurements
    per_user: dict[str, dict[int, dict[str, list[float]]]] = {
        v.name: defaultdict(lambda: defaultdict(list)) for v in variants
    }
    latencies: dict[str, list[float]] = defaultdict(list)
    latencies_generated: dict[str, list[float]] = defaultdict(list)
    items_seen: dict[str, set[int]] = defaultdict(set)
    n_exp: dict[str, int] = defaultdict(int)
    n_cached: dict[str, int] = defaultdict(int)
    for e in exposures:
        name = str(e.variant)
        if name not in per_user:
            continue
        n_exp[name] += 1
        n_cached[name] += int(bool(e.cached))
        if e.latency_ms is not None:
            latencies[name].append(float(e.latency_ms))
            if not e.cached:
                latencies_generated[name].append(float(e.latency_ms))
        listed = [int(i["movie_id"]) for i in sorted(e.items or [], key=lambda i: i["rank"])]
        items_seen[name].update(listed)
        outs = outcomes.get(e.id, [])
        in_list = [(m, k, v) for m, k, v, r in outs if r is not None]
        pos_all = {m for m, k, v, _ in outs if _positive(k, v)}
        pos_ranks = [r for _, k, v, r in outs if r is not None and _positive(k, v)]
        u = per_user[name][e.user_id]
        u["interaction"].append(float(any(k in INTERACTION_KINDS for _, k, _ in in_list)))
        u["positive"].append(float(any(_positive(k, v) for _, k, v in in_list)))
        u["rating"].append(float(any(k == "rating" for _, k, _ in in_list)))
        u["feedback"].append(float(any(k in FEEDBACK_KINDS for _, k, _ in in_list)))
        u["negative"].append(float(any(_negative(k, v) for _, k, v in in_list)))
        u["ndcg"].append(st.ndcg_at_k(pos_ranks, len(pos_all), 10))
        div = st.intra_list_diversity([genres.get(m, frozenset()) for m in listed[:10]])
        if div is not None:
            u["diversity"].append(div)
        nov = [info[m] for m in listed[:10] if m in info]
        if nov:
            u["novelty"].append(float(np.mean(nov)))

    def user_values(name: str, metric: str) -> list[float]:
        src = {"ndcg_at_10": "ndcg", "diversity": "diversity", "novelty": "novelty"}[metric]
        return [float(np.mean(m[src])) for m in per_user[name].values() if m[src]]

    rate_src = {
        "interaction_rate": "interaction",
        "positive_rate": "positive",
        "rating_rate": "rating",
        "feedback_rate": "feedback",
        "negative_rate": "negative",
    }

    def rate_counts(name: str, metric: str) -> tuple[int, int]:
        src = rate_src[metric]
        users = per_user[name]
        return sum(1 for m in users.values() if any(m[src])), len(users)

    variant_rows: list[dict[str, Any]] = []
    for v in variants:
        users = per_user[v.name]
        rates = {}
        for metric in RATE_METRICS:
            x, n = rate_counts(v.name, metric)
            src = rate_src[metric]
            flat = [x for m in users.values() for x in m[src]]
            rates[metric] = {
                "users_with_event": x,
                "users": n,
                "rate": st._r(x / n) if n else None,
                "per_exposure_rate": st._r(float(np.mean(flat))) if flat else None,
            }
        means = {}
        for metric in MEAN_METRICS:
            vals = user_values(v.name, metric)
            means[metric] = {"mean": st._r(float(np.mean(vals))) if vals else None, "users": len(vals)}
        lat = latencies[v.name]
        variant_rows.append(
            {
                "name": v.name,
                "is_control": v.is_control,
                "weight": v.weight,
                "config": v.config,
                "assigned_users": int(assigned.get(v.name, 0)),
                "exposed_users": len(users),
                "exposures": n_exp[v.name],
                "cache_hit_exposures": n_cached[v.name],
                "rates": rates,
                "means": means,
                "coverage": {
                    "distinct_items": len(items_seen[v.name]),
                    "catalogue": n_catalogue,
                    "value": st._r(len(items_seen[v.name]) / n_catalogue) if n_catalogue else None,
                },
                "latency_ms": {
                    "p50": st._r(st.percentile(lat, 50) or 0.0, 2) if lat else None,
                    "p95": st._r(st.percentile(lat, 95) or 0.0, 2) if lat else None,
                    "p95_generated": st._r(st.percentile(latencies_generated[v.name], 95) or 0.0, 2)
                    if latencies_generated[v.name]
                    else None,
                },
            }
        )
    by_name: dict[str, dict[str, Any]] = {str(r["name"]): r for r in variant_rows}

    # statistics: every treatment vs the control; the primary metric at a Bonferroni-adjusted alpha
    alpha_primary = alpha / max(1, len(treatments))
    resamples = settings.experiments_bootstrap_resamples
    comparisons: dict[str, dict[str, Any]] = {}
    for t in treatments:
        comp: dict[str, Any] = {}
        for metric in RATE_METRICS:
            a = alpha_primary if metric == exp.primary_metric else alpha
            xc, nc = rate_counts(control.name, metric)
            xt, nt = rate_counts(t.name, metric)
            comp[metric] = {**st.two_proportion(xc, nc, xt, nt, a), "alpha": a}
        for metric in MEAN_METRICS:
            a = alpha_primary if metric == exp.primary_metric else alpha
            comp[metric] = {
                **st.bootstrap_diff(
                    user_values(control.name, metric),
                    user_values(t.name, metric),
                    a,
                    resamples,
                    _seed(exp.salt, metric, t.name),
                ),
                "alpha": a,
            }
        comparisons[t.name] = comp

    # sample ratio mismatch: on assigned members, and on exposed members (a variant that fails to serve
    # shows up here)
    weights = [v.weight for v in variants]
    srm_assigned = st.srm(
        [int(assigned.get(v.name, 0)) for v in variants], weights, settings.experiments_srm_alpha
    )
    srm_exposed = st.srm([len(per_user[v.name]) for v in variants], weights, settings.experiments_srm_alpha)
    srm_detected = bool(srm_assigned["detected"] or srm_exposed["detected"])

    # sample size
    min_users = int(analysis["min_users_per_variant"])
    smallest = min((len(per_user[v.name]) for v in variants), default=0)
    required = None
    if exp.primary_metric in rate_src:
        xc, nc = rate_counts(control.name, exp.primary_metric)
        required = st.required_sample_size(
            xc / nc if nc else None, float(analysis["mde_relative"]), alpha_primary, float(analysis["power"])
        )
    else:
        vals = user_values(control.name, exp.primary_metric)
        if len(vals) >= 2 and float(np.mean(vals)) != 0:
            sd, mean = float(np.std(vals, ddof=1)), float(np.mean(vals))
            delta = abs(mean) * float(analysis["mde_relative"])
            za, zb = st.z_crit(alpha_primary), float(st.sps.norm.ppf(float(analysis["power"])))
            required = math.ceil(2 * (za + zb) ** 2 * sd**2 / delta**2) if delta > 0 and sd > 0 else None
    warnings: list[str] = []
    if smallest < min_users:
        warnings.append(
            f"only {smallest} exposed users in the smallest variant; the minimum is {min_users} per variant"
        )
    if required is not None and smallest < required:
        warnings.append(
            f"underpowered: detecting a {float(analysis['mde_relative']):.0%} relative change in "
            f"{exp.primary_metric} needs about {required} users per variant (smallest has {smallest})"
        )
    if required is None and smallest:
        warnings.append(
            f"no sample-size estimate: the control's {exp.primary_metric} is degenerate (0, 1 or no variance)"
        )
    if n_outcomes == 0:
        warnings.append("no outcomes attributed yet")

    guardrails = _guardrails(exp, control.name, treatments, by_name, comparisons)
    conclusion = _conclude(
        exp, control.name, treatments, comparisons, guardrails, srm_detected, smallest, min_users
    )
    return {
        "experiment": exp.key,
        "status": exp.status,
        "surface": exp.surface,
        "data_source": data_source,
        "label": REPLAY_LABEL if data_source == REPLAY else LIVE_LABEL,
        "primary_metric": exp.primary_metric,
        "alpha": alpha,
        "alpha_primary": alpha_primary,
        "attribution_window_hours": exp.attribution_window_hours,
        "unit_of_analysis": "member",
        "computed_at": datetime.now(UTC).isoformat(),
        "totals": {
            "assigned_users": int(sum(assigned.values())),
            "exposures": len(exposures),
            "outcomes": n_outcomes,
        },
        "variants": variant_rows,
        "comparisons": comparisons,
        "srm": {"assigned": srm_assigned, "exposed": srm_exposed, "detected": srm_detected},
        "sample_size": {
            "required_per_variant": required,
            "smallest_variant_users": smallest,
            "min_users_per_variant": min_users,
            "mde_relative": float(analysis["mde_relative"]),
            "power": float(analysis["power"]),
        },
        "warnings": warnings,
        "guardrails": guardrails,
        "conclusion": conclusion,
    }


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric in RATE_METRICS:
        return row["rates"][metric]["rate"]
    if metric in MEAN_METRICS:
        return row["means"][metric]["mean"]
    if metric == "coverage":
        return row["coverage"]["value"]
    if metric == "latency_p95_ms":
        return row["latency_ms"]["p95"]
    return None


def _guardrails(
    exp: AbExperiment,
    control: str,
    treatments: list[AbVariant],
    rows: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """A guardrail is breached when the observed degradation exceeds its threshold, or when a rate or
    mean guardrail degrades significantly (p < alpha) in the harmful direction."""
    out = []
    for g in exp.guardrails or []:
        metric = g["metric"]
        for t in treatments:
            c_val, t_val = _metric_value(rows[control], metric), _metric_value(rows[t.name], metric)
            breached, reason = False, None
            comp = comparisons[t.name].get(metric)
            if c_val is not None and t_val is not None:
                if metric == "latency_p95_ms":
                    ratio = t_val / c_val if c_val > 0 else None
                    max_ratio = g.get("max_ratio")
                    if (
                        max_ratio
                        and ratio is not None
                        and ratio > max_ratio
                        and t_val - c_val > g.get("min_absolute_ms", 5.0)
                    ):
                        breached, reason = True, f"p95 latency x{ratio:.2f} > x{max_ratio}"
                else:
                    diff = t_val - c_val
                    if g.get("max_increase") is not None and diff > g["max_increase"]:
                        breached, reason = True, f"increase {diff:+.4f} > {g['max_increase']}"
                    if g.get("max_decrease") is not None and -diff > g["max_decrease"]:
                        breached, reason = True, f"decrease {diff:+.4f} > {g['max_decrease']}"
                    harmful_up = metric == "negative_rate" or g.get("max_increase") is not None
                    if (
                        comp is not None
                        and comp.get("significant")
                        and comp.get("diff") is not None
                        and (comp["diff"] > 0) == harmful_up
                    ):
                        word = "increase" if harmful_up else "decrease"
                        breached, reason = True, f"significant {word} (p={comp['p_value']})"
            out.append(
                {
                    "metric": metric,
                    "variant": t.name,
                    "control_value": c_val,
                    "treatment_value": t_val,
                    "threshold": {k: v for k, v in g.items() if k != "metric" and v is not None},
                    "breached": breached,
                    "reason": reason,
                }
            )
    return out


def _conclude(
    exp: AbExperiment,
    control: str,
    treatments: list[AbVariant],
    comparisons: dict[str, dict[str, Any]],
    guardrails: list[dict[str, Any]],
    srm_detected: bool,
    smallest: int,
    min_users: int,
) -> dict[str, Any]:
    """ "ship" (a treatment wins), "keep_control" (every treatment is significantly worse) or
    "inconclusive". Never a winner without a significant primary effect and clean guardrails."""
    reasons: list[str] = []
    if srm_detected:
        return {
            "decision": "inconclusive",
            "winner": None,
            "reasons": [
                "sample ratio mismatch: assignment or exposure logging is broken; results are untrustworthy"
            ],
        }
    if smallest < min_users:
        return {
            "decision": "inconclusive",
            "winner": None,
            "reasons": [f"insufficient sample: {smallest} users in the smallest variant (< {min_users})"],
        }
    candidates: list[tuple[float, str]] = []
    worse = 0
    for t in treatments:
        comp = comparisons[t.name][exp.primary_metric]
        breached = [g for g in guardrails if g["variant"] == t.name and g["breached"]]
        sig = bool(comp.get("significant")) and comp.get("diff") is not None
        if sig and comp["diff"] > 0:
            if breached:
                reasons.append(
                    f"{t.name}: {exp.primary_metric} improved significantly but guardrails breached: "
                    + "; ".join(f"{g['metric']} ({g['reason']})" for g in breached)
                )
            else:
                candidates.append((float(comp["diff"]), t.name))
                reasons.append(
                    f"{t.name}: {exp.primary_metric} {comp['diff']:+.4f} "
                    f"(CI {comp['ci'][0]:+.4f}..{comp['ci'][1]:+.4f}, p={comp['p_value']}) "
                    "and no guardrail breached"
                )
        elif sig and comp["diff"] < 0:
            worse += 1
            reasons.append(
                f"{t.name}: {exp.primary_metric} significantly worse "
                f"({comp['diff']:+.4f}, p={comp['p_value']})"
            )
        else:
            reasons.append(
                f"{t.name}: {exp.primary_metric} difference not significant "
                f"(diff {comp.get('diff')}, p={comp.get('p_value')})"
            )
    if candidates:
        best = max(candidates)[1]
        return {"decision": "ship", "winner": best, "reasons": reasons}
    if treatments and worse == len(treatments):
        return {"decision": "keep_control", "winner": control, "reasons": reasons}
    return {"decision": "inconclusive", "winner": None, "reasons": reasons}
