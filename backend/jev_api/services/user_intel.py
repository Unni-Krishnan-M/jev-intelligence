"""Per-member intelligence (docs/platform.md, sections 4, 8 and 10): the recommendation-strategy
decision behind GET /recommendations, the GET /me/intelligence payload, preference scenarios and member
feedback on strategy decisions and recommendations.

The ML side is ``jev_ml.domains.movie.user_intel`` / ``user_scenario`` (pure functions). This module
feeds them exactly what the recommendation service uses: the member's interactions, genre preferences
and excluded (negative-feedback) movies from ``load_user_state``, plus the member's feedback:

* recommendation_feedback (like / dislike / not_interested / clicked; the newest verdict per movie,
  a click only where there is no verdict), and
* POST /me/intelligence/feedback verdicts on recommendations (accepted / rejected), which replace an
  older recommendation_feedback verdict on the same movie.

Strategy verdicts judge the decision, not a recommendation, so they are never an acceptance event.

The strategy decision is cached per member (profile version, model version, feedback marker), so a
cache hit on /recommendations costs two indexed aggregate queries and a cache read. Every freshly taken
decision is also persisted in ``member_decisions`` (Phase 2, gap P1 #7), so its id stays resolvable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from jev_api.cache import Cache
from jev_api.config import get_settings
from jev_api.metrics import metrics
from jev_api.models import RecommendationFeedback, User, UserIntelFeedback, utcnow
from jev_api.services.experiments import persist_decision
from jev_api.services.feedback import CLICK
from jev_api.services.profile import UserState, build_ml_profile, load_user_state
from jev_ml.domains.movie.user_intel import decay_profile, engine_with_overrides, strategy_for
from jev_ml.engine import RecommendationEngine
from jev_ml.signals import UserProfile

log = logging.getLogger(__name__)

# the strategy state's recommendation-confidence statistics use the top 10, as GET /me/intelligence
# does by default, so both endpoints report the same decision id for the same inputs
STRATEGY_K = 10
MAX_BLOCK_EVIDENCE = 8
DECISION_ID = re.compile(r"^dec-[A-Za-z0-9_.:-]{1,60}$")


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else _aware(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- feedback the drift report's acceptance aspect reads ------------------------------------------
def feedback_rows(db: Session, user_id: int) -> list[dict[str, Any]]:
    """One acceptance event per movie: {"timestamp": epoch seconds, "verdict": ...}."""
    fb = RecommendationFeedback
    per_movie: dict[int, tuple[datetime, str]] = {}
    for mid, kind, at in db.execute(
        select(fb.movie_id, fb.feedback, fb.created_at)
        .where(fb.user_id == user_id)
        .order_by(fb.created_at, fb.id)
    ):
        cur = per_movie.get(mid)
        # the newest verdict wins; a click only counts where the member gave no verdict
        if cur is None or kind != CLICK or cur[1] == CLICK:
            per_movie[mid] = (_aware(at), kind)
    uf = UserIntelFeedback
    for mid, verdict, at in db.execute(
        select(uf.movie_id, uf.verdict, uf.updated_at).where(
            uf.user_id == user_id, uf.target_type == "recommendation", uf.movie_id.is_not(None)
        )
    ):
        cur = per_movie.get(mid)
        if cur is None or _aware(at) >= cur[0]:
            per_movie[mid] = (_aware(at), verdict)
    return [{"timestamp": at.timestamp(), "verdict": v} for at, v in per_movie.values()]


def feedback_marker(db: Session, user_id: int) -> str:
    """Changes whenever the member's feedback changes (an upsert moves created_at / updated_at)."""
    fb, uf = RecommendationFeedback, UserIntelFeedback
    n1, t1 = db.execute(select(func.count(fb.id), func.max(fb.created_at)).where(fb.user_id == user_id)).one()
    n2, t2 = db.execute(select(func.count(uf.id), func.max(uf.updated_at)).where(uf.user_id == user_id)).one()
    raw = f"{n1}|{_iso(t1)}|{n2}|{_iso(t2)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# --- the strategy decision -----------------------------------------------------------------------------
def intelligence_block(dec: dict[str, Any]) -> dict[str, Any]:
    """The `intelligence` block of GET /recommendations (docs/platform.md, section 10)."""
    state = dec.get("state") or {}
    rationale = [str(r) for r in dec.get("rationale") or []]
    summary = dec.get("fallback_reason") or ("; ".join(rationale[:2]) if rationale else None)
    return {
        "decision_id": str(dec["id"]),
        "strategy": dec.get("answer"),
        "served_strategy": str(state.get("served_strategy") or "standard"),
        "confidence": dec.get("confidence"),
        "confidence_kind": str(dec.get("confidence_kind") or "margin"),
        "abstained": bool(dec.get("abstained")),
        "drift_detected": state.get("drift_detected"),
        "summary": summary,
        "policy_version": dec.get("policy_version"),
        "evidence": [e for e in (dec.get("evidence") or [])[:MAX_BLOCK_EVIDENCE] if isinstance(e, dict)],
    }


class StrategyChoice:
    """The member's current strategy: the response block plus how to serve it."""

    def __init__(self, block: dict[str, Any], profile_kwargs: dict[str, Any], overrides: dict[str, Any]):
        self.block = block
        self.profile_kwargs = profile_kwargs
        self.overrides = overrides

    @property
    def served(self) -> str:
        return str(self.block["served_strategy"])

    def to_cache(self) -> dict[str, Any]:
        return {"block": self.block, "profile_kwargs": self.profile_kwargs, "overrides": self.overrides}


def strategy_choice(
    db: Session,
    cache: Cache,
    engine: RecommendationEngine,
    user: User,
    state: UserState | None = None,
    profile: UserProfile | None = None,
) -> tuple[StrategyChoice, UserState | None, UserProfile | None]:
    """The member's recommendation_strategy decision (cached). Returns the choice plus the state and
    standard profile when they had to be built (so the caller does not build them twice)."""
    settings = get_settings()
    key = f"strat:{user.id}:{user.profile_version}:{engine.version}:{feedback_marker(db, user.id)}"
    hit = cache.get(key)
    if isinstance(hit, dict) and "block" in hit:
        metrics.inc("strategy_cache", "hit")
        return StrategyChoice(hit["block"], hit["profile_kwargs"], hit["overrides"]), state, profile
    metrics.inc("strategy_cache", "miss")
    t0 = time.perf_counter()
    if state is None:
        state = load_user_state(db, user)
    if profile is None:
        profile = build_ml_profile(engine, state)
    dec, pk, co = strategy_for(
        engine,
        state.interactions,
        feedback_rows(db, user.id),
        user_id=user.id,
        genre_prefs=state.genres,
        excluded_movie_ids=state.excluded,
        k=STRATEGY_K,
        profile=profile,
    )
    choice = StrategyChoice(intelligence_block(dec), dict(pk), dict(co))
    # P1 #7: persist the decision so decision ids on recommendation and feedback rows resolve after the
    # cache expires (member_decisions; never fails the request)
    persist_decision(db, user, dec, engine.version, "recommendations")
    metrics.observe("strategy_decision_ms", "all", 1000 * (time.perf_counter() - t0))
    metrics.inc("strategy_decisions", "abstained" if dec.get("abstained") else str(dec.get("answer")))
    metrics.inc("strategy_served", choice.served)
    cache.set(key, choice.to_cache(), settings.recommendation_cache_seconds)
    return choice, state, profile


def serve_profile(profile: UserProfile, choice: StrategyChoice) -> UserProfile:
    """The standard profile under the chosen strategy (adapt_to_recent: recency decay)."""
    pk = choice.profile_kwargs
    if not pk.get("recency_half_life_days"):
        return profile
    return decay_profile(profile, pk["recency_half_life_days"], pk.get("weight_floor", 0.5))


def serve_engine(engine: RecommendationEngine, choice: StrategyChoice, user: User) -> RecommendationEngine:
    """The strategy's HybridConfig overrides. A member's explicit diversity preference (focused /
    adventurous) is their own choice and wins over the strategy's MMR λ."""
    overrides = dict(choice.overrides)
    if (user.recommendation_prefs or {}).get("diversity") in ("focused", "adventurous"):
        overrides.pop("diversity_lambda", None)
    return engine_with_overrides(engine, overrides) if overrides else engine


# --- member feedback (POST /me/intelligence/feedback) ------------------------------------------------------
def me_feedback_out(row: UserIntelFeedback) -> dict[str, Any]:
    return {
        "id": row.id,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "verdict": row.verdict,
        "note": row.note,
        "decision_id": row.decision_id,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def upsert_me_feedback(
    db: Session,
    user: User,
    target_type: str,
    target_id: str,
    verdict: str,
    note: str | None,
    decision_id: str | None,
    movie_id: int | None,
) -> tuple[UserIntelFeedback, bool]:
    """One row per member per target: a repeat is a no-op, a new verdict (or note) replaces the old
    one. Returns (row, changed). Commits."""
    uf = UserIntelFeedback
    for attempt in range(2):
        row = db.scalar(
            select(uf).where(uf.user_id == user.id, uf.target_type == target_type, uf.target_id == target_id)
        )
        changed = True
        if row is None:
            row = uf(
                user_id=user.id,
                target_type=target_type,
                target_id=target_id,
                verdict=verdict,
                note=note,
                decision_id=decision_id,
                movie_id=movie_id,
            )
            db.add(row)
        elif row.verdict != verdict or (note is not None and row.note != note):
            if row.verdict != verdict:
                row.created_at = utcnow()  # the verdict changed now
            row.verdict = verdict
            row.note = note if note is not None else row.note
            row.decision_id = decision_id or row.decision_id
        else:
            changed = False
        try:
            db.flush()
        except IntegrityError:  # a concurrent request inserted the same target first
            db.rollback()
            if attempt:
                raise
            continue
        return row, changed
    raise AssertionError("unreachable")
