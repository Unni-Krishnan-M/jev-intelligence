"""Turn model contributions into human-readable reasons.

Rules:
  - Every sentence is filled from a Contribution the model actually produced (an anchor item with
    positive attribution, a shared feature with positive dot-product mass, a matched genre, …).
  - Reasons are ordered by the signal's contribution to the hybrid score, times an
    informativeness prior: personal signals count 1.0, non-personal ones (popularity, recency)
    count 0.5. So "Rated 4+ by N viewers" leads only when popularity contributes more than twice
    as much as any personal signal.
  - If a signal carries no attributable evidence, there is no reason for it; the sentence falls
    back to the next signal rather than inventing one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jev_ml.models.base import Contribution
from jev_ml.models.hybrid import SIGNALS, RankedItem
from jev_ml.signals import UserProfile


@dataclass
class Explanation:
    reason: str
    reason_code: str
    secondary: list[str]
    anchor_movie_ids: list[int]


def _liked(profile: UserProfile, item: int) -> bool:
    pos = {int(i): float(p) for i, p in zip(profile.items, profile.pref_weights, strict=True)}
    return pos.get(item, 0.0) > 0


def _item_reason(signal: str, c: Contribution, title: str, liked: bool) -> tuple[str, str]:
    if signal == "collaborative":
        return f"Viewers who enjoyed {title} also loved this", "cf_anchor"
    if signal == "latent":
        verb = "liked" if liked else "watched"
        return f"Because you {verb} {title}", "mf_anchor"
    return f"Similar to {title}, which you rated highly" if liked else f"Similar to {title}", "content_anchor"


def _feature_reason(c: Contribution) -> tuple[str, str] | None:
    label = c.label or ""
    if c.kind == "director":
        return f"Recommended because of your interest in {label}", "director"
    if c.kind == "cast":
        return f"Stars {label}, from films you enjoyed", "cast"
    if c.kind in ("keyword", "tag"):
        return f"Matches your interest in “{label}”", c.kind
    if c.kind == "genre":
        return f"Matches your preference for {label}", "genre"
    return None


# Non-personal signals are true, but they say little about *why this user*.
REASON_PRIOR = {
    "content": 1.0,
    "collaborative": 1.0,
    "latent": 1.0,
    "preference": 1.0,
    "popularity": 0.5,
    "recency": 0.5,
}


def explain(
    ranked: RankedItem,
    profile: UserProfile,
    title_of: Callable[[int], str],
    movie_id_of: Callable[[int], int],
) -> Explanation:
    order = sorted(SIGNALS, key=lambda s: -ranked.signals[s]["contribution"] * REASON_PRIOR[s])
    sentences: list[tuple[str, str]] = []
    anchors: list[int] = []

    for signal in order:
        if ranked.signals[signal]["contribution"] <= 1e-9:
            continue
        contribs = ranked.contributions.get(signal, [])
        sentence: tuple[str, str] | None = None
        if signal in ("collaborative", "latent"):
            items = [c for c in contribs if c.kind == "item" and c.item is not None]
            if items:
                a = items[0]
                assert a.item is not None
                sentence = _item_reason(signal, a, title_of(a.item), _liked(profile, a.item))
                anchors.append(movie_id_of(a.item))
            elif signal == "collaborative":
                sentence = ("Popular among viewers with similar tastes", "cf_neighborhood")
        elif signal == "content":
            feats = [c for c in contribs if c.kind in ("director", "cast", "keyword", "tag", "genre")]
            items = [c for c in contribs if c.kind == "item" and c.item is not None]
            strong_feature = feats[0] if feats and feats[0].kind in ("director", "cast", "keyword") else None
            if strong_feature is not None:
                sentence = _feature_reason(strong_feature)
            elif items:
                a = items[0]
                assert a.item is not None
                sentence = _item_reason("content", a, title_of(a.item), _liked(profile, a.item))
                anchors.append(movie_id_of(a.item))
            elif feats:
                sentence = _feature_reason(feats[0])
        elif signal == "preference" and contribs:
            genres = " & ".join(c.label or "" for c in contribs[:2])
            sentence = (f"Matches your preference for {genres}", "genre_preference")
        elif signal == "popularity" and contribs:
            sentence = (f"Rated 4+ stars by {contribs[0].label} viewers", "popular")
        elif signal == "recency" and contribs:
            sentence = (f"A {contribs[0].label} release", "recent")
        if sentence and sentence[0] not in {s for s, _ in sentences}:
            sentences.append(sentence)
        if len(sentences) >= 3:
            break

    if not sentences:
        return Explanation("Popular pick to start your profile", "fallback_popular", [], [])
    primary, code = sentences[0]
    return Explanation(primary, code, [s for s, _ in sentences[1:]], anchors)
