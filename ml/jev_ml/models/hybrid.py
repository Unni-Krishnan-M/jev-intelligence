"""Hybrid ranking engine.

Pipeline (deterministic for a fixed model state and input):
  1. candidate filtering    global mask: consumed, "not interested", request filters
  2. candidate generation   union of top-N from every active component
  3. feature computation    six signals per candidate (see SIGNALS)
  4. score normalization    per-signal min-max (or rank) within the candidate set
  5. hybrid scoring         weighted sum with interaction-adaptive weights
  6. business rules         quality floor
  7. diversity adjustment   Maximal Marginal Relevance over content similarity + optional genre cap
  8. final ranking / top-K  with offset-based pagination

Adaptive weighting (cold start → warm): collaborative and latent signals need behavioural data,
so their weights ramp linearly with the number of interactions up to `behavioral_ramp`. Popularity
and explicit genre preferences carry the ranking while that happens. Weights are renormalized to
sum to 1, so hybrid scores stay on a comparable [0, 1] scale for every user.

Cold stages (optional, `HybridConfig.cold_stages`): separate weight sets for short profiles, tuned
on the validation split per profile-size bucket. A stage either replaces the weights outright
(`weights`) or blends the adaptive weights with popularity (`popularity_blend`). Stages only change
the weights, so every served item still carries its per-signal contributions and its explanation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from jev_ml.data.dataset import split_list
from jev_ml.features import item_genre_matrix
from jev_ml.models.als import ALSRecommender
from jev_ml.models.base import Contribution, topk_indices
from jev_ml.models.content import ContentRecommender
from jev_ml.models.itemknn import ItemKNNRecommender
from jev_ml.models.popularity import PopularityRecommender
from jev_ml.signals import UserProfile

SIGNALS = ("content", "collaborative", "latent", "popularity", "preference", "recency")


@dataclass
class HybridConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "content": 0.20, "collaborative": 0.30, "latent": 0.35,
        "popularity": 0.08, "preference": 0.05, "recency": 0.02,
    })
    behavioral_ramp: int = 10
    candidates_per_source: int = 200
    normalization: str = "minmax"  # "minmax" | "rank"
    diversity_lambda: float = 0.9  # 1.0 disables MMR
    diversity_pool: int = 100
    max_per_genre: float | None = None  # e.g. 0.5 → no primary genre fills >50% of a page
    quality_floor: float = 2.0  # drop Bayesian-average rating below this (business rule)
    recency_tau_years: float = 15.0
    cold_start_boost: float = 2.0  # popularity/preference amplification for users w/o history
    # content scores are multiplied by a support prior ((log1p(n)+1)/(log1p(n_max)+1))^damping so
    # pure-metadata matches on obscure titles don't swamp the list; +1 keeps unrated (new) movies
    # discoverable. 0 disables it.
    content_support_damping: float = 1.0
    # profile-size conditioned weights, e.g. [{"name": "cold_0", "max_profile": 0, "weights": {...}},
    # {"name": "cold_1_3", "min_profile": 1, "max_profile": 3, "popularity_blend": 0.5}]. The first
    # stage with min_profile <= n_interactions <= max_profile applies (min_profile defaults to 0);
    # other profiles use the adaptive weights. None = off.
    cold_stages: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> HybridConfig:
        base = cls()
        if not d:
            return base
        merged = {**asdict(base), **d}
        merged["weights"] = {**base.weights, **(d.get("weights") or {})}
        return cls(**merged)


@dataclass
class RankedItem:
    item: int
    score: float
    signals: dict[str, dict[str, float]]
    dominant: str
    contributions: dict[str, list[Contribution]] = field(default_factory=dict)


@dataclass
class RecommendationFilters:
    genres: list[str] | None = None  # item must have at least one
    year_min: int | None = None
    year_max: int | None = None
    min_ratings: int | None = None
    max_ratings: int | None = None  # "discovery" shelves: only lesser-known titles


class HybridRanker:
    def __init__(self, movies: pd.DataFrame, popularity: PopularityRecommender,
                 content: ContentRecommender, itemknn: ItemKNNRecommender, als: ALSRecommender,
                 config: HybridConfig | None = None) -> None:
        self.movies = movies.reset_index(drop=True)
        self.popularity = popularity
        self.content = content
        self.itemknn = itemknn
        self.als = als
        self.config = config or HybridConfig()
        self.n_items = len(self.movies)
        self.genre_matrix, self.genres = item_genre_matrix(self.movies)
        gcount = np.asarray(self.genre_matrix.sum(axis=1)).ravel()
        self._genre_norm = 1.0 / np.sqrt(np.maximum(gcount, 1.0))
        self._primary_genre = np.asarray(
            [(split_list(g) or [""])[0] for g in self.movies["genres"]], dtype=object)
        years = pd.to_numeric(self.movies["year"], errors="coerce").to_numpy(dtype=np.float64)
        self._years = years
        ref = np.nanmax(years) if np.isfinite(years).any() else 2020.0
        age = np.where(np.isfinite(years), ref - years, 100.0)
        self._recency = np.exp(-np.maximum(age, 0.0) / self.config.recency_tau_years)
        self._n_ratings = self.movies["n_ratings"].to_numpy(dtype=np.float64)
        support = np.log1p(self.popularity.user_counts) + 1.0
        self._support_prior = support / support.max() if support.max() > 0 else np.ones(self.n_items)

    # --- per-signal raw scores ----------------------------------------------------------------
    def genre_preference_vector(self, profile: UserProfile) -> np.ndarray:
        """Explicit onboarding genres + genres of liked/disliked items, each L2-normalized."""
        g = np.zeros(len(self.genres))
        explicit = np.zeros(len(self.genres))
        for name, w in profile.genre_prefs.items():
            if name in self.genres:
                explicit[self.genres.index(name)] = w
        if np.linalg.norm(explicit) > 0:
            g += explicit / np.linalg.norm(explicit)
        if profile.n_interactions:
            derived = np.asarray(self.genre_matrix[profile.items].T @ profile.pref_weights).ravel()
            if np.linalg.norm(derived) > 0:
                g += derived / np.linalg.norm(derived)
        return g

    def preference_scores(self, profile: UserProfile) -> np.ndarray:
        g = self.genre_preference_vector(profile)
        if not np.any(g):
            return np.zeros(self.n_items)
        return np.asarray(self.genre_matrix @ g).ravel() * self._genre_norm

    def raw_signals(self, profile: UserProfile) -> dict[str, np.ndarray]:
        content = self.content.score(profile)
        if self.config.content_support_damping:
            content = np.maximum(content, 0.0) * self._support_prior ** self.config.content_support_damping
        return {
            "content": content,
            "collaborative": self.itemknn.score(profile),
            "latent": self.als.score(profile),
            "popularity": self.popularity.score(profile),
            "preference": self.preference_scores(profile),
            "recency": self._recency,
        }

    def cold_stage(self, profile: UserProfile) -> dict[str, Any] | None:
        """The cold stage that applies to this profile size, or None (adaptive weights)."""
        n = profile.n_interactions
        for st in self.config.cold_stages or []:
            if int(st.get("min_profile", 0)) <= n <= int(st["max_profile"]):
                return st
        return None

    def weight_strategy(self, profile: UserProfile) -> str:
        """Name of the weighting used for this profile: a cold stage's name or "adaptive"."""
        st = self.cold_stage(profile)
        return "adaptive" if st is None else str(st.get("name") or f"stage_le_{st['max_profile']}")

    def effective_weights(self, profile: UserProfile) -> dict[str, float]:
        st = self.cold_stage(profile)
        if st is None:
            return self._adaptive_weights(profile)
        if "popularity_blend" in st:
            a = float(np.clip(st["popularity_blend"], 0.0, 1.0))
            base = self._adaptive_weights(profile)
            return {k: (1.0 - a) * v + (a if k == "popularity" else 0.0) for k, v in base.items()}
        n = profile.n_interactions
        n_pos = int(np.sum(profile.pref_weights > 0)) if n else 0
        has_pref = bool(profile.genre_prefs) or n_pos > 0
        w = {k: max(float((st.get("weights") or {}).get(k, 0.0)), 0.0) for k in SIGNALS}
        if n == 0:  # behavioural signals are all zero without history
            w["collaborative"] = w["latent"] = 0.0
        w["content"] *= min(1.0, n_pos / 3.0)
        if not has_pref:
            w["preference"] = 0.0
        total = sum(w.values())
        if total <= 0:
            return {k: (1.0 if k == "popularity" else 0.0) for k in SIGNALS}
        return {k: v / total for k, v in w.items()}

    def _adaptive_weights(self, profile: UserProfile) -> dict[str, float]:
        cfg = self.config
        n = profile.n_interactions
        n_pos = int(np.sum(profile.pref_weights > 0)) if n else 0
        beta = min(1.0, n / cfg.behavioral_ramp) if cfg.behavioral_ramp > 0 else 1.0
        gamma = min(1.0, n_pos / 3.0)
        has_pref = bool(profile.genre_prefs) or n_pos > 0
        boost = 1.0 + cfg.cold_start_boost * (1.0 - beta)
        w = dict(cfg.weights)
        w["collaborative"] *= beta
        w["latent"] *= beta
        w["content"] *= gamma
        w["popularity"] *= boost
        w["preference"] = w["preference"] * boost if has_pref else 0.0
        total = sum(v for v in w.values() if v > 0)
        if total <= 0:
            return {k: (1.0 if k == "popularity" else 0.0) for k in SIGNALS}
        return {k: max(v, 0.0) / total for k, v in w.items()}

    # --- filtering --------------------------------------------------------------------------------
    def filter_mask(self, profile: UserProfile, filters: RecommendationFilters | None) -> np.ndarray:
        """True = item may NOT be recommended."""
        mask = np.zeros(self.n_items, dtype=bool)
        consumed = profile.consumed
        if len(consumed):
            mask[consumed] = True
        if filters:
            if filters.genres:
                wanted = [self.genres.index(g) for g in filters.genres if g in self.genres]
                has = np.asarray(self.genre_matrix[:, wanted].sum(axis=1)).ravel() > 0 if wanted else \
                    np.zeros(self.n_items, dtype=bool)
                mask |= ~has
            if filters.year_min is not None:
                mask |= ~(self._years >= filters.year_min)
            if filters.year_max is not None:
                mask |= ~(self._years <= filters.year_max)
            if filters.min_ratings is not None:
                mask |= self._n_ratings < filters.min_ratings
            if filters.max_ratings is not None:
                mask |= self._n_ratings > filters.max_ratings
        return mask

    # --- normalization ------------------------------------------------------------------------------
    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        if self.config.normalization == "rank":
            order = np.lexsort((np.arange(len(values)), values))
            ranks = np.empty(len(values))
            ranks[order] = np.arange(len(values))
            return ranks / max(len(values) - 1, 1)
        lo, hi = float(values.min()), float(values.max())
        if hi - lo <= 1e-12:
            return np.zeros_like(values) if hi <= 0 else np.ones_like(values)
        return (values - lo) / (hi - lo)

    # --- main entry point ------------------------------------------------------------------------------
    def rank(self, profile: UserProfile, k: int = 20, offset: int = 0,
             filters: RecommendationFilters | None = None, explain: bool = True) -> list[RankedItem]:
        cfg = self.config
        need = offset + k
        mask = self.filter_mask(profile, filters)
        weights = self.effective_weights(profile)
        raw = self.raw_signals(profile)

        # candidate generation: union of per-source top-N
        cand_set: set[int] = set()
        n_src = max(cfg.candidates_per_source, need)
        for name in ("content", "collaborative", "latent", "popularity", "preference"):
            if weights.get(name, 0.0) > 0:
                cand_set.update(topk_indices(raw[name], n_src, mask).tolist())
        if not cand_set:
            cand_set.update(topk_indices(raw["popularity"], n_src, mask).tolist())
        cands = np.asarray(sorted(cand_set), dtype=np.int64)
        if len(cands) == 0:
            return []

        # features + normalization + weighted score
        norm = {name: self._normalize(raw[name][cands]) for name in SIGNALS}
        score = np.zeros(len(cands))
        for name in SIGNALS:
            score += weights[name] * norm[name]

        # business rule: quality floor on Bayesian-average rating (only where evidence exists)
        if cfg.quality_floor:
            bayes = self.popularity.bayes_rating[cands]
            support = self.popularity.user_counts[cands]
            bad = (bayes < cfg.quality_floor) & (support >= 5)
            score[bad] = -np.inf

        valid = np.isfinite(score)
        cands, score = cands[valid], score[valid]
        norm = {n_: v[valid] for n_, v in norm.items()}
        order = np.lexsort((cands, -score))

        # diversity: MMR on the top pool
        pool_n = min(len(order), max(cfg.diversity_pool, need))
        pool = order[:pool_n]
        selected = self._mmr(cands[pool], score[pool], need) if cfg.diversity_lambda < 1.0 else \
            list(range(len(pool)))
        chosen = [pool[i] for i in selected]
        if cfg.max_per_genre:
            chosen = self._genre_cap(chosen, cands, need)

        results: list[RankedItem] = []
        for pos in chosen[offset:need]:
            item = int(cands[pos])
            sig = {name: {"raw": float(raw[name][item]), "normalized": float(norm[name][pos]),
                          "weight": float(weights[name]),
                          "contribution": float(weights[name] * norm[name][pos])} for name in SIGNALS}
            dominant = max(SIGNALS, key=lambda s: (sig[s]["contribution"], -SIGNALS.index(s)))
            ri = RankedItem(item=item, score=float(score[pos]), signals=sig, dominant=dominant)
            if explain:
                ri.contributions = self.contributions(profile, item, sig)
            results.append(ri)
        return results

    def _mmr(self, items: np.ndarray, rel: np.ndarray, k: int) -> list[int]:
        lam = self.config.diversity_lambda
        lo, hi = rel.min(), rel.max()
        r = (rel - lo) / (hi - lo) if hi > lo else np.ones_like(rel)
        sims = self.content.pairwise(items)
        selected: list[int] = []
        max_sim = np.zeros(len(items))
        available = np.ones(len(items), dtype=bool)
        for _ in range(min(k, len(items))):
            mmr = lam * r - (1.0 - lam) * max_sim
            mmr[~available] = -np.inf
            best = int(np.argmax(mmr))  # argmax takes the first max → deterministic
            selected.append(best)
            available[best] = False
            max_sim = np.maximum(max_sim, sims[best])
        return selected

    def _genre_cap(self, chosen: list[int], cands: np.ndarray, need: int) -> list[int]:
        cap = max(1, int(np.ceil((self.config.max_per_genre or 1.0) * min(need, 20))))
        counts: dict[str, int] = {}
        head, tail = [], []
        for pos in chosen:
            g = self._primary_genre[int(cands[pos])]
            if counts.get(g, 0) < cap:
                counts[g] = counts.get(g, 0) + 1
                head.append(pos)
            else:
                tail.append(pos)
        return head + tail

    def contributions(self, profile: UserProfile, item: int,
                      sig: dict[str, dict[str, float]]) -> dict[str, list[Contribution]]:
        out: dict[str, list[Contribution]] = {}
        if sig["content"]["contribution"] > 0:
            out["content"] = self.content.explain(profile, item)
        if sig["collaborative"]["contribution"] > 0:
            out["collaborative"] = self.itemknn.explain(profile, item)
        if sig["latent"]["contribution"] > 0:
            out["latent"] = self.als.explain(profile, item)
        if sig["preference"]["contribution"] > 0:
            g = self.genre_preference_vector(profile)
            row = self.genre_matrix.getrow(item)
            prefs = [(self.genres[j], float(g[j])) for j in row.indices if g[j] > 0]
            prefs.sort(key=lambda t: (-t[1], t[0]))
            out["preference"] = [Contribution(kind="genre", label=name, value=v) for name, v in prefs[:2]]
        if sig["popularity"]["contribution"] > 0:
            out["popularity"] = [Contribution(
                kind="popularity", value=float(self.popularity.like_counts[item]),
                label=f"{int(self.popularity.like_counts[item])}")]
        if sig["recency"]["contribution"] > 0 and np.isfinite(self._years[item]):
            out["recency"] = [Contribution(kind="recency", value=float(self._years[item]),
                                           label=str(int(self._years[item])))]
        return out

    # --- similar items (item-to-item) --------------------------------------------------------------
    def similar_items(self, item: int, k: int = 12, content_weight: float = 0.6
                      ) -> list[tuple[int, float, dict[str, float]]]:
        """Blend content neighbours (metadata; works for unrated items) with co-occurrence."""
        c_idx, c_val = self.content.similar(item, 100)
        cf_row = self.itemknn.sim.getrow(item)
        scores: dict[int, dict[str, float]] = {}
        for i, v in zip(c_idx.tolist(), c_val.tolist(), strict=True):
            scores.setdefault(i, {"content": 0.0, "collaborative": 0.0})["content"] = v
        cf_max = float(cf_row.data.max()) if cf_row.nnz else 0.0
        for i, v in zip(cf_row.indices.tolist(), cf_row.data.tolist(), strict=True):
            scores.setdefault(i, {"content": 0.0, "collaborative": 0.0})["collaborative"] = \
                v / cf_max if cf_max > 0 else 0.0
        cw = content_weight if cf_row.nnz else 1.0
        ranked = sorted(
            ((i, cw * s["content"] + (1 - cw) * s["collaborative"], s) for i, s in scores.items()
             if i != item),
            key=lambda t: (-t[1], t[0]))
        return ranked[:k]

    def similar_to_vector(self, vec: sp.csr_matrix, k: int = 12) -> list[tuple[int, float]]:
        sims = np.asarray((self.content.item_vectors @ vec.T).toarray()).ravel()
        idx = topk_indices(sims, k)
        return [(int(i), float(sims[i])) for i in idx]

    def trending(self, k: int = 20, offset: int = 0, exclude: np.ndarray | None = None,
                 boost: dict[int, float] | None = None) -> list[tuple[int, float]]:
        """Time-decayed popularity, optionally boosted with live (in-app) activity counts."""
        t = self.popularity.trending.astype(np.float64).copy()
        t = t / t.max() if t.max() > 0 else t
        if boost:
            live = np.zeros(self.n_items)
            for i, v in boost.items():
                live[i] = v
            if live.max() > 0:
                t = 0.7 * t + 0.3 * (live / live.max())
        mask = np.zeros(self.n_items, dtype=bool)
        if exclude is not None and len(exclude):
            mask[exclude] = True
        idx = topk_indices(t, offset + k, mask)[offset:]
        return [(int(i), float(t[i])) for i in idx]
