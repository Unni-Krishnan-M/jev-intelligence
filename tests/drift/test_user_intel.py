"""Movie user intelligence: drift report, strategy decision, §10 shapes, scenarios, latency."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from jev_ml.core.drift import OK, AspectResult, DriftConfig, WindowSplit
from jev_ml.domains.movie.user_intel import (
    ASPECTS,
    STRATEGIES,
    StrategyConfig,
    build_strategy_profile,
    engine_with_overrides,
    strategy_decision,
    strategy_for,
    user_intelligence,
)
from jev_ml.domains.movie.user_scenario import project, user_preference_scenarios, validate_spec
from jev_ml.engine import Interaction, RecommendationEngine

DAY = 86400.0
T0 = 1_500_000_000.0
DRIFT = DriftConfig(recent_n=12, n_permutations=199, n_bootstrap=199)
EFFECTS = {
    "adapt_to_recent": {"delta_ndcg10": 0.02, "ci95": [0.01, 0.03], "n_users": 100},
    "explore": {"delta_ndcg10": -0.002, "ci95": [-0.005, 0.001], "n_users": 100},
}
ELIGIBLE = StrategyConfig(drift=DRIFT, evaluated_effects=EFFECTS)
NOT_ELIGIBLE = StrategyConfig(drift=DRIFT, evaluated_effects={})


@pytest.fixture(scope="module")
def engine(tiny_model_dir: Path) -> RecommendationEngine:
    return RecommendationEngine(tiny_model_dir)


def drifting_user() -> list[Interaction]:
    """30 days of Sci-Fi/Comedy liked, then 12 days of Horror rated low (one event per day)."""
    hist = [
        Interaction(m, "rating", 4.5, T0 + i * DAY) for i, m in enumerate([*range(1, 16), *range(16, 31)])
    ]
    recent = [Interaction(m, "rating", 2.0, T0 + (40 + i) * DAY) for i, m in enumerate(range(46, 58))]
    return hist + recent


def stable_user() -> list[Interaction]:
    rng = np.random.default_rng(5)
    ids = rng.permutation(np.r_[1:16, 16:31, 31:46])[:42]
    return [
        Interaction(int(m), "rating", float(rng.choice([3.5, 4.0, 4.5])), T0 + i * DAY)
        for i, m in enumerate(ids)
    ]


def test_drifting_user_is_detected_and_adapted(engine: RecommendationEngine) -> None:
    out = user_intelligence(engine, drifting_user(), user_id=7, config=ELIGIBLE, k=5)
    d = out["drift"]
    assert d["status"] == OK and d["drift_detected"]
    sig = {a["aspect"] for a in d["aspects"] if a["significant"]}
    assert "genre_distribution" in sig
    s = out["strategy"]
    assert s["answer"] == "adapt_to_recent" and not s["abstained"]
    assert s["confidence_kind"] == "margin" and 0 < s["confidence"] <= 1
    assert all(
        r["decision_id"] == s["id"] and r["strategy"] == "adapt_to_recent" for r in out["recommendations"]
    )
    assert any(sg["dedup_key"].startswith("drift:genre_distribution") for sg in out["signals"])


def test_drift_without_evidence_keeps_standard(engine: RecommendationEngine) -> None:
    out = user_intelligence(engine, drifting_user(), config=NOT_ELIGIBLE, k=5)
    assert out["drift"]["drift_detected"]
    assert out["strategy"]["answer"] == "standard"
    assert any("no alternative strategy has shown a benefit" in r for r in out["strategy"]["rationale"])


def test_stable_user_standard(engine: RecommendationEngine) -> None:
    out = user_intelligence(engine, stable_user(), config=ELIGIBLE, k=5)
    assert out["drift"]["status"] == OK and not out["drift"]["drift_detected"]
    assert out["strategy"]["answer"] == "standard"


def test_abstains_on_small_profile(engine: RecommendationEngine) -> None:
    its = [Interaction(m, "rating", 4.0, T0 + m * 60) for m in range(1, 9)]
    out = user_intelligence(engine, its, config=ELIGIBLE, k=3)
    s = out["strategy"]
    assert out["drift"]["status"] == "insufficient_data"
    assert s["abstained"] and s["answer"] is None and s["confidence"] is None
    assert "serving standard" in s["fallback_reason"] and s["state"]["served_strategy"] == "standard"
    assert len(out["recommendations"]) == 3
    assert all(r["strategy"] == "standard" for r in out["recommendations"])


def _split() -> WindowSplit:
    rec = np.zeros(60, dtype=bool)
    rec[-20:] = True
    return WindowSplit(rec, OK, "", T0 + 60 * DAY, T0 + 40 * DAY, T0, T0 + 60 * DAY)


def _decide(p_genre: float, cfg: StrategyConfig, p_acc: float | None = None, acc_diff: float = -0.3) -> dict:
    aspects = [AspectResult(a, OK, "t", 0.1, 0.5, 0.5, False) for a in ASPECTS[:5]]
    aspects[0] = AspectResult("genre_distribution", OK, "t", 0.3, p_genre, p_genre, p_genre <= 0.05)
    if p_acc is not None:
        aspects.append(AspectResult("acceptance", OK, "t", acc_diff, p_acc, p_acc, p_acc <= 0.05))
    report = {"status": OK, "drift_detected": p_genre <= 0.05, "confidence": 1 - p_genre, "evidence": []}
    return strategy_decision(report, aspects, _split(), 60, {"mean": None}, cfg, "u", T0, 0.8)


def test_strategy_monotone_in_drift_evidence() -> None:
    ps = [0.9, 0.2, 0.06, 0.04, 0.01, 0.001, 1e-6]
    decs = [_decide(p, ELIGIBLE) for p in ps]
    adapt = [d["option_scores"]["adapt_to_recent"] for d in decs]
    assert adapt == sorted(adapt)
    answers = [d["answer"] for d in decs]
    assert answers[:3] == ["standard"] * 3 and answers[3:] == ["adapt_to_recent"] * 4
    for d in decs:
        assert set(d["options"]) == set(STRATEGIES) and math.isclose(
            sum(d["option_scores"].values()), 1, abs_tol=1e-3
        )
    # explore takes the drift route when only explore is eligible
    only_explore = StrategyConfig(drift=DRIFT, evaluated_effects={"explore": EFFECTS["explore"]})
    assert _decide(0.001, only_explore)["answer"] == "explore"
    # a significant acceptance drop supports explore even without preference drift
    assert _decide(0.9, NOT_ELIGIBLE, p_acc=0.001)["answer"] == "explore"
    assert _decide(0.9, NOT_ELIGIBLE, p_acc=0.001, acc_diff=0.3)["answer"] == "standard"


def test_small_evaluation_is_not_evidence() -> None:
    few = {k: {**v, "n_users": 7} for k, v in EFFECTS.items()}
    d = _decide(0.001, StrategyConfig(drift=DRIFT, evaluated_effects=few))
    assert d["answer"] == "standard"


def test_output_shape_matches_platform_section_10(engine: RecommendationEngine) -> None:
    fb = [{"timestamp": T0 + i * DAY, "feedback": "like" if i % 2 else "dislike"} for i in range(50)]
    out = user_intelligence(engine, drifting_user(), user_id=1, feedback=fb, config=ELIGIBLE, k=4)
    json.dumps(out, allow_nan=False)
    assert set(out) == {
        "user_id",
        "as_of",
        "profile",
        "preference_history",
        "drift",
        "strategy",
        "signals",
        "recommendations",
    }
    assert set(out["profile"]) == {"n_events", "first_event", "last_event"}
    ph = out["preference_history"]
    assert set(ph) == {"categories", "windows"}
    assert [w["label"] for w in ph["windows"][:2]] == ["historical", "recent"]
    for w in ph["windows"]:
        assert set(w) == {"label", "start", "end", "n", "shares"}
    d = out["drift"]
    assert set(d) == {
        "status",
        "drift_detected",
        "confidence",
        "confidence_kind",
        "historical_window",
        "recent_window",
        "aspects",
        "summary",
        "evidence",
    }
    assert set(d["historical_window"]) == {"start", "end", "n"}
    assert [a["aspect"] for a in d["aspects"]] == list(ASPECTS)
    for a in d["aspects"]:
        assert set(a) == {
            "aspect",
            "status",
            "test",
            "statistic",
            "p_value",
            "p_adjusted",
            "significant",
            "effect",
            "detail",
        }
    acc = d["aspects"][-1]
    assert acc["status"] == OK  # 50 feedback rows split by the window boundary
    s = out["strategy"]
    assert s["spec_id"] == "recommendation_strategy" and s["kind"] == "choice"
    for key in (
        "id",
        "policy_version",
        "question",
        "options",
        "answer",
        "option_scores",
        "confidence",
        "confidence_kind",
        "state",
        "rationale",
        "evidence",
        "abstained",
        "fallback_reason",
        "entity_type",
        "entity",
        "scale",
        "answer_interval",
        "batch_id",
    ):
        assert key in s
    for sg in out["signals"]:
        for key in (
            "id",
            "kind",
            "entity_type",
            "entity",
            "value",
            "unit",
            "strength",
            "direction",
            "source",
            "observed_at",
            "evidence",
            "entity_id",
            "baseline",
            "change",
            "confidence",
            "confidence_kind",
        ):
            assert key in sg
    for r in out["recommendations"]:
        assert {
            "item_id",
            "title",
            "rank",
            "score",
            "reason",
            "confidence",
            "decision_id",
            "evidence",
        } <= set(r)
        for e in r["evidence"]:
            assert set(e) == {"kind", "label", "value", "detail", "ref"}


def test_deterministic(engine: RecommendationEngine) -> None:
    a = user_intelligence(engine, drifting_user(), user_id=3, config=ELIGIBLE)
    b = user_intelligence(engine, drifting_user(), user_id=3, config=ELIGIBLE)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_strategy_for_matches_and_serves(engine: RecommendationEngine) -> None:
    its = drifting_user()
    dec, pk, co = strategy_for(engine, its, user_id=3, config=ELIGIBLE)
    full = user_intelligence(engine, its, user_id=3, config=ELIGIBLE)
    assert dec["id"] == full["strategy"]["id"] and dec["answer"] == full["strategy"]["answer"]
    assert pk["recency_half_life_days"] >= ELIGIBLE.adapt_half_life_min_days and co == {}
    prof = build_strategy_profile(engine, its, **pk)
    base = engine.build_profile(its)
    assert prof.weights[-1] / prof.weights[0] > base.weights[-1] / base.weights[0]
    ex = engine_with_overrides(engine, {"diversity_lambda": 0.6})
    assert ex.ranker.config.diversity_lambda == 0.6 and engine.ranker.config.diversity_lambda != 0.6


def test_empty_profile_decision_id_is_stable(
    engine: RecommendationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # with no events the reference time is the wall clock; the id must not change with it
    import jev_ml.domains.movie.user_intel as ui

    monkeypatch.setattr(ui.time, "time", lambda: 1_700_000_000.0)
    first, _, _ = strategy_for(engine, [], user_id=9)
    monkeypatch.setattr(ui.time, "time", lambda: 1_700_000_999.0)
    second, _, _ = strategy_for(engine, [], user_id=9)
    assert first["id"] == second["id"]


def test_strategy_for_latency_smoke(engine: RecommendationEngine) -> None:
    its = drifting_user()
    prof = engine.build_profile(its)
    strategy_for(engine, its, profile=prof)
    times = []
    for _ in range(15):
        t = time.perf_counter()
        strategy_for(engine, its, profile=prof)
        times.append(time.perf_counter() - t)
    assert np.percentile(times, 95) < 0.1


# --- scenarios -------------------------------------------------------------------------------------
def test_scenario_validation() -> None:
    assert validate_spec({})[0] == 10
    for bad in (
        {"k": 0},
        {"k": 51},
        {"k": "5"},
        {"scenarios": []},
        {"scenarios": [{"kind": "continue"}] * 5},
        {"scenarios": [{"kind": "boom"}]},
        {"scenarios": [{"kind": "accelerate", "factor": 1.0}]},
        {"scenarios": [{"kind": "reverse", "factor": 0}]},
        {"scenarios": [{"kind": "continue", "factor": float("nan")}]},
    ):
        with pytest.raises(ValueError):
            validate_spec(bad)


def test_projection_math() -> None:
    s_h, s_r = np.array([0.6, 0.3, 0.1]), np.array([0.4, 0.3, 0.3])
    rev = project(s_h, s_r, "reverse", 0.5)
    assert np.abs(rev - s_h).sum() < np.abs(s_r - s_h).sum()
    assert np.allclose(project(s_h, s_r, "reverse", 1.0), s_h)
    cont = project(s_h, s_r, "continue", 1.0)
    assert np.allclose(cont, [0.2, 0.3, 0.5])
    acc = project(s_h, s_r, "accelerate", 3.0)
    assert acc[0] == 0.0 and math.isclose(acc.sum(), 1.0)


def test_scenarios_payload(engine: RecommendationEngine) -> None:
    spec = {
        "k": 5,
        "scenarios": [
            {"name": "c", "kind": "continue"},
            {"name": "r", "kind": "reverse"},
            {"name": "a", "kind": "accelerate", "factor": 2.0},
        ],
    }
    out = user_preference_scenarios(engine, drifting_user(), spec, config=ELIGIBLE)
    json.dumps(out, allow_nan=False)
    assert set(out) == {"as_of", "assumptions", "uncertainty_note", "baseline", "scenarios", "evidence"}
    assert (
        set(out["baseline"]) == {"shares", "recommendations"} and len(out["baseline"]["recommendations"]) == 5
    )
    base = out["baseline"]["shares"]
    by = {s["name"]: s for s in out["scenarios"]}
    for s in out["scenarios"]:
        assert {
            "name",
            "kind",
            "assumptions",
            "projected_shares",
            "recommendations",
            "overlap_with_baseline",
        } <= set(s)
        assert 0 <= s["overlap_with_baseline"] <= 1
        for band in s["projected_shares"].values():
            assert band["lo80"] <= band["mean"] <= band["hi80"]
    # reverse moves back toward the historical mix: Horror (recent) down, Sci-Fi (historical) up
    assert by["r"]["projected_shares"]["Horror"]["mean"] < base["Horror"]
    assert by["c"]["projected_shares"]["Horror"]["mean"] >= base["Horror"]
    assert by["a"]["projected_shares"]["Horror"]["mean"] >= by["c"]["projected_shares"]["Horror"]["mean"]


def test_scenarios_without_change_reproduce_baseline(engine: RecommendationEngine) -> None:
    its = [Interaction(m, "rating", 4.0, T0 + m * 60) for m in range(1, 30)]  # one session: no trajectory
    out = user_preference_scenarios(engine, its, {"k": 5, "scenarios": [{"kind": "continue"}]})
    assert out["scenarios"][0]["overlap_with_baseline"] == 1.0
    assert "equals the current mix" in out["uncertainty_note"]


# --- real data -------------------------------------------------------------------------------------
def test_real_users_if_available() -> None:
    from jev_ml.data.dataset import load_interactions
    from jev_ml.paths import PROCESSED_DIR, ROOT

    models = ROOT / "models"
    if not (PROCESSED_DIR / "interactions.csv").exists() or not (models / "registry.json").exists():
        pytest.skip("real MovieLens data or model artifacts missing")
    try:
        engine = RecommendationEngine.load_active(models)
    except FileNotFoundError:
        pytest.skip("no active model")
    inter = load_interactions(PROCESSED_DIR / "interactions.csv")
    for uid in (318, 1):
        g = inter[inter["user_id"] == uid]
        its = [
            Interaction(int(m), "rating", float(r), float(t))
            for m, r, t in zip(g["movie_id"], g["rating"], g["timestamp"], strict=True)
        ]
        out = user_intelligence(engine, its, user_id=uid)
        json.dumps(out, allow_nan=False)
        assert out["profile"]["n_events"] == len(g)
        assert len(out["recommendations"]) == 10
        if uid == 1:  # every rating within 9 days: drift is not measurable
            assert out["drift"]["status"] == "insufficient_data" and out["strategy"]["abstained"]


def test_evaluated_effects_match_the_stored_drift_report():
    """EVALUATED_EFFECTS is a hand copy of the drift evaluation: pin it to the report it cites."""
    import json

    from jev_ml.domains.movie.user_intel import _EVAL_SOURCE, EVALUATED_EFFECTS
    from jev_ml.paths import ROOT

    path = ROOT / _EVAL_SOURCE
    if not path.exists():
        pytest.skip(f"{_EVAL_SOURCE} not present (gitignored experiment output)")
    rep = json.loads(path.read_text())
    stored = rep["adaptation"]["settings"]["refit"]["preference_drift"]["vs_standard"]
    for name, eff in EVALUATED_EFFECTS.items():
        s = stored[name]["ndcg10"]
        assert eff["delta_ndcg10"] == pytest.approx(s["delta"], abs=1e-9), name
        assert eff["ci95"] == pytest.approx(s["ci95"], abs=1e-9), name
        for k in ("n_users", "n_improved", "n_worse"):
            assert eff[k] == s[k], (name, k)
