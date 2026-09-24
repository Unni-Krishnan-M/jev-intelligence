"""Generic drift tests (jev_ml.core.drift): detection, false-positive rate, guards, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from jev_ml.core.drift import (
    INSUFFICIENT,
    OK,
    AspectResult,
    DriftConfig,
    categorical_shift,
    centroid_shift,
    holm,
    level_shift,
    median_shift,
    proportion_shift,
    rate_shift,
    split_windows,
    summarize,
)

CFG = DriftConfig(n_permutations=199, n_bootstrap=199)
CATS = ["a", "b", "c", "d"]


def onehot(idx: np.ndarray, k: int = 4) -> np.ndarray:
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def cat_data(rng: np.random.Generator, shift: bool, nh: int = 80, nr: int = 25):
    p_h = np.array([0.4, 0.3, 0.2, 0.1])
    p_r = np.array([0.05, 0.1, 0.25, 0.6]) if shift else p_h
    return onehot(rng.choice(4, nh, p=p_h)), onehot(rng.choice(4, nr, p=p_r))


def vec_data(rng: np.random.Generator, shift: bool, nh: int = 80, nr: int = 25):
    base = rng.normal(size=(2, 30))
    h = base[0] + 0.8 * rng.normal(size=(nh, 30))
    r = (base[1] if shift else base[0]) + 0.8 * rng.normal(size=(nr, 30))
    return h, r


RUNNERS = {
    "categorical": lambda rng, s, cfg: categorical_shift(*cat_data(rng, s), CATS, cfg, "genre"),
    "level": lambda rng, s, cfg: level_shift(
        rng.normal(3.5, 0.8, 80), rng.normal(2.5 if s else 3.5, 0.8, 25), cfg, "rating"
    ),
    "median": lambda rng, s, cfg: median_shift(
        rng.normal(1995, 8, 80), rng.normal(2010 if s else 1995, 8, 25), cfg, "year"
    ),
    "centroid": lambda rng, s, cfg: centroid_shift(*vec_data(rng, s), cfg, "content"),
    "rate": lambda rng, s, cfg: rate_shift(
        rng.poisson(0.2 * 300), 300, rng.poisson((0.6 if s else 0.2) * 60), 60, cfg, "rate"
    ),
    "proportion": lambda rng, s, cfg: proportion_shift(
        int(rng.binomial(80, 0.6)), 80, int(rng.binomial(40, 0.2 if s else 0.6)), 40, cfg, "acc"
    ),
}


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_detects_known_shift(name: str) -> None:
    res = RUNNERS[name](np.random.default_rng(1), True, CFG)
    assert res.status == OK
    assert res.p_value is not None and res.p_value < 0.01, res


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_false_positive_rate_near_alpha(name: str) -> None:
    cfg = DriftConfig(n_permutations=99, n_bootstrap=49)
    rng = np.random.default_rng(2)
    hits = [RUNNERS[name](rng, False, cfg).p_value <= 0.05 for _ in range(60)]
    # nominal 5 %; 60 repeats -> allow sampling noise (P(>=10 | p=.05) ~ 0.2 %)
    assert np.mean(hits) <= 0.15, np.mean(hits)


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_deterministic(name: str) -> None:
    a = RUNNERS[name](np.random.default_rng(3), True, CFG).to_dict()
    b = RUNNERS[name](np.random.default_rng(3), True, CFG).to_dict()
    assert a == b


def test_minimum_sample_guards() -> None:
    small = onehot(np.array([0, 1, 2]))
    assert categorical_shift(small, small, CATS, CFG).status == INSUFFICIENT
    assert level_shift(np.ones(3), np.ones(30), CFG).status == INSUFFICIENT
    assert median_shift(np.ones(30), np.ones(2), CFG).status == INSUFFICIENT
    assert centroid_shift(np.ones((5, 3)), np.ones((20, 3)), CFG).status == INSUFFICIENT
    assert rate_shift(3, 5.0, 2, 10.0, CFG).status == INSUFFICIENT
    assert proportion_shift(2, 5, 1, 30, CFG).status == INSUFFICIENT


def test_session_clusters_need_enough_sessions_and_are_conservative() -> None:
    rng = np.random.default_rng(4)
    h, r = cat_data(rng, True)
    few = categorical_shift(h, r, CATS, CFG, hist_clusters=np.arange(80) % 3, recent_clusters=np.zeros(25))
    assert few.status == INSUFFICIENT and "sessions" in few.detail
    ok = categorical_shift(h, r, CATS, CFG, hist_clusters=np.arange(80) // 2, recent_clusters=np.arange(25))
    assert ok.status == OK and ok.effect["permutation_unit"] == "session"
    assert ok.p_value is not None and ok.p_value < 0.01


def test_split_windows_last_n_days_and_guards() -> None:
    day = 86400.0
    ts = np.arange(60) * day
    s = split_windows(ts, DriftConfig(recent_n=20))
    assert s.status == OK and s.n_recent == 20 and s.recent[-20:].all()
    s2 = split_windows(ts, DriftConfig(recent_days=10.5))
    assert s2.n_recent == 11
    one_session = split_windows(np.arange(60) * 60.0, DriftConfig())
    assert one_session.status == INSUFFICIENT and "days" in one_session.detail
    # clusters: a day touched by the recent window joins it entirely
    ts3 = np.repeat(np.arange(30), 3) * day
    s3 = split_windows(ts3, DriftConfig(recent_n=20), clusters=(ts3 // day).astype(int))
    assert s3.n_recent == 21


def test_holm_and_summary() -> None:
    adj = holm([0.01, 0.04, None, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[2] is None
    assert adj[3] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    aspects = [
        AspectResult("x", OK, "t", 0.1, 0.001),
        AspectResult("y", OK, "t", 0.1, 0.5),
        AspectResult("z", INSUFFICIENT, "t"),
    ]
    head = summarize(aspects, CFG)
    assert head["drift_detected"] and head["confidence"] == pytest.approx(0.998)
    assert head["confidence_kind"] == "evidence"
    assert aspects[0].significant and not aspects[1].significant and aspects[2].p_adjusted is None
    assert summarize([AspectResult("z", INSUFFICIENT, "t")], CFG)["status"] == INSUFFICIENT
