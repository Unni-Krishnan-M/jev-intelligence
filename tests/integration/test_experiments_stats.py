"""Online-experiment statistics on known inputs (docs/EXPERIMENTATION.md, section 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jev_api.services import experiment_stats as st
from jev_api.services.experiments import bucket_of, enrolled, variant_index


def test_two_proportion_matches_the_textbook_formula():
    # 50/100 vs 65/100: pooled p = 0.575, SE = sqrt(.575 * .425 * (2/100)) = 0.069911, z = 2.14556
    out = st.two_proportion(50, 100, 65, 100, alpha=0.05)
    assert out["diff"] == pytest.approx(0.15)
    assert out["z"] == pytest.approx(2.1456, abs=1e-4)
    assert out["p_value"] == pytest.approx(math.erfc(2.145562 / math.sqrt(2)), abs=1e-5)
    assert out["p_value"] == pytest.approx(0.0319, abs=1e-4) and out["significant"] is True
    se = math.sqrt(0.5 * 0.5 / 100 + 0.65 * 0.35 / 100)
    assert out["ci"][0] == pytest.approx(0.15 - 1.959964 * se, abs=1e-5)
    assert out["ci"][1] == pytest.approx(0.15 + 1.959964 * se, abs=1e-5)
    assert out["relative_lift"] == pytest.approx(0.3)
    # a stricter alpha makes the same data non-significant
    assert st.two_proportion(50, 100, 65, 100, alpha=0.01)["significant"] is False


def test_two_proportion_degenerate_inputs_never_fabricate_an_effect():
    assert st.two_proportion(0, 0, 3, 10)["significant"] is False
    assert st.two_proportion(0, 0, 3, 10)["p_value"] == 1.0
    same = st.two_proportion(0, 50, 0, 50)  # no variance at all
    assert same["significant"] is False and same["p_value"] == 1.0 and same["diff"] == 0.0
    null = st.two_proportion(30, 100, 30, 100)
    assert null["z"] == 0.0 and null["p_value"] == pytest.approx(1.0) and not null["significant"]


def test_bootstrap_ci_covers_a_known_shift_and_is_reproducible():
    rng = np.random.default_rng(1)
    c = rng.normal(0.30, 0.1, 400)
    t = rng.normal(0.35, 0.1, 400)
    a = st.bootstrap_diff(c, t, 0.05, 2000, seed=7)
    b = st.bootstrap_diff(c, t, 0.05, 2000, seed=7)
    assert a == b  # seeded: the same report twice
    assert a["ci"][0] < 0.05 < a["ci"][1] and a["ci"][0] > 0
    assert a["significant"] is True and a["p_value"] < 0.01
    # the CI half-width is close to the normal-theory 1.96 * SE
    se = math.sqrt(c.var(ddof=1) / 400 + t.var(ddof=1) / 400)
    assert (a["ci"][1] - a["ci"][0]) / 2 == pytest.approx(1.96 * se, rel=0.15)
    null = st.bootstrap_diff(c, c + 0.0, 0.05, 1000, seed=3)
    assert null["diff"] == 0.0 and null["significant"] is False
    assert st.bootstrap_diff([1.0], [2.0])["significant"] is False  # too small to test


def test_srm_chi_square():
    ok = st.srm([550, 450], [1, 1])  # chi2 = 10, p = 0.00157: suspicious but above the 0.001 threshold
    assert ok["chi2"] == pytest.approx(10.0) and ok["p_value"] == pytest.approx(0.001565, abs=1e-5)
    assert ok["detected"] is False and ok["expected"] == [500.0, 500.0]
    bad = st.srm([600, 400], [1, 1])  # chi2 = 40
    assert bad["chi2"] == pytest.approx(40.0) and bad["detected"] is True
    weighted = st.srm([900, 100], [9, 1])  # exactly the configured 90/10 split
    assert weighted["chi2"] == pytest.approx(0.0) and weighted["detected"] is False
    assert st.srm([0, 0], [1, 1])["detected"] is False


def test_required_sample_size():
    # baseline 10 %, +10 % relative (to 11 %), alpha 0.05 two-sided, power 0.8
    n = st.required_sample_size(0.10, 0.10, 0.05, 0.8)
    expected = (1.959964 + 0.841621) ** 2 * (0.1 * 0.9 + 0.11 * 0.89) / 0.01**2
    assert n == math.ceil(expected) and 14_700 < n < 14_800
    assert st.required_sample_size(None, 0.1) is None
    assert st.required_sample_size(0.0, 0.1) is None


def test_ndcg_and_diversity():
    assert st.ndcg_at_k([1], 1) == pytest.approx(1.0)
    assert st.ndcg_at_k([2], 1) == pytest.approx(1 / math.log2(3))
    assert st.ndcg_at_k([], 2) == 0.0 and st.ndcg_at_k([], 0) == 0.0
    # two positives, one listed at rank 1, one missed by the list: DCG 1, IDCG 1 + 1/log2(3)
    assert st.ndcg_at_k([1], 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert st.ndcg_at_k([11], 1) == 0.0  # beyond the cut-off
    same = frozenset({"Drama"})
    assert st.intra_list_diversity([same, same]) == 0.0
    assert st.intra_list_diversity([frozenset({"A"}), frozenset({"B"})]) == 1.0
    assert st.intra_list_diversity([frozenset({"A", "B"}), frozenset({"B"})]) == pytest.approx(0.5)
    assert st.intra_list_diversity([same]) is None


def test_assignment_hash_is_deterministic_uniform_and_ramp_stable():
    salt = "fixed-salt"
    assert bucket_of(salt, 42) == bucket_of(salt, 42)  # no process state: same in every worker
    assert bucket_of(salt, 42) != bucket_of("other-salt", 42) or bucket_of(salt, 43) != bucket_of(
        "other-salt", 43
    )
    users = range(1, 20_001)
    split = np.bincount([variant_index(salt, u, [1.0, 1.0]) for u in users], minlength=2)
    assert abs(split[0] - 10_000) < 400  # ~50/50
    assert st.srm(split.tolist(), [1, 1])["detected"] is False
    weighted = np.bincount([variant_index(salt, u, [3.0, 1.0]) for u in users], minlength=2)
    assert abs(weighted[0] / 20_000 - 0.75) < 0.02
    # ramping 10 % -> 50 % -> 100 % only adds members, and nobody changes variant (the variant hash is
    # independent of the enrolment hash)
    at10 = {u for u in users if enrolled(salt, u, 10)}
    at50 = {u for u in users if enrolled(salt, u, 50)}
    assert at10 <= at50 and len(at10) == pytest.approx(2000, rel=0.1)
    assert all(enrolled(salt, u, 100) for u in users) and not any(enrolled(salt, u, 0) for u in users)
    # enrolment and variant are uncorrelated: the treatment share among the first 10 % is still ~50 %
    share = np.mean([variant_index(salt, u, [1.0, 1.0]) for u in at10])
    assert abs(share - 0.5) < 0.05


def test_aa_simulation_false_positive_rate():
    """Acceptance (P2.6): A/A tests on synthetic traffic are significant at alpha 0.05 about 5 % of the
    time. The literal "<= 10 of 200 runs" check is itself noisy: an exactly calibrated test passes it
    only ~58 % of the time (Binomial(200, 0.05)), so calibration is checked on more runs with a 3-sigma
    band: 2000 z-tests and 400 bootstraps."""
    rng = np.random.default_rng(2026)
    z_hits = 0
    for _ in range(2000):
        c = int((rng.random(500) < 0.2).sum())
        t = int((rng.random(500) < 0.2).sum())
        z_hits += st.two_proportion(c, 500, t, 500, 0.05)["significant"]
    assert 0.035 <= z_hits / 2000 <= 0.065, z_hits
    boot_hits = 0
    for run in range(400):
        gc, gt = rng.gamma(2.0, 0.1, 200), rng.gamma(2.0, 0.1, 200)  # skewed, like per-user NDCG
        boot_hits += st.bootstrap_diff(gc, gt, 0.05, 300, seed=run)["significant"]
    assert boot_hits / 400 <= 0.083, boot_hits
    print(f"A/A false-positive rate: z-test {z_hits / 2000:.3f}, bootstrap {boot_hits / 400:.3f}")
