"""Phase 2 (P2.4) decision-intelligence quality in the core: evidence lineage of every decision and
warning, determinism of decisions for the same as-of state, the change-point null under
autocorrelation, recovery-aware trends, confidence kinds and base-rate context in evaluations.

Real-data variants (MovieLens, FRED unemployment) skip when the data is not on this machine."""

from __future__ import annotations

import copy
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "intel"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains"))

from generic_golden import adapter as unemployment_adapter
from generic_golden import unemployment_frame
from intel_synth import make_series
from movie_golden import synthetic_variants

from jev_ml.core import run_domain
from jev_ml.core.config import CoreConfig
from jev_ml.core.decisions import CONFIDENCE_KINDS
from jev_ml.core.evaluation import confusion
from jev_ml.core.lineage import decision_lineage, warning_lineage
from jev_ml.core.trends import change_point, history_phi, trend_of
from jev_ml.domains import available, get_adapter
from jev_ml.domains.movie import MovieAdapter
from jev_ml.domains.movie.scenario import run_movie_pipeline

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def _real(key: str, as_of: str | None) -> dict:
    info = next((d for d in available() if d["key"] == key), None)
    if info is None or not info.get("available"):
        pytest.skip(f"{key} data not available: {info and info.get('reason')}")
    return run_domain(get_adapter(key), as_of=as_of, now=NOW).to_dict()


def _synthetic(tag: str) -> dict:
    if tag.startswith("movie:"):
        return run_movie_pipeline(synthetic_variants()[tag.split(":", 1)[1]]).to_dict()
    as_of = None if tag == "unemployment:default" else tag.split(":", 1)[1]
    return run_domain(unemployment_adapter(unemployment_frame()), as_of=as_of, now=NOW).to_dict()


def _assert_lineage(res: dict) -> None:
    assert res["decisions"], "a run without decisions proves nothing"
    for d in res["decisions"]:
        g = decision_lineage(res, d["id"])
        assert g is not None and g["complete"], (d["key"], d["id"], g and g["unresolved"])
        assert g["sources"] and g["root"]["key"] == d["key"]
        ids = {n["id"] for n in g["nodes"]}
        assert all(e["from"] in ids and e["to"] in ids for e in g["edges"])
        # every non-null evidence ref resolves to a node of the graph
        assert all(i["ref"] is None or i["resolves_to"] in ids for i in g["evidence"])
        # an object about a series reaches the series and, through it, the primary source
        for n in g["nodes"]:
            if n.get("series_id"):
                assert f"series:{n['series_id']}" in ids
    decision_ids = {d["id"] for d in res["decisions"]}
    for w in res["warnings"]:
        g = warning_lineage(res, w["key"])
        assert g is not None and g["complete"], (w["key"], g and g["unresolved"])
        assert w["decision_id"] in decision_ids and f"decision:{w['decision_id']}" in {
            n["id"] for n in g["nodes"]
        }


@pytest.mark.parametrize(
    "tag",
    [
        "movie:default",
        "movie:as_of_2013-06-15",
        "movie:full",
        "unemployment:default",
        "unemployment:2008-06-01",
        "unemployment:2020-05-01",
    ],
)
def test_lineage_synthetic(tag):
    _assert_lineage(_synthetic(tag))


@pytest.mark.parametrize(
    ("key", "as_of"),
    [
        ("movie", None),
        ("movie", "2017-07-01"),
        ("generic:us-unemployment", None),
        ("generic:us-unemployment", "2008-06-01"),
    ],
)
def test_lineage_real_runs(key, as_of):
    res = _real(key, as_of)
    _assert_lineage(res)
    if key != "movie":
        assert res["warnings"], "the real unemployment run should raise warnings to trace"


def test_lineage_flags_unresolved_refs():
    res = copy.deepcopy(_synthetic("movie:full"))
    d = next(x for x in res["decisions"] if x["evidence"])
    d["evidence"][0]["ref"] = "trend-doesnotexist"
    g = decision_lineage(res, d["id"])
    assert g is not None and not g["complete"]
    assert g["unresolved"][0]["ref"] == "trend-doesnotexist"
    assert decision_lineage(res, "dec-nope") is None and warning_lineage(res, "nope") is None


# --- determinism -----------------------------------------------------------------------------------------
def _decisions(res: dict) -> list[tuple]:
    return [
        (
            d["id"],
            d["key"],
            d["answer"],
            d["confidence"],
            d["confidence_kind"],
            d["policy_version"],
            d["abstained"],
        )
        for d in res["decisions"]
    ]


def _strip(res: dict) -> dict:
    out = copy.deepcopy(res)
    out["run"].pop("stage_ms")
    out["run"].pop("now")
    return out


@pytest.mark.parametrize("tag", ["movie:full", "movie:as_of_2013-06-15", "unemployment:2008-06-01"])
def test_same_as_of_state_gives_identical_decisions(tag):
    a, b = _synthetic(tag), _synthetic(tag)
    assert _decisions(a) == _decisions(b)
    assert [x["state_hash"] for x in a["decision_batches"]] == [
        x["state_hash"] for x in b["decision_batches"]
    ]
    assert all(x["state_hash"] for x in a["decision_batches"])
    assert a["run"]["config_hash"] == b["run"]["config_hash"]
    assert a["run"]["input_fingerprint"] == b["run"]["input_fingerprint"]
    assert _strip(a) == _strip(b)  # the whole result, not only the decisions


def test_replay_decisions_do_not_depend_on_the_wall_clock():
    """A replay (explicit as_of) gives the same decisions whenever it is run."""
    inp = synthetic_variants()["as_of_2013-06-15"]
    later = copy.copy(inp)
    later.now = datetime(2027, 3, 1, tzinfo=UTC)
    a = run_domain(MovieAdapter(inp), as_of=inp.as_of, now=inp.now)
    b = run_domain(MovieAdapter(later), as_of=later.as_of, now=later.now)
    assert _decisions(a.to_dict()) == _decisions(b.to_dict())


def test_input_fingerprint_changes_with_the_inputs():
    a, b = _synthetic("unemployment:2008-06-01"), _synthetic("unemployment:2020-05-01")
    assert a["run"]["input_fingerprint"] != b["run"]["input_fingerprint"]
    assert a["run"]["config_hash"] == b["run"]["config_hash"]


def test_replay_clock_for_app_events():
    """App events after as_of are invisible to a replay and live-window stages run on as_of."""
    inp = copy.copy(synthetic_variants()["full"])
    as_of = datetime(2016, 6, 1, tzinfo=UTC)
    replay = copy.copy(inp)
    replay.as_of = as_of
    res = run_domain(MovieAdapter(replay), as_of=as_of, now=inp.now).to_dict()
    app = next(s for s in res["data"]["sources"] if s["source"] == "app")
    assert app["rows"] == 0  # the feedback is from 2025-26: not yet recorded in 2016
    assert not any(a["kind"] == "live_feedback" for a in res["anomalies"])
    live = run_movie_pipeline(inp).to_dict()
    assert next(s for s in live["data"]["sources"] if s["source"] == "app")["rows"] > 0


# --- change points ----------------------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, phi: float, n: int) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.standard_normal() / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_change_point_null_is_calibrated_under_autocorrelation():
    """phi 0.56 (the real MovieLens volume): the history-estimated null keeps false alarms near the
    nominal 1 %, the in-window estimate (core-1.0.0) does not; detection of a 2-sigma shift survives."""
    rng = np.random.default_rng(2026)
    phi, n_hist, n, trials = 0.56, 60, 24, 300
    fa_hist = fa_win = det = 0
    for t in range(trials):
        full = _ar1(rng, phi, n_hist + n)
        hist, x = full[:n_hist], full[n_hist:]
        fa_hist += change_point(x, 4, 199, t, history=hist)["p_value"] < 0.01
        fa_win += change_point(x, 4, 199, t)["p_value"] < 0.01
        y = x.copy()
        y[12:] += 2.0 / np.sqrt(1 - phi**2)
        det += change_point(y, 4, 199, t, history=hist)["p_value"] < 0.01
    assert fa_hist / trials <= 0.03
    assert fa_win / trials >= 2 * fa_hist / trials and fa_win / trials >= 0.05
    assert det / trials >= 0.15


def test_change_point_uses_history_only_when_long_enough():
    rng = np.random.default_rng(1)
    x = _ar1(rng, 0.5, 24)
    assert change_point(x, 4, 99, 1, history=_ar1(rng, 0.5, 60))["phi_source"] == "history"
    assert change_point(x, 4, 99, 1, history=x[:10])["phi_source"] == "window"
    assert change_point(x, 4, 99, 1)["phi_source"] == "window"
    assert history_phi(np.full(30, 3.0), 24) == 0.0 and history_phi(None, 24) is None


# --- recovery-aware trends --------------------------------------------------------------------------------
def test_reversing_trend_earns_no_adverse_trend_points():
    rng = np.random.default_rng(5)
    rise = np.linspace(4, 9, 60)
    fall = np.array([8.0, 7.0, 6.2])  # the last periods fall away from the peak
    s = make_series(np.r_[rise, fall] + rng.normal(0, 0.05, 63), metric="level", sid="level:all")
    s.adverse_direction = "up"
    cfg = CoreConfig()
    t = trend_of(s, cfg, "k")
    assert t is not None and t["direction"] == "up"
    assert t["recent_move"]["reversing"] is True and t["recent_move"]["z"] < -cfg.trend_reversal_z
    off = trend_of(s, CoreConfig(trend_reversal_periods=0), "k")
    assert off is not None and off["recent_move"] is None
    steady = make_series(rise + rng.normal(0, 0.05, 60), metric="level", sid="level:all")
    st = trend_of(steady, cfg, "k")
    assert st is not None and st["recent_move"]["reversing"] is False

    from jev_ml.core.early_warning import build_situations
    from jev_ml.core.risk import trend_risks

    t["q_value"], t["confidence"] = 0.001, 0.999
    assert trend_risks([t], {s.id: s}, 1.0, cfg, "k") == []
    (sit,) = build_situations({s.id: s}, [t], [], [], {}, [], {}, None, cfg)
    comp = next(c for c in sit.components if c["stage"] == "trend")
    assert comp["points"] == 0.0 and comp["reversing"] is True and "reversing" in comp["detail"]


# --- confidence semantics ---------------------------------------------------------------------------------
def _confidences(o, path="$"):
    if isinstance(o, dict):
        if isinstance(o.get("confidence"), int | float) and not isinstance(o.get("confidence"), bool):
            yield path, o
        for k, v in o.items():
            if k != "config":
                yield from _confidences(v, f"{path}.{k}")
    elif isinstance(o, list):
        for v in o:
            yield from _confidences(v, f"{path}[]")


@pytest.mark.parametrize("tag", ["movie:full", "unemployment:2020-05-01"])
def test_every_emitted_confidence_has_a_kind(tag):
    res = _synthetic(tag)
    found = list(_confidences(res))
    assert found
    for path, o in found:
        assert o.get("confidence_kind") in CONFIDENCE_KINDS, (path, o.get("confidence_kind"))
    for w in res["warnings"]:
        if w["source"]["type"] == "risk":
            risk = next(r for r in res["risks"] if r["id"] == w["source"]["id"])
            assert w["confidence_kind"] == risk["confidence_kind"]


def test_confusion_carries_base_rate_and_lift():
    units = [{"warned": w, "confirmed": c} for w, c in [(1, 1), (1, 0), (0, 1), (0, 0), (0, 0), (1, 1)]]
    c = confusion([{**u, "warned": bool(u["warned"]), "confirmed": bool(u["confirmed"])} for u in units])
    assert c["precision"] == pytest.approx(2 / 3, abs=1e-4) and c["base_rate"] == 0.5
    assert c["lift"] == pytest.approx((2 / 3) / 0.5, abs=1e-3) and c["flag_rate"] == 0.5
    assert confusion([{"warned": False, "confirmed": False}])["lift"] is None


def test_warning_rows_and_month_cluster_lift_ci():
    """keep_rows stores every scored unit; the lift CI resamples whole replay months."""
    from jev_ml.core.evaluation import lift_uncertainty, warning_replay

    reps = [{"as_of": f"20{y:02d}-{m:02d}-01"} for y in (10, 11) for m in range(1, 13)]

    def units(r):
        return [{"kind": "k", "unit": f"u{i}", "warned": i < 2} for i in range(5)]

    def outcome(r, u):
        month = int(r["as_of"][5:7])
        if u["unit"] == "u4":
            return None, "unobservable"
        return (u["unit"] == "u0" and month % 2 == 0) or (u["unit"] == "u3" and month == 1), ""

    res = warning_replay(reps, units, outcome, keep_rows=True)
    assert len(res["rows"]) == 24 * 5
    assert set(res["rows"][0]) == {"as_of", "kind", "unit", "warned", "confirmed"}
    lu = res["lift_uncertainty"]
    assert lu["observable"] == 24 * 4 and lu["months"] == 24
    assert lu["lift"] == pytest.approx(res["overall"]["lift"], abs=1e-4)
    lo, hi = lu["lift_ci"]
    assert lo <= lu["lift"] <= hi
    assert lu["lift_year_stratified"] == pytest.approx(lu["lift"], rel=1e-3)  # same rate both years
    assert lift_uncertainty(res["rows"]) == lu  # deterministic
    assert "rows" not in warning_replay(reps, units, outcome)
