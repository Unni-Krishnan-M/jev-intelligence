"""Retraining and model governance, ML side (no database): snapshots, registry states, the gate.

Runs on the synthetic fixture data (tests/conftest.py), in private models/experiments directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from jev_ml.governance import (
    GateConfig,
    build_snapshot,
    evaluate_candidate,
    promotion_blockers,
    train_candidate,
)
from jev_ml.governance.snapshot import PSEUDO_RATINGS, SEMANTICS
from jev_ml.paths import CONFIGS_DIR
from jev_ml.registry import (
    active_version,
    read_registry,
    register_version,
    rollback_target,
    set_active,
    set_state,
    version_state,
)
from jev_ml.signals import FAVORITE_WEIGHT, ONBOARDING_PICK_WEIGHT, rating_weight

OFFSET = 10_000_000
FAST_GATE = GateConfig(bootstrap_b=200, permutations=200, latency_requests=10, latency_p95_ms=2000)


def _events(rows: list[tuple[int, int, str, float | None, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["app_user_id", "movie_id", "kind", "value", "ts"])


APP_EVENTS = [
    (1, 1, "rating", 4.5, 2_000_000_000),
    (1, 1, "favorite", None, 2_000_000_100),  # the explicit rating wins
    (1, 2, "favorite", None, 2_000_000_010),
    (1, 3, "watch", None, 2_000_000_020),
    (1, 3, "watch", None, 2_000_000_030),  # repeated watch: one row, first watch time
    (1, 4, "like", None, 2_000_000_040),
    (1, 5, "dislike", None, 2_000_000_050),
    (1, 6, "watch", None, 2_000_000_060),
    (1, 6, "not_interested", None, 2_000_000_070),  # exclusion drops the watch
    (1, 7, "like", None, 2_000_000_080),
    (1, 7, "dislike", None, 2_000_000_090),  # the latest verdict counts
    (2, 8, "onboarding", None, 2_000_000_000),
    (2, 9, "clicked", None, 2_000_000_000),  # ignored
    (2, 99_999, "favorite", None, 2_000_000_000),  # not in the catalogue
]


@pytest.fixture(scope="module")
def gov_dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("gov-ml")
    return {k: root / k for k in ("models", "experiments", "snapshots", "configs")}


@pytest.fixture(scope="module")
def snapshot(processed_dir: Path, gov_dirs: dict[str, Path]) -> dict:
    return build_snapshot(processed_dir, _events(APP_EVENTS), gov_dirs["snapshots"], user_offset=OFFSET)


def _config(gov_dirs: dict[str, Path], name: str, **hybrid_overrides) -> Path:
    cfg = yaml.safe_load((CONFIGS_DIR / "experiment.yaml").read_text())
    cfg["models"]["hybrid"].update(hybrid_overrides)
    gov_dirs["configs"].mkdir(parents=True, exist_ok=True)
    path = gov_dirs["configs"] / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# --- snapshots -------------------------------------------------------------------------------------------
def test_pseudo_ratings_follow_signal_weights():
    assert rating_weight(PSEUDO_RATINGS["favorite"]) == FAVORITE_WEIGHT
    assert rating_weight(PSEUDO_RATINGS["onboarding"]) == ONBOARDING_PICK_WEIGHT
    assert PSEUDO_RATINGS["like"] >= 4.0 > PSEUDO_RATINGS["watch"] > PSEUDO_RATINGS["dislike"]
    assert SEMANTICS["version"]


def test_snapshot_includes_app_feedback(snapshot: dict, gov_dirs: dict[str, Path]):
    d = gov_dirs["snapshots"] / snapshot["snapshot_id"]
    app = pd.read_csv(d / "app_interactions.csv")
    got = {(int(r.app_user_id), int(r.movie_id)): (r.signal, float(r.rating)) for r in app.itertuples()}
    assert got == {
        (1, 1): ("rating", 4.5),
        (1, 2): ("favorite", 5.0),
        (1, 3): ("watch", 3.5),
        (1, 4): ("like", 4.0),
        (1, 5): ("dislike", 0.5),
        (1, 7): ("dislike", 0.5),
        (2, 8): ("onboarding", 4.5),
    }
    assert (app["user_id"] == app["app_user_id"] + OFFSET).all()
    assert int(app.loc[app["movie_id"] == 3, "timestamp"].iloc[0]) == 2_000_000_020
    excl = pd.read_csv(d / "exclusions.csv")
    assert list(zip(excl["user_id"], excl["movie_id"], strict=True)) == [(1 + OFFSET, 6)]
    inter = pd.read_csv(d / "interactions.csv")
    assert (inter["user_id"] >= OFFSET).sum() == 7
    m = snapshot
    assert m["row_counts"]["app"] == 7 and m["row_counts"]["exclusions"] == 1
    assert m["sources"]["app"]["dropped_unknown_movie"] == 1
    assert m["sources"]["app"]["events_clicked"] == 1
    assert m["cutoff_ts"] == 2_000_000_100  # defaults to the newest event
    meta = json.loads((d / "dataset_meta.json").read_text())
    assert meta["dataset_version"] == m["snapshot_id"]
    assert meta["base_dataset_version"] == "synthetic-test-v1"


def test_snapshot_is_deterministic(processed_dir: Path, snapshot: dict, gov_dirs: dict[str, Path]):
    shuffled = _events(APP_EVENTS).sample(frac=1.0, random_state=3)
    again = build_snapshot(processed_dir, shuffled, gov_dirs["snapshots"], user_offset=OFFSET)
    assert again["snapshot_id"] == snapshot["snapshot_id"]
    assert again["content_hash"] == snapshot["content_hash"]
    assert again["reused"] is True
    # a different cut-off is a different snapshot
    earlier = build_snapshot(
        processed_dir,
        _events(APP_EVENTS),
        gov_dirs["snapshots"],
        cutoff=pd.Timestamp(2_000_000_045, unit="s", tz="UTC").to_pydatetime(),
        user_offset=OFFSET,
    )
    assert earlier["snapshot_id"] != snapshot["snapshot_id"]
    assert earlier["sources"]["app"]["dropped_after_cutoff"] > 0


def test_snapshot_rejects_colliding_user_ids(processed_dir: Path, gov_dirs: dict[str, Path]):
    with pytest.raises(ValueError, match="offset"):
        build_snapshot(processed_dir, _events(APP_EVENTS), gov_dirs["snapshots"], user_offset=10)


# --- registry states ---------------------------------------------------------------------------------------
def _fake_version(models_dir: Path, name: str) -> None:
    (models_dir / name).mkdir(parents=True)
    (models_dir / name / "manifest.json").write_text(json.dumps({"version": name, "created_at": name}))


def test_registry_register_does_not_activate(tmp_path: Path):
    for v in ("a", "b", "c"):
        _fake_version(tmp_path, v)
    register_version({"version": "a"}, tmp_path)
    assert active_version(tmp_path) is None  # not even the first one (no implicit bootstrap)
    assert version_state("a", tmp_path) == "candidate"
    register_version({"version": "a"}, tmp_path, activate=True)  # explicit bootstrap
    register_version({"version": "b"}, tmp_path)
    assert active_version(tmp_path) == "a" and version_state("b", tmp_path) == "candidate"
    set_active("b", tmp_path, action="promote")
    assert version_state("a", tmp_path) == "retired" and rollback_target(tmp_path) == "a"
    register_version({"version": "c"}, tmp_path)
    set_active("c", tmp_path, action="promote")
    assert rollback_target(tmp_path) == "b"
    set_active("b", tmp_path, action="rollback")
    assert active_version(tmp_path) == "b" and version_state("c", tmp_path) == "retired"
    assert rollback_target(tmp_path) == "a"  # walks back, does not toggle to c
    set_state("c", "rejected", tmp_path, reason="test")
    assert read_registry(tmp_path)["states"]["c"] == "rejected"
    with pytest.raises(ValueError):
        set_state("b", "retired", tmp_path)  # the active one


def test_v1_registry_is_read_with_states(tmp_path: Path):
    (tmp_path / "registry.json").write_text(json.dumps({"active": "x", "versions": ["w", "x"]}))
    reg = read_registry(tmp_path)
    assert reg["states"] == {"w": "retired", "x": "active"}


# --- training + gate ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(snapshot: dict, gov_dirs: dict[str, Path]) -> dict[str, dict]:
    """An incumbent, an equivalent candidate (same recipe) and a deliberately broken candidate."""
    snap_dir = gov_dirs["snapshots"] / snapshot["snapshot_id"]
    md, ed = gov_dirs["models"], gov_dirs["experiments"]
    good = _config(gov_dirs, "good")
    inc = train_candidate(snap_dir, good, quick=True, models_dir=md, experiments_dir=ed)
    set_active(inc["version"], md, action="activate")  # the bootstrap model is activated explicitly
    same = train_candidate(
        snap_dir, good, quick=True, models_dir=md, experiments_dir=ed, lineage_extra={"job_id": "j1"}
    )
    # broken: popularity-only ranking ignores every member's taste
    broken_cfg = _config(
        gov_dirs,
        "broken",
        weights={
            "content": 0,
            "collaborative": 0,
            "latent": 0,
            "popularity": 1,
            "preference": 0,
            "recency": 0,
        },
    )
    broken = train_candidate(snap_dir, broken_cfg, quick=True, models_dir=md, experiments_dir=ed)
    return {"incumbent": inc, "same": same, "broken": broken}


def test_candidate_is_never_auto_activated(trained: dict, gov_dirs: dict[str, Path]):
    md = gov_dirs["models"]
    assert active_version(md) == trained["incumbent"]["version"]
    for key in ("same", "broken"):
        assert version_state(trained[key]["version"], md) == "candidate"


def test_lineage_is_complete(trained: dict, snapshot: dict):
    lin = trained["same"]
    for key in (
        "snapshot_id",
        "snapshot_content_hash",
        "config_hash",
        "seed",
        "experiment_run",
        "metrics",
        "quick",
    ):
        assert lin.get(key) is not None, key
    assert "git_commit" in lin
    assert lin["snapshot_id"] == snapshot["snapshot_id"]
    assert lin["dataset_version"] == snapshot["snapshot_id"]
    assert lin["job_id"] == "j1"
    assert lin["incumbent_at_training"] == trained["incumbent"]["version"]
    assert lin["metrics"]["test_hybrid"]["ndcg@10"] > 0


def test_gate_passes_an_equivalent_model(trained: dict, gov_dirs: dict[str, Path], snapshot: dict):
    snap_dir = gov_dirs["snapshots"] / snapshot["snapshot_id"]
    inc = trained["incumbent"]["version"]
    res = evaluate_candidate(trained["same"]["version"], inc, snap_dir, FAST_GATE, gov_dirs["models"])
    assert res["passed"], res["reasons"]
    g = res["gates"]
    assert g["ndcg@10"]["diff"] == pytest.approx(0.0, abs=1e-12)
    assert g["artifact"]["status"] == "pass" and g["latency"]["status"] == "pass"
    assert g["coverage@10"]["status"] == "pass" and g["cold_start"]["status"] == "pass"
    assert g["calibration"]["status"] == "skipped"
    assert res["split"]["test_fingerprint"]
    assert promotion_blockers(trained["same"]["version"], "candidate", res, inc) == []


def test_gate_rejects_a_worse_model(trained: dict, gov_dirs: dict[str, Path], snapshot: dict):
    snap_dir = gov_dirs["snapshots"] / snapshot["snapshot_id"]
    inc = trained["incumbent"]["version"]
    res = evaluate_candidate(trained["broken"]["version"], inc, snap_dir, FAST_GATE, gov_dirs["models"])
    assert not res["passed"]
    assert res["gates"]["ndcg@10"]["status"] == "fail"
    assert res["gates"]["ndcg@10"]["diff"] < 0
    assert any("ndcg@10" in r for r in res["reasons"])
    assert promotion_blockers(trained["broken"]["version"], "rejected", res, inc)


def test_gate_fails_a_broken_artifact(trained: dict, gov_dirs: dict[str, Path], snapshot: dict):
    import shutil

    md = gov_dirs["models"]
    src = md / trained["broken"]["version"]
    bad = md / "jev-broken-artifact"
    shutil.copytree(src, bad)
    manifest = json.loads((bad / "manifest.json").read_text())
    manifest["version"] = bad.name
    (bad / "manifest.json").write_text(json.dumps(manifest))
    (bad / "hybrid.json").write_text("{not json")
    res = evaluate_candidate(
        bad.name,
        trained["incumbent"]["version"],
        gov_dirs["snapshots"] / snapshot["snapshot_id"],
        FAST_GATE,
        md,
    )
    assert not res["passed"] and res["gates"]["artifact"]["status"] == "fail"
    assert set(res["gates"]) == {"artifact"}


def test_gate_is_stale_after_the_incumbent_changes(trained: dict):
    gate = {"passed": True, "reasons": [], "incumbent": "old"}
    assert promotion_blockers("v", "candidate", gate, "new")
    assert promotion_blockers("v", "candidate", None, "new") == ["no gate result: evaluate the version first"]
    assert promotion_blockers("v", "active", gate, "v")


def test_gate_margins_follow_the_power_rule():
    """An equivalent candidate passes each accuracy gate ~80 % of the time at the observed SEs."""
    from jev_ml.governance.gates import pass_probability_if_equivalent, power_margin

    cfg = GateConfig()
    # paired SEs observed on the MovieLens split (models/jev-20260924T174707Z-0298b516/gate.json)
    for se, margin in (
        (0.0039, cfg.ndcg_margin),
        (0.0044, cfg.recall_margin),
        (0.0022, cfg.cold_start_margin),
    ):
        assert margin == pytest.approx(power_margin(se), abs=0.001)
        assert pass_probability_if_equivalent(se, margin) >= 0.78
    # the old 0.005 NDCG margin: an equal model passed only about a third of the time
    assert pass_probability_if_equivalent(0.0039, 0.005) == pytest.approx(0.36, abs=0.02)
    assert cfg.quick().bootstrap_b >= 1000


def test_gate_records_its_power(trained: dict, gov_dirs: dict[str, Path], snapshot: dict):
    snap_dir = gov_dirs["snapshots"] / snapshot["snapshot_id"]
    inc = trained["incumbent"]["version"]
    res = evaluate_candidate(trained["broken"]["version"], inc, snap_dir, FAST_GATE, gov_dirs["models"])
    g = res["gates"]["ndcg@10"]
    assert g["se"] > 0 and 0 <= g["power_if_equivalent"] <= 1 and g["margin_for_80pct"] > 0
    # synthetic snapshot: the real raw tags must never be attached to other data
    assert res["split"]["tags"].startswith("all tags")
