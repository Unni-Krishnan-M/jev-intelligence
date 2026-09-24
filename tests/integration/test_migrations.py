"""CI guards for the Alembic chain (Phase 2 enabler, docs/PHASE2_ARCHITECTURE_AUDIT.md "Enabler — done").

- exactly one head and one base (parallel workstreams must not fork the chain);
- upgrade head -> downgrade base -> upgrade head, on SQLite and, when JEV_TEST_POSTGRES_URL is set, on
  PostgreSQL (the CI "postgres" job runs every test with "postgres" in its name);
- every CHECK constraint of a freshly migrated database equals the tuples in jev_api.models.enums;
- migration 0006: the intel_runs.mode backfill, the widened constraints and a lossless-where-possible
  downgrade.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

PG_URL = os.environ.get("JEV_TEST_POSTGRES_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="JEV_TEST_POSTGRES_URL not set")
QUOTED = re.compile(r"'((?:[^']|'')*)'")


def _cfg(url: str):
    from alembic.config import Config

    from jev_ml.paths import ROOT

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


@pytest.fixture
def engine_for() -> Iterator:
    engines: list[Engine] = []

    def make(url: str) -> Engine:
        eng = create_engine(url)
        engines.append(eng)
        return eng

    yield make
    for eng in engines:
        eng.dispose()


# --- chain shape ---------------------------------------------------------------------------------------
def test_single_alembic_head_and_base():
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_cfg("sqlite://"))
    heads, bases = script.get_heads(), script.get_bases()
    assert len(heads) == 1, f"multiple Alembic heads {heads}: merge or renumber (integrator)"
    assert len(bases) == 1, bases
    revisions = list(script.walk_revisions())
    # strictly linear: no merge points and no branch points
    assert all(isinstance(r.down_revision, str | None) for r in revisions)
    downs = [r.down_revision for r in revisions]
    assert len(downs) == len(set(downs)), "two revisions share a down_revision"
    # the revision id matches the file's number prefix (0007_events.py -> "0007")
    for r in revisions:
        assert r.path is not None and os.path.basename(r.path).startswith(f"{r.revision}_"), r.path


# --- round trips ---------------------------------------------------------------------------------------
def _roundtrip(url: str, engine_for) -> None:
    from alembic import command

    cfg = _cfg(url)
    command.downgrade(cfg, "base")  # a reused PostgreSQL database starts clean
    command.upgrade(cfg, "head")
    eng = engine_for(url)
    tables = set(inspect(eng).get_table_names())
    assert {"intel_runs", "audit_logs", "users"} <= tables
    command.downgrade(cfg, "base")
    assert set(inspect(eng).get_table_names()) <= {"alembic_version"}
    command.upgrade(cfg, "head")
    assert "mode" in {c["name"] for c in inspect(eng).get_columns("intel_runs")}


def test_migration_roundtrip_head_base_head_sqlite(tmp_path, engine_for):
    _roundtrip(f"sqlite:///{tmp_path / 'roundtrip.db'}", engine_for)


@needs_pg
def test_migration_roundtrip_head_base_head_postgres(engine_for):
    assert PG_URL is not None
    _roundtrip(PG_URL, engine_for)


# --- CHECK parity --------------------------------------------------------------------------------------
def _db_checks(eng: Engine) -> dict[str, tuple[str, str]]:
    insp = inspect(eng)
    out: dict[str, tuple[str, str]] = {}
    for table in insp.get_table_names():
        for ck in insp.get_check_constraints(table):
            assert ck["name"], f"unnamed CHECK constraint on {table}: {ck['sqltext']}"
            assert ck["name"] not in out, f"CHECK name {ck['name']} used twice"
            out[ck["name"]] = (table, ck["sqltext"])
    return out


def _check_parity(url: str, engine_for) -> None:
    from alembic import command

    from jev_api.models.enums import CHECK_ENUMS, FEEDBACK_VERDICTS, NON_ENUM_CHECKS

    cfg = _cfg(url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    checks = _db_checks(engine_for(url))
    unknown = set(checks) - set(CHECK_ENUMS) - NON_ENUM_CHECKS
    assert not unknown, f"CHECK constraints missing from enums.CHECK_ENUMS / NON_ENUM_CHECKS: {unknown}"
    missing = (set(CHECK_ENUMS) | NON_ENUM_CHECKS) - set(checks)
    assert not missing, f"enums.py lists constraints the migrated database lacks: {missing}"
    for name, (table, column, values, nullable) in CHECK_ENUMS.items():
        db_table, sql = checks[name]
        assert db_table == table, (name, db_table, table)
        assert column in sql, (name, sql)
        # PostgreSQL rewrites IN (...) to = ANY (ARRAY[...]) but keeps the literals and their order
        assert tuple(v.replace("''", "'") for v in QUOTED.findall(sql)) == values, (name, sql, values)
        assert ("IS NULL" in sql.upper()) == nullable, (name, sql)
    _, verdict_sql = checks["ck_intel_feedback_verdict"]
    expected = set(FEEDBACK_VERDICTS) | {v for vs in FEEDBACK_VERDICTS.values() for v in vs}
    assert set(QUOTED.findall(verdict_sql)) == expected, verdict_sql


def test_check_constraints_match_enums_sqlite(tmp_path, engine_for):
    _check_parity(f"sqlite:///{tmp_path / 'parity.db'}", engine_for)


@needs_pg
def test_check_constraints_match_enums_postgres(engine_for):
    assert PG_URL is not None
    _check_parity(PG_URL, engine_for)


def test_orm_checks_match_enums():
    """The ORM builds its constraints from enums.py; this pins the table/column mapping too."""
    from sqlalchemy import CheckConstraint

    from jev_api.db import Base
    from jev_api.models.enums import CHECK_ENUMS, NON_ENUM_CHECKS

    seen = {}
    for table in Base.metadata.tables.values():
        for c in table.constraints:
            if isinstance(c, CheckConstraint):
                seen[c.name] = (table.name, str(c.sqltext))
    assert set(seen) == set(CHECK_ENUMS) | NON_ENUM_CHECKS
    for name, (table, _column, values, _nullable) in CHECK_ENUMS.items():
        assert seen[name][0] == table
        assert tuple(QUOTED.findall(seen[name][1])) == values, name


def test_core_confidence_kinds_fit_the_decision_check():
    """Trap 2 (audit §4): a kind the core can emit but the CHECK rejects fails the whole run."""
    from jev_api.models.enums import CONFIDENCE_KINDS, DECISION_KINDS
    from jev_ml.core import decisions

    assert set(decisions.CONFIDENCE_KINDS) <= set(CONFIDENCE_KINDS)
    assert set(decisions.DECISION_KINDS) <= set(DECISION_KINDS)


# --- 0006 ----------------------------------------------------------------------------------------------
_NOW = "2026-09-01 00:00:00.000000"  # raw SQL: bound as text (sqlite3's datetime adapter is deprecated)


def _insert_run(db: Session, domain: str, requested: str | None, as_of: str | None, status: str) -> str:
    run_id = str(uuid.uuid4())
    db.execute(
        text(
            'INSERT INTO intel_runs (run_id, domain, "trigger", status, requested_as_of, as_of, started_at, '
            "pipeline_version, stage_ms) VALUES (:r, :d, 'manual', :s, :req, :a, :t, 'intel-1.2.0', '{}')"
        ),
        {"r": run_id, "d": domain, "s": status, "req": requested, "a": as_of, "t": _NOW},
    )
    return run_id


def _migration_0006(url: str, engine_for) -> None:
    from alembic import command

    cfg = _cfg(url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "0005")
    eng = engine_for(url)
    end, before = "2018-09-24 00:00:00.000000", "2017-01-01 00:00:00.000000"
    with Session(eng) as db:
        runs = {
            "live": _insert_run(db, "movie", None, end, "succeeded"),  # default as_of = the data end
            "replay": _insert_run(db, "movie", before, before, "succeeded"),
            "at_end": _insert_run(db, "movie", end, end, "succeeded"),  # explicit, but the current data
            "failed_replay": _insert_run(db, "movie", before, None, "failed"),  # judged by requested_as_of
            "no_reference": _insert_run(db, "synthetic", before, before, "succeeded"),  # data end unknown
        }
        db.commit()
    command.upgrade(cfg, "0006")
    with Session(eng) as db:
        modes = dict(db.execute(text("SELECT run_id, mode FROM intel_runs")).tuples().all())
    assert {k: modes[v] for k, v in runs.items()} == {
        "live": "live",
        "replay": "replay",
        "at_end": "live",
        "failed_replay": "replay",
        "no_reference": "live",
    }
    with Session(eng) as db, pytest.raises(IntegrityError):
        db.execute(text("UPDATE intel_runs SET mode = 'shadow'"))
        db.commit()

    decision = (
        "INSERT INTO intel_decisions (run_id, decision_id, domain, key, spec_id, policy_version, question, "
        "kind, options, option_scores, confidence_kind, state, rationale, evidence, abstained, as_of, "
        "created_at) VALUES (:r, :d, 'movie', 'k', 's', 'p', 'q', 'boolean', '[]', '{}', :ck, '{}', '[]', "
        "'[]', :f, :t, :t)"
    )
    audit = "INSERT INTO audit_logs (at, actor, action, detail) VALUES (:t, 'system', :a, '{}')"
    with Session(eng) as db:  # the widened constraints accept the Phase 2 values
        db.execute(text(decision), {"r": runs["live"], "d": "dec-1", "ck": "evidence", "f": False, "t": _NOW})
        db.execute(text(decision), {"r": runs["live"], "d": "dec-2", "ck": "margin", "f": False, "t": _NOW})
        for action in ("events.ingest", "model.promote", "experiment.start", "token.revoke", "intel.run"):
            db.execute(text(audit), {"t": _NOW, "a": action})
        db.commit()
    with Session(eng) as db, pytest.raises(IntegrityError):
        db.execute(text(audit), {"t": _NOW, "a": "not.an.action"})
        db.commit()

    command.downgrade(cfg, "0005")
    assert "mode" not in {c["name"] for c in inspect(eng).get_columns("intel_runs")}
    with Session(eng) as db:
        assert db.execute(text("SELECT count(*) FROM intel_runs")).scalar_one() == len(runs)  # runs stay
        kinds = db.execute(text("SELECT confidence_kind FROM intel_decisions")).scalars().all()
        assert kinds == ["margin"]  # the evidence decision is gone, the other stays
        actions = db.execute(text("SELECT action FROM audit_logs")).scalars().all()
        assert actions == ["intel.run"]
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")


def test_migration_0006_sqlite(tmp_path, engine_for):
    _migration_0006(f"sqlite:///{tmp_path / 'm0006.db'}", engine_for)


@needs_pg
def test_migration_0006_postgres(engine_for):
    assert PG_URL is not None
    _migration_0006(PG_URL, engine_for)


def test_new_runs_record_their_mode():
    from datetime import UTC, datetime

    from jev_api.models import run_mode

    assert run_mode(None) == "live"
    assert run_mode(datetime(2017, 1, 1, tzinfo=UTC)) == "replay"
