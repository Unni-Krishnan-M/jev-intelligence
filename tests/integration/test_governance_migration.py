"""Migration 0008 (model governance): upgrade -> downgrade to 0007 -> upgrade, on SQLite and PostgreSQL."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect
from test_migrations import _cfg

PG_URL = os.environ.get("JEV_TEST_POSTGRES_URL")
TABLES = {"dataset_snapshots", "training_jobs", "model_governance", "governance_locks"}


def _roundtrip(url: str) -> None:
    from alembic import command

    cfg = _cfg(url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    eng = create_engine(url)
    try:
        assert set(inspect(eng).get_table_names()) >= TABLES
        checks = {c["name"] for c in inspect(eng).get_check_constraints("training_jobs")}
        assert {"ck_training_job_kind", "ck_training_job_status", "ck_training_job_trigger"} <= checks
        command.downgrade(cfg, "0007")
        assert not TABLES & set(inspect(eng).get_table_names())
        command.upgrade(cfg, "head")
        assert set(inspect(eng).get_table_names()) >= TABLES
    finally:
        eng.dispose()


def test_migration_0008_roundtrip_sqlite(tmp_path):
    _roundtrip(f"sqlite:///{tmp_path / 'gov.db'}")


@pytest.mark.skipif(not PG_URL, reason="JEV_TEST_POSTGRES_URL not set")
def test_migration_0008_roundtrip_postgres():
    assert PG_URL is not None
    _roundtrip(PG_URL)
