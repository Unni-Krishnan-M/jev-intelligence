"""Repair a local SQLite dev database whose schema does not match its alembic stamp.

    uv run python scripts/repair_dev_db.py [--db jev.db] [--dry-run]

Why. A dev database that the API auto-migrated while migrations 0006-0010 were still empty stubs
is stamped at head but misses the Phase-2 tables and some older columns, so ``alembic upgrade``
does nothing and ``alembic check`` reports drift. Stamping it back is unsafe here because its
schema matches no single revision (some 0005/0006 changes are present, others are not).

What it does (nothing is deleted):

1. copies the database to ``<db>.bak-<UTC ts>`` with the SQLite backup API;
2. builds a fresh database at head with ``alembic upgrade head`` next to it;
3. copies every row of every table the two share, column by column (columns the old schema lacks
   take their server default or NULL), with foreign keys off, then runs ``PRAGMA foreign_key_check``;
4. if a table cannot be copied (a constraint added since rejects an old row), the repair stops
   unless the table only holds derived data (intelligence runs and their objects, served
   recommendation logs, experiment metrics, audit rows), which the API can recompute; those are
   reported and left empty;
5. replaces the database with the rebuilt one (skipped with ``--dry-run``) and prints the row
   counts before and after.

User data (users, ratings, favorites, watch history, genre preferences, recommendation feedback,
member feedback) must copy completely, or nothing is replaced.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_TABLES = (
    "users",
    "ratings",
    "favorites",
    "watch_history",
    "user_genre_preferences",
    "recommendation_feedback",
    "user_intel_feedback",
)
DERIVED_PREFIXES = ("intel_", "recommendations", "evaluation_metrics", "experiments", "audit_logs")


def tables(con: sqlite3.Connection, schema: str = "main") -> list[str]:
    rows = con.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table' "  # noqa: S608 - schema is a literal
        "AND name NOT LIKE 'sqlite_%' AND name <> 'alembic_version' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def columns(con: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA {schema}.table_info("{table}")').fetchall()]


def counts(path: Path) -> dict[str, int]:
    con = sqlite3.connect(path)
    try:
        return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables(con)}  # noqa: S608
    finally:
        con.close()


def build_head(path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend/jev_api/migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(cfg, "head")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=ROOT / "jev.db")
    p.add_argument(
        "--dry-run", action="store_true", help="build and check the rebuilt copy, keep the old file"
    )
    args = p.parse_args()
    db = args.db.resolve()
    if not db.exists():
        print(f"no database at {db}")
        return 1
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = db.with_name(f"{db.name}.bak-{ts}")
    rebuilt = db.with_name(f"{db.name}.rebuild-{ts}")

    src = sqlite3.connect(db)
    dst = sqlite3.connect(backup)
    src.backup(dst)
    dst.close()
    src.close()
    print(f"backup: {backup}")
    before = counts(db)

    build_head(rebuilt)
    con = sqlite3.connect(rebuilt)
    con.execute("PRAGMA foreign_keys = OFF")
    con.execute("ATTACH DATABASE ? AS old", (str(db),))
    old_tables = set(tables(con, "old"))
    skipped: dict[str, str] = {}
    for t in tables(con):
        if t not in old_tables:
            continue
        common = [c for c in columns(con, t) if c in set(columns(con, t, "old"))]
        cols = ", ".join(f'"{c}"' for c in common)
        try:
            with con:
                con.execute(f'INSERT INTO main."{t}" ({cols}) SELECT {cols} FROM old."{t}"')  # noqa: S608
        except sqlite3.DatabaseError as exc:
            if t in USER_TABLES or not t.startswith(DERIVED_PREFIXES):
                con.close()
                rebuilt.unlink(missing_ok=True)
                print(f"ABORT: could not copy {t}: {exc}; {db} is unchanged (backup {backup})")
                return 2
            skipped[t] = str(exc)
    fk = con.execute("PRAGMA main.foreign_key_check").fetchall()
    con.execute("DETACH DATABASE old")
    con.close()
    after = counts(rebuilt)
    lost = {t: (n, after.get(t, 0)) for t, n in before.items() if t in USER_TABLES and after.get(t, 0) != n}
    if lost:
        rebuilt.unlink(missing_ok=True)
        print(f"ABORT: user rows would change {lost}; {db} is unchanged (backup {backup})")
        return 2

    print(f"{'table':32s} {'before':>8s} {'after':>8s}")
    for t in sorted(set(before) | set(after)):
        print(f"{t:32s} {before.get(t, '-')!s:>8s} {after.get(t, '-')!s:>8s}")
    for t, why in skipped.items():
        print(f"not copied (derived, recomputed by the API): {t}: {why}")
    if fk:
        print(f"foreign-key check: {len(fk)} orphan rows (first: {fk[0]})")
    if args.dry_run:
        print(f"dry run: rebuilt copy at {rebuilt}; {db} unchanged")
        return 0
    rebuilt.replace(db)
    print(f"repaired: {db} rebuilt at head; the original is {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
