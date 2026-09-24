"""Export a versioned training snapshot: MovieLens processed data + app feedback from the database.

    uv run python scripts/export_training_snapshot.py [--cutoff 2026-09-24T00:00:00Z]

Writes data/snapshots/<snapshot_id>/ (JEV_GOVERNANCE_SNAPSHOTS_DIR) and registers it in the
dataset_snapshots table. The same database and cut-off always give the same snapshot id; the default
cut-off is the newest app event. docs/RETRAINING_AND_MODEL_GOVERNANCE.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cutoff", default=None, help="ISO timestamp; app events after it are left out")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from jev_api.config import get_settings
    from jev_api.db import SessionLocal
    from jev_api.main import run_migrations
    from jev_api.services.governance import export_snapshot, snapshot_manifest

    settings = get_settings()
    if settings.auto_migrate:
        run_migrations(settings)
    cutoff = None
    if args.cutoff:
        cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
        cutoff = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=UTC)
    with SessionLocal() as db:
        row, new = export_snapshot(db, settings, cutoff=cutoff, actor="cli")
        manifest = snapshot_manifest(row) or {}
    print(
        json.dumps(
            {
                "snapshot_id": row.snapshot_id,
                "new": new,
                "path": row.path,
                "content_hash": row.content_hash,
                "cutoff": manifest.get("cutoff"),
                "row_counts": manifest.get("row_counts"),
                "app": (manifest.get("sources") or {}).get("app"),
                "watermark": manifest.get("watermark"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
