"""Retraining and model governance CLI (docs/RETRAINING_AND_MODEL_GOVERNANCE.md).

    uv run python scripts/retrain.py run [--quick | --full] [--config experiment.yaml]
        snapshot -> train -> register a CANDIDATE -> gate against the active model (never activates,
        unless JEV_GOVERNANCE_AUTO_PROMOTE=true and the gate passes)
    uv run python scripts/retrain.py evaluate VERSION [--quick]   gate an existing version
    uv run python scripts/retrain.py promote VERSION [--force --reason "..."]
    uv run python scripts/retrain.py rollback --reason "..."
    uv run python scripts/retrain.py status                       active, previous, states, last jobs
    uv run python scripts/retrain.py schedule [--once]            the scheduler loop (compose "scheduler")

Jobs take the same retrain lock as the API (PostgreSQL advisory lock / SQLite lease), so only one
runs at a time across processes. A running API follows promotions made here within
JEV_GOVERNANCE_ENGINE_POLL_SECONDS (it polls models/registry.json).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from typing import Any


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="snapshot, train a candidate and gate it")
    speed = run.add_mutually_exclusive_group()
    speed.add_argument("--quick", action="store_true", default=None, help="skip tuning (laptop/CI budget)")
    speed.add_argument("--full", action="store_true", help="tune hyper-parameters (minutes)")
    run.add_argument("--config", default=None, help="a YAML file name inside configs/")
    ev = sub.add_parser("evaluate", help="gate an existing version against the active model")
    ev.add_argument("version")
    ev.add_argument("--full", action="store_true")
    pr = sub.add_parser("promote", help="promote a version whose gate passed")
    pr.add_argument("version")
    pr.add_argument("--force", action="store_true", help="override a failing gate (needs --reason)")
    pr.add_argument("--reason", default=None)
    rb = sub.add_parser("rollback", help="restore the previously active version")
    rb.add_argument("--reason", required=True, help="why (audited; the API requires one too)")
    sub.add_parser("status", help="active model, states and the latest jobs")
    sc = sub.add_parser("schedule", help="run the scheduler loop")
    sc.add_argument("--once", action="store_true", help="one tick, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from sqlalchemy import select

    from jev_api.config import get_settings
    from jev_api.db import SessionLocal
    from jev_api.main import run_migrations
    from jev_api.models.governance import ModelGovernance, TrainingJob
    from jev_api.services.governance import GovernanceError, GovernanceService
    from jev_api.services.ml import EngineHolder
    from jev_api.services.sync import sync_model_versions
    from jev_ml.registry import read_registry

    settings = get_settings()
    if settings.auto_migrate:
        run_migrations(settings)
    # poll 0: this process loads a version only to verify it before pointing the registry at it
    svc = GovernanceService(settings, EngineHolder(settings.models_dir, poll_seconds=0))

    def job_out(job: TrainingJob) -> dict[str, Any]:
        return {
            c: getattr(job, c)
            for c in (
                "job_id",
                "kind",
                "trigger",
                "status",
                "snapshot_id",
                "model_version",
                "incumbent_version",
                "gate_passed",
                "promoted",
                "duration_ms",
                "error",
            )
        }

    try:
        with SessionLocal() as db:
            if args.cmd == "run":
                quick = not args.full
                job = svc.submit(
                    db,
                    kind="retrain",
                    trigger="cli",
                    actor="cli",
                    quick=quick,
                    config=args.config,
                    background=False,
                )
                db.refresh(job)
                gate = None
                if job.model_version:
                    row = db.scalar(
                        select(ModelGovernance).where(ModelGovernance.version == job.model_version)
                    )
                    gate = row.gate if row else None
                _print({"job": job_out(job), "steps": job.steps, "gate": gate})
                return 0 if job.status == "succeeded" else 1
            if args.cmd == "evaluate":
                job = svc.submit(
                    db,
                    kind="evaluate",
                    trigger="cli",
                    actor="cli",
                    quick=not args.full,
                    version=args.version,
                    background=False,
                )
                db.refresh(job)
                row = db.scalar(select(ModelGovernance).where(ModelGovernance.version == args.version))
                _print({"job": job_out(job), "gate": row.gate if row else None})
                return 0 if job.status == "succeeded" else 1
            if args.cmd == "promote":
                if args.force and not args.reason:
                    parser.error("--force needs --reason")
                _print(svc.promote(db, args.version, "cli", force=args.force, reason=args.reason))
                return 0
            if args.cmd == "rollback":
                _print(svc.rollback(db, "cli", reason=args.reason))
                return 0
            if args.cmd == "status":
                sync_model_versions(db)
                reg = read_registry(settings.models_dir)
                jobs = db.scalars(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(5)).all()
                _print(
                    {
                        "active": reg["active"],
                        "previous": reg["previous"],
                        "states": reg["states"],
                        "jobs": [job_out(j) for j in jobs],
                    }
                )
                return 0
    except GovernanceError as exc:
        _print({"error": str(exc), "detail": exc.detail})
        return 2
    if args.cmd == "schedule":
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        svc.run_scheduler(stop, once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
