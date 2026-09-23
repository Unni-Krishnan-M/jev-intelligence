"""Audit log: who did what, when (docs/intelligence.md, section 9.3).

`record` adds one audit_logs row to the caller's session, so the entry commits (or rolls back) with
the change it describes. Pass `commit=True` where nothing else is committed (a failed login).
Recording never raises: a failure is logged and counted, and the request carries on.

Never pass secrets. `detail` is still redacted by key (password, token, secret, ...) and by value
(bearer tokens, JWTs, URL credentials) before it is stored, as a second line of defence.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from jev_api.logging_setup import redact, request_id_var
from jev_api.metrics import metrics
from jev_api.models import AUDIT_ACTIONS, AuditLog, User

log = logging.getLogger(__name__)

SYSTEM = "system"
MAX_DETAIL_CHARS = 4000


def _clean(detail: dict[str, Any] | None) -> dict[str, Any]:
    out = redact(json.loads(json.dumps(detail or {}, default=str, allow_nan=False)))
    if len(json.dumps(out)) > MAX_DETAIL_CHARS:  # never let one entry grow without bound
        return {"truncated": True, "keys": sorted(out)[:50]}
    return out if isinstance(out, dict) else {"value": out}


def record(
    db: Session,
    action: str,
    actor: User | str | None,
    target_type: str | None = None,
    target_id: str | int | None = None,
    detail: dict[str, Any] | None = None,
    commit: bool = False,
) -> AuditLog | None:
    """Add an audit entry for `action` (one of AUDIT_ACTIONS). `actor` is a user, an email (for
    example the address a failed login attempted) or None for the system."""
    try:
        if action not in AUDIT_ACTIONS:
            raise ValueError(f"unknown audit action {action!r}")
        if isinstance(actor, User):
            actor_id, actor_name = actor.id, actor.email
        else:
            actor_id, actor_name = None, (actor or SYSTEM)
        entry = AuditLog(
            at=datetime.now(UTC),
            actor_user_id=actor_id,
            actor=str(actor_name)[:320],
            action=action,
            target_type=None if target_type is None else str(target_type)[:32],
            target_id=None if target_id is None else str(target_id)[:200],
            detail=_clean(detail),
            request_id=request_id_var.get(),
        )
        db.add(entry)
        if commit:
            db.commit()
    except Exception:
        if commit:
            db.rollback()
        metrics.inc("audit", "errors")
        log.exception("audit entry failed", extra={"extra_fields": {"action": action}})
        return None
    metrics.inc("audit_entries_by_action", action)
    return entry


def entry_out(e: AuditLog) -> dict[str, Any]:
    at = e.at if e.at.tzinfo else e.at.replace(tzinfo=UTC)
    return {
        "id": e.id,
        "at": at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actor_user_id": e.actor_user_id,
        "actor": e.actor,
        "action": e.action,
        "target_type": e.target_type,
        "target_id": e.target_id,
        "detail": e.detail or {},
        "request_id": e.request_id,
    }
