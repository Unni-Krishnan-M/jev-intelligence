"""Every closed vocabulary that a database CHECK constraint enforces, defined once.

The ORM models build their CHECK constraints from these tuples. Migrations keep literal copies on purpose
(a migration must not import the models), and ``tests/integration/test_migrations.py`` asserts that the
CHECK constraints of a freshly migrated database equal these tuples (``CHECK_ENUMS`` below).

Rules (docs/PHASE2_ARCHITECTURE_AUDIT.md, "Enabler — done"):
- integrator-owned: a new value goes through the integrator, bundled with the migration that widens the
  constraint, and a line in ``CHECK_ENUMS`` when a new constrained column appears;
- append only: never reorder or remove a value (the downgrade of the migration that added it does that).
"""

from __future__ import annotations

# the domain of rows created before migration 0005 (v1.2, docs/platform.md)
DEFAULT_DOMAIN = "movie"

# --- users / recommender ---------------------------------------------------------------------------
REC_FEEDBACK_KINDS = ("like", "dislike", "not_interested", "clicked")  # recommendation_feedback.feedback
REC_CONFIDENCE_KINDS = ("probability",)  # recommendations.confidence_kind (nullable)
ME_FEEDBACK_TARGETS = ("strategy", "recommendation")
ME_FEEDBACK_VERDICTS = ("accepted", "rejected")

# --- intelligence layer (docs/intelligence.md, section 5) -------------------------------------------
INTEL_RUN_TRIGGERS = ("startup", "manual", "script", "schedule")
INTEL_RUN_STATUSES = ("running", "succeeded", "failed")
# Phase 2 (migration 0006): a replay (explicit as_of) never touches live state; "latest" means latest live
INTEL_RUN_MODES = ("live", "replay")
INTEL_SEVERITIES = ("low", "medium", "high", "critical")
WARNING_STATUSES = ("new", "acknowledged", "investigating", "resolved", "dismissed")
WARNING_OPEN_STATUSES = ("new", "acknowledged", "investigating")
DECISION_KINDS = ("boolean", "choice", "score")
# = jev_ml.core.decisions.CONFIDENCE_KINDS ("evidence" added in migration 0006)
CONFIDENCE_KINDS = ("probability", "margin", "rule", "interval", "evidence")
TREND_DIRECTIONS = ("up", "down", "flat")
EVIDENCE_OWNERS = ("signal", "trend", "anomaly", "forecast", "risk", "decision", "warning", "action")
FEEDBACK_VERDICTS = {
    "decision": ("correct", "incorrect"),
    "warning": ("useful", "not_useful", "false_positive"),
    "action": ("useful", "not_useful"),
    "prediction": ("correct", "incorrect"),
}

# --- audit log (ck_audit_action) ----------------------------------------------------------------------
AUDIT_ACTIONS_V12 = (
    "intel.run",
    "warning.transition",
    "feedback.create",
    "scenario.save",
    "model.activate",
    "auth.login.success",
    "auth.login.failure",
    "auth.register",
    "me.feedback",  # v1.2: a member's verdict on a strategy decision or recommendation
)
# Phase 2: pre-registered in migration 0006 so no workstream has to migrate ck_audit_action
AUDIT_ACTIONS_PHASE2 = (
    # WS1 events and ingestion
    "events.ingest",
    "events.replay",
    # WS2 retraining and model governance
    "model.register",  # a trained model registered as a candidate
    "model.retrain",
    "model.promote",  # detail.passed: the gate result (a failing gate is audited too)
    "model.reject",
    "model.rollback",
    "dataset.snapshot",
    # WS4 decision intelligence
    "warning.auto_resolve",  # a stale warning resolved by the system after N live runs without its key
    # WS5 online experimentation
    "experiment.create",
    "experiment.start",
    "experiment.ramp",
    "experiment.pause",
    "experiment.stop",
    "experiment.conclude",
    # security hardening
    "token.create",
    "token.revoke",
    "auth.logout",
    "auth.revoke_all",
    "auth.password_change",
    "auth.login.throttled",  # the per-account login throttle tripped
    "user.role_change",  # admin promotion or demotion (invalidates the member's tokens)
    # operations
    "retention.prune",
)
AUDIT_ACTIONS = AUDIT_ACTIONS_V12 + AUDIT_ACTIONS_PHASE2

# --- CHECK parity registry -----------------------------------------------------------------------------
# constraint name -> (table, column, allowed values, NULL allowed). The parity test compares each with
# the migrated database; constraints that are not a plain "column IN (...)" are listed in
# NON_ENUM_CHECKS and tested on their own.
CHECK_ENUMS: dict[str, tuple[str, str, tuple[str, ...], bool]] = {
    "ck_feedback_kind": ("recommendation_feedback", "feedback", REC_FEEDBACK_KINDS, False),
    "ck_rec_confidence_kind": ("recommendations", "confidence_kind", REC_CONFIDENCE_KINDS, True),
    "ck_user_intel_feedback_target": ("user_intel_feedback", "target_type", ME_FEEDBACK_TARGETS, False),
    "ck_user_intel_feedback_verdict": ("user_intel_feedback", "verdict", ME_FEEDBACK_VERDICTS, False),
    "ck_intel_run_trigger": ("intel_runs", "trigger", INTEL_RUN_TRIGGERS, False),
    "ck_intel_run_status": ("intel_runs", "status", INTEL_RUN_STATUSES, False),
    "ck_intel_run_mode": ("intel_runs", "mode", INTEL_RUN_MODES, False),
    "ck_intel_warning_severity": ("intel_warnings", "severity", INTEL_SEVERITIES, False),
    "ck_intel_warning_status": ("intel_warnings", "status", WARNING_STATUSES, False),
    "ck_intel_warning_dismissed_severity": ("intel_warnings", "dismissed_severity", INTEL_SEVERITIES, True),
    "ck_intel_event_from": ("intel_warning_events", "from_status", WARNING_STATUSES, True),
    "ck_intel_event_to": ("intel_warning_events", "to_status", WARNING_STATUSES, False),
    "ck_intel_decision_kind": ("intel_decisions", "kind", DECISION_KINDS, False),
    "ck_intel_decision_confidence_kind": ("intel_decisions", "confidence_kind", CONFIDENCE_KINDS, False),
    "ck_intel_feedback_target": ("intel_feedback", "target_type", tuple(FEEDBACK_VERDICTS), False),
    "ck_intel_signal_direction": ("intel_signals", "direction", TREND_DIRECTIONS, True),
    "ck_intel_trend_direction": ("intel_trends", "direction", TREND_DIRECTIONS, True),
    "ck_intel_anomaly_severity": ("intel_anomalies", "severity", INTEL_SEVERITIES, True),
    "ck_intel_risk_level": ("intel_risks", "level", INTEL_SEVERITIES, True),
    "ck_intel_evidence_owner": ("intel_evidence", "owner_type", EVIDENCE_OWNERS, False),
    "ck_audit_action": ("audit_logs", "action", AUDIT_ACTIONS, False),
}
# CHECK constraints that are not a single IN list (range checks, the verdict-per-target rule)
NON_ENUM_CHECKS = frozenset({"ck_rating_range", "ck_intel_warning_occurrences", "ck_intel_feedback_verdict"})
