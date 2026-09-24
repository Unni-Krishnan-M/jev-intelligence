/**
 * Every action the API writes to the audit log. This mirrors AUDIT_ACTIONS in
 * backend/jev_api/models/enums.py (AUDIT_ACTIONS_V12 + AUDIT_ACTIONS_PHASE2, in that order); no
 * endpoint exposes the list, so src/__tests__/audit-actions.test.ts reads the Python file and fails
 * when the two diverge.
 */
export const AUDIT_ACTIONS = [
  // v1.2
  "intel.run",
  "warning.transition",
  "feedback.create",
  "scenario.save",
  "model.activate",
  "auth.login.success",
  "auth.login.failure",
  "auth.register",
  "me.feedback",
  // Phase 2: events and ingestion
  "events.ingest",
  "events.replay",
  // retraining and model governance
  "model.register",
  "model.retrain",
  "model.promote",
  "model.reject",
  "model.rollback",
  "dataset.snapshot",
  // decision intelligence
  "warning.auto_resolve",
  // online experimentation
  "experiment.create",
  "experiment.start",
  "experiment.ramp",
  "experiment.pause",
  "experiment.stop",
  "experiment.conclude",
  // security
  "token.create",
  "token.revoke",
  "auth.logout",
  "auth.revoke_all",
  "auth.password_change",
  "auth.login.throttled",
  "user.role_change",
  // operations
  "retention.prune",
] as const;

export type AuditAction = (typeof AUDIT_ACTIONS)[number];

/** Filter groups for the audit page: the prefix before the first dot. */
export function auditGroups(actions: readonly string[] = AUDIT_ACTIONS): { group: string; actions: string[] }[] {
  const out = new Map<string, string[]>();
  for (const a of actions) {
    const g = a.split(".")[0];
    out.set(g, [...(out.get(g) ?? []), a]);
  }
  return [...out].map(([group, list]) => ({ group, actions: list }));
}

/** An action that records a refusal or failure, set apart by icon and word. */
export function isFailureAction(a: string): boolean {
  return a.endsWith(".failure") || a.endsWith(".throttled") || a === "model.reject";
}
