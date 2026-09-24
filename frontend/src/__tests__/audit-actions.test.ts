import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { AUDIT_ACTIONS, auditGroups, isFailureAction } from "@/lib/audit-actions";

/** The string literals of a Python tuple assignment `NAME = ( ... )` in enums.py. */
function pythonTuple(src: string, name: string): string[] {
  const m = new RegExp(`^${name}\\s*=\\s*\\(([\\s\\S]*?)^\\)`, "m").exec(src);
  if (!m) throw new Error(`${name} not found in enums.py`);
  return [...m[1].replace(/#.*$/gm, "").matchAll(/"([^"]+)"/g)].map((x) => x[1]);
}

describe("audit action parity with backend/jev_api/models/enums.py", () => {
  const enums = readFileSync(resolve(process.cwd(), "../backend/jev_api/models/enums.py"), "utf8");

  it("AUDIT_ACTIONS is still V12 + PHASE2", () => {
    expect(enums).toMatch(/^AUDIT_ACTIONS\s*=\s*AUDIT_ACTIONS_V12\s*\+\s*AUDIT_ACTIONS_PHASE2\s*$/m);
  });

  it("mirrors every action, in order, with nothing extra", () => {
    const backend = [...pythonTuple(enums, "AUDIT_ACTIONS_V12"), ...pythonTuple(enums, "AUDIT_ACTIONS_PHASE2")];
    expect(backend.length).toBeGreaterThan(20);
    expect([...AUDIT_ACTIONS]).toEqual(backend);
  });

  it("includes me.feedback and the Phase 2 actions the old list missed", () => {
    for (const a of ["me.feedback", "events.ingest", "model.promote", "model.rollback", "experiment.conclude", "warning.auto_resolve"]) expect(AUDIT_ACTIONS).toContain(a);
  });

  it("groups by prefix and marks refusals", () => {
    const groups = auditGroups().map((g) => g.group);
    expect(groups).toEqual([...new Set(groups)]);
    expect(groups).toContain("experiment");
    expect(auditGroups().flatMap((g) => g.actions)).toHaveLength(AUDIT_ACTIONS.length);
    expect(isFailureAction("auth.login.failure")).toBe(true);
    expect(isFailureAction("auth.login.throttled")).toBe(true);
    expect(isFailureAction("model.reject")).toBe(true);
    expect(isFailureAction("model.promote")).toBe(false);
  });
});
