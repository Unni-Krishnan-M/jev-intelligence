import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ConclusionBadge, GateStatusBadge } from "@/components/jev/ops/badges";
import { LabelBanner } from "@/components/jev/ops/label-banner";
import {
  conclusionWording,
  eventLagLevel,
  eventsAlert,
  expectedEventLagS,
  experimentFormProblems,
  fmtSeconds,
  fmtSrm,
  gateStatusMeta,
  gateVerdict,
  ingestLagLevel,
  isAutoResolve,
  labelBanner,
  labelKind,
  lifecycleActions,
  nonInferiority,
  orderedGates,
  pendingText,
  promoteControls,
  reasonProblem,
  srmRows,
} from "@/lib/ops";
import type { GateSummary } from "@/lib/ops-types";

import { renderUi } from "./render";

describe("gate status mapping", () => {
  it("maps each status to a word and tone", () => {
    expect(gateStatusMeta("pass")).toEqual({ label: "Pass", tone: "good" });
    expect(gateStatusMeta("fail")).toEqual({ label: "Fail", tone: "bad" });
    expect(gateStatusMeta("skipped")).toEqual({ label: "Skipped", tone: "neutral" });
    expect(gateStatusMeta(undefined).label).toBe("Not run");
  });

  it("renders icon + label, never colour alone", () => {
    const { container } = renderUi(<GateStatusBadge status="fail" />);
    expect(screen.getByText("Fail")).toBeInTheDocument();
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("orders gates as the gate runs them and reads non-inferiority numbers", () => {
    const g = (status: GateSummary["status"], numbers: GateSummary["numbers"] = {}): GateSummary => ({ status, reason: null, numbers });
    const order = orderedGates({ latency: g("pass"), "ndcg@10": g("fail"), artifact: g("pass"), zzz: g("skipped") }).map(([k]) => k);
    expect(order).toEqual(["artifact", "ndcg@10", "latency", "zzz"]);
    expect(nonInferiority(g("fail", { diff: -0.002, ci_lo: -0.0073, ci_hi: 0.003, margin: 0.005, ci_level: 0.9, p_value: 0.4, n_users: 594 }))).toMatchObject({ lo: -0.0073, margin: 0.005, level: 0.9, n: 594 });
    expect(nonInferiority(g("pass", { load_seconds: 0.15 }))).toBeNull();
  });

  it("flags a gate run against a model that no longer serves as stale", () => {
    expect(gateVerdict({ gate_passed: true, gated_against: "a", is_active: false }, "a")).toBe("pass");
    expect(gateVerdict({ gate_passed: true, gated_against: "a", is_active: false }, "b")).toBe("stale_pass");
    expect(gateVerdict({ gate_passed: false, gated_against: "a", is_active: false }, "a")).toBe("fail");
    expect(gateVerdict({ gate_passed: null, gated_against: null, is_active: false }, "a")).toBe("not_evaluated");
    expect(gateVerdict({ gate_passed: true, gated_against: null, is_active: false }, null)).toBe("pass");
  });
});

describe("promote and force", () => {
  const base = { is_active: false, state: "candidate" as const, promotable: false, blockers: ["ndcg@10: CI lower bound below -0.005"] };

  it("promote only when the API says promotable", () => {
    expect(promoteControls({ ...base, promotable: true, blockers: [] })).toEqual({ canPromote: true, canForce: false, why: [] });
    const blocked = promoteControls(base);
    expect(blocked.canPromote).toBe(false);
    expect(blocked.canForce).toBe(true);
    expect(blocked.why).toEqual(base.blockers);
  });

  it("offers neither on the serving version", () => {
    expect(promoteControls({ ...base, is_active: true, state: "active" })).toMatchObject({ canPromote: false, canForce: false });
  });

  it("gives a reason even when the API lists no blocker", () => {
    expect(promoteControls({ ...base, blockers: [] }).why.length).toBe(1);
  });

  it("requires a real reason for force and rollback", () => {
    expect(reasonProblem("")).toMatch(/required/);
    expect(reasonProblem("  ")).toMatch(/required/);
    expect(reasonProblem("ok")).toMatch(/at least 3/);
    expect(reasonProblem("x".repeat(501))).toMatch(/under 500/);
    expect(reasonProblem("incident 42")).toBeNull();
  });
});

describe("experiment conclusion wording", () => {
  const WINNER = /winner|\bwins?\b|\bwon\b|better|beats?/i;

  it("never implies a winner when inconclusive, even if a winner field leaked through", () => {
    for (const winner of [null, "recency"]) {
      const w = conclusionWording({ decision: "inconclusive", winner });
      expect(`${w.title} ${w.body}`).not.toMatch(WINNER);
      expect(`${w.title} ${w.body}`).not.toContain("recency");
      expect(w.tone).toBe("neutral");
    }
  });

  it("names the treatment only when shipping, and the control when keeping it", () => {
    expect(conclusionWording({ decision: "ship", winner: "recency" }).title).toBe("Ship recency");
    expect(conclusionWording({ decision: "keep_control", winner: "control" }).title).toBe("Keep the control");
  });

  it("the badge reads the same", () => {
    renderUi(<ConclusionBadge conclusion={{ decision: "inconclusive", winner: "diverse" }} />);
    expect(screen.getByText("Inconclusive")).toBeInTheDocument();
    expect(screen.queryByText(/diverse/)).toBeNull();
  });
});

describe("results label banner", () => {
  it("recognises an offline replay by data source or label", () => {
    expect(labelKind("offline_replay", "offline replay using held-out ratings, not live traffic")).toBe("replay");
    expect(labelKind("live", "OFFLINE REPLAY of logged data")).toBe("replay");
    expect(labelKind("live", "live traffic")).toBe("live");
    expect(labelKind(null, null)).toBe("live");
  });

  it("shouts a replay label and keeps its text", () => {
    const b = labelBanner("offline_replay", "offline replay using held-out ratings, not live traffic");
    expect(b.title).toBe("OFFLINE REPLAY USING HELD-OUT RATINGS, NOT LIVE TRAFFIC");
    expect(b.body).toMatch(/cannot measure/);
    expect(labelBanner("live", "live traffic").title).toBe("Live traffic");
  });

  it("renders the banner as a note", () => {
    renderUi(<LabelBanner dataSource="offline_replay" label="offline replay using held-out ratings, not live traffic" />);
    expect(screen.getByRole("note")).toHaveTextContent("OFFLINE REPLAY");
  });
});

describe("SRM formatting", () => {
  it("formats the test and the verdict", () => {
    expect(fmtSrm({ chi2: 1.3865, p_value: 0.49995, detected: false })).toBe("χ² 1.387 · p = 0.500 · no mismatch");
    expect(fmtSrm({ chi2: 22.1, p_value: 0.00001, detected: true })).toBe("χ² 22.100 · p = < 0.001 · mismatch detected");
    expect(fmtSrm({ chi2: null, p_value: null, detected: false })).toMatch(/not computed/);
  });

  it("pairs observed with expected per variant", () => {
    const rows = srmRows(["control", "diverse"], { observed: [198, 195], expected: [196.5, 196.5] });
    expect(rows[0]).toMatchObject({ name: "control", observed: 198, expected: 196.5 });
    expect(rows[1].deltaPct).toBeCloseTo(-0.00763, 4);
    expect(srmRows(["a"], { observed: [3], expected: null })[0].deltaPct).toBeNull();
  });
});

describe("events lag thresholds", () => {
  it("ingest: on time within an hour, late within a day, stale beyond", () => {
    expect(ingestLagLevel(5)).toBe("ok");
    expect(ingestLagLevel(3600)).toBe("ok");
    expect(ingestLagLevel(3601)).toBe("watch");
    expect(ingestLagLevel(86401)).toBe("stale");
    expect(ingestLagLevel(null)).toBe("none");
  });

  it("event time is judged against the domain's own expectation", () => {
    expect(expectedEventLagS("movie", "month")).toBe(3600);
    const monthly = expectedEventLagS("generic:us-unemployment", "month");
    expect(eventLagLevel(30 * 86400, monthly)).toBe("ok");
    expect(eventLagLevel(60 * 86400, monthly)).toBe("watch");
    expect(eventLagLevel(120 * 86400, monthly)).toBe("stale");
  });

  it("rejections are the alert", () => {
    const z = { accepted: 3, duplicates: 1, rejected: 0 };
    expect(eventsAlert({ today: z, last_7_days: z })).toBeNull();
    expect(eventsAlert({ today: { ...z, rejected: 2 }, last_7_days: { ...z, rejected: 2 } })).toBe("rejected_today");
    expect(eventsAlert({ today: z, last_7_days: { ...z, rejected: 1 } })).toBe("rejected_week");
  });

  it("formats durations and the pending count", () => {
    expect(fmtSeconds(5.2)).toBe("5 s");
    expect(fmtSeconds(720)).toBe("12 min");
    expect(fmtSeconds(3 * 3600)).toBe("3.0 h");
    expect(fmtSeconds(61 * 86400)).toBe("61 d");
    expect(pendingText(0)).toBe("Every event is in intelligence");
    expect(pendingText(1)).toBe("1 event not yet in intelligence");
    expect(pendingText(1200)).toBe("1,200 events not yet in intelligence");
  });
});

describe("experiment form and lifecycle", () => {
  const ok = { key: "mmr-06", name: "MMR", variants: [{ name: "control", is_control: true, weight: 1 }, { name: "diverse", is_control: false, weight: 1 }] };
  it("mirrors the API's create rules", () => {
    expect(experimentFormProblems(ok)).toEqual([]);
    expect(experimentFormProblems({ ...ok, key: "list" }).join()).toMatch(/reserved/);
    expect(experimentFormProblems({ ...ok, variants: ok.variants.map((v) => ({ ...v, is_control: true })) }).join()).toMatch(/Exactly one/);
    expect(experimentFormProblems({ ...ok, variants: [ok.variants[0], { ...ok.variants[1], name: "control" }] }).join()).toMatch(/unique/);
  });
  it("shows only allowed lifecycle actions, in order", () => {
    expect(lifecycleActions(["ramp", "stop", "pause"])).toEqual(["pause", "stop"]);
    expect(lifecycleActions(["start", "ramp", "update", "delete"])).toEqual(["start", "delete"]);
  });
  it("recognises a system auto-resolve", () => {
    expect(isAutoResolve({ actor: "system", note: "auto-resolved: not seen in 3 live runs" })).toBe(true);
    expect(isAutoResolve({ actor: "admin@example.com", note: "auto-resolved: manual" })).toBe(false);
  });
});
