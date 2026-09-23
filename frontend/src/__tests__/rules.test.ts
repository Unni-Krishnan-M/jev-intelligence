import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/api";
import {
  calibrationView,
  canTransition,
  groupByBatch,
  historyField,
  isNoRun,
  isNotDeployed,
  OPEN_STATUSES,
  ownerHref,
  refHref,
  riskHistoryKey,
  sourceHref,
  TRANSITIONS,
} from "@/lib/intel";
import type { DecisionRecord, HistoryPoint, WarningStatus } from "@/lib/intel-types";

describe("warning transitions", () => {
  it("allows the documented moves out of each open status", () => {
    expect(canTransition("new", "acknowledged")).toBe(true);
    expect(canTransition("new", "dismissed")).toBe(true);
    expect(canTransition("acknowledged", "investigating")).toBe(true);
    expect(canTransition("investigating", "resolved")).toBe(true);
  });

  it("never moves backwards or out of a terminal status", () => {
    expect(canTransition("investigating", "acknowledged")).toBe(false);
    expect(canTransition("acknowledged", "new")).toBe(false);
    for (const to of Object.keys(TRANSITIONS) as WarningStatus[]) {
      expect(canTransition("resolved", to)).toBe(false);
      expect(canTransition("dismissed", to)).toBe(false);
    }
  });

  it("treats exactly the statuses with a way out as open", () => {
    const withMoves = (Object.keys(TRANSITIONS) as WarningStatus[]).filter((s) => TRANSITIONS[s].length > 0);
    expect(withMoves.sort()).toEqual([...OPEN_STATUSES].sort());
  });
});

describe("no-run and not-deployed detection", () => {
  it("reads 404 and 503 from a run-backed endpoint as no run yet", () => {
    expect(isNoRun(new ApiError(404, "no intelligence run yet"))).toBe(true);
    expect(isNoRun(new ApiError(503, "unavailable"))).toBe(true);
    expect(isNoRun(new ApiError(500, "boom"))).toBe(false);
    expect(isNoRun(new Error("network"))).toBe(false);
  });

  it("reads only a route miss as a missing endpoint", () => {
    expect(isNotDeployed(new ApiError(404, "Not Found"))).toBe(true);
    expect(isNotDeployed(new ApiError(405, "Method Not Allowed"))).toBe(true);
    expect(isNotDeployed(new ApiError(404, "decision not found"))).toBe(false);
    expect(isNotDeployed(new ApiError(500, "Not Found"))).toBe(false);
  });
});

describe("evidence references → console links", () => {
  it("maps series ids to the series chart", () => {
    expect(refHref("share:genre:Drama")).toBe("/intel/series/share%3Agenre%3ADrama");
    expect(refHref("volume:all")).toBe("/intel/series/volume%3Aall");
  });

  it("maps object ids to their list anchors", () => {
    expect(refHref("sig-abc")).toBe("/intel/signals#sig-abc");
    expect(refHref("trend-1")).toBe("/intel/trends#trend-1");
    expect(refHref("anom-9")).toBe("/intel/anomalies#anom-9");
    expect(refHref("fc-2")).toBe("/intel/predictions#fc-2");
    expect(refHref("risk-3")).toBe("/intel/risks#risk-3");
    expect(refHref("act-4")).toBe("/intel/actions#act-4");
  });

  it("returns null for refs that point nowhere", () => {
    expect(refHref(null)).toBeNull();
    expect(refHref("")).toBeNull();
    expect(refHref("user 42")).toBeNull();
  });

  it("links evidence owners, preferring detail pages", () => {
    expect(ownerHref("decision", "dec-1")).toBe("/intel/decisions/dec-1");
    expect(ownerHref("warning", "12")).toBe("/intel/warnings/12");
    expect(ownerHref("warning", "risk:x")).toBe("/intel/warnings");
    expect(ownerHref("forecast", "fc-7")).toBe("/intel/predictions#fc-7");
    expect(ownerHref("forecast", "whatever")).toBe("/intel/predictions");
    expect(ownerHref("risk", "risk-5")).toBe("/intel/risks#risk-5");
    expect(ownerHref("signal", "")).toBeNull();
    expect(sourceHref({ type: "anomaly", id: "x1" })).toBe("/intel/anomalies#x1");
  });
});

function dec(db_id: number, run_id: string, batch_id: string | null): DecisionRecord {
  return {
    id: `dec-${db_id}`, key: "k", spec_id: "k", policy_version: "p", question: "q", kind: "boolean", options: ["yes", "no"],
    answer: "no", option_scores: {}, confidence: 0.5, confidence_kind: "margin", state: {}, rationale: [], evidence: [],
    abstained: false, fallback_reason: null, entity_type: "model", entity: "m", batch_id,
    db_id, run_id, as_of: "2018-09-24T00:00:00Z", created_at: "2026-09-23T00:00:00Z", feedback: { correct: 0, incorrect: 0 },
  };
}

describe("decision batches", () => {
  it("keeps a batch together at its first position and leaves singles alone", () => {
    const groups = groupByBatch([dec(1, "r1", "b1"), dec(2, "r1", null), dec(3, "r1", "b1"), dec(4, "r2", "b1")]);
    expect(groups.map((g) => g.items.map((d) => d.db_id))).toEqual([[1, 3], [2], [4]]);
    expect(groups.map((g) => g.batchId)).toEqual(["b1", null, "b1"]);
  });
});

describe("history and calibration helpers", () => {
  const pt = (score: number | null, value: number | null): HistoryPoint => ({ run_id: "r", as_of: "", created_at: "", value, score, level: null, direction: null });

  it("plots score when every run has one, else value", () => {
    expect(historyField([pt(1, 2), pt(3, null)])).toBe("score");
    expect(historyField([pt(1, 2), pt(null, 4)])).toBe("value");
    expect(historyField([pt(null, null)])).toBeNull();
    expect(historyField([])).toBeNull();
  });

  it("keys a risk across runs by kind and entity", () => {
    expect(riskHistoryKey({ kind: "genre_demand_decline", entity: "Horror" })).toBe("risk:genre_demand_decline:Horror");
  });

  it("prefers the primary held-out test block of a calibration", () => {
    const v = calibrationView({
      method: "isotonic_regression_on_score",
      fitted_on: { split: "validation" },
      validation: { ece: 0.001, brier: 0.1 },
      test: { next_10: { ece: 0.02, brier: 0.15, base_rate_brier: 0.16, n: 900, primary: true, reliability: [{ bin: "0.0-0.1", predicted: 0.05, observed: 0.04, n: 10 }] }, protocol: "…" },
    });
    expect(v?.split).toBe("test · next 10");
    expect(v?.ece).toBe(0.02);
    expect(v?.bins).toHaveLength(1);
    expect(v?.fittedOn).toBe("validation split");
    expect(calibrationView(null)).toBeNull();
    const h = calibrationView({ headline: { split: "test", stratum: "profile_full", ece: 0.00075, brier: 0.0179, n: 28350 }, test: { x: { ece: 1, brier: 1 } } });
    expect(h?.split).toBe("test · profile full");
    expect(h?.n).toBe(28350);
    expect(calibrationView({ validation: { ece: 0.01, brier: 0.1 } })?.split).toBe("validation (in-sample)");
  });
});
