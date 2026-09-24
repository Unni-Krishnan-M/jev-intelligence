import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/api";
import { EARLY_WARNING_LEVELS, ewlStateGroups, isEarlyWarningDecision, levelIndex, levelScale, strategyLabel } from "@/lib/decisions";
import { capabilityState, DEFAULT_DOMAIN, intelHref, MOVIE_ONLY, normalizeDomain, switchDomainHref, withDomain } from "@/lib/domain";
import { isNotDeployed } from "@/lib/intel";
import type { DomainCapability, DriftAspect } from "@/lib/intel-types";
import { driftAspectView, driftHeadline, projectedRows, shareAxisMax, shareSegments, topCategories } from "@/lib/me-intel";

describe("domain param propagation", () => {
  it("adds domain to every API path, keeping the query and the hash", () => {
    expect(withDomain("/intel/status", "movie")).toBe("/intel/status?domain=movie");
    expect(withDomain("/intel/trends?limit=25&offset=0", "generic:us-unemployment")).toBe("/intel/trends?limit=25&offset=0&domain=generic%3Aus-unemployment");
    expect(withDomain("/intel/trends?domain=movie&limit=5", "generic:x")).toBe("/intel/trends?domain=generic%3Ax&limit=5");
    expect(withDomain("/intel/risks#risk-1", "movie")).toBe("/intel/risks?domain=movie#risk-1");
  });

  it("keeps a non-default domain on console links and leaves the default out", () => {
    expect(intelHref("/intel/warnings/3", "generic:us-unemployment")).toBe("/intel/warnings/3?domain=generic%3Aus-unemployment");
    expect(intelHref("/intel/signals#sig-1", "generic:x")).toBe("/intel/signals?domain=generic%3Ax#sig-1");
    expect(intelHref("/intel/decisions?batch_id=b1", "generic:x")).toBe("/intel/decisions?batch_id=b1&domain=generic%3Ax");
    expect(intelHref("/intel", DEFAULT_DOMAIN)).toBe("/intel");
    expect(intelHref("/intel/trends?domain=generic%3Ax", "movie")).toBe("/intel/trends");
  });

  it("never touches links outside the console", () => {
    expect(intelHref("/admin/experiments", "generic:x")).toBe("/admin/experiments");
    expect(intelHref("/intelligence", "generic:x")).toBe("/intelligence");
    expect(intelHref("/me/intelligence", "generic:x")).toBe("/me/intelligence");
  });

  it("falls back to movie for a missing or malformed domain", () => {
    expect(normalizeDomain(null)).toBe("movie");
    expect(normalizeDomain("  ")).toBe("movie");
    expect(normalizeDomain("<script>")).toBe("movie");
    expect(normalizeDomain("generic:us-unemployment")).toBe("generic:us-unemployment");
  });

  it("switches domain on the same section, dropping detail ids", () => {
    expect(switchDomainHref("/intel/decisions/412", "generic:x")).toBe("/intel/decisions?domain=generic%3Ax");
    expect(switchDomainHref("/intel/series/share%3Agenre%3ADrama", "generic:x")).toBe("/intel/trends?domain=generic%3Ax");
    expect(switchDomainHref("/intel", "movie")).toBe("/intel");
  });
});

describe("capability gating", () => {
  const generic = { name: "US unemployment", capabilities: { recommendation: false, user_intelligence: false, lapse: false, raters: false, model_governance: false, scenarios: true } };

  it("honours the domain's declared capabilities, with a one-line reason", () => {
    const lapse = capabilityState("generic:us-unemployment", generic, "lapse");
    expect(lapse.available).toBe(false);
    expect(lapse.reason).toMatch(/^Audience-lapse model is not shown for US unemployment: /);
    expect(capabilityState("generic:us-unemployment", generic, "scenarios")).toEqual({ available: true, reason: null });
  });

  it("keeps every section for movie when capabilities are unknown", () => {
    for (const cap of [...MOVIE_ONLY, "scenarios"] as DomainCapability[]) expect(capabilityState("movie", null, cap).available).toBe(true);
  });

  it("hides movie-only sections for another domain when capabilities are unknown", () => {
    for (const cap of MOVIE_ONLY) {
      const s = capabilityState("generic:x", null, cap);
      expect(s.available).toBe(false);
      expect(s.reason).toContain("capabilities not reported");
    }
    expect(capabilityState("generic:x", null, "scenarios").available).toBe(true);
  });

  it("lets a declared false hide a section even for movie", () => {
    expect(capabilityState("movie", { name: "Movies", capabilities: { lapse: false } }, "lapse").available).toBe(false);
  });

  it("treats 501 as not deployed", () => {
    expect(isNotDeployed(new ApiError(501, "Not Implemented"))).toBe(true);
    expect(isNotDeployed(new ApiError(404, "User has no events"))).toBe(false);
  });
});

describe("early-warning level scale", () => {
  const base = { answer: "WARNING", option_scores: { NO_ACTION: 0.1, MONITOR: 0.3, WARNING: 0.7, URGENT_ACTION: 0.4 }, abstained: false };

  it("maps the four answers in order", () => {
    expect(EARLY_WARNING_LEVELS.map(levelIndex)).toEqual([0, 1, 2, 3]);
    expect(levelIndex("warning")).toBe(-1);
    expect(levelIndex(null)).toBe(-1);
  });

  it("marks the chosen level, the reached steps and which levels raise a warning", () => {
    const steps = levelScale(base);
    expect(steps.map((s) => s.step)).toEqual([1, 2, 3, 4]);
    expect(steps.filter((s) => s.chosen).map((s) => s.level)).toEqual(["WARNING"]);
    expect(steps.map((s) => s.reached)).toEqual([true, true, true, false]);
    expect(steps.map((s) => s.raisesWarning)).toEqual([false, false, true, true]);
    expect(steps[2].score).toBe(0.7);
  });

  it("chooses nothing when the decision abstained", () => {
    const steps = levelScale({ ...base, abstained: true, answer: null, option_scores: {} });
    expect(steps.some((s) => s.chosen || s.reached)).toBe(false);
    expect(steps.every((s) => s.score === null)).toBe(true);
  });

  it("recognises the decision by spec, key or option set", () => {
    expect(isEarlyWarningDecision({ spec_id: "early_warning_level", key: "x", options: [] })).toBe(true);
    expect(isEarlyWarningDecision({ spec_id: "s", key: "early_warning_level:genre:Drama", options: [] })).toBe(true);
    expect(isEarlyWarningDecision({ spec_id: "s", key: "k", options: ["URGENT_ACTION", "WARNING", "MONITOR", "NO_ACTION"] })).toBe(true);
    expect(isEarlyWarningDecision({ spec_id: "retrain_model", key: "retrain_model", options: ["yes", "no"] })).toBe(false);
  });

  it("groups the state by pipeline stage, nested or flat", () => {
    const { groups, other } = ewlStateGroups({ signal_strength: 0.8, trend: { direction: "down", q: 0.012 }, risk_score: 64, risk_level: "high", adverse_direction: "down" });
    expect(groups.map((g) => g.stage)).toEqual(["signal", "trend", "anomaly", "forecast", "risk"]);
    expect(groups.find((g) => g.stage === "trend")?.rows.map((r) => [r.label, r.value])).toEqual([["direction", "down"], ["q", "0.012"]]);
    expect(groups.find((g) => g.stage === "risk")?.rows.map((r) => r.label)).toEqual(["score", "level"]);
    expect(groups.find((g) => g.stage === "anomaly")?.observed).toBe(false);
    expect(other.map((r) => r.label)).toEqual(["adverse direction"]);
  });

  it("labels strategies", () => {
    expect(strategyLabel("adapt_to_recent")).toBe("Adapt to recent taste");
    expect(strategyLabel("something_new")).toBe("something new");
    expect(strategyLabel(null)).toBe("—");
  });
});

describe("drift aspect formatting", () => {
  const aspect = (over: Partial<DriftAspect>): DriftAspect => ({
    aspect: "genre_distribution",
    status: "ok",
    test: "Jensen–Shannon permutation",
    statistic: 0.21345,
    p_value: 0.0004,
    p_adjusted: 0.0024,
    significant: true,
    effect: {},
    detail: null,
    ...over,
  });

  it("formats a significant aspect", () => {
    const v = driftAspectView(aspect({}));
    expect(v.label).toBe("Genre mix");
    expect(v.statistic).toBe("0.213");
    expect(v.p).toBe("< 0.001");
    expect(v.pAdjusted).toBe("0.002");
    expect(v.verdict).toBe("significant");
    expect(v.verdictLabel).toBe("Significant");
  });

  it("prints dashes, never zeros, below the minimum sample", () => {
    const v = driftAspectView(aspect({ aspect: "acceptance", status: "insufficient_data", statistic: null, p_value: null, p_adjusted: null, significant: false }));
    expect([v.statistic, v.p, v.pAdjusted]).toEqual(["—", "—", "—"]);
    expect(v.verdictLabel).toBe("Insufficient data");
    expect(v.label).toBe("Acceptance of recommendations");
  });

  it("humanises unknown aspects and reports not significant", () => {
    const v = driftAspectView(aspect({ aspect: "watch_time", significant: false, p_adjusted: 0.4 }));
    expect(v.label).toBe("watch time");
    expect(v.verdict).toBe("not_significant");
  });

  it("states the headline and its evidence strength", () => {
    expect(driftHeadline({ status: "ok", drift_detected: true, confidence: 0.9976 })).toEqual({ verdict: "detected", label: "Drift detected", confidence: "1.00" });
    expect(driftHeadline({ status: "ok", drift_detected: false, confidence: 0.4 }).label).toBe("No drift detected");
    expect(driftHeadline({ status: "insufficient_data", drift_detected: false, confidence: null }).confidence).toBe("—");
  });
});

describe("preference shares", () => {
  const history = {
    categories: ["Drama", "Comedy", "Action", "Horror", "Sci-Fi", "Romance", "War"],
    windows: [
      { label: "historical", start: "2016-01-01", end: "2017-01-01", n: 100, shares: { Drama: 0.4, Comedy: 0.2, Action: 0.1, Horror: 0.1, "Sci-Fi": 0.1, Romance: 0.05, War: 0.05 } },
      { label: "recent", start: "2017-01-01", end: "2017-06-01", n: 20, shares: { Drama: 0.1, Comedy: 0.1, Action: 0.5, Horror: 0.1, "Sci-Fi": 0.1, Romance: 0.1, War: 0 } },
    ],
  };

  it("fixes a category order for the whole history and folds the tail", () => {
    const { shown, folded } = topCategories(history, 5);
    expect(shown).toEqual(["Action", "Drama", "Comedy", "Horror", "Sci-Fi"]);
    expect(folded).toEqual(["Romance", "War"]);
  });

  it("puts the folded share in Other, last, with no colour slot", () => {
    const segs = shareSegments(history.windows[0].shares, ["Action", "Drama", "Comedy", "Horror", "Sci-Fi"]);
    expect(segs.map((s) => s.slot)).toEqual([0, 1, 2, 3, 4, null]);
    expect(segs[5].category).toBe("Other");
    expect(segs[5].share).toBeCloseTo(0.1);
  });

  it("sorts projected rows and sizes a shared axis", () => {
    const rows = projectedRows({ projected_shares: { Drama: { mean: 0.2, lo80: 0.1, hi80: 0.3 }, Action: { mean: 0.4, lo80: 0.33, hi80: 0.52 } } }, { Drama: 0.3 });
    expect(rows.map((r) => r.category)).toEqual(["Action", "Drama"]);
    expect(rows[0].baseline).toBeNull();
    expect(rows[1].baseline).toBe(0.3);
    expect(shareAxisMax(rows)).toBe(0.6);
  });
});

describe("state flattening and evaluation helpers", () => {
  it("flattens nested state without JSON", async () => {
    const { flattenState } = await import("@/lib/decisions");
    const rows = flattenState({ drift: { status: "ok", q: 0.0004 }, effects: { adapt: { delta: 0.0063, ci95: [0.0014, 0.0134], n_users: 7 } }, skipped: [] });
    expect(rows).toContainEqual({ label: "drift · status", value: "ok" });
    expect(rows).toContainEqual({ label: "drift · q", value: "< 0.001" });
    expect(rows.find((r) => r.label === "effects · adapt")?.value).toBe("delta 0.0063 · ci95 [0.0014, 0.0134] · n users 7");
    expect(rows).toContainEqual({ label: "skipped", value: "none" });
    expect(rows.every((r) => !r.value.includes("{"))).toBe(true);
  });

  it("collects early-warning point components, highest first", () => {
    const { components, groups } = ewlStateGroups({ forecast: null, components: [{ stage: "trend", points: 2.5, ref: "trend-1" }, { stage: "risk", points: 3.6, title: "t" }] });
    expect(components.map((c) => c.stage)).toEqual(["risk", "trend"]);
    expect(groups.find((g) => g.stage === "forecast")?.observed).toBe(false);
  });

  it("formats paired deltas and their verdict", async () => {
    const { deltaVerdict, fmtDelta, fmtWithCi, replayBars, consistencyOf } = await import("@/lib/evaluation");
    const d = { delta: 0.00633, ci95: [0.00138, 0.01343] as [number, number], p_better: 0.98, n_users: 7, n_improved: 3, n_worse: 0 };
    expect(fmtDelta(d)).toBe("+0.0063 [0.0014, 0.0134]");
    expect(deltaVerdict(d)).toBe("better");
    expect(deltaVerdict({ ...d, ci95: [-0.001, 0.007] })).toBe("unclear");
    expect(fmtWithCi(0.2195, [0.1436, 0.3205])).toBe("0.22 [0.14, 0.32]");
    expect(fmtWithCi(null)).toBe("—");
    expect(replayBars([0, 2])).toEqual([{ bin: "1", n: 0 }, { bin: "2", n: 2 }]);
    const pooled = consistencyOf({ consistency_pooled_within_windows: { situation_pairs: 946, flips: 156, warning_boundary_flips: 69 } } as never);
    expect(pooled?.pooled).toBe(true);
    expect(pooled?.flipRate).toBeCloseTo(156 / 946);
  });
});

describe("review fixes", () => {
  const situation = {
    key: "series:unemployment_rate:region:Midwest",
    entity_type: "census_region",
    entity: "Midwest",
    series_id: "unemployment_rate:region:Midwest",
    adverse_direction: "up",
    label: "Midwest unemployment_rate",
    components: [
      { stage: "trend", stage_group: "trend", ref: "trend-1", points: 2.5, title: "Midwest trend up", detail: "trend up (q 0.0)" },
      { stage: "change_point", stage_group: "trend", ref: "trend-1", points: 2.497, title: "Midwest shifted up" },
      { stage: "risk", stage_group: "risk", ref: "risk-1", points: 3.59, title: "Midwest rising (adverse)" },
    ],
    skipped: [{ stage: "forecast", reason: "series too short for a backtest" }],
  };

  it("derives stage cells from the components and the skipped list", () => {
    const { groups, other } = ewlStateGroups(situation);
    const by = Object.fromEntries(groups.map((g) => [g.stage, g]));
    expect(by.trend.observed).toBe(true);
    expect(by.trend.rows).toEqual([{ key: "trend.points", label: "points", value: "2.5" }, { key: "trend.n", label: "components", value: "2" }]);
    expect(by.trend.summary).toBe("Midwest trend up");
    expect(by.risk.observed).toBe(true);
    expect(by.risk.rows[0].value).toBe("3.59");
    expect(by.forecast.observed).toBe(false);
    expect(by.forecast.skippedReason).toBe("series too short for a backtest");
    expect(by.anomaly.observed).toBe(false);
    expect(by.anomaly.skippedReason).toBeNull();
    expect(by.signal.skippedReason).toBeNull();
    expect(other.map((r) => r.label)).toEqual(["key", "entity type", "entity", "series id", "adverse direction", "label"]);
  });

  it("never prints raw JSON for lists of objects or deep objects", async () => {
    const { fmtStateValue, flattenState, isScalarMap } = await import("@/lib/decisions");
    expect(fmtStateValue([{ stage: "forecast", reason: "too short" }, { stage: "anomaly", reason: "no data" }])).toBe("forecast: too short; anomaly: no data");
    expect(fmtStateValue({ a: { b: { c: 1 } } })).toBe("a (b (c 1))");
    expect(fmtStateValue([{ x: 1, y: null }])).toBe("x 1 · y —");
    const rows = flattenState({ skipped: [{ stage: "forecast", reason: "too short" }], deep: { a: { b: { c: { d: 1 } } } } });
    expect(rows).toContainEqual({ label: "skipped · forecast", value: "too short" });
    expect(rows.every((r) => !/[{}"]/.test(r.value))).toBe(true);
    expect(isScalarMap({ a: 1, b: null, c: [1, null] })).toBe(true);
    expect(isScalarMap({ a: { b: 1 } })).toBe(false);
  });

  it("uses level words only for early-warning decisions", async () => {
    const { fmtAnswer } = await import("@/lib/intel");
    expect(fmtAnswer({ kind: "choice", answer: "URGENT_ACTION", scale: null, spec_id: "early_warning_level", key: "k", options: [] })).toBe("Urgent action");
    expect(fmtAnswer({ kind: "choice", answer: "WARNING", scale: null, spec_id: "some_other_spec", key: "k", options: ["WARNING", "OK"] })).toBe("WARNING");
    expect(fmtAnswer({ kind: "choice", answer: "MONITOR", scale: null })).toBe("MONITOR");
  });

  it("labels replays by month only when they are provably monthly", async () => {
    const { replayBars, replayMonths } = await import("@/lib/evaluation");
    const months = replayMonths(["2016-09-01", "2018-08-01"], 24);
    expect(months?.[0]).toBe("2016-09");
    expect(months?.[4]).toBe("2017-01");
    expect(months?.[23]).toBe("2018-08");
    expect(replayMonths(["2016-09-01", "2018-08-01"], 23)).toBeNull();
    expect(replayMonths(undefined, 3)).toBeNull();
    expect(replayBars([1, 2], ["2020-01", "2020-02"])).toEqual([{ bin: "2020-01", n: 1 }, { bin: "2020-02", n: 2 }]);
    expect(replayBars([1, 2], null).map((b) => b.bin)).toEqual(["1", "2"]);
  });
});
