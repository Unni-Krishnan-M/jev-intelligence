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
