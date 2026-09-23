import { describe, expect, it } from "vitest";

import pkg from "../../package.json";
import { fmtAnswer, fmtConfidence, fmtPct, fmtScaleValue, fmtSigned, fmtValue } from "@/lib/intel";
import { APP_VERSION, shortVersion } from "@/lib/version";

describe("fmtValue", () => {
  it("prints integers without decimals and with separators", () => {
    expect(fmtValue(3)).toBe("3");
    expect(fmtValue(100836)).toBe("100,836");
  });

  it("rounds large floats to whole numbers and small ones to 3 significant digits", () => {
    expect(fmtValue(1234.5)).toBe("1,235");
    expect(fmtValue(0.123456)).toBe("0.123");
    expect(fmtValue(12.3456)).toBe("12.3");
  });

  it("recognises day counts by field name", () => {
    expect(fmtValue(2.6, "days_since_last")).toBe("3");
    expect(fmtValue(2.6, "age_days")).toBe("3");
    expect(fmtValue(2.6, "ratio")).toBe("2.6");
  });

  it("handles missing, boolean, array, object and non-finite values", () => {
    expect(fmtValue(null)).toBe("—");
    expect(fmtValue(undefined)).toBe("—");
    expect(fmtValue(true)).toBe("true");
    expect(fmtValue([1, 0.5])).toBe("1, 0.5");
    expect(fmtValue({ a: 1 })).toBe('{"a":1}');
    expect(fmtValue(Number.POSITIVE_INFINITY)).toBe("Infinity");
    expect(fmtValue("text")).toBe("text");
  });
});

describe("small formatters", () => {
  it("fmtPct and fmtSigned", () => {
    expect(fmtPct(0.625)).toBe("63 %");
    expect(fmtPct(null)).toBe("—");
    expect(fmtSigned(1.25, 1, " %")).toBe("+1.3 %");
    expect(fmtSigned(-0.5)).toBe("−0.5");
    expect(fmtSigned(0)).toBe("±0.0");
  });
});

describe("fmtConfidence", () => {
  it("writes only a probability as a chance", () => {
    expect(fmtConfidence(0.62, "probability")).toBe("62 %");
    expect(fmtConfidence(0.62, "margin")).toBe("0.62");
    expect(fmtConfidence(0.62, null)).toBe("0.62");
  });

  it("marks a fired rule and keeps other rule values as scores", () => {
    expect(fmtConfidence(1, "rule")).toBe("fired");
    expect(fmtConfidence(0.4, "rule")).toBe("0.40");
  });

  it("writes an interval as its coverage with the range", () => {
    expect(fmtConfidence(0.8, "interval", [2.5, 7])).toBe("80 % interval [2.5, 7]");
    expect(fmtConfidence(0.8, "interval", [2.5, 7], (v) => `${v} %`)).toBe("80 % interval [2.5 %, 7 %]");
    expect(fmtConfidence(0.8, "interval")).toBe("80 % interval");
    expect(fmtConfidence(null, "interval", [1, 2])).toBe("—");
  });
});

describe("score answers", () => {
  const scale = { min: 0, max: 40, unit: "%" };

  it("formats a value on its scale", () => {
    expect(fmtScaleValue(12.345, scale)).toBe("12.3 %");
    expect(fmtScaleValue(3, { min: 0, max: 10, unit: "slots" })).toBe("3 slots");
    expect(fmtScaleValue(3, null)).toBe("3");
    expect(fmtScaleValue(null, scale)).toBe("—");
  });

  it("formats the answer by decision kind", () => {
    expect(fmtAnswer({ kind: "score", answer: 12.5, scale })).toBe("12.5 %");
    expect(fmtAnswer({ kind: "boolean", answer: "no", scale: null })).toBe("no");
    expect(fmtAnswer({ kind: "choice", answer: null })).toBe("—");
  });
});

describe("version", () => {
  it("matches package.json and shortens to major.minor", () => {
    expect(APP_VERSION).toBe(pkg.version);
    expect(shortVersion("1.1.0")).toBe("v1.1");
    expect(shortVersion("2")).toBe("v2.0");
  });
});
