import { describe, expect, it } from "vitest";

import { safeNextPath } from "@/lib/safe-redirect";

const ORIGIN = "http://localhost:3000";

describe("safeNextPath", () => {
  it.each([
    ["/\\evil.com"], // browsers read "\" as "/": http://evil.com/
    ["/%5Cevil.com"], // the same, percent-encoded once more
    ["/\tevil.com"], // tab (dropped by the URL parser)
    ["/\t/evil.com"], // "/<TAB>/evil.com" resolves to http://evil.com/
    ["/\n/evil.com"],
    ["//evil.com"],
    ["/%2F/evil.com"],
    ["https://evil.com"],
    ["http://localhost:3000.evil.com/home"],
    ["javascript:alert(1)"],
    ["home"],
    ["/%E0%A4%A"], // malformed percent-encoding
    [""],
  ])("rejects %j", (value) => {
    expect(safeNextPath(value, ORIGIN)).toBeNull();
  });

  it("rejects a missing value", () => {
    expect(safeNextPath(null, ORIGIN)).toBeNull();
    expect(safeNextPath(undefined, ORIGIN)).toBeNull();
  });

  it("keeps same-origin paths with their query and hash", () => {
    expect(safeNextPath("/home?x=1#y", ORIGIN)).toBe("/home?x=1#y");
    expect(safeNextPath("/intel/series/volume%3Aall", ORIGIN)).toBe("/intel/series/volume%3Aall");
    expect(safeNextPath("/movies/42", ORIGIN)).toBe("/movies/42");
  });

  it("normalises dot segments without leaving the origin", () => {
    expect(safeNextPath("/a/../admin", ORIGIN)).toBe("/admin");
  });
});
