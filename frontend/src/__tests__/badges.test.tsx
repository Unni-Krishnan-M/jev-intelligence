import { describe, expect, it } from "vitest";

import { ConfidenceBadge, FreshnessBadge, SeverityBadge } from "@/components/jev/intel/badges";

import { renderUi } from "./render";

describe("ConfidenceBadge", () => {
  it("shows a probability as a percentage", () => {
    const { container } = renderUi(<ConfidenceBadge value={0.62} kind="probability" />);
    expect(container).toHaveTextContent("62 %");
    expect(container).toHaveTextContent("probability");
  });

  it("never shows a margin as a percentage", () => {
    const { container } = renderUi(<ConfidenceBadge value={0.62} kind="margin" />);
    expect(container).toHaveTextContent("0.62");
    expect(container.textContent).not.toContain("%");
  });

  it("shows a fired rule as fired, not as 100 %", () => {
    const { container } = renderUi(<ConfidenceBadge value={1} kind="rule" />);
    expect(container).toHaveTextContent("fired");
    expect(container.textContent).not.toContain("%");
  });

  it("shows an interval as its coverage and range", () => {
    const { container } = renderUi(<ConfidenceBadge value={0.8} kind="interval" interval={[4, 9.5]} format={(v) => `${v} %`} />);
    expect(container).toHaveTextContent("80 % interval [4 %, 9.5 %]");
    expect(container.textContent?.match(/interval/g)).toHaveLength(1);
    expect(container.querySelector("[aria-label]")?.getAttribute("aria-label")).toMatch(/not a probability/);
  });

  it("falls back to a plain score without a kind", () => {
    const { container } = renderUi(<ConfidenceBadge value={0.351} />);
    expect(container).toHaveTextContent("0.35");
    expect(container).toHaveTextContent("score");
  });
});

describe("SeverityBadge", () => {
  it.each([
    ["critical", "Critical"],
    ["high", "High"],
    ["medium", "Medium"],
    ["low", "Low"],
  ] as const)("carries %s by an icon and a word", (severity, label) => {
    const { container } = renderUi(<SeverityBadge severity={severity} />);
    expect(container).toHaveTextContent(label);
    const icon = container.querySelector("svg");
    expect(icon).not.toBeNull();
    expect(icon).toHaveAttribute("aria-hidden", "true");
  });
});

describe("FreshnessBadge", () => {
  it("reports an archival snapshot's age without alarming", () => {
    const { container } = renderUi(<FreshnessBadge kind="static_snapshot" days={2921.4} fresh={false} />);
    expect(container).toHaveTextContent("Archival");
    expect(container).toHaveTextContent("2,921 d old");
    expect(container.textContent).not.toContain("Stale");
  });

  it("maps signal source names to kinds", () => {
    expect(renderUi(<FreshnessBadge kind="movielens" days={10} />).container).toHaveTextContent("Archival");
    expect(renderUi(<FreshnessBadge kind="model" days={3.2} />).container).toHaveTextContent("3.2 d old");
  });

  it("distinguishes an empty, fresh, stale and unassessed live source", () => {
    expect(renderUi(<FreshnessBadge kind="live" rows={0} days={null} />).container).toHaveTextContent("No events yet");
    expect(renderUi(<FreshnessBadge kind="live" rows={5} days={0.5} fresh />).container).toHaveTextContent("Fresh12.0 h old");
    expect(renderUi(<FreshnessBadge kind="live" rows={5} days={9} fresh={false} />).container).toHaveTextContent("Stale9.0 d old");
    expect(renderUi(<FreshnessBadge kind="live" rows={5} days={null} fresh={null} />).container).toHaveTextContent("Not assessed");
  });
});
