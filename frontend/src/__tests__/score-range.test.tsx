import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RunTimeline, ScoreRange, scoreWindow } from "@/components/jev/intel/charts";
import { confidenceSentence, fmtChance, RecConfidence } from "@/components/jev/rec-confidence";

import { renderUi } from "./render";

const scale = { min: 0, max: 20, unit: "%" };

describe("ScoreRange", () => {
  it("places the answer and its interval on the scale", () => {
    renderUi(<ScoreRange value={5} interval={[2, 10]} scale={scale} coverage={0.8} />);
    expect(screen.getByTestId("score-range-value").style.left).toBe("25%");
    const band = screen.getByTestId("score-range-interval");
    expect(band.style.left).toBe("10%");
    expect(band.style.width).toBe("40%");
  });

  it("describes the whole range in words", () => {
    renderUi(<ScoreRange value={5} interval={[2, 10]} scale={scale} coverage={0.8} />);
    expect(screen.getByRole("img")).toHaveAccessibleName("Answer 5 %, 80 % interval 2 % to 10 %, on a scale from 0 % to 20 %");
    expect(screen.getByTestId("score-range")).toHaveTextContent("80 % interval 2–10");
    expect(screen.getByTestId("score-range")).toHaveTextContent("20 %");
    expect(screen.queryByTestId("score-range-zoom")).toBeNull();
  });

  it("clamps values outside the scale and omits a missing interval", () => {
    renderUi(<ScoreRange value={25} interval={null} scale={scale} />);
    expect(screen.getByTestId("score-range-value").style.left).toBe("100%");
    expect(screen.queryByTestId("score-range-interval")).toBeNull();
  });

  it("zooms to a labelled window when the interval is a sliver of the scale", () => {
    const wide = { min: 0, max: 100, unit: "% of home-rail slots" };
    expect(scoreWindow(1.36, [0.74, 2.04], wide)).toEqual({ min: 0, max: 3, zoomed: true });
    expect(scoreWindow(50, [20, 80], wide).zoomed).toBe(false);
    renderUi(<ScoreRange value={1.36} interval={[0.74, 2.04]} scale={wide} coverage={0.8} />);
    expect(screen.getByTestId("score-range-zoom")).toHaveTextContent("the full scale is 0–100 % of home-rail slots");
    expect(Number.parseFloat(screen.getByTestId("score-range-value").style.left)).toBeCloseTo(45.33, 1);
    expect(screen.getByRole("img")).toHaveAccessibleName(/on a scale from 0 % of home-rail slots to 100 % of home-rail slots$/);
  });

  it("shows no tick for an abstained answer", () => {
    renderUi(<ScoreRange value={null} scale={scale} />);
    expect(screen.queryByTestId("score-range-value")).toBeNull();
    expect(screen.getByRole("img")).toHaveAccessibleName(/^No answer/);
  });
});

describe("RunTimeline", () => {
  it("draws nothing with fewer than two runs", () => {
    const { container } = renderUi(<RunTimeline points={[{ key: "a", v: 1, title: "a" }]} format={String} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("prints the first and the latest reading", () => {
    const { container } = renderUi(<RunTimeline points={[{ key: "a", v: 10, title: "a" }, { key: "b", v: 30, title: "b" }]} format={(v) => `${v}`} />);
    expect(container).toHaveTextContent("10");
    expect(container).toHaveTextContent("30");
    expect(container.querySelectorAll("[title]")).toHaveLength(2);
  });
});

describe("RecConfidence", () => {
  it("states a calibrated probability honestly", () => {
    const { container } = renderUi(<RecConfidence item={{ confidence: 0.62, confidence_kind: "probability" }} />);
    expect(container).toHaveTextContent("≈62 % chance this is one of your next five 4★+ ratings");
    expect(container).toHaveTextContent("calibrated on held-out ratings");
  });

  it("keeps small chances visible instead of rounding them to zero", () => {
    expect(fmtChance(0.0091)).toBe("≈0.9 %");
    expect(fmtChance(0.057)).toBe("≈6 %");
    expect(fmtChance(0.0004)).toBe("under 0.1 %");
    expect(confidenceSentence(0.03)).not.toMatch(/rate it/);
  });

  it("is hidden without a calibrated confidence", () => {
    expect(renderUi(<RecConfidence item={{ confidence: null, confidence_kind: null }} />).container).toBeEmptyDOMElement();
    expect(renderUi(<RecConfidence item={{ confidence: 0.5, confidence_kind: null }} />).container).toBeEmptyDOMElement();
    expect(renderUi(<RecConfidence item={{}} />).container).toBeEmptyDOMElement();
  });
});
