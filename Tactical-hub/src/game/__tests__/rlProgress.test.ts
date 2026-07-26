import { describe, expect, it } from "vitest";
import { formatRlElapsed, formatRlRate } from "../cpu/rlProgress";

describe("RL progress formatting", () => {
  it("formats elapsed time and rates without affecting processing", () => {
    expect(formatRlElapsed(0)).toBe("00:00:00");
    expect(formatRlElapsed(3_661_999)).toBe("01:01:01");
    expect(formatRlRate(12.345)).toBe("12.3");
    expect(formatRlRate(Number.NaN)).toBe("0.0");
  });
});
