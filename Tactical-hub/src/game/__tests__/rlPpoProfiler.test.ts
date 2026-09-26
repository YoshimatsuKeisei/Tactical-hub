import { afterEach, describe, expect, it, vi } from "vitest";
import { PpoTimingProfiler } from "../cpu/rlPpoProfiler";

const original = process.env.PPO_PROFILE;
const originalNode = process.env.PPO_NODE_PROFILE;
afterEach(() => {
  if (original === undefined) delete process.env.PPO_PROFILE;
  else process.env.PPO_PROFILE = original;
  if (originalNode === undefined) delete process.env.PPO_NODE_PROFILE;
  else process.env.PPO_NODE_PROFILE = originalNode;
  vi.restoreAllMocks();
});

describe("PPO opt-in timing", () => {
  it("stays silent when profiling is disabled", async () => {
    delete process.env.PPO_PROFILE;
    delete process.env.PPO_NODE_PROFILE;
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const profiler = new PpoTimingProfiler();
    expect(profiler.measure("step", () => 7)).toBe(7);
    expect(await profiler.measureAsync("act", async () => 9)).toBe(9);
    profiler.report("test");
    expect(stderr).not.toHaveBeenCalled();
  });

  it("collects stage timings without altering returned values", async () => {
    process.env.PPO_PROFILE = "1";
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const profiler = new PpoTimingProfiler();
    expect(profiler.measure("step", () => 7)).toBe(7);
    expect(await profiler.measureAsync("act", async () => 9)).toBe(9);
    expect(() => profiler.measure("step", () => { throw Error("sentinel"); })).toThrow("sentinel");
    profiler.report("test");
    const line = String(stderr.mock.calls[0][0]);
    expect(line).toContain("[PPO profile node]");
    const payload = JSON.parse(line.slice(line.indexOf("{")));
    expect(payload.stages.step.count).toBe(2);
    expect(payload.stages.act.count).toBe(1);
    expect(payload.stages.step.totalMs).toBeGreaterThanOrEqual(0);
  });

  it("supports node-only profiling without enabling Python PPO_PROFILE", () => {
    delete process.env.PPO_PROFILE;
    process.env.PPO_NODE_PROFILE = "1";
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const profiler = new PpoTimingProfiler();
    expect(profiler.measure("replay", () => 3)).toBe(3);
    profiler.report("node-only");
    const line = String(stderr.mock.calls[0][0]);
    expect(line).toContain("[PPO profile node]");
    expect(line).toContain("replay");
  });
});
