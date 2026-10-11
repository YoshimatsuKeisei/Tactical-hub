import { describe, expect, it, vi } from "vitest";
import {
  LAKE_ANIMATION_CYCLE_MS,
  LAKE_CROSSFADE_DURATION_MS,
  LAKE_FRAME_DURATION_MS,
  LAKE_HOLD_DURATION_MS,
  LakeAnimationPlayer,
  lakeAnimationStateAt,
  type LakeAnimationTiming,
} from "./pixiLakeAnimation";

class FakeTiming implements LakeAnimationTiming {
  time = 0;
  isHidden = false;
  private nextId = 1;
  private timers = new Map<number, { at: number; callback: () => void }>();
  private visibilityListener: (() => void) | undefined;
  now = () => this.time;
  setTimeout = (callback: () => void, delayMs: number) => {
    const id = this.nextId++;
    this.timers.set(id, { at: this.time + delayMs, callback });
    return id as unknown as ReturnType<typeof setTimeout>;
  };
  clearTimeout = (handle: ReturnType<typeof setTimeout>) => {
    this.timers.delete(handle as unknown as number);
  };
  hidden = () => this.isHidden;
  subscribeVisibility = (callback: () => void) => {
    this.visibilityListener = callback;
    return () => { this.visibilityListener = undefined; };
  };
  setHidden(hidden: boolean) {
    this.isHidden = hidden;
    this.visibilityListener?.();
  }
  advance(ms: number) {
    const end = this.time + ms;
    for (;;) {
      const due = [...this.timers].sort((a, b) => a[1].at - b[1].at)[0];
      if (!due || due[1].at > end) break;
      this.time = due[1].at;
      this.timers.delete(due[0]);
      due[1].callback();
    }
    this.time = end;
  }
  get pendingCount() { return this.timers.size; }
}

describe("synchronized lake animation timing", () => {
  it("cycles Frame 1 → 2 → 3 → 4 → 1 every 800ms", () => {
    expect(LAKE_FRAME_DURATION_MS).toBe(800);
    expect(LAKE_ANIMATION_CYCLE_MS).toBe(3200);
    expect([0, 800, 1600, 2400, 3200].map((time) => lakeAnimationStateAt(time).current))
      .toEqual(["lake", "lake_frame_2", "lake_frame_3", "lake_frame_4", "lake"]);
    expect(lakeAnimationStateAt(3199).next).toBe("lake");
  });

  it("holds for 600ms and cross-fades over the final 200ms without leaving 0…1", () => {
    expect(LAKE_HOLD_DURATION_MS).toBe(600);
    expect(LAKE_CROSSFADE_DURATION_MS).toBe(200);
    expect(lakeAnimationStateAt(599)).toMatchObject({ current: "lake", next: "lake_frame_2", alpha: 0, fading: false });
    expect(lakeAnimationStateAt(600)).toMatchObject({ alpha: 0, fading: true });
    expect(lakeAnimationStateAt(700).alpha).toBe(0.5);
    expect(lakeAnimationStateAt(799).alpha).toBeCloseTo(0.995);
    expect(lakeAnimationStateAt(800)).toMatchObject({ current: "lake_frame_2", alpha: 0, fading: false });
    for (let time = -6400; time <= 6400; time += 17) {
      expect(lakeAnimationStateAt(time).alpha).toBeGreaterThanOrEqual(0);
      expect(lakeAnimationStateAt(time).alpha).toBeLessThanOrEqual(1);
    }
  });

  it("uses one timer, sleeps during holds, stops while hidden/off, and cleans up", () => {
    const timing = new FakeTiming();
    const onFrame = vi.fn();
    const player = new LakeAnimationPlayer(onFrame, timing);
    expect(player.configure(true)).toMatchObject({ current: "lake", alpha: 0 });
    player.configure(true);
    expect(timing.pendingCount).toBe(1);
    timing.advance(599);
    expect(onFrame).not.toHaveBeenCalled();
    timing.advance(1);
    expect(onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ fading: true, alpha: 0 }));
    expect(timing.pendingCount).toBe(1);

    timing.setHidden(true);
    expect(timing.pendingCount).toBe(0);
    const callsWhileVisible = onFrame.mock.calls.length;
    timing.advance(1000);
    expect(onFrame).toHaveBeenCalledTimes(callsWhileVisible);
    timing.setHidden(false);
    expect(onFrame).toHaveBeenCalledTimes(callsWhileVisible + 1);
    expect(timing.pendingCount).toBe(1);

    player.configure(false);
    expect(timing.pendingCount).toBe(0);
    timing.advance(5000);
    expect(onFrame).toHaveBeenCalledTimes(callsWhileVisible + 1);
    player.configure(true);
    expect(timing.pendingCount).toBe(1);
    player.dispose();
    expect(timing.pendingCount).toBe(0);
    timing.setHidden(true);
    timing.setHidden(false);
    expect(player.configure(true)).toBeUndefined();
  });
});
