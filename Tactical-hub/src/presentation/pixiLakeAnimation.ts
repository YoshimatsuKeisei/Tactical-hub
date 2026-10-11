import {
  LAKE_FRAME_KEYS,
  type LakeFrameAssetKey,
} from "./tileAssets";

export const LAKE_FRAME_DURATION_MS = 800;
export const LAKE_HOLD_DURATION_MS = 600;
export const LAKE_CROSSFADE_DURATION_MS = 200;
export const LAKE_ANIMATION_CYCLE_MS = LAKE_FRAME_DURATION_MS * LAKE_FRAME_KEYS.length;
export const LAKE_ANIMATION_MAX_FPS = 30;

export interface LakeAnimationState {
  current: LakeFrameAssetKey;
  next: LakeFrameAssetKey;
  alpha: number;
  frameIndex: number;
  frameElapsedMs: number;
  fading: boolean;
}

export function lakeAnimationStateAt(elapsedMs: number): LakeAnimationState {
  const cycleElapsed = ((elapsedMs % LAKE_ANIMATION_CYCLE_MS) + LAKE_ANIMATION_CYCLE_MS) % LAKE_ANIMATION_CYCLE_MS;
  const frameIndex = Math.floor(cycleElapsed / LAKE_FRAME_DURATION_MS);
  const frameElapsedMs = cycleElapsed - frameIndex * LAKE_FRAME_DURATION_MS;
  const fading = frameElapsedMs >= LAKE_HOLD_DURATION_MS;
  const alpha = fading
    ? Math.min(1, Math.max(0, (frameElapsedMs - LAKE_HOLD_DURATION_MS) / LAKE_CROSSFADE_DURATION_MS))
    : 0;
  return {
    current: LAKE_FRAME_KEYS[frameIndex],
    next: LAKE_FRAME_KEYS[(frameIndex + 1) % LAKE_FRAME_KEYS.length],
    alpha,
    frameIndex,
    frameElapsedMs,
    fading,
  };
}

export interface LakeAnimationTiming {
  now(): number;
  setTimeout(callback: () => void, delayMs: number): ReturnType<typeof setTimeout>;
  clearTimeout(handle: ReturnType<typeof setTimeout>): void;
  hidden(): boolean;
  subscribeVisibility(callback: () => void): () => void;
}

function browserTiming(): LakeAnimationTiming {
  return {
    now: () => typeof performance === "undefined" ? Date.now() : performance.now(),
    setTimeout: (callback, delayMs) => globalThis.setTimeout(callback, delayMs),
    clearTimeout: (handle) => globalThis.clearTimeout(handle),
    hidden: () => typeof document !== "undefined" && document.visibilityState === "hidden",
    subscribeVisibility: (callback) => {
      if (typeof document === "undefined") return () => {};
      document.addEventListener("visibilitychange", callback);
      return () => document.removeEventListener("visibilitychange", callback);
    },
  };
}

export class LakeAnimationPlayer {
  private active = false;
  private disposed = false;
  private startedAt = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private readonly unsubscribeVisibility: () => void;

  constructor(
    private readonly onFrame: (state: LakeAnimationState) => void,
    private readonly timing: LakeAnimationTiming = browserTiming(),
  ) {
    this.unsubscribeVisibility = timing.subscribeVisibility(() => this.visibilityChanged());
  }

  configure(active: boolean): LakeAnimationState | undefined {
    if (this.disposed) return undefined;
    if (active !== this.active) {
      this.cancelTimer();
      this.active = active;
      if (active) this.startedAt = this.timing.now();
    }
    if (!active) return undefined;
    const state = this.currentState();
    this.schedule(state);
    return state;
  }

  private currentState() {
    return lakeAnimationStateAt(this.timing.now() - this.startedAt);
  }

  private schedule(state: LakeAnimationState) {
    if (this.timer !== undefined || this.timing.hidden() || !this.active || this.disposed) return;
    const delay = state.fading
      ? 1000 / LAKE_ANIMATION_MAX_FPS
      : Math.max(1, LAKE_HOLD_DURATION_MS - state.frameElapsedMs);
    this.timer = this.timing.setTimeout(() => {
      this.timer = undefined;
      if (!this.active || this.disposed || this.timing.hidden()) return;
      const nextState = this.currentState();
      this.onFrame(nextState);
      this.schedule(nextState);
    }, delay);
  }

  private visibilityChanged() {
    this.cancelTimer();
    if (!this.active || this.disposed || this.timing.hidden()) return;
    const state = this.currentState();
    this.onFrame(state);
    this.schedule(state);
  }

  private cancelTimer() {
    if (this.timer === undefined) return;
    this.timing.clearTimeout(this.timer);
    this.timer = undefined;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.active = false;
    this.cancelTimer();
    this.unsubscribeVisibility();
  }
}
