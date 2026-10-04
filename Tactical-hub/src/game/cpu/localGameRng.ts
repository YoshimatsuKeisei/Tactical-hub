import type { CpuRuntime } from "./types";

/**
 * Advances the persistent LOCAL game random stream.
 *
 * CpuRuntime owns this state so a future LOCAL session snapshot can restore the
 * exact next random value without depending on the browser's Math.random state.
 */
export function nextLocalGameRandom(runtime: CpuRuntime) {
  runtime.rngState = (Math.imul(runtime.rngState, 1664525) + 1013904223) >>> 0;
  return runtime.rngState / 0x1_0000_0000;
}

export function createLocalGameRng(runtime: CpuRuntime): () => number {
  return () => nextLocalGameRandom(runtime);
}
