import type { GameState } from "../types";
import { advanceCpuOneStep, type CpuStepResult } from "./cpuStep";
import type { CpuRuntime, CpuTeamSettings } from "./types";

export { resolveBattleWithHiddenCpuIntents } from "./cpuStep";

export function advanceVisualCpuTick(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, control: { running: boolean; paused: boolean }): CpuStepResult {
  if (!control.running || control.paused) return { state, runtime, applied: false };
  return advanceCpuOneStep(state, runtime, settings);
}

export function advanceVisualCpuOneStep(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings): CpuStepResult {
  return advanceCpuOneStep(state, runtime, settings);
}
