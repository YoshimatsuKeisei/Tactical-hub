import type { GameState } from "../types";
import { advanceCpuOneStep, type CpuStepResult } from "./cpuStep";
import type { CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";

export { resolveBattleWithHiddenCpuIntents } from "./cpuStep";

export function advanceVisualCpuTick(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, control: { running: boolean; paused: boolean }, policy?: CpuPolicy): CpuStepResult {
  if (!control.running || control.paused) return { state, runtime, applied: false };
  return advanceCpuOneStep(state, runtime, settings, policy);
}

export function advanceVisualCpuOneStep(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, policy?: CpuPolicy): CpuStepResult {
  return advanceCpuOneStep(state, runtime, settings, policy);
}
