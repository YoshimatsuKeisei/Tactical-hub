import type { GameState } from "../types";
import { resolveStrategistActions } from "../engine/construction";
import { resolveMovement } from "../engine/movement";
import { advanceCpuOneStep, type CpuStepResult } from "./cpuStep";
import { createLocalGameRng } from "./localGameRng";
import type { CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";

export { resolveBattleWithHiddenCpuIntents } from "./cpuStep";

export function advanceVisualCpuTick(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, control: { running: boolean; paused: boolean }, policy?: CpuPolicy): CpuStepResult {
  if (!control.running || control.paused) return { state, runtime, applied: false };
  return advanceCpuOneStep(state, runtime, settings, policy);
}

export function advanceVisualCpuOneStep(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, policy?: CpuPolicy): CpuStepResult {
  return advanceCpuOneStep(state, runtime, settings, policy);
}

export type LocalGameResolutionResult = { state: GameState; runtime: CpuRuntime };

function resolveWithLocalGameRng(
  state: GameState,
  sourceRuntime: CpuRuntime,
  resolve: (state: GameState, rng: () => number) => GameState,
): LocalGameResolutionResult {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  return { state: resolve(state, createLocalGameRng(runtime)), runtime };
}

export function resolveLocalMovement(state: GameState, runtime: CpuRuntime) {
  return resolveWithLocalGameRng(state, runtime, resolveMovement);
}

export function resolveLocalStrategistActions(state: GameState, runtime: CpuRuntime) {
  return resolveWithLocalGameRng(state, runtime, resolveStrategistActions);
}
