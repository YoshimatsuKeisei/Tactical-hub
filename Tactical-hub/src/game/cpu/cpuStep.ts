import { resolveBattle, saveAttackIntent } from "../engine/battle";
import { resolveStrategistActions, saveStrategistActionIntent, submitStrategistActions } from "../engine/construction";
import { saveMovementIntent, submitMovement } from "../engine/movement";
import { resolveProduction, saveProductionChoice, submitTeamProduction } from "../engine/production";
import { placeRewardUnit } from "../engine/reward";
import { saveTeleportIntent } from "../engine/teleport";
import type { GameState } from "../types";
import { positionKey } from "../utils/position";
import { getRandomCpuDecision } from "./randomCpuPolicy";
import type { CpuActionLog, CpuDecision, CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";

function contextKey(state: GameState) { return `${state.turnNumber}:${state.phase}`; }
export function syncCpuContext(runtime: CpuRuntime, state: GameState) {
  const key = contextKey(state);
  if (runtime.contextKey === key) return;
  runtime.contextKey = key;
  runtime.processedKeys = [];
  runtime.completedProductionTeamIds = [];
  runtime.completedAttackTeamIds = [];
  runtime.hiddenAttackIntents = [];
}
export type CpuStepInstrumentation = {
  logMode?: "full" | "ring" | "none";
  logLimit?: number;
  onPolicy?: (milliseconds: number) => void;
  onApply?: (milliseconds: number) => void;
  onLog?: (milliseconds: number) => void;
  onDecision?: (decision: CpuDecision) => void;
};
function log(runtime: CpuRuntime, state: GameState, teamId: string | undefined, action: string, detail?: string, error?: string, instrumentation?: CpuStepInstrumentation) {
  const started = instrumentation?.onLog ? performance.now() : 0;
  if (instrumentation?.logMode === "none") { instrumentation.onLog?.(performance.now() - started); return; }
  const entry: CpuActionLog = { id: `cpu-${runtime.logs.length}`, turnNumber: state.turnNumber, phase: state.phase, teamId, action, detail, error };
  runtime.logs.push(entry);
  if (instrumentation?.logMode === "ring" && runtime.logs.length > (instrumentation.logLimit ?? 50)) runtime.logs.splice(0, runtime.logs.length - (instrumentation.logLimit ?? 50));
  instrumentation?.onLog?.(performance.now() - started);
}
function injectedRng(runtime: CpuRuntime) {
  return () => {
    runtime.rngState = (Math.imul(runtime.rngState, 1664525) + 1013904223) >>> 0;
    return runtime.rngState / 0x1_0000_0000;
  };
}

export type CpuStepResult = { state: GameState; runtime: CpuRuntime; applied: boolean; waitingForHuman?: boolean };

export function advanceCpuOneStep(state: GameState, sourceRuntime: CpuRuntime, settings: CpuTeamSettings, policy: CpuPolicy = getRandomCpuDecision, instrumentation?: CpuStepInstrumentation): CpuStepResult {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  syncCpuContext(runtime, state);
  if (runtime.stoppedReason) return { state, runtime, applied: false };
  if (runtime.appliedStepCount >= runtime.maxAppliedSteps) {
    runtime.stoppedReason = `CPU safety limit ${runtime.maxAppliedSteps} reached`;
    log(runtime, state, undefined, "stop", undefined, runtime.stoppedReason, instrumentation);
    return { state, runtime, applied: false };
  }
  const policyStarted = instrumentation?.onPolicy ? performance.now() : 0;
  const decision = policy(state, runtime, settings);
  instrumentation?.onPolicy?.(performance.now() - policyStarted);
  if (!decision) return { state, runtime, applied: false, waitingForHuman: true };
  instrumentation?.onDecision?.(decision);
  const applyStarted = instrumentation?.onApply ? performance.now() : 0;
  let actionLogMs = 0;
  const actionInstrumentation = instrumentation ? { ...instrumentation, onLog: (milliseconds: number) => { actionLogMs += milliseconds; instrumentation.onLog?.(milliseconds); } } : undefined;
  const writeLog = (teamId: string | undefined, action: string, detail?: string, error?: string) => log(runtime, state, teamId, action, detail, error, actionInstrumentation);
  let next = state;
  switch (decision.kind) {
    case "production":
      if (decision.choice) next = saveProductionChoice(state, decision.choice);
      runtime.processedKeys.push(decision.actorKey);
      writeLog(decision.teamId, decision.choice ? "production" : "production pass", decision.choice ? `${decision.choice.baseId}:${decision.choice.unitType}` : undefined);
      break;
    case "resolve_production": next = resolveProduction(state); writeLog(undefined, "confirm production"); break;
    case "submit_team_production": next = submitTeamProduction(state, decision.teamId); writeLog(decision.teamId, "confirm production / skip"); break;
    case "movement": {
      const unit = state.units.find((entry) => entry.id === decision.unitId);
      if (decision.to && unit) next = saveMovementIntent(state, { teamId: decision.teamId, unitId: unit.id, from: unit.position, to: decision.to, stay: false });
      runtime.processedKeys.push(decision.actorKey);
      writeLog(decision.teamId, decision.to ? "move" : "movement pass", `${decision.unitId}${decision.to ? ` -> ${positionKey(decision.to)}` : ""}`);
      break;
    }
    case "teleport":
      if (decision.intent) next = saveTeleportIntent(state, decision.intent);
      runtime.processedKeys.push(decision.actorKey);
      writeLog(decision.teamId, decision.intent ? "teleport" : "teleport pass", decision.intent ? `${decision.intent.strategistUnitId}:${decision.intent.targetUnitId} -> ${positionKey(decision.intent.to)}` : decision.strategistUnitId);
      break;
    case "submit_movement": next = submitMovement(state, decision.teamId); writeLog(decision.teamId, "confirm movement"); break;
    case "attack":
      runtime.hiddenAttackIntents = [...runtime.hiddenAttackIntents.filter((entry) => entry.attackerUnitId !== decision.intent.attackerUnitId), decision.intent];
      runtime.processedKeys.push(decision.actorKey);
      writeLog(decision.teamId, "attack planned (hidden)", decision.intent.attackerUnitId);
      break;
    case "complete_attack_team": runtime.completedAttackTeamIds.push(decision.teamId); writeLog(decision.teamId, "confirm attacks"); break;
    case "resolve_battle": {
      for (const intent of runtime.hiddenAttackIntents) writeLog(intent.teamId, intent.pass ? "attack pass" : "attack", `${intent.attackerUnitId}${intent.target ? ` -> ${intent.target.unitId}` : ""}`);
      next = runtime.hiddenAttackIntents.reduce((current, intent) => saveAttackIntent(current, intent), state);
      next = resolveBattle(next, injectedRng(runtime));
      writeLog(undefined, "resolve simultaneous battle");
      break;
    }
    case "reward": next = placeRewardUnit(state, decision.requestId, decision.baseId, decision.unitType); writeLog(decision.teamId, "place reward", `${decision.requestId}:${decision.baseId}:${decision.unitType}`); break;
    case "strategist":
      next = saveStrategistActionIntent(state, decision.intent);
      runtime.processedKeys.push(decision.actorKey);
      writeLog(decision.teamId, decision.intent.action, decision.intent.constructionId ?? decision.intent.tiles?.map((cell) => `${cell.x},${cell.y}`).join("/") ?? decision.intent.strategistUnitId);
      break;
    case "submit_strategist": next = submitStrategistActions(state, decision.teamId); writeLog(decision.teamId, "confirm strategist actions"); break;
    case "resolve_strategists": next = resolveStrategistActions(state, injectedRng(runtime)); writeLog(undefined, "resolve strategist actions"); break;
  }
  instrumentation?.onApply?.(Math.max(0, performance.now() - applyStarted - actionLogMs));
  runtime.appliedStepCount += 1;
  return { state: next, runtime, applied: true };
}

export function resolveBattleWithHiddenCpuIntents(state: GameState, sourceRuntime: CpuRuntime) {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  for (const intent of runtime.hiddenAttackIntents) log(runtime, state, intent.teamId, intent.pass ? "attack pass" : "attack", `${intent.attackerUnitId}${intent.target ? ` -> ${intent.target.unitId}` : ""}`);
  const withCpu = runtime.hiddenAttackIntents.reduce((current, intent) => saveAttackIntent(current, intent), state);
  const next = resolveBattle(withCpu, injectedRng(runtime));
  log(runtime, state, undefined, "resolve simultaneous battle after human confirmation");
  runtime.hiddenAttackIntents = [];
  return { state: next, runtime };
}
