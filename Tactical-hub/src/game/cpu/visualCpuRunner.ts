import { resolveBattle, saveAttackIntent } from "../engine/battle";
import { resolveStrategistActions, saveStrategistActionIntent, submitStrategistActions } from "../engine/construction";
import { saveMovementIntent, submitMovement } from "../engine/movement";
import { resolveProduction, saveProductionChoice, submitTeamProduction } from "../engine/production";
import { placeRewardUnit } from "../engine/reward";
import { saveTeleportIntent } from "../engine/teleport";
import type { GameState } from "../types";
import { positionKey } from "../utils/position";
import { getRandomCpuDecision } from "./randomCpuPolicy";
import type { CpuActionLog, CpuRuntime, CpuTeamSettings } from "./types";

function contextKey(state: GameState) { return `${state.turnNumber}:${state.phase}`; }
function syncContext(runtime: CpuRuntime, state: GameState) {
  const key = contextKey(state);
  if (runtime.contextKey === key) return;
  runtime.contextKey = key;
  runtime.processedKeys = [];
  runtime.completedProductionTeamIds = [];
  runtime.completedAttackTeamIds = [];
  runtime.hiddenAttackIntents = [];
}
function log(runtime: CpuRuntime, state: GameState, teamId: string | undefined, action: string, detail?: string, error?: string) {
  const entry: CpuActionLog = { id: `cpu-${runtime.logs.length}`, turnNumber: state.turnNumber, phase: state.phase, teamId, action, detail, error };
  runtime.logs.push(entry);
}
function injectedRng(runtime: CpuRuntime) {
  return () => {
    runtime.rngState = (Math.imul(runtime.rngState, 1664525) + 1013904223) >>> 0;
    return runtime.rngState / 0x1_0000_0000;
  };
}

export type CpuStepResult = { state: GameState; runtime: CpuRuntime; applied: boolean; waitingForHuman?: boolean };

export function advanceVisualCpuTick(
  state: GameState,
  runtime: CpuRuntime,
  settings: CpuTeamSettings,
  control: { running: boolean; paused: boolean },
): CpuStepResult {
  if (!control.running || control.paused) return { state, runtime, applied: false };
  return advanceVisualCpuOneStep(state, runtime, settings);
}

export function advanceVisualCpuOneStep(state: GameState, sourceRuntime: CpuRuntime, settings: CpuTeamSettings): CpuStepResult {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  syncContext(runtime, state);
  if (runtime.stoppedReason) return { state, runtime, applied: false };
  if (runtime.appliedStepCount >= runtime.maxAppliedSteps) {
    runtime.stoppedReason = `CPU safety limit ${runtime.maxAppliedSteps} reached`;
    log(runtime, state, undefined, "stop", undefined, runtime.stoppedReason);
    return { state, runtime, applied: false };
  }
  const decision = getRandomCpuDecision(state, runtime, settings);
  if (!decision) return { state, runtime, applied: false, waitingForHuman: true };
  let next = state;
  switch (decision.kind) {
    case "production":
      if (decision.choice) next = saveProductionChoice(state, decision.choice);
      runtime.processedKeys.push(decision.actorKey);
      log(runtime, state, decision.teamId, decision.choice ? "production" : "production pass", decision.choice ? `${decision.choice.baseId}:${decision.choice.unitType}` : undefined);
      break;
    case "resolve_production":
      next = resolveProduction(state); log(runtime, state, undefined, "confirm production"); break;
    case "submit_team_production":
      next = submitTeamProduction(state, decision.teamId); log(runtime, state, decision.teamId, "confirm production / skip"); break;
    case "movement": {
      const unit = state.units.find((entry) => entry.id === decision.unitId);
      if (decision.to && unit) next = saveMovementIntent(state, { teamId: decision.teamId, unitId: unit.id, from: unit.position, to: decision.to, stay: false });
      runtime.processedKeys.push(decision.actorKey);
      log(runtime, state, decision.teamId, decision.to ? "move" : "movement pass", `${decision.unitId}${decision.to ? ` -> ${positionKey(decision.to)}` : ""}`);
      break;
    }
    case "teleport":
      if (decision.intent) next = saveTeleportIntent(state, decision.intent);
      runtime.processedKeys.push(decision.actorKey);
      log(runtime, state, decision.teamId, decision.intent ? "teleport" : "teleport pass", decision.intent ? `${decision.intent.strategistUnitId}:${decision.intent.targetUnitId} -> ${positionKey(decision.intent.to)}` : decision.strategistUnitId);
      break;
    case "submit_movement":
      next = submitMovement(state, decision.teamId); log(runtime, state, decision.teamId, "confirm movement"); break;
    case "attack":
      runtime.hiddenAttackIntents = [...runtime.hiddenAttackIntents.filter((entry) => entry.attackerUnitId !== decision.intent.attackerUnitId), decision.intent];
      runtime.processedKeys.push(decision.actorKey);
      log(runtime, state, decision.teamId, "attack planned (hidden)", decision.intent.attackerUnitId);
      break;
    case "complete_attack_team":
      runtime.completedAttackTeamIds.push(decision.teamId); log(runtime, state, decision.teamId, "confirm attacks"); break;
    case "resolve_battle": {
      for (const intent of runtime.hiddenAttackIntents)
        log(runtime, state, intent.teamId, intent.pass ? "attack pass" : "attack", `${intent.attackerUnitId}${intent.target ? ` -> ${intent.target.unitId}` : ""}`);
      next = runtime.hiddenAttackIntents.reduce((current, intent) => saveAttackIntent(current, intent), state);
      next = resolveBattle(next, injectedRng(runtime));
      log(runtime, state, undefined, "resolve simultaneous battle");
      break;
    }
    case "reward":
      next = placeRewardUnit(state, decision.requestId, decision.baseId, decision.unitType);
      log(runtime, state, decision.teamId, "place reward", `${decision.requestId}:${decision.baseId}:${decision.unitType}`);
      break;
    case "strategist":
      next = saveStrategistActionIntent(state, decision.intent);
      runtime.processedKeys.push(decision.actorKey);
      log(runtime, state, decision.teamId, decision.intent.action, decision.intent.constructionId ?? decision.intent.tiles?.map((cell) => `${cell.x},${cell.y}`).join("/") ?? decision.intent.strategistUnitId);
      break;
    case "submit_strategist":
      next = submitStrategistActions(state, decision.teamId); log(runtime, state, decision.teamId, "confirm strategist actions"); break;
    case "resolve_strategists":
      next = resolveStrategistActions(state, injectedRng(runtime)); log(runtime, state, undefined, "resolve strategist actions"); break;
  }
  runtime.appliedStepCount += 1;
  return { state: next, runtime, applied: true };
}

export function resolveBattleWithHiddenCpuIntents(state: GameState, sourceRuntime: CpuRuntime) {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  for (const intent of runtime.hiddenAttackIntents)
    log(runtime, state, intent.teamId, intent.pass ? "attack pass" : "attack", `${intent.attackerUnitId}${intent.target ? ` -> ${intent.target.unitId}` : ""}`);
  const withCpu = runtime.hiddenAttackIntents.reduce((current, intent) => saveAttackIntent(current, intent), state);
  const next = resolveBattle(withCpu, injectedRng(runtime));
  log(runtime, state, undefined, "resolve simultaneous battle after human confirmation");
  runtime.hiddenAttackIntents = [];
  return { state: next, runtime };
}
