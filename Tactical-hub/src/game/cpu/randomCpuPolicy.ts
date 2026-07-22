import { getAttackCandidates, getTeamAttackerUnitIds } from "../engine/battle";
import { getBuilderUnits, getStrategistActionCandidatesForUnit } from "../engine/construction";
import { getMovementCandidates, getTeamMovementUnitIds } from "../engine/movement";
import { getProductionCandidatesForBase } from "../engine/production";
import { isTeamProductionPending } from "../engine/productionSchedule";
import { getRewardPlacementCandidates } from "../engine/reward";
import { getTeleportDestinationCandidates, getTeleportStrategists, getTeleportTargetCandidates } from "../engine/teleport";
import type { GameState } from "../types";
import { positionKey } from "../utils/position";
import type { CpuDecision, CpuRuntime, CpuTeamSettings } from "./types";
import { withLegalProfileSink } from "./legalEnumerationProfile";

function nextRandom(runtime: CpuRuntime) {
  runtime.rngState = (Math.imul(runtime.rngState, 1664525) + 1013904223) >>> 0;
  return runtime.rngState / 0x1_0000_0000;
}

function choose<T>(runtime: CpuRuntime, values: T[]) {
  return values[Math.floor(nextRandom(runtime) * values.length)];
}

const activeTeamIds = (state: GameState) => state.teams.filter((team) => team.status === "active").map((team) => team.id);
const isCpu = (settings: CpuTeamSettings, teamId: string) => settings[teamId] === "random_cpu";

export type LegalEnumerationTiming = { category: string; phase: GameState["phase"]; milliseconds: number };
type PolicyTiming = { enumerationMs: number; record?: (entry: LegalEnumerationTiming) => void };
const timed = <T>(timing: PolicyTiming | undefined, phase: GameState["phase"], category: string, operation: () => T) => {
  if (!timing) return operation();
  const start = performance.now();
  const value = operation();
  const milliseconds = performance.now() - start;
  timing.enumerationMs += milliseconds;
  timing.record?.({ category, phase, milliseconds });
  return value;
};

function decideRandomCpu(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, timing?: PolicyTiming): CpuDecision | undefined {
  if (state.phase === "production") {
    const teamId = activeTeamIds(state).find((id) => isCpu(settings, id) && !runtime.completedProductionTeamIds.includes(id));
    if (teamId) {
      const baseIds = timed(timing, state.phase, "unitBaseEquipmentSearch", () => state.bases.filter((base) => base.ownerTeamId === teamId).map((base) => base.id).sort());
      for (const baseId of baseIds) {
        if (runtime.processedKeys.includes(`production:${teamId}:${baseId}`)) continue;
        const candidates = timed(timing, state.phase, "production", () => getProductionCandidatesForBase(state, teamId, baseId));
        if (candidates.length) return { kind: "production", teamId, actorKey: `production:${teamId}:${baseId}`, choice: choose(runtime, candidates) };
      }
      runtime.completedProductionTeamIds.push(teamId);
      return decideRandomCpu(state, runtime, settings, timing);
    }
    if (activeTeamIds(state).every((id) => isCpu(settings, id))) return timed(timing, state.phase, "confirmPass", () => ({ kind: "resolve_production", teamId: "all" } as const));
    return undefined;
  }

  if (state.phase === "movement_input") {
    const teamId = state.currentMovementTeamId;
    if (!teamId || !isCpu(settings, teamId)) return undefined;
    if (isTeamProductionPending(state, teamId)) {
      const baseIds = timed(timing, state.phase, "unitBaseEquipmentSearch", () => state.bases.filter((base) => base.ownerTeamId === teamId).map((base) => base.id).sort());
      for (const baseId of baseIds) {
        if (runtime.processedKeys.includes(`movement-production:${teamId}:${baseId}`)) continue;
        const candidates = timed(timing, state.phase, "production", () => getProductionCandidatesForBase(state, teamId, baseId));
        if (candidates.length) return { kind: "production", teamId, actorKey: `movement-production:${teamId}:${baseId}`, choice: choose(runtime, candidates) };
      }
      return timed(timing, state.phase, "confirmPass", () => ({ kind: "submit_team_production", teamId } as const));
    }
    const unitId = timed(timing, state.phase, "unitBaseEquipmentSearch", () => getTeamMovementUnitIds(state, teamId).find((id) => !runtime.processedKeys.includes(`movement:${teamId}:${id}`)));
    if (unitId) {
      const unit = state.units.find((entry) => entry.id === unitId);
      const category = unit?.statuses.some((status) => status.kind === "retreating") ? "retreat" : unit?.type === "ninja" ? "ninjaMovement" : "movement";
      const destinations = timed(timing, state.phase, category, () => getMovementCandidates(state, unitId));
      timed(timing, state.phase, "postProcessing", () => destinations.sort((left, right) => positionKey(left).localeCompare(positionKey(right))));
      const options = timed(timing, state.phase, "confirmPass", () => [undefined, ...destinations]);
      return { kind: "movement", teamId, actorKey: `movement:${teamId}:${unitId}`, unitId, to: choose(runtime, options) };
    }
    const teleportId = timed(timing, state.phase, "unitBaseEquipmentSearch", () => getTeleportStrategists(state, teamId).map((entry) => entry.id).find((id) => !runtime.processedKeys.includes(`teleport:${teamId}:${id}`)));
    if (!teleportId) return timed(timing, state.phase, "confirmPass", () => ({ kind: "submit_movement", teamId } as const));
    const teleport = timed(timing, state.phase, "teleport", () => ({ targets: getTeleportTargetCandidates(state, teleportId), destinations: getTeleportDestinationCandidates(state, teleportId) }));
    const target = teleport.targets.length ? choose(runtime, [undefined, ...teleport.targets]) : undefined;
    const destination = target && teleport.destinations.length ? choose(runtime, teleport.destinations) : undefined;
    return {
      kind: "teleport",
      teamId,
      actorKey: `teleport:${teamId}:${teleportId}`,
      strategistUnitId: teleportId,
      intent: target && destination ? { teamId, strategistUnitId: teleportId, targetUnitId: target.id, to: destination } : undefined,
    };
  }

  if (state.phase === "attack_input") {
    const teamId = activeTeamIds(state).find((id) => isCpu(settings, id) && !runtime.completedAttackTeamIds.includes(id));
    if (teamId) {
      const attackerUnitId = timed(timing, state.phase, "unitBaseEquipmentSearch", () => getTeamAttackerUnitIds(state, teamId).find((id) => !runtime.processedKeys.includes(`attack:${teamId}:${id}`)));
      if (!attackerUnitId) return timed(timing, state.phase, "confirmPass", () => ({ kind: "complete_attack_team", teamId } as const));
      const targets = timed(timing, state.phase, "attack", () => getAttackCandidates(state, attackerUnitId));
      const target = choose(runtime, timed(timing, state.phase, "confirmPass", () => [undefined, ...targets]));
      return {
        kind: "attack",
        teamId,
        actorKey: `attack:${teamId}:${attackerUnitId}`,
        intent: { teamId, attackerUnitId, target, pass: !target },
      };
    }
    if (activeTeamIds(state).every((id) => isCpu(settings, id))) return timed(timing, state.phase, "confirmPass", () => ({ kind: "resolve_battle", teamId: "all" } as const));
    return undefined;
  }

  if (state.phase === "reward_placement") {
    const candidate = state.teams
      .filter((team) => team.status === "active" && isCpu(settings, team.id))
      .flatMap((team) => timed(timing, state.phase, "reward", () => getRewardPlacementCandidates(state, team.id)).map((placement) => ({ teamId: team.id, ...placement })))
      .sort((left, right) => `${left.requestId}:${left.baseId}:${left.unitType}`.localeCompare(`${right.requestId}:${right.baseId}:${right.unitType}`));
    if (!candidate.length) return undefined;
    return { kind: "reward", ...choose(runtime, candidate) };
  }

  if (state.phase === "strategist_action_input") {
    const teamId = activeTeamIds(state).find((id) => isCpu(settings, id) && !state.strategistSubmittedTeamIds.includes(id));
    if (!teamId) return undefined;
    const builderId = timed(timing, state.phase, "unitBaseEquipmentSearch", () => getBuilderUnits(state, teamId).map((unit) => unit.id).sort().find((id) => !runtime.processedKeys.includes(`strategist:${teamId}:${id}`)));
    if (!builderId) return timed(timing, state.phase, "confirmPass", () => ({ kind: "submit_strategist", teamId } as const));
    const candidates = timed(timing, state.phase, "constructionStrategist", () => getStrategistActionCandidatesForUnit(state, teamId, builderId));
    return { kind: "strategist", teamId, actorKey: `strategist:${teamId}:${builderId}`, intent: choose(runtime, candidates) };
  }

  if (state.phase === "strategist_action_resolution") return timed(timing, state.phase, "confirmPass", () => ({ kind: "resolve_strategists", teamId: "all" } as const));
  return undefined;
}

export function getRandomCpuDecision(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings): CpuDecision | undefined {
  return decideRandomCpu(state, runtime, settings);
}

export function createProfiledRandomCpuPolicy(record: (enumerationMs: number, policyTotalMs: number, details: LegalEnumerationTiming[]) => void) {
  return (state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings) => {
    const details: LegalEnumerationTiming[] = [];
    const timing: PolicyTiming = { enumerationMs: 0, record: (entry) => details.push(entry) };
    const start = performance.now();
    const decision = withLegalProfileSink((category, milliseconds) => details.push({ category, phase: state.phase, milliseconds }), () => decideRandomCpu(state, runtime, settings, timing));
    record(timing.enumerationMs, performance.now() - start, details);
    return decision;
  };
}
