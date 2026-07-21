import { getTeamAttackCandidates } from "../engine/battle";
import { getStrategistActionCandidates } from "../engine/construction";
import { getTeamMovementCandidates } from "../engine/movement";
import { getProductionCandidates } from "../engine/production";
import { isTeamProductionPending } from "../engine/productionSchedule";
import { getRewardPlacementCandidates } from "../engine/reward";
import { getTeamTeleportCandidates } from "../engine/teleport";
import type { GameState } from "../types";
import type { CpuDecision, CpuRuntime, CpuTeamSettings } from "./types";

function nextRandom(runtime: CpuRuntime) {
  runtime.rngState = (Math.imul(runtime.rngState, 1664525) + 1013904223) >>> 0;
  return runtime.rngState / 0x1_0000_0000;
}

function choose<T>(runtime: CpuRuntime, values: T[]) {
  return values[Math.floor(nextRandom(runtime) * values.length)];
}

const activeTeamIds = (state: GameState) => state.teams.filter((team) => team.status === "active").map((team) => team.id);
const isCpu = (settings: CpuTeamSettings, teamId: string) => settings[teamId] === "random_cpu";

type PolicyTiming = { enumerationMs: number };
const timed = <T>(timing: PolicyTiming | undefined, operation: () => T) => {
  if (!timing) return operation();
  const start = performance.now();
  const value = operation();
  timing.enumerationMs += performance.now() - start;
  return value;
};

function decideRandomCpu(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings, timing?: PolicyTiming): CpuDecision | undefined {
  if (state.phase === "production") {
    const teamId = activeTeamIds(state).find((id) => isCpu(settings, id) && !runtime.completedProductionTeamIds.includes(id));
    if (teamId) {
      const byBase = new Map<string, ReturnType<typeof getProductionCandidates>>();
      for (const candidate of timed(timing, () => getProductionCandidates(state, teamId))) byBase.set(candidate.baseId, [...(byBase.get(candidate.baseId) ?? []), candidate]);
      const baseId = [...byBase.keys()].sort().find((id) => !runtime.processedKeys.includes(`production:${teamId}:${id}`));
      if (baseId) return { kind: "production", teamId, actorKey: `production:${teamId}:${baseId}`, choice: choose(runtime, byBase.get(baseId)!) };
      runtime.completedProductionTeamIds.push(teamId);
      return decideRandomCpu(state, runtime, settings, timing);
    }
    if (activeTeamIds(state).every((id) => isCpu(settings, id))) return { kind: "resolve_production", teamId: "all" };
    return undefined;
  }

  if (state.phase === "movement_input") {
    const teamId = state.currentMovementTeamId;
    if (!teamId || !isCpu(settings, teamId)) return undefined;
    if (isTeamProductionPending(state, teamId)) {
      const byBase = new Map<string, ReturnType<typeof getProductionCandidates>>();
      for (const candidate of timed(timing, () => getProductionCandidates(state, teamId))) byBase.set(candidate.baseId, [...(byBase.get(candidate.baseId) ?? []), candidate]);
      const baseId = [...byBase.keys()].sort().find((id) => !runtime.processedKeys.includes(`movement-production:${teamId}:${id}`));
      if (baseId) return { kind: "production", teamId, actorKey: `movement-production:${teamId}:${baseId}`, choice: choose(runtime, byBase.get(baseId)!) };
      return { kind: "submit_team_production", teamId };
    }
    const movement = timed(timing, () => getTeamMovementCandidates(state, teamId));
    const teleports = timed(timing, () => getTeamTeleportCandidates(state, teamId));
    const actors = [
      ...movement.map((entry) => ({ kind: "movement" as const, key: `movement:${teamId}:${entry.unitId}`, entry })),
      ...teleports.map((entry) => ({ kind: "teleport" as const, key: `teleport:${teamId}:${entry.strategistUnitId}`, entry })),
    ].sort((left, right) => left.key.localeCompare(right.key));
    const actor = actors.find((entry) => !runtime.processedKeys.includes(entry.key));
    if (!actor) return { kind: "submit_movement", teamId };
    if (actor.kind === "movement") {
      const options = [undefined, ...actor.entry.destinations];
      return { kind: "movement", teamId, actorKey: actor.key, unitId: actor.entry.unitId, to: choose(runtime, options) };
    }
    const target = actor.entry.targets.length ? choose(runtime, [undefined, ...actor.entry.targets]) : undefined;
    const destination = target && actor.entry.destinations.length ? choose(runtime, actor.entry.destinations) : undefined;
    return {
      kind: "teleport",
      teamId,
      actorKey: actor.key,
      strategistUnitId: actor.entry.strategistUnitId,
      intent: target && destination ? { teamId, strategistUnitId: actor.entry.strategistUnitId, targetUnitId: target.id, to: destination } : undefined,
    };
  }

  if (state.phase === "attack_input") {
    const teamId = activeTeamIds(state).find((id) => isCpu(settings, id) && !runtime.completedAttackTeamIds.includes(id));
    if (teamId) {
      const attacker = timed(timing, () => getTeamAttackCandidates(state, teamId)).find((entry) => !runtime.processedKeys.includes(`attack:${teamId}:${entry.attackerUnitId}`));
      if (!attacker) return { kind: "complete_attack_team", teamId };
      const target = choose(runtime, [undefined, ...attacker.targets]);
      return {
        kind: "attack",
        teamId,
        actorKey: `attack:${teamId}:${attacker.attackerUnitId}`,
        intent: { teamId, attackerUnitId: attacker.attackerUnitId, target, pass: !target },
      };
    }
    if (activeTeamIds(state).every((id) => isCpu(settings, id))) return { kind: "resolve_battle", teamId: "all" };
    return undefined;
  }

  if (state.phase === "reward_placement") {
    const candidate = state.teams
      .filter((team) => team.status === "active" && isCpu(settings, team.id))
      .flatMap((team) => timed(timing, () => getRewardPlacementCandidates(state, team.id)).map((placement) => ({ teamId: team.id, ...placement })))
      .sort((left, right) => `${left.requestId}:${left.baseId}:${left.unitType}`.localeCompare(`${right.requestId}:${right.baseId}:${right.unitType}`));
    if (!candidate.length) return undefined;
    return { kind: "reward", ...choose(runtime, candidate) };
  }

  if (state.phase === "strategist_action_input") {
    const teamId = activeTeamIds(state).find((id) => isCpu(settings, id) && !state.strategistSubmittedTeamIds.includes(id));
    if (!teamId) return undefined;
    const candidates = timed(timing, () => getStrategistActionCandidates(state, teamId));
    const builderId = [...new Set(candidates.map((candidate) => candidate.strategistUnitId))].sort().find((id) => !runtime.processedKeys.includes(`strategist:${teamId}:${id}`));
    if (!builderId) return { kind: "submit_strategist", teamId };
    return { kind: "strategist", teamId, actorKey: `strategist:${teamId}:${builderId}`, intent: choose(runtime, candidates.filter((candidate) => candidate.strategistUnitId === builderId)) };
  }

  if (state.phase === "strategist_action_resolution") return { kind: "resolve_strategists", teamId: "all" };
  return undefined;
}

export function getRandomCpuDecision(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings): CpuDecision | undefined {
  return decideRandomCpu(state, runtime, settings);
}

export function createProfiledRandomCpuPolicy(record: (enumerationMs: number, policyTotalMs: number) => void) {
  return (state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings) => {
    const timing: PolicyTiming = { enumerationMs: 0 };
    const start = performance.now();
    const decision = decideRandomCpu(state, runtime, settings, timing);
    record(timing.enumerationMs, performance.now() - start);
    return decision;
  };
}
