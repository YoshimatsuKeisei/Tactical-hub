import type { GameState, RewardPlacementRequest, SiegeState, Unit } from "../types";
import { getBaseConnectedRoadSectionIds, getRoadSectionIdForPosition } from "../utils/roadTopology";
import { resetSiegeState } from "./siege";

function alive(unit: Unit) { return unit.hp > 0 && unit.position.kind !== "removed"; }

function routeDistance(state: GameState, unit: Unit, targetBaseId: string) {
  if (unit.position.kind === "base" && unit.position.baseId === targetBaseId) return 0;
  const targetSections = new Set(getBaseConnectedRoadSectionIds(state, targetBaseId));
  const startSections = unit.position.kind === "base"
    ? getBaseConnectedRoadSectionIds(state, unit.position.baseId)
    : [getRoadSectionIdForPosition(state, unit.position)].filter((id): id is string => Boolean(id));
  if (startSections.some((id) => targetSections.has(id))) return 1;
  const adjacency = new Map<string, Set<string>>();
  for (const base of state.bases) {
    const sections = getBaseConnectedRoadSectionIds(state, base.id);
    for (const a of sections) for (const b of sections) if (a !== b) {
      if (!adjacency.has(a)) adjacency.set(a, new Set());
      adjacency.get(a)!.add(b);
    }
  }
  const queue = startSections.map((id) => ({ id, distance: 1 }));
  const seen = new Set(startSections);
  while (queue.length) {
    const current = queue.shift()!;
    if (targetSections.has(current.id)) return current.distance;
    for (const next of adjacency.get(current.id) ?? []) if (!seen.has(next)) {
      seen.add(next); queue.push({ id: next, distance: current.distance + 1 });
    }
  }
  return Number.POSITIVE_INFINITY;
}

function closestTeamDistance(state: GameState, teamId: string, baseId: string) {
  const distances = state.units.filter((unit) => unit.ownerTeamId === teamId && alive(unit)).map((unit) => routeDistance(state, unit, baseId));
  return distances.length ? Math.min(...distances) : Number.POSITIVE_INFINITY;
}

export function selectCaptureTeam(state: GameState, siege: SiegeState, candidateTeamIds: string[], rng: () => number = Math.random) {
  let candidates = [...new Set(candidateTeamIds)].filter((id) => id !== siege.defendingTeamId);
  if (candidates.length <= 1) return candidates[0];
  const record = (id: string) => siege.teamRecords.find((entry) => entry.teamId === id);
  const maxKills = Math.max(...candidates.map((id) => record(id)?.defenderKills ?? 0));
  candidates = candidates.filter((id) => (record(id)?.defenderKills ?? 0) === maxKills);
  if (candidates.length <= 1) return candidates[0];
  const minDistance = Math.min(...candidates.map((id) => closestTeamDistance(state, id, siege.baseId)));
  candidates = candidates.filter((id) => closestTeamDistance(state, id, siege.baseId) === minDistance);
  if (candidates.length <= 1) return candidates[0];
  const maxTurns = Math.max(...candidates.map((id) => record(id)?.effectiveAttackTurns ?? 0));
  candidates = candidates.filter((id) => (record(id)?.effectiveAttackTurns ?? 0) === maxTurns);
  return candidates.length === 1 ? candidates[0] : candidates[Math.min(candidates.length - 1, Math.floor(rng() * candidates.length))];
}

function requestRewards(state: GameState, siege: SiegeState, captureTeamId: string) {
  const captureKills = siege.teamRecords.find((entry) => entry.teamId === captureTeamId)?.defenderKills ?? 0;
  const requests: RewardPlacementRequest[] = [{ id: `reward-${state.turnNumber}-${state.rewardPlacementRequests.length}`, teamId: captureTeamId, rewardType: "capture_reward", sourceBaseId: siege.baseId, destinationKind: "fixed", fixedBaseId: siege.baseId, eligibleBaseIds: [siege.baseId], completed: false, expired: false }];
  for (const record of siege.teamRecords.filter((entry) => entry.teamId !== captureTeamId && entry.defenderKills > captureKills)) {
    const eligibleBaseIds = state.bases.filter((base) => base.ownerTeamId === record.teamId && base.slots.some((slot) => !slot.unitId)).map((base) => base.id);
    requests.push({ id: `reward-${state.turnNumber}-${state.rewardPlacementRequests.length + requests.length}`, teamId: record.teamId, rewardType: "contribution_compensation", sourceBaseId: siege.baseId, destinationKind: "selectable", eligibleBaseIds, completed: false, expired: eligibleBaseIds.length === 0, expirationReason: eligibleBaseIds.length ? undefined : "no_available_owned_base_slot" });
  }
  state.rewardPlacementRequests.push(...requests);
  for (const request of requests) state.logs.push({ id: `log-reward-request-${state.logs.length}`, turnNumber: state.turnNumber, type: "reward", message: `${request.rewardType === "capture_reward" ? "占領褒賞" : "攻略功労補償"}配置要求: ${request.teamId} / ${siege.baseId}`, relatedIds: [request.id, request.teamId, siege.baseId] });
}

export function transferBaseOwnership(state: GameState, baseId: string, teamId: string) {
  const base = state.bases.find((entry) => entry.id === baseId);
  if (!base || base.ownerTeamId === teamId) return;
  state.teams = state.teams.map((team) => ({ ...team, controlledBaseIds: team.id === teamId ? [...new Set([...team.controlledBaseIds, baseId])] : team.controlledBaseIds.filter((id) => id !== baseId) }));
  base.ownerTeamId = teamId;
  base.occupationPriorityTeamId = undefined;
  resetSiegeState(state, baseId, "owner_changed");
  state.logs.push({ id: `log-capture-owner-${state.logs.length}`, turnNumber: state.turnNumber, type: "capture", message: `所有権移転: ${baseId} → ${teamId}`, relatedIds: [baseId, teamId] });
}

export function completeSiegeCapture(state: GameState, siege: SiegeState, candidateTeamIds: string[], reason: "annihilation" | "combat_abandonment", rng: () => number = Math.random) {
  const teamId = selectCaptureTeam(state, siege, candidateTeamIds, rng);
  if (!teamId) {
    state.logs.push({ id: `log-capture-held-${state.logs.length}`, turnNumber: state.turnNumber, type: "capture", message: `占領チームを決定できないため処理保留: ${siege.baseId}`, relatedIds: [siege.baseId] });
    return false;
  }
  const snapshot = structuredClone(siege);
  state.logs.push({ id: `log-capture-${state.logs.length}`, turnNumber: state.turnNumber, type: "capture", message: `${reason === "annihilation" ? "守備隊全滅" : "戦闘中放棄"}による占領: ${snapshot.baseId} → ${teamId}`, relatedIds: [snapshot.baseId, teamId] });
  transferBaseOwnership(state, snapshot.baseId, teamId);
  requestRewards(state, snapshot, teamId);
  return true;
}
