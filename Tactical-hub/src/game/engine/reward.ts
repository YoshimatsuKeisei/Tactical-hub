import { PRODUCIBLE_UNIT_TYPES, UNIT_STATS } from "../constants";
import type { GameState, RewardPlacementRequest, RewardType, Unit, UnitType } from "../types";
import { getAvailableProductionTypes } from "./production";
import { beginStrategistActionPhase } from "./construction";

export function getPendingRewardRequests(state: GameState) { return state.rewardPlacementRequests.filter((request) => !request.completed && !request.expired); }

export function getRewardPlacementCandidates(state: GameState, teamId: string) {
  if (state.phase !== "reward_placement") return [];
  return getPendingRewardRequests(state)
    .filter((request) => request.teamId === teamId)
    .sort((left, right) => left.id.localeCompare(right.id))
    .flatMap((request) => request.eligibleBaseIds
      .slice()
      .sort((left, right) => left.localeCompare(right))
      .flatMap((baseId) => getAvailableProductionTypes(state, teamId, baseId).map((unitType) => ({ requestId: request.id, baseId, unitType }))));
}

export function enqueueRewardRequest(state: GameState, input: { teamId: string; rewardType: RewardType; sourceBaseId: string; sourceKingUnitId?: string; fixedBaseId?: string }) {
  const destinationKind = input.fixedBaseId ? "fixed" as const : "selectable" as const;
  const eligibleBaseIds = input.fixedBaseId ? [input.fixedBaseId] : state.bases.filter((base) => base.ownerTeamId === input.teamId && base.slots.some((slot) => !slot.unitId)).map((base) => base.id);
  const request: RewardPlacementRequest = {
    id: `reward-${state.turnNumber}-${state.rewardPlacementRequests.length}`,
    teamId: input.teamId,
    rewardType: input.rewardType,
    sourceBaseId: input.sourceBaseId,
    sourceKingUnitId: input.sourceKingUnitId,
    destinationKind,
    fixedBaseId: input.fixedBaseId,
    eligibleBaseIds,
    completed: false,
    expired: eligibleBaseIds.length === 0,
    expirationReason: eligibleBaseIds.length ? undefined : "no_available_owned_base_slot",
  };
  state.rewardPlacementRequests.push(request);
  state.logs.push({ id: `log-reward-request-${state.logs.length}`, turnNumber: state.turnNumber, type: "reward", message: `褒賞配置要求: ${request.rewardType} / ${request.teamId}`, relatedIds: [request.id, request.teamId, input.sourceKingUnitId ?? input.sourceBaseId] });
  return request;
}

export function placeRewardUnit(state: GameState, requestId: string, baseId: string, unitType: UnitType): GameState {
  const next = structuredClone(state) as GameState;
  const request = next.rewardPlacementRequests.find((entry) => entry.id === requestId);
  const base = next.bases.find((entry) => entry.id === baseId);
  const slot = base?.slots.find((entry) => !entry.unitId);
  const eligible = request && (request.destinationKind === "fixed" ? request.fixedBaseId === baseId : request.eligibleBaseIds.includes(baseId));
  const legalTypes = request ? getAvailableProductionTypes(next, request.teamId, baseId) : [];
  if (!request || request.completed || request.expired || !base || !slot || !eligible || !PRODUCIBLE_UNIT_TYPES.includes(unitType) || !legalTypes.includes(unitType)) return state;
  const unit: Unit = { id: `${base.id}-${unitType}-reward-${next.units.length + 1}`, ownerTeamId: request.teamId, type: unitType, hp: UNIT_STATS[unitType].hp, position: { kind: "base", baseId, slotId: slot.id }, statuses: [] };
  if (unitType === "strategist") unit.role = "encourage";
  slot.unitId = unit.id; next.units.push(unit); request.selectedUnitType = unitType; request.completed = true;
  next.logs.push({ id: `log-reward-place-${next.logs.length}`, turnNumber: next.turnNumber, type: "reward", message: `${request.rewardType === "capture_reward" ? "褒賞駒" : "補償駒"}の配置: ${request.teamId} / ${baseId}`, relatedIds: [request.id, unit.id, baseId] });
  if (!getPendingRewardRequests(next).length) {
    next.phase = next.phaseAfterRewards ?? "strategist_action_input"; next.turnState.phase = next.phase; next.phaseAfterRewards = undefined;
    if (next.phase === "strategist_action_input") return beginStrategistActionPhase(next);
  }
  return next;
}
