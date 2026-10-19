import { PRODUCIBLE_UNIT_TYPES, UNIT_STATS } from "../constants";
import { UNIFIED_STRATEGIST_ROLE, type GameState, type RewardPlacementRequest, type RewardType, type Unit, type UnitType } from "../types";
import { getAvailableProductionTypes } from "./production";
import { beginStrategistActionPhase } from "./construction";

function isEligibleRewardRecipient(state: GameState, teamId: string) {
  const team = state.teams.find((candidate) => candidate.id === teamId);
  return Boolean(team && team.status === "active" && !team.isNeutral);
}

function getRewardRequestInvalidReason(state: GameState, request: RewardPlacementRequest) {
  if (!isEligibleRewardRecipient(state, request.teamId)) return "recipient_not_active_player";
  if (request.destinationKind === "fixed") {
    const base = request.fixedBaseId
      ? state.bases.find((candidate) => candidate.id === request.fixedBaseId)
      : undefined;
    if (!base || base.ownerTeamId !== request.teamId) return "fixed_base_not_owned_by_recipient";
  }
  if (!getEligibleRewardBaseIds(state, request).length) return "no_available_owned_base_slot";
  return undefined;
}

export function getPendingRewardRequests(state: GameState) {
  return state.rewardPlacementRequests.filter((request) =>
    !request.completed
    && !request.expired
    && !getRewardRequestInvalidReason(state, request));
}

export function getEligibleRewardBaseIds(state: GameState, request: RewardPlacementRequest) {
  if (!isEligibleRewardRecipient(state, request.teamId)) return [];
  if (request.destinationKind === "fixed") {
    const base = request.fixedBaseId
      ? state.bases.find((candidate) => candidate.id === request.fixedBaseId)
      : undefined;
    return base
      && base.ownerTeamId === request.teamId
      && base.slots.some((slot) => !slot.unitId)
      ? [base.id]
      : [];
  }
  return state.bases
    .filter((base) => base.ownerTeamId === request.teamId && base.slots.some((slot) => !slot.unitId))
    .map((base) => base.id)
    .sort((left, right) => left.localeCompare(right));
}

export function getRewardPlacementCandidates(state: GameState, teamId: string) {
  if (state.phase !== "reward_placement") return [];
  return getPendingRewardRequests(state)
    .filter((request) => request.teamId === teamId)
    .sort((left, right) => left.id.localeCompare(right.id))
    .flatMap((request) => getEligibleRewardBaseIds(state, request)
      .flatMap((baseId) => getAvailableProductionTypes(state, teamId, baseId).map((unitType) => ({ requestId: request.id, baseId, unitType }))));
}

export function enqueueRewardRequest(state: GameState, input: { teamId: string; rewardType: RewardType; sourceBaseId: string; sourceKingUnitId?: string; fixedBaseId?: string }) {
  if (!isEligibleRewardRecipient(state, input.teamId)) return undefined;
  if (input.fixedBaseId && state.bases.find((base) => base.id === input.fixedBaseId)?.ownerTeamId !== input.teamId) return undefined;
  const destinationKind = input.fixedBaseId ? "fixed" as const : "selectable" as const;
  const eligibleBaseIds = input.fixedBaseId
    ? state.bases.find((base) => base.id === input.fixedBaseId)?.slots.some((slot) => !slot.unitId) ? [input.fixedBaseId] : []
    : state.bases.filter((base) => base.ownerTeamId === input.teamId && base.slots.some((slot) => !slot.unitId)).map((base) => base.id);
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

export function expireInvalidRewardRequests(state: GameState) {
  let expired = false;
  for (const request of state.rewardPlacementRequests) {
    if (request.completed || request.expired) continue;
    const reason = getRewardRequestInvalidReason(state, request);
    if (!reason) continue;
    request.expired = true;
    request.expirationReason = reason;
    expired = true;
    state.logs.push({
      id: `log-reward-expired-${state.logs.length}`,
      turnNumber: state.turnNumber,
      type: "reward",
      message: `Invalid reward request expired: ${request.id} / ${reason}`,
      relatedIds: [request.id, request.teamId, request.fixedBaseId ?? request.sourceBaseId],
    });
  }
  return expired;
}

function resumeAfterRewardPlacement(state: GameState) {
  if (state.phase !== "reward_placement" || getPendingRewardRequests(state).length) return state;
  state.phase = state.phaseAfterRewards ?? "strategist_action_input";
  state.turnState.phase = state.phase;
  state.phaseAfterRewards = undefined;
  return state.phase === "strategist_action_input" ? beginStrategistActionPhase(state) : state;
}

export function normalizeRewardPlacementState(state: GameState): GameState {
  const hasInvalidRequest = state.rewardPlacementRequests.some((request) =>
    !request.completed
    && !request.expired
    && Boolean(getRewardRequestInvalidReason(state, request)));
  const shouldResumeSettledRewardPhase = state.phase === "reward_placement"
    && getPendingRewardRequests(state).length === 0;
  if (!hasInvalidRequest && !shouldResumeSettledRewardPhase) return state;
  const next = structuredClone(state) as GameState;
  expireInvalidRewardRequests(next);
  return resumeAfterRewardPlacement(next);
}

export function placeRewardUnit(state: GameState, requestId: string, baseId: string, unitType: UnitType): GameState {
  if (state.phase !== "reward_placement") return state;
  const next = structuredClone(state) as GameState;
  const invalidRequestsExpired = expireInvalidRewardRequests(next);
  const request = next.rewardPlacementRequests.find((entry) => entry.id === requestId);
  const base = next.bases.find((entry) => entry.id === baseId);
  const slot = base?.slots.find((entry) => !entry.unitId);
  const eligible = request && getEligibleRewardBaseIds(next, request).includes(baseId);
  const legalTypes = request ? getAvailableProductionTypes(next, request.teamId, baseId) : [];
  if (!request || request.completed || request.expired || !base || base.ownerTeamId !== request.teamId || !slot || !eligible || !PRODUCIBLE_UNIT_TYPES.includes(unitType) || !legalTypes.includes(unitType)) {
    return invalidRequestsExpired ? resumeAfterRewardPlacement(next) : state;
  }
  const unit: Unit = { id: `${base.id}-${unitType}-reward-${next.units.length + 1}`, ownerTeamId: request.teamId, type: unitType, hp: UNIT_STATS[unitType].hp, position: { kind: "base", baseId, slotId: slot.id }, statuses: [] };
  if (unitType === "strategist") unit.role = UNIFIED_STRATEGIST_ROLE;
  slot.unitId = unit.id; next.units.push(unit); request.selectedUnitType = unitType; request.completed = true;
  next.logs.push({ id: `log-reward-place-${next.logs.length}`, turnNumber: next.turnNumber, type: "reward", message: `${request.rewardType === "capture_reward" ? "褒賞駒" : "補償駒"}の配置: ${request.teamId} / ${baseId}`, relatedIds: [request.id, unit.id, baseId] });
  expireInvalidRewardRequests(next);
  return resumeAfterRewardPlacement(next);
}
