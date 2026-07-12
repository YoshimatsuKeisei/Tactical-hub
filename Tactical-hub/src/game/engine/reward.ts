import { PRODUCIBLE_UNIT_TYPES, UNIT_STATS } from "../constants";
import type { GameState, Unit, UnitType } from "../types";
import { getAvailableProductionTypes } from "./production";

export function getPendingRewardRequests(state: GameState) { return state.rewardPlacementRequests.filter((request) => !request.completed && !request.expired); }

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
    next.phase = next.phaseAfterRewards ?? "movement_input"; next.turnState.phase = next.phase; next.phaseAfterRewards = undefined;
  }
  return next;
}
