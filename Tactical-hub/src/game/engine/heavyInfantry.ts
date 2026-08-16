import type { GameState, Unit } from "../types";
import { getPositionCoord } from "../utils/roadTopology";
import { isRetreating } from "./retreat";

export function isHeavyInfantry(unit: Unit | undefined): unit is Unit & { formation: "heavy" } {
  return Boolean(unit?.type === "infantry" && unit.formation === "heavy");
}

function isAvailableNormalInfantry(state: GameState, unit: Unit, teamId: string) {
  if (unit.ownerTeamId !== teamId || unit.type !== "infantry" || unit.formation || unit.hp <= 0) return false;
  if (unit.position.kind === "removed" || unit.position.kind === "water" || isRetreating(unit)) return false;
  if (state.teleportIntents.some((intent) => intent.targetUnitId === unit.id)) return false;
  return !state.turnState.actionIntents.some((intent) =>
    intent.movementIntents.some((movement) => movement.unitId === unit.id),
  );
}

function occupiedCells(state: GameState, unit: Unit) {
  if (unit.position.kind === "base") {
    const baseId = unit.position.baseId;
    return state.bases.find((base) => base.id === baseId)?.coords ?? [];
  }
  const coord = getPositionCoord(state, unit.position);
  return coord ? [coord] : [];
}

function areMergeAdjacent(state: GameState, left: Unit, right: Unit) {
  if (left.position.kind === "base" && right.position.kind === "base" && left.position.baseId === right.position.baseId) return true;
  return occupiedCells(state, left).some((leftCell) => occupiedCells(state, right).some((rightCell) =>
    Math.max(Math.abs(leftCell.x - rightCell.x), Math.abs(leftCell.y - rightCell.y)) === 1,
  ));
}

export function getHeavyInfantryMergeCandidates(state: GameState, unitId: string): Unit[] {
  if (state.phase !== "movement_input") return [];
  const unit = state.units.find((candidate) => candidate.id === unitId);
  if (!unit || unit.ownerTeamId !== state.currentMovementTeamId || !isAvailableNormalInfantry(state, unit, unit.ownerTeamId)) return [];
  return state.units
    .filter((candidate) => candidate.id !== unit.id
      && isAvailableNormalInfantry(state, candidate, unit.ownerTeamId)
      && areMergeAdjacent(state, unit, candidate))
    .sort((left, right) => left.id.localeCompare(right.id));
}

export function mergeHeavyInfantry(state: GameState, retainedUnitId: string, mergedUnitId: string): GameState {
  if (!getHeavyInfantryMergeCandidates(state, retainedUnitId).some((unit) => unit.id === mergedUnitId)) return state;
  const next = structuredClone(state) as GameState;
  const retained = next.units.find((unit) => unit.id === retainedUnitId)!;
  const merged = next.units.find((unit) => unit.id === mergedUnitId)!;
  if (merged.position.kind === "base") {
    const { baseId, slotId } = merged.position;
    const slot = next.bases.find((base) => base.id === baseId)?.slots.find((candidate) => candidate.id === slotId);
    if (slot) slot.unitId = undefined;
  }
  retained.formation = "heavy";
  retained.hp = 2;
  merged.hp = 0;
  merged.position = { kind: "removed", reason: "merged" };
  merged.statuses = [];
  next.movedUnitIdsThisMovementPhase = [...new Set([...next.movedUnitIdsThisMovementPhase, retained.id, merged.id])];
  next.logs.push({
    id: `log-heavy-merge-${next.logs.length}`,
    turnNumber: next.turnNumber,
    type: "movement",
    message: `${retained.id} merged with ${merged.id} into heavy infantry.`,
    relatedIds: [retained.id, merged.id],
  });
  return next;
}
