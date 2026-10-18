import type { Base, GameState, Unit, UnitPosition, UnitTurnFlags } from "../types";
import { chebyshevDistance } from "../utils/distance";

type DistanceResult = {
  distance: number;
  baseIds: string[];
};

function isAlive(unit: Unit) {
  return unit.position.kind !== "removed" && unit.hp > 0;
}

function positionCells(state: GameState, position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water") return [{ x: position.x, y: position.y }];
  if (position.kind === "base") return state.bases.find((base) => base.id === position.baseId)?.coords ?? [];
  if (position.kind === "bridge") {
    const bridge = state.constructions.find((entry) => entry.active && entry.kind === "bridge" && entry.id === position.bridgeId);
    return bridge?.tiles[position.cellIndex] ? [bridge.tiles[position.cellIndex]] : [];
  }
  return [];
}

function minDistanceToBase(state: GameState, position: UnitPosition, base: Base) {
  if (position.kind === "base" && position.baseId === base.id) return 0;
  const ownCells = positionCells(state, position);
  if (!ownCells.length || !base.coords.length) return Number.POSITIVE_INFINITY;
  return Math.min(...ownCells.flatMap((own) => base.coords.map((cell) => chebyshevDistance(own, cell))));
}

function nearestDistance(state: GameState, position: UnitPosition, bases: Base[]): DistanceResult | undefined {
  const distances = bases
    .map((base) => ({ baseId: base.id, distance: minDistanceToBase(state, position, base) }))
    .filter((entry) => Number.isFinite(entry.distance));
  if (!distances.length) return undefined;
  const distance = Math.min(...distances.map((entry) => entry.distance));
  return {
    distance,
    baseIds: distances.filter((entry) => entry.distance === distance).map((entry) => entry.baseId).sort((a, b) => a.localeCompare(b)),
  };
}

export function getBaseControllerTeamId(state: GameState, base: Base): string | undefined {
  const controlledByTeam = state.teams.find((team) => team.controlledBaseIds.includes(base.id));
  if (controlledByTeam) return controlledByTeam.id;
  if (base.ownerTeamId) return base.ownerTeamId;
  return state.teams.find((team) => team.homeBaseId === base.id)?.id;
}

export function getEnemyControlledBases(state: GameState, teamId: string) {
  return state.bases.filter((base) => {
    const controllerTeamId = getBaseControllerTeamId(state, base);
    if (!controllerTeamId || controllerTeamId === teamId) return false;
    return state.teams.find((team) => team.id === controllerTeamId)?.status !== "eliminated";
  });
}

function getNearestEnemyBaseDistance(state: GameState, teamId: string, position: UnitPosition) {
  return nearestDistance(state, position, getEnemyControlledBases(state, teamId));
}

/**
 * Removes only serialized data that belonged to the retired retreat-mode
 * rules. Unit placement and independent battle history remain unchanged.
 */
export function clearLegacyRetreatState(state: GameState) {
  state.units = state.units.map((unit) => {
    const statuses = unit.statuses.filter((status) => status.kind !== "retreating");
    return statuses.length === unit.statuses.length ? unit : { ...unit, statuses };
  });
  state.unitTurnFlags = state.unitTurnFlags.map((flag) => ({
    ...flag,
    retreatFriendlyBaseIdsAtEligibility: [],
    retreatHostileBaseIdsAtEligibility: [],
    retreatEligible: false,
    retreatEligibilityReason: "retreat mode removed",
  }));
}

/**
 * Battle participation and enemy-base proximity remain observation history.
 * Legacy retreat-only fields stay structurally present for PPO/checkpoint
 * compatibility but are always disabled.
 */
export function buildUnitTurnFlag(
  state: GameState,
  unit: Unit,
  battleTurnNumber: number,
  wasAliveAtBattleStart: boolean,
  attackedInPreviousBattle: boolean,
  wasTargetedInPreviousBattle: boolean,
  positionAtBattleStart = unit.position,
): UnitTurnFlags {
  const enemyBaseDistance = getNearestEnemyBaseDistance(state, unit.ownerTeamId, positionAtBattleStart)?.distance
    ?? Number.POSITIVE_INFINITY;
  return {
    unitId: unit.id,
    battleTurnNumber,
    positionAtBattleStart,
    enemyBaseDistanceAtBattleStart: Number.isFinite(enemyBaseDistance) ? enemyBaseDistance : undefined,
    enemyBaseWithin3AtBattleStart: enemyBaseDistance <= 3,
    retreatFriendlyBaseIdsAtEligibility: [],
    retreatHostileBaseIdsAtEligibility: [],
    wasAliveAtBattleStart,
    survivedPreviousBattle: isAlive(unit),
    attackedInPreviousBattle,
    wasTargetedInPreviousBattle,
    retreatEligible: false,
    retreatEligibilityReason: "retreat mode removed",
  };
}
