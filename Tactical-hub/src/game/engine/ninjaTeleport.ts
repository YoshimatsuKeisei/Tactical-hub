import type { GameState, Unit, UnitPosition } from "../types";
import { getBridgePositionAt, getPositionCoord } from "../utils/roadTopology";
import { getTile, getUnitAtBoardCell, positionKey, tileKey } from "../utils/position";
import { createTeamVisibleState } from "../visibility";
import { getOperationalAreaTiles } from "./construction";

const ADJACENT_DIRECTIONS = [-1, 0, 1]
  .flatMap((dx) => [-1, 0, 1].map((dy) => ({ dx, dy })))
  .filter(({ dx, dy }) => dx !== 0 || dy !== 0);

export function hasUsedNinjaTeleport(state: GameState, unitId: string) {
  return state.ninjaTeleportUsedUnitIds?.includes(unitId) === true;
}

function canUseNinjaTeleport(state: GameState, unit: Unit | undefined): unit is Unit {
  return Boolean(
    unit &&
    unit.type === "ninja" &&
    unit.hp > 0 &&
    unit.position.kind !== "water" &&
    unit.position.kind !== "removed" &&
    !hasUsedNinjaTeleport(state, unit.id) &&
    state.phase === "movement_input" &&
    state.currentMovementTeamId === unit.ownerTeamId &&
    !state.movedUnitIdsThisMovementPhase.includes(unit.id),
  );
}

function groundPositionAt(state: GameState, x: number, y: number): UnitPosition | undefined {
  const bridge = getBridgePositionAt(state, x, y);
  if (bridge) return bridge;
  const tile = getTile(state.map.tiles, x, y);
  return tile && ["road", "baseGate", "reorganize"].includes(tile.terrain)
    ? { kind: "tile", x, y }
    : undefined;
}

function isUnoccupiedGroundCell(state: GameState, x: number, y: number) {
  return !getUnitAtBoardCell(state, x, y) && !state.constructions.some(
    (entry) => entry.active && entry.kind === "obstacle" && entry.tiles.some((cell) => cell.x === x && cell.y === y),
  );
}

/**
 * Returns one-use ninja teleport destinations. Enemy discovery and occupancy
 * are evaluated through the acting team's visible state to avoid hidden-unit
 * information leaking into legal actions.
 */
export function getNinjaTeleportDestinations(state: GameState, unitId: string): UnitPosition[] {
  const source = state.units.find((candidate) => candidate.id === unitId);
  if (!canUseNinjaTeleport(state, source)) return [];

  const visibleState = createTeamVisibleState(state, source.ownerTeamId);
  const ninja = visibleState.units.find((candidate) => candidate.id === unitId);
  if (!ninja) return [];
  const operationalCells = getOperationalAreaTiles(visibleState, ninja.ownerTeamId);
  const operationalKeys = new Set(operationalCells.map((cell) => tileKey(cell.x, cell.y)));
  const destinations = new Map<string, UnitPosition>();

  const enemiesInOperationalArea = visibleState.units.filter((unit) => {
    if (unit.ownerTeamId === ninja.ownerTeamId || unit.hp <= 0 || unit.position.kind === "removed") return false;
    const coord = getPositionCoord(visibleState, unit.position);
    return Boolean(coord && operationalKeys.has(tileKey(coord.x, coord.y)));
  });

  for (const enemy of enemiesInOperationalArea) {
    const enemyCoord = getPositionCoord(visibleState, enemy.position);
    if (!enemyCoord) continue;
    for (const { dx, dy } of ADJACENT_DIRECTIONS) {
      const x = enemyCoord.x + dx;
      const y = enemyCoord.y + dy;
      if (!operationalKeys.has(tileKey(x, y)) || !isUnoccupiedGroundCell(visibleState, x, y)) continue;
      const destination = groundPositionAt(visibleState, x, y);
      if (destination) destinations.set(positionKey(destination), destination);
    }
  }

  return [...destinations.values()].sort((left, right) => positionKey(left).localeCompare(positionKey(right)));
}

export function isNinjaTeleportDestination(state: GameState, unitId: string, destination: UnitPosition) {
  return getNinjaTeleportDestinations(state, unitId).some((candidate) => positionKey(candidate) === positionKey(destination));
}
