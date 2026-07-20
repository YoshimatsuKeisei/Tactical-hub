import type { GameState, Unit, UnitPosition } from "../types";
import { chebyshevDistance } from "../utils/distance";
import { tileKey } from "../utils/position";

function isAlive(unit: Unit) {
  return unit.position.kind !== "removed" && unit.hp > 0;
}

export function positionCells(state: GameState, position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water") return [{ x: position.x, y: position.y }];
  if (position.kind === "base") return state.bases.find((base) => base.id === position.baseId)?.coords ?? [];
  if (position.kind === "bridge") {
    const bridge = state.constructions.find((entry) => entry.active && entry.kind === "bridge" && entry.id === position.bridgeId);
    return bridge?.tiles[position.cellIndex] ? [bridge.tiles[position.cellIndex]] : [];
  }
  return [];
}

function minDistanceToCells(position: UnitPosition, cells: { x: number; y: number }[], state: GameState) {
  const ownCells = positionCells(state, position);
  if (!ownCells.length || !cells.length) return Number.POSITIVE_INFINITY;
  return Math.min(...ownCells.flatMap((own) => cells.map((cell) => chebyshevDistance(own, cell))));
}

function isEncourageStrategist(unit: Unit) {
  return unit.type === "strategist" && unit.role === "encourage";
}

function distanceToNearestBase(state: GameState, unit: Unit) {
  if (unit.position.kind === "base") return 0;
  const ownedBases = state.bases.filter((base) => base.ownerTeamId === unit.ownerTeamId);
  return Math.min(...ownedBases.map((base) => minDistanceToCells(unit.position, base.coords, state)));
}

function isEncouragablePosition(position: UnitPosition) {
  return position.kind === "tile" || position.kind === "base" || position.kind === "bridge";
}

export function getEncourageRadius(state: GameState, strategist: Unit): 1 | 2 {
  return distanceToNearestBase(state, strategist) <= 2 ? 1 : 2;
}

export function isWithinEncourageRange(state: GameState, unit: Unit, strategist: Unit) {
  return minDistanceToCells(unit.position, positionCells(state, strategist.position), state) <= getEncourageRadius(state, strategist);
}

export function isUnitEncouragedByStrategist(state: GameState, unit: Unit, strategist: Unit) {
  if (!isAlive(unit) || !isAlive(strategist)) return false;
  if (!isEncourageStrategist(strategist)) return false;
  if (unit.id === strategist.id) return false;
  if (unit.ownerTeamId !== strategist.ownerTeamId) return false;
  if (!isEncouragablePosition(unit.position)) return false;

  return isWithinEncourageRange(state, unit, strategist);
}

export function isUnitEncouraged(state: GameState, unit: Unit) {
  return state.units.some((strategist) => isUnitEncouragedByStrategist(state, unit, strategist));
}

export function getEncouragedUnitIds(state: GameState) {
  return new Set(state.units.filter((unit) => isUnitEncouraged(state, unit)).map((unit) => unit.id));
}

export function getEncouragedUnitIdsByStrategist(state: GameState, strategist: Unit) {
  return state.units
    .filter((unit) => isUnitEncouragedByStrategist(state, unit, strategist))
    .map((unit) => unit.id)
    .sort((a, b) => a.localeCompare(b));
}

export function getEncourageAreaTileKeys(state: GameState, strategist: Unit) {
  if (!isAlive(strategist) || !isEncourageStrategist(strategist)) return new Set<string>();

  const radius = getEncourageRadius(state, strategist);
  const originCells = positionCells(state, strategist.position);
  const keys = new Set<string>();

  for (const tile of state.map.tiles) {
    if (!originCells.some((origin) => chebyshevDistance(origin, tile) <= radius)) continue;
    keys.add(tileKey(tile.x, tile.y));
  }

  return keys;
}
