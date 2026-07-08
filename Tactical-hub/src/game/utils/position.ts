import type { Base, GameState, Tile, UnitPosition } from "../types";

export function samePosition(a: UnitPosition, b: UnitPosition) {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function positionKey(position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water") return `${position.x},${position.y}`;
  if (position.kind === "base") return `base:${position.baseId}:${position.slotId}`;
  if (position.kind === "bridge") return `bridge:${position.bridgeId}:${position.cellIndex}`;
  return `removed:${position.reason}`;
}

export function tileKey(x: number, y: number) {
  return `${x},${y}`;
}

export function getTile(mapTiles: Tile[], x: number, y: number) {
  return mapTiles.find((tile) => tile.x === x && tile.y === y);
}

export function getBaseAtTile(bases: Base[], x: number, y: number) {
  return bases.find((base) => base.coords.some((coord) => coord.x === x && coord.y === y));
}

export function getUnitAtBoardCell(state: GameState, x: number, y: number) {
  return state.units.find(
    (unit) =>
      (unit.position.kind === "tile" || unit.position.kind === "water") &&
      unit.position.x === x &&
      unit.position.y === y,
  );
}

export function getUnitInBaseSlot(state: GameState, baseId: string, slotId: string) {
  return state.units.find(
    (unit) => unit.position.kind === "base" && unit.position.baseId === baseId && unit.position.slotId === slotId,
  );
}

export function firstEmptySlot(base: Base) {
  return base.slots.find((slot) => !slot.unitId);
}
