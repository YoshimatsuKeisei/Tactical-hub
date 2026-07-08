import { UNIT_STATS } from "../constants";
import type { GameState, MovementIntent, Unit, UnitPosition } from "../types";
import { getBaseAtTile, getTile, getUnitAtBoardCell, positionKey, samePosition, tileKey } from "../utils/position";

const directions = [-1, 0, 1].flatMap((dx) => [-1, 0, 1].map((dy) => ({ dx, dy }))).filter(({ dx, dy }) => dx || dy);

function canEnterWater(unit: Unit) {
  return unit.type === "ninja";
}

function positionToCells(state: GameState, position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water") return [{ x: position.x, y: position.y }];
  if (position.kind === "base") {
    return state.bases.find((base) => base.id === position.baseId)?.coords ?? [];
  }
  return [];
}

function baseCanBeEntered(state: GameState, baseId: string, unit: Unit) {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  if (!base) return false;
  const hasEnemy = base.slots.some((slot) => {
    if (!slot.unitId) return false;
    const occupant = state.units.find((candidate) => candidate.id === slot.unitId);
    return occupant?.ownerTeamId !== unit.ownerTeamId;
  });
  return !hasEnemy && base.slots.some((slot) => !slot.unitId);
}

function firstEmptyBasePosition(state: GameState, baseId: string): UnitPosition | undefined {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  const slot = base?.slots.find((candidate) => !candidate.unitId);
  return base && slot ? { kind: "base", baseId, slotId: slot.id } : undefined;
}

export function isLegalDestination(state: GameState, unit: Unit, destination: UnitPosition): boolean {
  if (destination.kind === "tile" || destination.kind === "water") {
    const tile = getTile(state.map.tiles, destination.x, destination.y);
    if (!tile || tile.terrain === "outside" || tile.terrain === "base") return false;
    if (getUnitAtBoardCell(state, destination.x, destination.y)) return false;
    if (destination.kind === "water") return tile.terrain === "lake" && canEnterWater(unit);
    return ["road", "baseGate", "reorganize"].includes(tile.terrain);
  }

  if (destination.kind === "base") return baseCanBeEntered(state, destination.baseId, unit);
  return false;
}

export function getMovementCandidates(state: GameState, unitId: string): UnitPosition[] {
  const unit = state.units.find((candidate) => candidate.id === unitId);
  if (!unit || unit.position.kind === "removed" || unit.position.kind === "bridge") return [];

  const starts = positionToCells(state, unit.position);
  const maxMove = UNIT_STATS[unit.type].move;
  const visited = new Set<string>();
  const queue = starts.map((cell) => ({ ...cell, distance: 0 }));
  const candidates = new Map<string, UnitPosition>();

  for (const start of starts) visited.add(tileKey(start.x, start.y));

  while (queue.length) {
    const current = queue.shift()!;
    if (current.distance >= maxMove) continue;

    for (const { dx, dy } of directions) {
      const x = current.x + dx;
      const y = current.y + dy;
      const key = tileKey(x, y);
      if (visited.has(key)) continue;
      visited.add(key);

      const base = getBaseAtTile(state.bases, x, y);
      if (base) {
        const basePosition = firstEmptyBasePosition(state, base.id);
        if (basePosition && isLegalDestination(state, unit, basePosition)) candidates.set(positionKey(basePosition), basePosition);
        continue;
      }

      const tile = getTile(state.map.tiles, x, y);
      if (!tile) continue;
      const destination: UnitPosition = tile.terrain === "lake" ? { kind: "water", x, y } : { kind: "tile", x, y };
      if (!isLegalDestination(state, unit, destination)) continue;
      candidates.set(positionKey(destination), destination);
      queue.push({ x, y, distance: current.distance + 1 });
    }
  }

  return [...candidates.values()];
}

export function saveMovementIntent(state: GameState, intent: MovementIntent): GameState {
  const existing = state.turnState.actionIntents.find((candidate) => candidate.teamId === intent.teamId);
  const actionIntents = existing
    ? state.turnState.actionIntents.map((candidate) =>
        candidate.teamId === intent.teamId
          ? {
              ...candidate,
              movementIntents: [
                ...candidate.movementIntents.filter((movement) => movement.unitId !== intent.unitId),
                intent,
              ],
            }
          : candidate,
      )
    : [
        ...state.turnState.actionIntents,
        { teamId: intent.teamId, productionChoices: [], movementIntents: [intent], attackIntents: [] },
      ];
  return { ...state, turnState: { ...state.turnState, actionIntents } };
}

function applyPosition(state: GameState, unitId: string, to: UnitPosition) {
  const previous = state.units.find((unit) => unit.id === unitId)?.position;
  if (previous?.kind === "base") {
    const oldBase = state.bases.find((base) => base.id === previous.baseId);
    const oldSlot = oldBase?.slots.find((slot) => slot.id === previous.slotId);
    if (oldSlot) oldSlot.unitId = undefined;
  }
  if (to.kind === "base") {
    const newBase = state.bases.find((base) => base.id === to.baseId);
    const newSlot = newBase?.slots.find((slot) => slot.id === to.slotId);
    if (newSlot) newSlot.unitId = unitId;
  }
  state.units = state.units.map((unit) => (unit.id === unitId ? { ...unit, position: to } : unit));
}

export function resolveMovement(state: GameState): GameState {
  const next = structuredClone(state) as GameState;
  const intents = next.turnState.actionIntents
    .flatMap((intent) => intent.movementIntents)
    .sort((a, b) => {
      const aUnit = next.units.find((unit) => unit.id === a.unitId);
      const bUnit = next.units.find((unit) => unit.id === b.unitId);
      const priority = (aUnit ? UNIT_STATS[aUnit.type].priority : 99) - (bUnit ? UNIT_STATS[bUnit.type].priority : 99);
      return priority || a.unitId.localeCompare(b.unitId);
    });

  for (const intent of intents) {
    const unit = next.units.find((candidate) => candidate.id === intent.unitId);
    if (!unit || intent.stay) {
      next.logs.push({ id: `log-move-stay-${next.logs.length}`, turnNumber: next.turnNumber, type: "movement", message: `${intent.unitId} stayed.` });
      continue;
    }

    const stillAtSource = samePosition(unit.position, intent.from);
    const legal = stillAtSource && isLegalDestination(next, unit, intent.to);
    if (legal) {
      applyPosition(next, unit.id, intent.to);
      next.logs.push({
        id: `log-move-ok-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "movement",
        message: `${unit.id} moved to ${positionKey(intent.to)}.`,
        relatedIds: [unit.id],
      });
    } else {
      next.logs.push({
        id: `log-move-fail-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "movement",
        message: `${intent.unitId} failed to move to ${positionKey(intent.to)}.`,
        relatedIds: [intent.unitId],
      });
    }
  }

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) => ({ ...intent, movementIntents: [] }));
  next.turnNumber += 1;
  next.phase = "movement_input";
  next.turnState.turnNumber = next.turnNumber;
  next.turnState.phase = "movement_input";
  return next;
}
