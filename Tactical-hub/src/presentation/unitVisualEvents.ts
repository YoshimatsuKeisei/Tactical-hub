import type { AttackIntent, GameState, Unit } from "../game/types";
import type { BattlePresentationMetadata } from "../game/engine/battle";
import { isHeavyInfantry } from "../game/engine/heavyInfantry";
import { getPositionCoord } from "../game/utils/roadTopology";

export const UNIT_DIRECTION_LABELS = [
  "down",
  "down-right",
  "right",
  "up-right",
  "up",
  "up-left",
  "left",
  "down-left",
] as const;

export type UnitDirection = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type UnitSpriteKind =
  | "catapult"
  | "6Crusader"
  | "10Caster"
  | "1Knight"
  | "2Archer"
  | "4Paladin"
  | "7DeathKnight";
export type UnitVisualEventKind = "attack" | "hit" | "death";

export type UnitVisualEvent = {
  unitId: string;
  kind: UnitVisualEventKind;
  direction?: UnitDirection;
  eventId: string;
};

export type UnitDeathOverlay = {
  overlayId: string;
  unit: Unit;
  coord: { x: number; y: number };
  direction: UnitDirection;
  events: UnitVisualEvent[];
};

export type UnitVisualPresentation = {
  events: UnitVisualEvent[];
  deathOverlays: UnitDeathOverlay[];
};

export type UnitDirectionMap = ReadonlyMap<string, UnitDirection>;

export type UnitVisualPresentationOptions = {
  attackIntents?: readonly AttackIntent[];
  battle?: BattlePresentationMetadata;
  directions?: UnitDirectionMap;
};

// Long enough for Catapult attack+break and HD Enemy attack+damage+death queues.
export const UNIT_VISUAL_EVENT_RETENTION_MS = 7_000;

export function getUnitSpriteKind(unit: Unit): UnitSpriteKind | undefined {
  if (unit.type === "engineer") return "catapult";
  if (unit.type === "king") return "6Crusader";
  if (unit.type === "strategist") return "10Caster";
  if (isHeavyInfantry(unit)) return "4Paladin";
  if (unit.type === "infantry") return "1Knight";
  if (unit.type === "archer") return "2Archer";
  if (unit.type === "ninja") return "7DeathKnight";
  return undefined;
}

export function unitDirectionFromDelta(dx: number, dy: number): UnitDirection {
  const horizontal = Math.sign(dx);
  const vertical = Math.sign(dy);
  if (horizontal === 0 && vertical >= 0) return 0;
  if (horizontal > 0 && vertical > 0) return 1;
  if (horizontal > 0 && vertical === 0) return 2;
  if (horizontal > 0 && vertical < 0) return 3;
  if (horizontal === 0 && vertical < 0) return 4;
  if (horizontal < 0 && vertical < 0) return 5;
  if (horizontal < 0 && vertical === 0) return 6;
  return 7;
}

export function getUnitPresentationCoord(state: GameState, unit: Unit) {
  const position = unit.position;
  const direct = getPositionCoord(state, position);
  if (direct || position.kind !== "base") return direct;
  const base = state.bases.find((candidate) => candidate.id === position.baseId);
  const slot = base?.slots.find((candidate) => candidate.id === position.slotId);
  if (!base || !slot) return undefined;
  const minX = Math.min(...base.coords.map((coord) => coord.x));
  const minY = Math.min(...base.coords.map((coord) => coord.y));
  return { x: minX + slot.localCol, y: minY + slot.localRow };
}

export function getUnitAttackDirection(
  state: GameState,
  attacker: Unit,
  intent: AttackIntent,
  fallback: UnitDirection = 0,
): UnitDirection {
  if (!intent.target) return fallback;
  const target = state.units.find((unit) => unit.id === intent.target?.unitId);
  const from = getUnitPresentationCoord(state, attacker);
  const to = target ? getUnitPresentationCoord(state, target) : undefined;
  if (!from || !to || (from.x === to.x && from.y === to.y)) return fallback;
  return unitDirectionFromDelta(to.x - from.x, to.y - from.y);
}

function directionBetweenUnits(
  state: GameState,
  fromUnitId: string,
  toUnitId: string,
  fallback: UnitDirection,
) {
  const fromUnit = state.units.find((unit) => unit.id === fromUnitId);
  const toUnit = state.units.find((unit) => unit.id === toUnitId);
  const from = fromUnit ? getUnitPresentationCoord(state, fromUnit) : undefined;
  const to = toUnit ? getUnitPresentationCoord(state, toUnit) : undefined;
  if (!from || !to || (from.x === to.x && from.y === to.y)) return fallback;
  return unitDirectionFromDelta(to.x - from.x, to.y - from.y);
}

export function getInitialUnitDirection(state: GameState, unit: Unit): UnitDirection {
  const position = unit.position;
  if (position.kind === "base" && position.baseId === "neutral-center") return 0;
  const base = position.kind === "base"
    ? state.bases.find((candidate) => candidate.id === position.baseId)
    : undefined;
  const from = base?.coords.length ? {
    x: base.coords.reduce((sum, coord) => sum + coord.x, 0) / base.coords.length,
    y: base.coords.reduce((sum, coord) => sum + coord.y, 0) / base.coords.length,
  } : getUnitPresentationCoord(state, unit);
  if (!from) return 0;
  const center = { x: (state.map.width - 1) / 2, y: (state.map.height - 1) / 2 };
  if (from.x === center.x && from.y === center.y) return 0;
  return unitDirectionFromDelta(center.x - from.x, center.y - from.y);
}

export function createUnitDirectionMap(state: GameState): Map<string, UnitDirection> {
  return new Map(state.units
    .filter((unit) => unit.hp > 0 && unit.position.kind !== "removed")
    .map((unit) => [unit.id, getInitialUnitDirection(state, unit)]));
}

export function updateUnitDirectionMap(
  before: GameState,
  after: GameState,
  current: UnitDirectionMap,
): Map<string, UnitDirection> {
  if (before.turnNumber !== 1 && after.turnNumber === 1
    && after.logs.length === 1 && after.logs[0]?.id === "log-setup") {
    return createUnitDirectionMap(after);
  }
  const next = new Map(current);
  const visibleAfterIds = new Set(after.units
    .filter((unit) => unit.hp > 0 && unit.position.kind !== "removed")
    .map((unit) => unit.id));
  for (const unitId of next.keys()) if (!visibleAfterIds.has(unitId)) next.delete(unitId);
  for (const unit of after.units) {
    if (!visibleAfterIds.has(unit.id)) continue;
    const previous = before.units.find((candidate) => candidate.id === unit.id);
    if (!previous || previous.hp <= 0 || previous.position.kind === "removed") {
      next.set(unit.id, getInitialUnitDirection(after, unit));
      continue;
    }
    const from = getUnitPresentationCoord(before, previous);
    const to = getUnitPresentationCoord(after, unit);
    if (from && to && (from.x !== to.x || from.y !== to.y)) {
      next.set(unit.id, unitDirectionFromDelta(to.x - from.x, to.y - from.y));
    } else if (!next.has(unit.id)) {
      next.set(unit.id, getInitialUnitDirection(after, unit));
    }
  }
  return next;
}

export function applyUnitVisualEventDirections(
  current: UnitDirectionMap,
  events: readonly UnitVisualEvent[],
): Map<string, UnitDirection> {
  const next = new Map(current);
  for (const event of events) if (event.direction !== undefined) next.set(event.unitId, event.direction);
  return next;
}

function closestHitDirection(
  state: GameState,
  targetUnitId: string,
  battle: BattlePresentationMetadata | undefined,
): UnitDirection | undefined {
  const target = state.units.find((unit) => unit.id === targetUnitId);
  const targetCoord = target ? getUnitPresentationCoord(state, target) : undefined;
  if (!targetCoord || !battle) return undefined;
  const candidates = battle.hits
    .filter((event) => event.targetUnitId === targetUnitId)
    .flatMap((event) => {
      const attacker = state.units.find((unit) => unit.id === event.attackerUnitId);
      const coord = attacker ? getUnitPresentationCoord(state, attacker) : undefined;
      if (!coord || (coord.x === targetCoord.x && coord.y === targetCoord.y)) return [];
      const dx = coord.x - targetCoord.x;
      const dy = coord.y - targetCoord.y;
      return [{ attackerUnitId: event.attackerUnitId, dx, dy, distanceSquared: dx * dx + dy * dy }];
    })
    .sort((left, right) => left.distanceSquared - right.distanceSquared
      || left.attackerUnitId.localeCompare(right.attackerUnitId));
  const closest = candidates[0];
  return closest ? unitDirectionFromDelta(closest.dx, closest.dy) : undefined;
}

function copyUnitForPresentation(unit: Unit): Unit {
  return {
    ...unit,
    position: { ...unit.position },
    statuses: unit.statuses.map((status) => ({ ...status })),
  };
}

export function createUnitVisualPresentation(
  before: GameState,
  after: GameState,
  optionsOrIntents: UnitVisualPresentationOptions | readonly AttackIntent[] = {},
): UnitVisualPresentation {
  const options: UnitVisualPresentationOptions = Array.isArray(optionsOrIntents)
    ? { attackIntents: optionsOrIntents }
    : optionsOrIntents as UnitVisualPresentationOptions;
  const attackIntents = options.attackIntents ?? before.turnState.actionIntents.flatMap(
    (entry) => entry.attackIntents ?? [],
  );
  const events: UnitVisualEvent[] = [];
  const deathCandidates: { unit: Unit; coord: { x: number; y: number }; direction: UnitDirection }[] = [];
  const eventPrefix = `battle-${before.turnNumber}`;

  const attacks = options.battle?.attacks ?? attackIntents.flatMap((intent) =>
    intent.pass || !intent.target ? [] : [{
      eventId: eventPrefix,
      attackerUnitId: intent.attackerUnitId,
      targetUnitId: intent.target.unitId,
    }]);
  for (const attack of attacks) {
    const attacker = before.units.find((unit) => unit.id === attack.attackerUnitId);
    if (!attacker || !getUnitSpriteKind(attacker)) continue;
    const fallback = options.directions?.get(attacker.id) ?? getInitialUnitDirection(before, attacker);
    const direction = directionBetweenUnits(before, attacker.id, attack.targetUnitId, fallback);
    events.push({
      unitId: attacker.id,
      kind: "attack",
      direction,
      eventId: `${attack.eventId}:attack:${attacker.id}`,
    });
  }

  for (const previous of before.units) {
    if (!getUnitSpriteKind(previous)
      || previous.position.kind === "removed"
      || previous.hp <= 0) continue;
    const current = after.units.find((unit) => unit.id === previous.id);
    const removed = !current || current.hp <= 0 || current.position.kind === "removed";
    const previousDirection = options.directions?.get(previous.id) ?? getInitialUnitDirection(before, previous);
    const hitDirection = closestHitDirection(before, previous.id, options.battle);
    if (removed) {
      const direction = hitDirection ?? previousDirection;
      events.push({
        unitId: previous.id,
        kind: "death",
        direction,
        eventId: `${eventPrefix}:death:${previous.id}`,
      });
      const coord = getUnitPresentationCoord(before, previous);
      if (coord) deathCandidates.push({ unit: previous, coord, direction });
    } else if (current.hp < previous.hp) {
      events.push({
        unitId: previous.id,
        kind: "hit",
        direction: hitDirection ?? previousDirection,
        eventId: `${eventPrefix}:hit:${previous.id}`,
      });
    }
  }

  const deathOverlays = deathCandidates.map(({ unit, coord, direction }) => ({
    overlayId: `${eventPrefix}:overlay:${unit.id}`,
    unit: copyUnitForPresentation(unit),
    coord: { ...coord },
    direction,
    events: events.filter((event) => event.unitId === unit.id),
  }));
  return { events, deathOverlays };
}
