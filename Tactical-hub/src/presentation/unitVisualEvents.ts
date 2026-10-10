import type { AttackIntent, GameState, Unit } from "../game/types";
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
  attackIntents: readonly AttackIntent[] = before.turnState.actionIntents.flatMap(
    (entry) => entry.attackIntents ?? [],
  ),
): UnitVisualPresentation {
  const events: UnitVisualEvent[] = [];
  const deathCandidates: { unit: Unit; coord: { x: number; y: number } }[] = [];
  const directionByUnit = new Map<string, UnitDirection>();
  const eventPrefix = `battle-${before.turnNumber}`;

  for (const intent of attackIntents) {
    if (intent.pass || !intent.target) continue;
    const attacker = before.units.find((unit) => unit.id === intent.attackerUnitId);
    if (!attacker || !getUnitSpriteKind(attacker)) continue;
    const direction = getUnitAttackDirection(before, attacker, intent);
    directionByUnit.set(attacker.id, direction);
    events.push({
      unitId: attacker.id,
      kind: "attack",
      direction,
      eventId: `${eventPrefix}:attack:${attacker.id}`,
    });
  }

  for (const previous of before.units) {
    if (!getUnitSpriteKind(previous)
      || previous.position.kind === "removed"
      || previous.hp <= 0) continue;
    const current = after.units.find((unit) => unit.id === previous.id);
    const removed = !current || current.hp <= 0 || current.position.kind === "removed";
    if (removed) {
      const direction = directionByUnit.get(previous.id) ?? 0;
      events.push({
        unitId: previous.id,
        kind: "death",
        direction,
        eventId: `${eventPrefix}:death:${previous.id}`,
      });
      const coord = getUnitPresentationCoord(before, previous);
      if (coord) deathCandidates.push({ unit: previous, coord });
    } else if (current.hp < previous.hp) {
      events.push({
        unitId: previous.id,
        kind: "hit",
        direction: directionByUnit.get(previous.id),
        eventId: `${eventPrefix}:hit:${previous.id}`,
      });
    }
  }

  const deathOverlays = deathCandidates.map(({ unit, coord }) => ({
    overlayId: `${eventPrefix}:overlay:${unit.id}`,
    unit: copyUnitForPresentation(unit),
    coord: { ...coord },
    direction: directionByUnit.get(unit.id) ?? 0,
    events: events.filter((event) => event.unitId === unit.id),
  }));
  return { events, deathOverlays };
}
