import { UNIT_STATS } from "../constants";
import type { AttackIntent, AttackTarget, BattleEvent, GameState, Unit, UnitPosition } from "../types";
import { chebyshevDistance } from "../utils/distance";
import { positionKey } from "../utils/position";

const BASE_HIT_CHANCE = 1 / 6;

function isAlive(unit: Unit) {
  return unit.position.kind !== "removed" && unit.hp > 0;
}

function positionCells(state: GameState, position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water") return [{ x: position.x, y: position.y }];
  if (position.kind === "base") return state.bases.find((base) => base.id === position.baseId)?.coords ?? [];
  return [];
}

function minDistance(aCells: { x: number; y: number }[], bCells: { x: number; y: number }[]) {
  if (!aCells.length || !bCells.length) return Number.POSITIVE_INFINITY;
  return Math.min(...aCells.flatMap((a) => bCells.map((b) => chebyshevDistance(a, b))));
}

function targetForUnit(unit: Unit): AttackTarget {
  const position = unit.position;
  if (position.kind === "base") {
    return { kind: "unit", unitId: unit.id, baseId: position.baseId, slotId: position.slotId };
  }
  return { kind: "unit", unitId: unit.id };
}

function isProtectedByOkuzashiki(state: GameState, target: Unit) {
  const position = target.position;
  if (position.kind !== "base") return false;
  const base = state.bases.find((candidate) => candidate.id === position.baseId);
  if (!base || base.type !== "home" || position.slotId !== base.protectedSlotId) return false;

  return base.slots.some((slot) => {
    if (!slot.unitId || slot.id === position.slotId) return false;
    const other = state.units.find((unit) => unit.id === slot.unitId);
    return Boolean(other && isAlive(other) && other.ownerTeamId === target.ownerTeamId);
  });
}

export function getAttackCandidates(state: GameState, attackerUnitId: string): AttackTarget[] {
  const attacker = state.units.find((unit) => unit.id === attackerUnitId);
  if (!attacker || !isAlive(attacker)) return [];

  const range = UNIT_STATS[attacker.type].range;
  if (range <= 0) return [];

  const attackerCells = positionCells(state, attacker.position);
  return state.units
    .filter((target) => target.id !== attacker.id)
    .filter((target) => isAlive(target))
    .filter((target) => target.ownerTeamId !== attacker.ownerTeamId)
    .filter((target) => !isProtectedByOkuzashiki(state, target))
    .filter((target) => minDistance(attackerCells, positionCells(state, target.position)) <= range)
    .map(targetForUnit);
}

export function saveAttackIntent(state: GameState, intent: AttackIntent): GameState {
  const existing = state.turnState.actionIntents.find((candidate) => candidate.teamId === intent.teamId);
  const actionIntents = existing
    ? state.turnState.actionIntents.map((candidate) =>
        candidate.teamId === intent.teamId
          ? {
              ...candidate,
              attackIntents: [
                ...(candidate.attackIntents ?? []).filter((attack) => attack.attackerUnitId !== intent.attackerUnitId),
                intent,
              ],
            }
          : candidate,
      )
    : [
        ...state.turnState.actionIntents,
        { teamId: intent.teamId, productionChoices: [], movementIntents: [], attackIntents: [intent] },
      ];
  return { ...state, turnState: { ...state.turnState, actionIntents } };
}

function clearBaseSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const base = state.bases.find((candidate) => candidate.id === position.baseId);
  const slot = base?.slots.find((candidate) => candidate.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function defeatUnit(state: GameState, target: Unit) {
  clearBaseSlot(state, target.position);
  state.units = state.units.map((unit) =>
    unit.id === target.id ? { ...unit, hp: 0, position: { kind: "removed", reason: "defeated" } } : unit,
  );

  if (target.type === "king") {
    state.teams = state.teams.map((team) =>
      team.id === target.ownerTeamId ? { ...team, status: "defeated" } : team,
    );
    state.logs.push({
      id: `log-king-defeated-${state.logs.length}`,
      turnNumber: state.turnNumber,
      type: "battle",
      message: `${target.ownerTeamId} king was defeated.`,
      relatedIds: [target.id, target.ownerTeamId],
    });
  }
}

function targetStillLegal(state: GameState, attackerUnitId: string, target: AttackTarget) {
  return getAttackCandidates(state, attackerUnitId).some((candidate) => candidate.unitId === target.unitId);
}

export function resolveBattle(state: GameState, rng: () => number = Math.random): GameState {
  const next = structuredClone(state) as GameState;
  const intents = next.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []);
  const events: BattleEvent[] = intents.flatMap((intent, index) =>
    intent.pass || !intent.target
      ? []
      : [{ id: `battle-${next.turnNumber}-${index}`, attackerUnitId: intent.attackerUnitId, target: intent.target, hitChance: BASE_HIT_CHANCE }],
  );

  for (const event of events) {
    const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
    const target = next.units.find((unit) => unit.id === event.target.unitId);

    if (!attacker || !target || !isAlive(attacker) || !isAlive(target) || !targetStillLegal(next, attacker.id, event.target)) {
      next.logs.push({
        id: `log-battle-invalid-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "battle",
        message: `${event.attackerUnitId} attack against ${event.target.unitId} was invalid.`,
        relatedIds: [event.attackerUnitId, event.target.unitId],
      });
      continue;
    }

    const hit = rng() < event.hitChance;
    if (!hit) {
      next.logs.push({
        id: `log-battle-miss-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "battle",
        message: `${attacker.id} missed ${target.id}.`,
        relatedIds: [attacker.id, target.id],
      });
      continue;
    }

    const nextHp = target.hp - 1;
    next.units = next.units.map((unit) => (unit.id === target.id ? { ...unit, hp: nextHp } : unit));
    next.logs.push({
      id: `log-battle-hit-${next.logs.length}`,
      turnNumber: next.turnNumber,
      type: "battle",
      message: `${attacker.id} hit ${target.id} for 1 damage (${Math.max(nextHp, 0)} HP left).`,
      relatedIds: [attacker.id, target.id],
    });

    if (nextHp <= 0) {
      const updatedTarget = next.units.find((unit) => unit.id === target.id)!;
      defeatUnit(next, updatedTarget);
      next.logs.push({
        id: `log-unit-defeated-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "battle",
        message: `${target.id} was defeated at ${positionKey(target.position)}.`,
        relatedIds: [target.id],
      });
    }
  }

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) => ({ ...intent, attackIntents: [] }));
  next.phase = "attack_input";
  next.turnState.phase = "attack_input";
  return next;
}
