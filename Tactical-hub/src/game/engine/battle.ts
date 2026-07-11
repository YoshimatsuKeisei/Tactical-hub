import { UNIT_STATS } from "../constants";
import type { AttackIntent, AttackTarget, BattleEvent, GameState, Unit, UnitPosition, UnitType } from "../types";
import { chebyshevDistance } from "../utils/distance";
import { positionKey } from "../utils/position";

type AttackDenominatorContext = {
  targetInBase: boolean;
  encouraged?: boolean;
};

type BattleLogDraft = {
  message: string;
  relatedIds?: string[];
};

const SUCCESS_DENOMINATORS: Partial<Record<UnitType, Partial<Record<UnitType, number>>>> = {
  infantry: {
    archer: 5,
    cavalry: 7,
    infantry: 6,
    strategist: 5,
    ninja: 5,
    king: 7,
    engineer: 5,
    apprentice_ninja: 5,
  },
  archer: {
    infantry: 7,
    cavalry: 5,
    archer: 6,
    strategist: 5,
    ninja: 5,
    king: 7,
    engineer: 5,
    apprentice_ninja: 5,
  },
  cavalry: {
    infantry: 5,
    archer: 7,
    cavalry: 6,
    strategist: 5,
    ninja: 5,
    king: 7,
    engineer: 5,
    apprentice_ninja: 5,
  },
  ninja: {
    infantry: 7,
    archer: 7,
    cavalry: 7,
    strategist: 5,
    ninja: 6,
    king: 7,
    engineer: 5,
    apprentice_ninja: 6,
  },
  king: {
    infantry: 5,
    archer: 5,
    cavalry: 5,
    strategist: 5,
    ninja: 5,
    king: 6,
    engineer: 5,
    apprentice_ninja: 5,
  },
  apprentice_ninja: {
    infantry: 6,
    archer: 6,
    cavalry: 6,
    strategist: 6,
    ninja: 6,
    king: 6,
    engineer: 6,
    apprentice_ninja: 6,
  },
};

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

export function getBaseAttackDenominator(
  attackerType: UnitType,
  targetType: UnitType,
  context: AttackDenominatorContext,
): number | null {
  if (attackerType === "strategist") return null;
  if (attackerType === "engineer") {
    if (!context.targetInBase) return null;
    return targetType === "engineer" ? 6 : 5;
  }
  return SUCCESS_DENOMINATORS[attackerType]?.[targetType] ?? 6;
}

export function applyEncouragementToDenominator(denominator: number, encouraged: boolean) {
  return encouraged ? Math.max(1, denominator - 1) : denominator;
}

export function getFinalAttackDenominator(
  attackerType: UnitType,
  targetType: UnitType,
  context: AttackDenominatorContext,
): number | null {
  const baseDenominator = getBaseAttackDenominator(attackerType, targetType, context);
  if (baseDenominator === null) return null;
  return applyEncouragementToDenominator(baseDenominator, Boolean(context.encouraged));
}

function getAttackDenominators(attacker: Unit, target: Unit, encouraged = false) {
  const context = { targetInBase: target.position.kind === "base", encouraged };
  const baseSuccessDenominator = getBaseAttackDenominator(attacker.type, target.type, context);
  if (baseSuccessDenominator === null) return undefined;
  const finalSuccessDenominator = applyEncouragementToDenominator(baseSuccessDenominator, encouraged);
  return { baseSuccessDenominator, finalSuccessDenominator, encouraged };
}

function isEncouragedForBattle(_state: GameState, attacker: Unit) {
  return attacker.position.kind !== "water" && attacker.statuses.some((status) => status.kind === "encouraged");
}

function canAttackByPositionRule(attacker: Unit, target: Unit) {
  if (attacker.position.kind === "water") {
    return attacker.type === "ninja" && target.type === "ninja" && target.position.kind === "water";
  }
  if (target.position.kind === "water") return false;
  return true;
}

function canAttackUnit(state: GameState, attacker: Unit, target: Unit) {
  if (!canAttackByPositionRule(attacker, target)) return false;
  if (!getAttackDenominators(attacker, target, isEncouragedForBattle(state, attacker))) return false;
  return true;
}

function candidateDistance(state: GameState, attacker: Unit, target: Unit) {
  return minDistance(positionCells(state, attacker.position), positionCells(state, target.position));
}

function targetSortKey(state: GameState, attacker: Unit, target: Unit): [number, number, number, number, number, number, string] {
  const baseHp = UNIT_STATS[target.type].hp;
  const denominator = getAttackDenominators(attacker, target, isEncouragedForBattle(state, attacker))?.finalSuccessDenominator;
  return [
    target.type === "king" ? 0 : 1,
    denominator ?? 99,
    target.type === "engineer" || target.type === "strategist" ? 0 : 1,
    target.hp < baseHp ? 0 : 1,
    target.position.kind === "base" ? 0 : 1,
    candidateDistance(state, attacker, target),
    target.id,
  ];
}

function compareSortKey(a: [number, number, number, number, number, number, string], b: [number, number, number, number, number, number, string]) {
  for (let index = 0; index < a.length - 1; index += 1) {
    const diff = (a[index] as number) - (b[index] as number);
    if (diff) return diff;
  }
  return a[6].localeCompare(b[6]);
}

export function sortAttackCandidates(state: GameState, attacker: Unit, targets: Unit[]) {
  return [...targets].sort((a, b) => compareSortKey(targetSortKey(state, attacker, a), targetSortKey(state, attacker, b)));
}

export function getAttackCandidates(state: GameState, attackerUnitId: string): AttackTarget[] {
  const attacker = state.units.find((unit) => unit.id === attackerUnitId);
  if (!attacker || !isAlive(attacker)) return [];

  const range = UNIT_STATS[attacker.type].range;
  if (range <= 0) return [];

  const attackerCells = positionCells(state, attacker.position);
  const targets = state.units
    .filter((target) => target.id !== attacker.id)
    .filter((target) => isAlive(target))
    .filter((target) => target.ownerTeamId !== attacker.ownerTeamId)
    .filter((target) => !isProtectedByOkuzashiki(state, target))
    .filter((target) => canAttackUnit(state, attacker, target))
    .filter((target) => minDistance(attackerCells, positionCells(state, target.position)) <= range);

  return sortAttackCandidates(state, attacker, targets).map((target) => ({
    ...targetForUnit(target),
    ...getAttackDenominators(attacker, target, isEncouragedForBattle(state, attacker)),
  }));
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

function battleEventForIntent(state: GameState, intent: AttackIntent, index: number): BattleEvent[] {
  if (intent.pass || !intent.target) return [];
  const attacker = state.units.find((unit) => unit.id === intent.attackerUnitId);
  const target = state.units.find((unit) => unit.id === intent.target?.unitId);
  if (!attacker || !target || !isAlive(attacker) || !isAlive(target) || !targetStillLegal(state, attacker.id, intent.target)) return [];

  const denominators = getAttackDenominators(attacker, target, isEncouragedForBattle(state, attacker));
  if (!denominators) return [];

  return [
    {
      id: `battle-${state.turnNumber}-${index}`,
      attackerUnitId: intent.attackerUnitId,
      target: { ...intent.target, ...denominators },
      ...denominators,
    },
  ];
}

function battleLog(logs: BattleLogDraft[], message: string, relatedIds?: string[]) {
  logs.push({ message, relatedIds });
}

export function resolveBattle(state: GameState, rng: () => number = Math.random): GameState {
  const next = structuredClone(state) as GameState;
  const intents = next.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []);
  const events = intents.flatMap((intent, index) => battleEventForIntent(next, intent, index));
  const battleLogs: BattleLogDraft[] = [];

  const hitEvents = events.filter((event) => {
    const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
    const target = next.units.find((unit) => unit.id === event.target.unitId);
    const hit = rng() < 1 / event.finalSuccessDenominator;
    const prefix = `${event.attackerUnitId} -> ${event.target.unitId} base 1/${event.baseSuccessDenominator}, encouraged: ${
      event.encouraged ? "yes" : "no"
    }, final 1/${event.finalSuccessDenominator}`;

    battleLog(
      battleLogs,
      `${prefix}, result: ${hit ? "success" : "failure"}, damage: ${hit ? 1 : 0}.`,
      [event.attackerUnitId, event.target.unitId],
    );

    return Boolean(attacker && target && hit);
  });

  const damageByUnitId = new Map<string, number>();
  for (const event of hitEvents) {
    damageByUnitId.set(event.target.unitId, (damageByUnitId.get(event.target.unitId) ?? 0) + 1);
  }

  for (const [targetUnitId, damage] of damageByUnitId) {
    const target = next.units.find((unit) => unit.id === targetUnitId);
    if (!target || !isAlive(target)) continue;

    const nextHp = target.hp - damage;
    next.units = next.units.map((unit) => (unit.id === target.id ? { ...unit, hp: nextHp } : unit));

    battleLog(
      battleLogs,
      `${target.id} took ${damage} total damage (${Math.max(nextHp, 0)} HP left).`,
      [target.id],
    );

    if (nextHp <= 0) {
      const updatedTarget = next.units.find((unit) => unit.id === target.id)!;
      defeatUnit(next, updatedTarget);
      battleLog(battleLogs, `${target.id} was defeated at ${positionKey(target.position)}.`, [target.id]);
    }
  }

  next.logs.push(
    ...battleLogs.map((log, index) => ({
      id: `log-battle-${next.logs.length + index}`,
      turnNumber: next.turnNumber,
      type: "battle" as const,
      message: log.message,
      relatedIds: log.relatedIds,
    })),
  );

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) => ({ ...intent, attackIntents: [] }));
  next.phase = "attack_input";
  next.turnState.phase = "attack_input";
  return next;
}
