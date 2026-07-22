import { UNIT_STATS } from "../constants";
import type {
  AttackIntent,
  AttackTarget,
  BattleEvent,
  GameState,
  Unit,
  UnitPosition,
  UnitType,
} from "../types";
import { positionKey } from "../utils/position";
import { canAttackAcrossRoadTopology, createRoadAttackTopologyContext, getPositionCoord, getRoadAttackDistance, type RoadAttackTopologyContext } from "../utils/roadTopology";
import { getEncouragedUnitIds } from "./encouragement";
import { buildUnitTurnFlag, clearInvalidRetreatTargets, getLegalRetreatRouteDistance, isRetreating } from "./retreat";
import { getMovementCandidates } from "./movement";
import { beginStrategistActionPhase } from "./construction";
import { completeSiegeCapture, selectCaptureTeam } from "./capture";
import { getSiegeState, recordDefenderKill, recordEffectiveBaseAttacks, resetInactiveSieges } from "./siege";
import { getKingCampaign, recordKingAttackTurns, recordKingDamage } from "./kingCampaign";
import { measureLegalSegment } from "../cpu/legalEnumerationProfile";
import { defeatTeamsWithoutBases, resolveKingDefeats, type DefeatedKingPlan, type FallenBasePlan } from "./defeat";

type AttackDenominatorContext = {
  targetInBase: boolean;
  encouraged?: boolean;
};

type BattleLogDraft = {
  message: string;
  relatedIds?: string[];
};

const SUCCESS_DENOMINATORS: Partial<
  Record<UnitType, Partial<Record<UnitType, number>>>
> = {
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

function targetForUnit(unit: Unit): AttackTarget {
  const position = unit.position;
  if (position.kind === "base") {
    return {
      kind: "unit",
      unitId: unit.id,
      baseId: position.baseId,
      slotId: position.slotId,
    };
  }
  return { kind: "unit", unitId: unit.id };
}

function isProtectedByOkuzashiki(state: GameState, target: Unit) {
  const position = target.position;
  if (position.kind !== "base") return false;
  const base = state.bases.find(
    (candidate) => candidate.id === position.baseId,
  );
  if (!base || base.type !== "home" || position.slotId !== base.protectedSlotId)
    return false;

  return base.slots.some((slot) => {
    if (!slot.unitId || slot.id === position.slotId) return false;
    const other = state.units.find((unit) => unit.id === slot.unitId);
    return Boolean(
      other && isAlive(other) && other.ownerTeamId === target.ownerTeamId,
    );
  });
}

export type AttackEnumerationContext = {
  readonly sourceState: GameState;
  readonly unitsRef: GameState["units"];
  readonly basesRef: GameState["bases"];
  readonly constructionsRef: GameState["constructions"];
  readonly unitById: ReadonlyMap<string, Unit>;
  readonly enemiesByTeamId: ReadonlyMap<string, readonly Unit[]>;
  readonly protectedUnitIds: ReadonlySet<string>;
  readonly encouragedUnitIds: ReadonlySet<string>;
  readonly coordByUnitId: ReadonlyMap<string, { x: number; y: number } | undefined>;
  readonly roadTopology: RoadAttackTopologyContext;
  readonly distances: Map<string, number>;
};

const attackContextCache = new WeakMap<GameState["units"], AttackEnumerationContext>();

export function getAttackEnumerationContext(state: GameState): AttackEnumerationContext {
  const cached = attackContextCache.get(state.units);
  if (cached && state.phase === "attack_input" && cached.unitsRef === state.units && cached.basesRef === state.bases && cached.constructionsRef === state.constructions) return cached;
  const context = measureLegalSegment("attackUnitCoordinateBaseSearch", () => {
    const unitById = new Map(state.units.map((unit) => [unit.id, unit]));
    const living = measureLegalSegment("attackLivingEnemyList", () => state.units.filter(isAlive));
    const teamIds = state.teams.map((team) => team.id);
    const enemiesByTeamId = new Map(teamIds.map((teamId) => [teamId, living.filter((unit) => unit.ownerTeamId !== teamId)]));
    const protectedUnitIds = measureLegalSegment("attackBaseBlocking", () => new Set(living.filter((unit) => isProtectedByOkuzashiki(state, unit)).map((unit) => unit.id)));
    return {
      sourceState: state,
      unitsRef: state.units,
      basesRef: state.bases,
      constructionsRef: state.constructions,
      unitById,
      enemiesByTeamId,
      protectedUnitIds,
      encouragedUnitIds: getEncouragedUnitIds(state),
      coordByUnitId: new Map(living.map((unit) => [unit.id, getPositionCoord(state, unit.position)])),
      roadTopology: createRoadAttackTopologyContext(state),
      distances: new Map<string, number>(),
    } satisfies AttackEnumerationContext;
  });
  if (state.phase === "attack_input") attackContextCache.set(state.units, context);
  return context;
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

export function applyEncouragementToDenominator(
  denominator: number,
  encouraged: boolean,
) {
  return encouraged ? Math.max(1, denominator - 1) : denominator;
}

export function getFinalAttackDenominator(
  attackerType: UnitType,
  targetType: UnitType,
  context: AttackDenominatorContext,
): number | null {
  const baseDenominator = getBaseAttackDenominator(
    attackerType,
    targetType,
    context,
  );
  if (baseDenominator === null) return null;
  return applyEncouragementToDenominator(
    baseDenominator,
    Boolean(context.encouraged),
  );
}

function getAttackDenominators(
  attacker: Unit,
  target: Unit,
  encouraged = false,
) {
  const context = { targetInBase: target.position.kind === "base", encouraged };
  const baseSuccessDenominator = getBaseAttackDenominator(
    attacker.type,
    target.type,
    context,
  );
  if (baseSuccessDenominator === null) return undefined;
  let finalSuccessDenominator = applyEncouragementToDenominator(
    baseSuccessDenominator,
    encouraged,
  );
  if (target.type === "infantry" && isRetreating(target)) {
    finalSuccessDenominator *= 2;
  }
  return { baseSuccessDenominator, finalSuccessDenominator, encouraged };
}

function isEncouragedForBattle(encouragedUnitIds: Set<string>, attacker: Unit) {
  return encouragedUnitIds.has(attacker.id);
}

function canAttackByPositionRule(attacker: Unit, target: Unit) {
  if (attacker.position.kind === "water") {
    return (
      attacker.type === "ninja" &&
      target.type === "ninja" &&
      target.position.kind === "water"
    );
  }
  if (target.position.kind === "water") return false;
  return true;
}

function candidateDistance(state: GameState, attacker: Unit, target: Unit, topology?: RoadAttackTopologyContext) {
  return getRoadAttackDistance(state, attacker.position, target.position, topology);
}

function contextDistance(state: GameState, attacker: Unit, target: Unit, context?: AttackEnumerationContext) {
  if (!context) return candidateDistance(state, attacker, target);
  const key = `${attacker.id}\u0000${target.id}`;
  const cached = context.distances.get(key);
  if (cached !== undefined) return cached;
  const distance = candidateDistance(state, attacker, target, context.roadTopology);
  context.distances.set(key, distance);
  return distance;
}

function targetSortKey(
  state: GameState,
  attacker: Unit,
  target: Unit,
  encouragedUnitIds: Set<string>,
  distance = candidateDistance(state, attacker, target),
): [number, number, number, number, number, number, string] {
  const baseHp = UNIT_STATS[target.type].hp;
  const denominator = getAttackDenominators(
    attacker,
    target,
    encouragedUnitIds.has(attacker.id),
  )?.finalSuccessDenominator;
  return [
    target.type === "king" ? 0 : 1,
    denominator ?? 99,
    target.type === "engineer" || target.type === "strategist" ? 0 : 1,
    target.hp < baseHp ? 0 : 1,
    target.position.kind === "base" ? 0 : 1,
    distance,
    target.id,
  ];
}

function compareSortKey(
  a: [number, number, number, number, number, number, string],
  b: [number, number, number, number, number, number, string],
) {
  for (let index = 0; index < a.length - 1; index += 1) {
    const diff = (a[index] as number) - (b[index] as number);
    if (diff) return diff;
  }
  return a[6].localeCompare(b[6]);
}

export function sortAttackCandidates(
  state: GameState,
  attacker: Unit,
  targets: Unit[],
  encouragedUnitIds = getEncouragedUnitIds(state),
  context?: AttackEnumerationContext,
) {
  const keys = new Map(targets.map((target) => [target.id, targetSortKey(state, attacker, target, encouragedUnitIds, contextDistance(state, attacker, target, context))]));
  return [...targets].sort((a, b) => compareSortKey(keys.get(a.id)!, keys.get(b.id)!));
}

export function getAttackCandidates(
  state: GameState,
  attackerUnitId: string,
  context = getAttackEnumerationContext(state),
): AttackTarget[] {
  return measureLegalSegment("attackTargetSearch", () => getAttackCandidatesCore(state, attackerUnitId, context));
}

function getAttackCandidatesCore(state: GameState, attackerUnitId: string, context: AttackEnumerationContext): AttackTarget[] {
  const attacker = context.unitById.get(attackerUnitId);
  if (!attacker || !isAlive(attacker)) return [];
  if (isRetreating(attacker)) return [];

  const range = UNIT_STATS[attacker.type].range;
  if (range <= 0) return [];

  const enemies = context.enemiesByTeamId.get(attacker.ownerTeamId) ?? [];
  const targets = measureLegalSegment("attackBasicFilter", () => enemies.filter((target) => target.id !== attacker.id && !context.protectedUnitIds.has(target.id)));
  const legal: Unit[] = [];
  for (const target of targets) {
    if (!measureLegalSegment("attackLakeNinjaRule", () => canAttackByPositionRule(attacker, target))) continue;
    if (!measureLegalSegment("attackFinalLegalCheck", () => Boolean(getAttackDenominators(attacker, target, false)))) continue;
    const attackerCoord = context.coordByUnitId.get(attacker.id), targetCoord = context.coordByUnitId.get(target.id);
    if (attackerCoord && targetCoord && !measureLegalSegment("attackStaticRangePrefilter", () => Math.max(Math.abs(attackerCoord.x - targetCoord.x), Math.abs(attackerCoord.y - targetCoord.y)) <= range)) continue;
    const topologyCategory = attacker.position.kind === "base" || target.position.kind === "base"
      ? "attackAcrossBaseBlocking"
      : attacker.position.kind === "bridge" || target.position.kind === "bridge"
        ? "attackBridgeConnection"
        : "attackRoadSectionConnection";
    if (!measureLegalSegment(topologyCategory, () => canAttackAcrossRoadTopology(state, attacker.position, target.position, context.roadTopology))) continue;
    const distance = measureLegalSegment("attackRangeDistance", () => contextDistance(state, attacker, target, context));
    if (distance > range) continue;
    legal.push(target);
  }

  const sorted = measureLegalSegment("attackPostProcessing", () => sortAttackCandidates(state, attacker, legal, context.encouragedUnitIds as Set<string>, context));
  return measureLegalSegment("attackCandidateIdGeneration", () => sorted.map((target) => ({
    ...targetForUnit(target),
    ...getAttackDenominators(attacker, target, context.encouragedUnitIds.has(attacker.id)),
  })));
}

export function getTeamAttackCandidates(state: GameState, teamId: string) {
  if (state.phase !== "attack_input" || state.teams.find((team) => team.id === teamId)?.status !== "active") return [];
  const context = getAttackEnumerationContext(state);
  return getTeamAttackerUnitIds(state, teamId)
    .map((unitId) => ({ attackerUnitId: unitId, targets: getAttackCandidates(state, unitId, context) }));
}

export function getTeamAttackerUnitIds(state: GameState, teamId: string) {
  if (state.phase !== "attack_input" || state.teams.find((team) => team.id === teamId)?.status !== "active") return [];
  return state.units
    .filter((unit) => unit.ownerTeamId === teamId && unit.hp > 0 && unit.position.kind !== "removed")
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((unit) => unit.id);
}

export function saveAttackIntent(
  state: GameState,
  intent: AttackIntent,
): GameState {
  const attacker = state.units.find((unit) => unit.id === intent.attackerUnitId);
  if (attacker && isRetreating(attacker) && !intent.pass) return state;
  const existing = state.turnState.actionIntents.find(
    (candidate) => candidate.teamId === intent.teamId,
  );
  const actionIntents = existing
    ? state.turnState.actionIntents.map((candidate) =>
        candidate.teamId === intent.teamId
          ? {
              ...candidate,
              attackIntents: [
                ...(candidate.attackIntents ?? []).filter(
                  (attack) => attack.attackerUnitId !== intent.attackerUnitId,
                ),
                intent,
              ],
            }
          : candidate,
      )
    : [
        ...state.turnState.actionIntents,
        {
          teamId: intent.teamId,
          productionChoices: [],
          movementIntents: [],
          attackIntents: [intent],
        },
      ];
  return { ...state, turnState: { ...state.turnState, actionIntents } };
}

function clearBaseSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const base = state.bases.find(
    (candidate) => candidate.id === position.baseId,
  );
  const slot = base?.slots.find(
    (candidate) => candidate.id === position.slotId,
  );
  if (slot) slot.unitId = undefined;
}

function defeatUnit(state: GameState, target: Unit) {
  clearBaseSlot(state, target.position);
  state.units = state.units.map((unit) =>
    unit.id === target.id
      ? {
          ...unit,
          hp: 0,
          position: { kind: "removed", reason: "defeated" },
          statuses: unit.statuses.filter(
            (status) => status.kind !== "retreating",
          ),
        }
      : unit,
  );
  if (target.type === "strategist") {
    state.constructions = state.constructions.map((construction) =>
      construction.managerUnitId === target.id
        ? { ...construction, managerUnitId: undefined }
        : construction,
    );
    state.teams = state.teams.map((team) =>
      team.constructionCapacityBonusStrategistId === target.id
        ? { ...team, constructionCapacityBonusStrategistId: undefined }
        : team,
    );
  }

  if (target.type === "king") {
    state.logs.push({
      id: `log-king-defeated-${state.logs.length}`,
      turnNumber: state.turnNumber,
      type: "battle",
      message: `${target.ownerTeamId} king was defeated.`,
      relatedIds: [target.id, target.ownerTeamId],
    });
  }
}

function targetStillLegal(
  state: GameState,
  attackerUnitId: string,
  target: AttackTarget,
) {
  return getAttackCandidates(state, attackerUnitId).some(
    (candidate) => candidate.unitId === target.unitId,
  );
}

function battleEventForIntent(
  state: GameState,
  encouragedUnitIds: Set<string>,
  intent: AttackIntent,
  index: number,
): BattleEvent[] {
  if (intent.pass || !intent.target) return [];
  const attacker = state.units.find(
    (unit) => unit.id === intent.attackerUnitId,
  );
  const target = state.units.find((unit) => unit.id === intent.target?.unitId);
  if (
    !attacker ||
    !target ||
    !isAlive(attacker) ||
    !isAlive(target) ||
    !targetStillLegal(state, attacker.id, intent.target)
  )
    return [];

  const denominators = getAttackDenominators(
    attacker,
    target,
    isEncouragedForBattle(encouragedUnitIds, attacker),
  );
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

function battleLog(
  logs: BattleLogDraft[],
  message: string,
  relatedIds?: string[],
) {
  logs.push({ message, relatedIds });
}

export function resolveBattle(
  state: GameState,
  rng: () => number = Math.random,
): GameState {
  const next = structuredClone(state) as GameState;
  const intents = next.turnState.actionIntents.flatMap(
    (intent) => intent.attackIntents ?? [],
  );
  const encouragedUnitIds = getEncouragedUnitIds(next);
  const battleLogs: BattleLogDraft[] = [];
  const battleStartPositionsByUnitId = new Map(
    next.units.map((unit) => [
      unit.id,
      structuredClone(unit.position) as UnitPosition,
    ]),
  );
  const events = intents.flatMap((intent, index) => {
    if (intent.pass || !intent.target) return [];
    const attacker = next.units.find(
      (unit) => unit.id === intent.attackerUnitId,
    );
    if (attacker && isRetreating(attacker)) {
      battleLog(
        battleLogs,
        `${attacker.id} attack intent was invalid because the unit is retreating.`,
        [attacker.id],
      );
      return [];
    }
    return battleEventForIntent(next, encouragedUnitIds, intent, index);
  });
  recordEffectiveBaseAttacks(next, events.flatMap((event) => {
    const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
    const target = next.units.find((unit) => unit.id === event.target.unitId);
    if (!attacker || !target || target.position.kind !== "base") return [];
    const targetBaseId = target.position.baseId;
    const base = next.bases.find((entry) => entry.id === targetBaseId);
    if (!base || target.ownerTeamId !== base.ownerTeamId || attacker.ownerTeamId === base.ownerTeamId) return [];
    return [{ baseId: base.id, defendingTeamId: base.ownerTeamId, attackingTeamId: attacker.ownerTeamId }];
  }));
  recordKingAttackTurns(next, events.flatMap((event) => {
    const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
    const target = next.units.find((unit) => unit.id === event.target.unitId);
    return attacker && target?.type === "king" && attacker.ownerTeamId !== target.ownerTeamId
      ? [{ kingUnitId: target.id, kingTeamId: target.ownerTeamId, attackingTeamId: attacker.ownerTeamId }]
      : [];
  }));
  const defenderCountsAtStart = new Map(next.bases.map((base) => [base.id, next.units.filter((unit) => isAlive(unit) && unit.ownerTeamId === base.ownerTeamId && unit.position.kind === "base" && unit.position.baseId === base.id).length]));
  const aliveAtBattleStart = new Set(
    next.units.filter(isAlive).map((unit) => unit.id),
  );

  const hitEvents = events.filter((event) => {
    const attacker = next.units.find(
      (unit) => unit.id === event.attackerUnitId,
    );
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
    damageByUnitId.set(
      event.target.unitId,
      (damageByUnitId.get(event.target.unitId) ?? 0) + 1,
    );
    const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
    const target = next.units.find((unit) => unit.id === event.target.unitId);
    if (attacker && target?.type === "king") recordKingDamage(next, target.id, target.ownerTeamId, attacker.ownerTeamId, 1);
  }

  for (const [targetUnitId, damage] of damageByUnitId) {
    const target = next.units.find((unit) => unit.id === targetUnitId);
    if (!target || !isAlive(target)) continue;

    const nextHp = target.hp - damage;
    next.units = next.units.map((unit) =>
      unit.id === target.id ? { ...unit, hp: nextHp } : unit,
    );

    battleLog(
      battleLogs,
      `${target.id} took ${damage} total damage (${Math.max(nextHp, 0)} HP left).`,
      [target.id],
    );

    if (nextHp <= 0) {
      const updatedTarget = next.units.find((unit) => unit.id === target.id)!;
      defeatUnit(next, updatedTarget);
      if (target.position.kind === "base") {
        const targetBaseId = target.position.baseId;
        const base = next.bases.find((entry) => entry.id === targetBaseId);
        if (base && target.ownerTeamId === base.ownerTeamId) {
          const killerTeamIds = hitEvents.filter((event) => event.target.unitId === target.id).flatMap((event) => {
            const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
            return attacker ? [attacker.ownerTeamId] : [];
          });
          if (killerTeamIds.length) recordDefenderKill(next, base.id, base.ownerTeamId, killerTeamIds);
        }
      }
      battleLog(
        battleLogs,
        `${target.id} was defeated at ${positionKey(target.position)}.`,
        [target.id],
      );
    }
  }

  const attackedUnitIds = new Set(events.map((event) => event.attackerUnitId));
  const targetedUnitIds = new Set(events.map((event) => event.target.unitId));
  const participantUnitIds = new Set([...attackedUnitIds, ...targetedUnitIds]);
  next.unitTurnFlags = [...participantUnitIds]
    .map((unitId) => {
      const unit = next.units.find((candidate) => candidate.id === unitId);
      if (!unit) return undefined;
      return buildUnitTurnFlag(
        next,
        unit,
        next.turnNumber,
        aliveAtBattleStart.has(unitId),
        attackedUnitIds.has(unitId),
        targetedUnitIds.has(unitId),
        battleStartPositionsByUnitId.get(unitId) ?? unit.position,
      );
    })
    .filter((flag): flag is NonNullable<typeof flag> => Boolean(flag));

  for (const flag of next.unitTurnFlags.filter(
    (candidate) => candidate.retreatEligible,
  )) {
    const role =
      flag.attackedInPreviousBattle && flag.wasTargetedInPreviousBattle
        ? "attacker and target"
        : flag.attackedInPreviousBattle
          ? "attacker"
          : "target";
    battleLog(
      battleLogs,
      `${flag.unitId} became retreat eligible as a valid battle ${role} near an enemy-controlled base.`,
      [flag.unitId],
    );
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

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) => ({
    ...intent,
    attackIntents: [],
  }));
  let captured = false;
  const fallenBases: FallenBasePlan[] = [];
  for (const base of [...next.bases]) {
    if (!(defenderCountsAtStart.get(base.id) ?? 0)) continue;
    const remaining = next.units.some((unit) => isAlive(unit) && unit.ownerTeamId === base.ownerTeamId && unit.position.kind === "base" && unit.position.baseId === base.id);
    if (remaining) continue;
    const siege = getSiegeState(next, base.id);
    if (!siege) continue;
    const candidates = [...new Set(hitEvents.flatMap((event) => {
      const targetStart = battleStartPositionsByUnitId.get(event.target.unitId);
      if (targetStart?.kind !== "base" || targetStart.baseId !== base.id) return [];
      const target = next.units.find((unit) => unit.id === event.target.unitId);
      const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
      return target?.position.kind === "removed" && attacker ? [attacker.ownerTeamId] : [];
    }))];
    const intendedCaptureTeamId = selectCaptureTeam(next, siege, candidates, rng);
    fallenBases.push({ baseId: base.id, defendingTeamId: base.ownerTeamId, siege: structuredClone(siege), candidateTeamIds: candidates, intendedCaptureTeamId });
  }
  const defeatedKings: DefeatedKingPlan[] = next.units.filter((unit) => unit.type === "king" && unit.position.kind === "removed" && aliveAtBattleStart.has(unit.id)).flatMap((king) => {
    const campaign = getKingCampaign(next, king.id);
    if (!campaign) return [];
    const candidateTeamIds = [...new Set(hitEvents.flatMap((event) => {
      if (event.target.unitId !== king.id) return [];
      const attacker = next.units.find((unit) => unit.id === event.attackerUnitId);
      return attacker ? [attacker.ownerTeamId] : [];
    }))];
    return [{ kingUnitId: king.id, kingTeamId: king.ownerTeamId, candidateTeamIds, campaign: structuredClone(campaign) }];
  });
  const kingDefeatApplied = resolveKingDefeats(next, defeatedKings, fallenBases, rng);
  const kingDefeatedTeamIds = new Set(defeatedKings.map((plan) => plan.kingTeamId));
  for (const fallen of fallenBases.filter((entry) => !kingDefeatApplied || !kingDefeatedTeamIds.has(entry.defendingTeamId))) {
    captured = completeSiegeCapture(next, fallen.siege, fallen.candidateTeamIds, "annihilation", rng) || captured;
  }
  defeatTeamsWithoutBases(next, kingDefeatedTeamIds);
  next.unitTurnFlags = next.unitTurnFlags.map((flag) => {
    if (!flag.retreatEligible) return flag;
    const unit = next.units.find((candidate) => candidate.id === flag.unitId);
    const team = unit ? next.teams.find((candidate) => candidate.id === unit.ownerTeamId) : undefined;
    const currentRoute = unit ? getLegalRetreatRouteDistance(next, unit.ownerTeamId, unit.position) : undefined;
    const hasRetreatStep = Boolean(unit && currentRoute && getMovementCandidates(next, unit.id).some((destination) => {
      const route = getLegalRetreatRouteDistance(next, unit.ownerTeamId, destination);
      return route && route.distance < currentRoute.distance;
    }));
    if (unit && isAlive(unit) && team?.status === "active" && currentRoute && hasRetreatStep) return flag;
    return { ...flag, retreatEligible: false, retreatEligibilityReason: "no legal route to a currently controlled friendly base" };
  });
  next.units = next.units.map((unit) => isAlive(unit) && next.teams.find((team) => team.id === unit.ownerTeamId)?.status === "active"
    ? unit
    : { ...unit, statuses: unit.statuses.filter((status) => status.kind !== "retreating") });
  clearInvalidRetreatTargets(next);
  captured = captured || kingDefeatApplied;
  resetInactiveSieges(next);
  if (next.rewardPlacementRequests.some((request) => !request.completed && !request.expired)) {
    next.phaseAfterRewards = "strategist_action_input";
    next.phase = "reward_placement";
  } else next.phase = "strategist_action_input";
  next.turnState.phase = next.phase;
  return next.phase === "strategist_action_input" ? beginStrategistActionPhase(next) : next;
}
