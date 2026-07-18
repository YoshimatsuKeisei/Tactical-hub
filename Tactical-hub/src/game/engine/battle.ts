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
import { canAttackAcrossRoadTopology, getRoadAttackDistance } from "../utils/roadTopology";
import { getEncouragedUnitIds } from "./encouragement";
import { buildUnitTurnFlag, clearInvalidRetreatTargets, getLegalRetreatRouteDistance, isRetreating } from "./retreat";
import { getMovementCandidates } from "./movement";
import { beginStrategistActionPhase } from "./construction";
import { completeSiegeCapture, selectCaptureTeam } from "./capture";
import { getSiegeState, recordDefenderKill, recordEffectiveBaseAttacks, resetInactiveSieges } from "./siege";
import { getKingCampaign, recordKingAttackTurns, recordKingDamage } from "./kingCampaign";
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

function canAttackUnit(state: GameState, attacker: Unit, target: Unit) {
  if (!canAttackByPositionRule(attacker, target)) return false;
  if (!getAttackDenominators(attacker, target, false)) return false;
  return true;
}

function candidateDistance(state: GameState, attacker: Unit, target: Unit) {
  return getRoadAttackDistance(state, attacker.position, target.position);
}

function targetSortKey(
  state: GameState,
  attacker: Unit,
  target: Unit,
  encouragedUnitIds: Set<string>,
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
    candidateDistance(state, attacker, target),
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
) {
  return [...targets].sort((a, b) =>
    compareSortKey(
      targetSortKey(state, attacker, a, encouragedUnitIds),
      targetSortKey(state, attacker, b, encouragedUnitIds),
    ),
  );
}

export function getAttackCandidates(
  state: GameState,
  attackerUnitId: string,
): AttackTarget[] {
  const attacker = state.units.find((unit) => unit.id === attackerUnitId);
  if (!attacker || !isAlive(attacker)) return [];
  if (isRetreating(attacker)) return [];

  const range = UNIT_STATS[attacker.type].range;
  if (range <= 0) return [];

  const targets = state.units
    .filter((target) => target.id !== attacker.id)
    .filter((target) => isAlive(target))
    .filter((target) => target.ownerTeamId !== attacker.ownerTeamId)
    .filter((target) => !isProtectedByOkuzashiki(state, target))
    .filter((target) => canAttackUnit(state, attacker, target))
    .filter((target) =>
      canAttackAcrossRoadTopology(state, attacker.position, target.position),
    )
    .filter(
      (target) =>
        getRoadAttackDistance(state, attacker.position, target.position) <= range,
    );

  const encouragedUnitIds = getEncouragedUnitIds(state);
  return sortAttackCandidates(state, attacker, targets, encouragedUnitIds).map(
    (target) => ({
      ...targetForUnit(target),
      ...getAttackDenominators(
        attacker,
        target,
        encouragedUnitIds.has(attacker.id),
      ),
    }),
  );
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
  if (captured && next.rewardPlacementRequests.some((request) => !request.completed && !request.expired)) {
    next.phaseAfterRewards = "strategist_action_input";
    next.phase = "reward_placement";
  } else next.phase = "strategist_action_input";
  next.turnState.phase = next.phase;
  return next.phase === "strategist_action_input" ? beginStrategistActionPhase(next) : next;
}
