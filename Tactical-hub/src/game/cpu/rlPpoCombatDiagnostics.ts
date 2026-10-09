import type { GameState, Unit } from "../types";

export const PPO_COMBAT_UNIT_CATEGORIES = [
  "king",
  "infantry",
  "heavy_infantry",
  "cavalry",
  "archer",
  "engineer",
  "ninja",
  "apprentice_ninja",
  "strategist",
] as const;

export type PpoCombatUnitCategory =
  (typeof PPO_COMBAT_UNIT_CATEGORIES)[number];

export type PpoCombatDiagnostics = {
  enemyDefeatedUnitCount: number;
  unattributedEnemyDefeatedUnitCount: number;
  enemyDefeatParticipationCountByAttackerType: Record<PpoCombatUnitCategory, number>;
  enemyDefeatCreditByAttackerType: Record<PpoCombatUnitCategory, number>;
};

export type PpoCombatSnapshot = {
  logCount: number;
  nonNeutralTeamIds: Set<string>;
  units: Map<string, {
    ownerTeamId: string;
    category: PpoCombatUnitCategory;
    alive: boolean;
  }>;
};

function unitCategory(unit: Unit): PpoCombatUnitCategory {
  if (unit.type === "infantry" && unit.formation === "heavy") {
    return "heavy_infantry";
  }
  return unit.type;
}

function zeroByCategory(): Record<PpoCombatUnitCategory, number> {
  return Object.fromEntries(
    PPO_COMBAT_UNIT_CATEGORIES.map((category) => [category, 0]),
  ) as Record<PpoCombatUnitCategory, number>;
}

export function createPpoCombatDiagnostics(): PpoCombatDiagnostics {
  return {
    enemyDefeatedUnitCount: 0,
    unattributedEnemyDefeatedUnitCount: 0,
    enemyDefeatParticipationCountByAttackerType: zeroByCategory(),
    enemyDefeatCreditByAttackerType: zeroByCategory(),
  };
}

export function clonePpoCombatDiagnostics(
  source: PpoCombatDiagnostics,
): PpoCombatDiagnostics {
  return {
    enemyDefeatedUnitCount: source.enemyDefeatedUnitCount,
    unattributedEnemyDefeatedUnitCount: source.unattributedEnemyDefeatedUnitCount,
    enemyDefeatParticipationCountByAttackerType: {
      ...source.enemyDefeatParticipationCountByAttackerType,
    },
    enemyDefeatCreditByAttackerType: {
      ...source.enemyDefeatCreditByAttackerType,
    },
  };
}

export function capturePpoCombatSnapshot(
  state: GameState,
): PpoCombatSnapshot {
  return {
    logCount: state.logs.length,
    nonNeutralTeamIds: new Set(
      state.teams
        .filter((team) => team.status !== "neutral")
        .map((team) => team.id),
    ),
    units: new Map(
      state.units.map((unit) => [
        unit.id,
        {
          ownerTeamId: unit.ownerTeamId,
          category: unitCategory(unit),
          alive: unit.hp > 0 && unit.position.kind !== "removed",
        },
      ]),
    ),
  };
}

export function observePpoBattleDefeats(
  diagnostics: PpoCombatDiagnostics,
  before: PpoCombatSnapshot,
  after: GameState,
) {
  const successfulAttackersByTarget = new Map<string, Set<string>>();
  for (const log of after.logs.slice(before.logCount)) {
    if (
      log.type !== "battle"
      || !log.message.includes("result: success")
      || !log.relatedIds
      || log.relatedIds.length < 2
    ) {
      continue;
    }
    const [attackerUnitId, targetUnitId] = log.relatedIds;
    const attackers = successfulAttackersByTarget.get(targetUnitId)
      ?? new Set<string>();
    attackers.add(attackerUnitId);
    successfulAttackersByTarget.set(targetUnitId, attackers);
  }

  for (const targetAfter of after.units) {
    const targetBefore = before.units.get(targetAfter.id);
    if (
      !targetBefore?.alive
      || !before.nonNeutralTeamIds.has(targetBefore.ownerTeamId)
      || targetAfter.position.kind !== "removed"
      || targetAfter.position.reason !== "defeated"
    ) {
      continue;
    }

    const attackers = [...(successfulAttackersByTarget.get(targetAfter.id) ?? [])]
      .flatMap((attackerUnitId) => {
        const attacker = before.units.get(attackerUnitId);
        if (
          !attacker?.alive
          || !before.nonNeutralTeamIds.has(attacker.ownerTeamId)
          || attacker.ownerTeamId === targetBefore.ownerTeamId
        ) {
          return [];
        }
        return [attacker];
      });

    diagnostics.enemyDefeatedUnitCount += 1;
    if (!attackers.length) {
      diagnostics.unattributedEnemyDefeatedUnitCount += 1;
      continue;
    }

    const participantCategories = new Set(
      attackers.map((attacker) => attacker.category),
    );
    for (const category of participantCategories) {
      diagnostics.enemyDefeatParticipationCountByAttackerType[category] += 1;
    }

    const credit = 1 / attackers.length;
    for (const attacker of attackers) {
      diagnostics.enemyDefeatCreditByAttackerType[attacker.category] += credit;
    }
  }
}
