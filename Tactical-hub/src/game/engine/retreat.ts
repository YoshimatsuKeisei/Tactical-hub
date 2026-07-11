import type {
  Base,
  GameState,
  Unit,
  UnitPosition,
  UnitStatus,
  UnitTurnFlags,
} from "../types";
import { chebyshevDistance } from "../utils/distance";

type DistanceResult = {
  distance: number;
  baseIds: string[];
};

type DirectionLabel =
  | "up"
  | "up-right"
  | "right"
  | "down-right"
  | "down"
  | "down-left"
  | "left"
  | "up-left";

export type RetreatDirectionIndicator = {
  key: "friendly" | "hostile";
  label: string;
  directionX: -1 | 0 | 1;
  directionY: -1 | 0 | 1;
  directionLabel: DirectionLabel;
};

export type RetreatMoveEffect =
  | "none"
  | "start"
  | "maintain"
  | "release"
  | "complete";

export type RetreatDebugInfo = {
  unitId: string;
  eligible: boolean;
  wasAttacker: boolean;
  wasTargeted: boolean;
  participatedInBattle: boolean;
  survivedBattle: boolean;
  battlePositionKind?: UnitPosition["kind"];
  nearestHostileBaseId?: string;
  nearestHostileBaseController?: string;
  nearestHostileBaseDistanceAtBattle?: number;
  withinHostileBaseRangeAtBattle: boolean;
  failureReasons: string[];
};

function isAlive(unit: Unit) {
  return unit.position.kind !== "removed" && unit.hp > 0;
}

function positionCells(state: GameState, position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water")
    return [{ x: position.x, y: position.y }];
  if (position.kind === "base")
    return (
      state.bases.find((base) => base.id === position.baseId)?.coords ?? []
    );
  return [];
}

function minDistanceToBase(
  state: GameState,
  position: UnitPosition,
  base: Base,
) {
  if (position.kind === "base" && position.baseId === base.id) return 0;
  const ownCells = positionCells(state, position);
  if (!ownCells.length || !base.coords.length) return Number.POSITIVE_INFINITY;
  return Math.min(
    ...ownCells.flatMap((own) =>
      base.coords.map((cell) => chebyshevDistance(own, cell)),
    ),
  );
}

export function isRetreating(unit: Unit) {
  return unit.statuses.some((status) => status.kind === "retreating");
}

export function withRetreatingStatus(unit: Unit, retreating: boolean): Unit {
  const statuses = unit.statuses.filter(
    (status) => status.kind !== "retreating",
  );
  return retreating
    ? {
        ...unit,
        statuses: [...statuses, { kind: "retreating" } satisfies UnitStatus],
      }
    : { ...unit, statuses };
}

export function getBaseControllerTeamId(
  state: GameState,
  base: Base,
): string | undefined {
  const controlledByTeam = state.teams.find((team) =>
    team.controlledBaseIds.includes(base.id),
  );
  if (controlledByTeam) return controlledByTeam.id;
  if (base.ownerTeamId) return base.ownerTeamId;
  return state.teams.find((team) => team.homeBaseId === base.id)?.id;
}

export function getControlledFriendlyBases(state: GameState, teamId: string) {
  const team = state.teams.find((candidate) => candidate.id === teamId);
  const controlled = new Set(team?.controlledBaseIds ?? []);
  return state.bases.filter(
    (base) =>
      controlled.has(base.id) ||
      getBaseControllerTeamId(state, base) === teamId,
  );
}

export function getEnemyControlledBases(state: GameState, teamId: string) {
  return state.bases.filter((base) => {
    const controllerTeamId = getBaseControllerTeamId(state, base);

    // 未占領など、支配チームが存在しない拠点は対象外。
    if (!controllerTeamId) {
      return false;
    }

    // 自軍拠点は対象外。
    if (controllerTeamId === teamId) {
      return false;
    }

    const controller = state.teams.find((team) => team.id === controllerTeamId);

    /*
     * 中立CPUが守備する拠点も、戦闘中のプレイヤーから見れば
     * 敵対拠点として扱う。
     *
     * isNeutral や status === "neutral" を理由に除外しない。
     */
    return controller?.status !== "eliminated";
  });
}

function nearestDistance(
  state: GameState,
  position: UnitPosition,
  bases: Base[],
): DistanceResult | undefined {
  const distances = bases
    .map((base) => ({
      baseId: base.id,
      distance: minDistanceToBase(state, position, base),
    }))
    .filter((entry) => Number.isFinite(entry.distance));
  if (!distances.length) return undefined;

  const distance = Math.min(...distances.map((entry) => entry.distance));
  return {
    distance,
    baseIds: distances
      .filter((entry) => entry.distance === distance)
      .map((entry) => entry.baseId)
      .sort((a, b) => a.localeCompare(b)),
  };
}

export function getNearestFriendlyBaseDistance(
  state: GameState,
  teamId: string,
  position: UnitPosition,
) {
  return nearestDistance(
    state,
    position,
    getControlledFriendlyBases(state, teamId),
  );
}

export function getNearestEnemyBaseDistance(
  state: GameState,
  teamId: string,
  position: UnitPosition,
) {
  return nearestDistance(
    state,
    position,
    getEnemyControlledBases(state, teamId),
  );
}

export function getNearestEnemyBaseInfo(
  state: GameState,
  teamId: string,
  position: UnitPosition,
) {
  const distance = getNearestEnemyBaseDistance(state, teamId, position);
  const baseId = distance?.baseIds[0];
  const base = baseId
    ? state.bases.find((candidate) => candidate.id === baseId)
    : undefined;
  return {
    ...distance,
    baseId,
    controllerTeamId: base ? getBaseControllerTeamId(state, base) : undefined,
  };
}

function directionLabel(dx: -1 | 0 | 1, dy: -1 | 0 | 1): DirectionLabel {
  if (dx === 0 && dy < 0) return "up";
  if (dx > 0 && dy < 0) return "up-right";
  if (dx > 0 && dy === 0) return "right";
  if (dx > 0 && dy > 0) return "down-right";
  if (dx === 0 && dy > 0) return "down";
  if (dx < 0 && dy > 0) return "down-left";
  if (dx < 0 && dy === 0) return "left";
  return "up-left";
}

function signDirection(value: number): -1 | 0 | 1 {
  return value === 0 ? 0 : value > 0 ? 1 : -1;
}

function nearestBaseVector(
  state: GameState,
  position: UnitPosition,
  baseIds: string[] | undefined,
) {
  const unitCells = positionCells(state, position);
  if (!unitCells.length || !baseIds?.length) return undefined;

  const base = state.bases.find((candidate) => candidate.id === baseIds[0]);
  if (!base?.coords.length) return undefined;

  const pairs = unitCells.flatMap((unitCell) =>
    base.coords.map((baseCell) => ({
      unitCell,
      baseCell,
      distance: chebyshevDistance(unitCell, baseCell),
    })),
  );
  const nearest = pairs.sort(
    (a, b) =>
      a.distance - b.distance ||
      a.unitCell.y - b.unitCell.y ||
      a.unitCell.x - b.unitCell.x ||
      a.baseCell.y - b.baseCell.y ||
      a.baseCell.x - b.baseCell.x,
  )[0];
  if (!nearest) return undefined;

  const dx = signDirection(nearest.baseCell.x - nearest.unitCell.x);
  const dy = signDirection(nearest.baseCell.y - nearest.unitCell.y);
  return { dx, dy };
}

export function getRetreatDirectionIndicators(
  state: GameState,
  unitId: string | undefined,
): RetreatDirectionIndicator[] {
  const unit = unitId ? state.units.find((candidate) => candidate.id === unitId) : undefined;
  if (!unit || unit.position.kind === "removed") return [];
  const retreating = isRetreating(unit);
  if (!retreating && !isUnitRetreatEligible(state, unit)) return [];

  const friendly = getNearestFriendlyBaseDistance(
    state,
    unit.ownerTeamId,
    unit.position,
  );
  const hostile = getNearestEnemyBaseDistance(
    state,
    unit.ownerTeamId,
    unit.position,
  );

  return [
    { kind: "friendly" as const, vector: nearestBaseVector(state, unit.position, friendly?.baseIds) },
    { kind: "hostile" as const, vector: nearestBaseVector(state, unit.position, hostile?.baseIds) },
  ]
    .filter((entry): entry is { kind: "friendly" | "hostile"; vector: { dx: -1 | 0 | 1; dy: -1 | 0 | 1 } } =>
      Boolean(entry.vector && (entry.vector.dx !== 0 || entry.vector.dy !== 0)),
    )
    .map((entry) => ({
      key: entry.kind,
      label:
        entry.kind === "friendly"
          ? retreating
            ? "撤退を継続する"
            : "撤退する"
          : retreating
            ? "撤退を解除する"
            : "継戦する",
      directionX: entry.vector.dx,
      directionY: entry.vector.dy,
      directionLabel: directionLabel(entry.vector.dx, entry.vector.dy),
    }));
}

export function isRetreatEligiblePosition(position: UnitPosition) {
  return position.kind === "tile" || position.kind === "bridge";
}

export function getUnitTurnFlags(state: GameState, unitId: string) {
  return state.unitTurnFlags.find((flag) => flag.unitId === unitId);
}

export function isUnitRetreatEligible(state: GameState, unit: Unit) {
  return Boolean(getUnitTurnFlags(state, unit.id)?.retreatEligible);
}

export function getRetreatDebugInfo(
  state: GameState,
  unitId: string,
): RetreatDebugInfo {
  const unit = state.units.find((candidate) => candidate.id === unitId);
  const flags = getUnitTurnFlags(state, unitId);
  const battlePosition = flags?.positionAtBattleStart ?? unit?.position;
  const hostileBase =
    unit && battlePosition
      ? getNearestEnemyBaseInfo(state, unit.ownerTeamId, battlePosition)
      : undefined;
  const survivedBattle =
    flags?.survivedPreviousBattle ?? Boolean(unit && isAlive(unit));
  const wasAttacker = flags?.attackedInPreviousBattle ?? false;
  const wasTargeted = flags?.wasTargetedInPreviousBattle ?? false;
  const participatedInBattle = wasAttacker || wasTargeted;
  const withinHostileBaseRangeAtBattle =
    flags?.enemyBaseWithin3AtBattleStart ??
    Boolean(hostileBase?.distance !== undefined && hostileBase.distance <= 3);
  const eligible = Boolean(flags?.retreatEligible);
  const failureReasons = eligible
    ? []
    : [
        !unit ? "unit not found" : undefined,
        flags && !flags.wasAliveAtBattleStart
          ? "unit was not alive at battle start"
          : undefined,
        !survivedBattle ? "unit did not survive battle" : undefined,
        !participatedInBattle
          ? "unit was not recorded as a valid battle participant"
          : undefined,
        battlePosition && !isRetreatEligiblePosition(battlePosition)
          ? `battle position was ${battlePosition.kind}`
          : undefined,
        !withinHostileBaseRangeAtBattle
          ? `nearest hostile base distance at battle was ${flags?.enemyBaseDistanceAtBattleStart ?? hostileBase?.distance ?? "none"}`
          : undefined,
        flags?.retreatEligibilityReason &&
        flags.retreatEligibilityReason !== "eligible"
          ? flags.retreatEligibilityReason
          : undefined,
      ].filter((reason): reason is string => Boolean(reason));

  return {
    unitId,
    eligible,
    wasAttacker,
    wasTargeted,
    participatedInBattle,
    survivedBattle,
    battlePositionKind: battlePosition?.kind,
    nearestHostileBaseId: hostileBase?.baseId,
    nearestHostileBaseController: hostileBase?.controllerTeamId,
    nearestHostileBaseDistanceAtBattle:
      flags?.enemyBaseDistanceAtBattleStart ?? hostileBase?.distance,
    withinHostileBaseRangeAtBattle,
    failureReasons,
  };
}

export function buildUnitTurnFlag(
  state: GameState,
  unit: Unit,
  battleTurnNumber: number,
  wasAliveAtBattleStart: boolean,
  attackedInPreviousBattle: boolean,
  wasTargetedInPreviousBattle: boolean,
  positionAtBattleStart = unit.position,
): UnitTurnFlags {
  const survivedPreviousBattle = isAlive(unit);
  const enemyBaseDistance =
    getNearestEnemyBaseInfo(state, unit.ownerTeamId, positionAtBattleStart)
      ?.distance ?? Number.POSITIVE_INFINITY;
  const hasFriendlyBase =
    getControlledFriendlyBases(state, unit.ownerTeamId).length > 0;
  const enemyBaseWithin3AtBattleStart = enemyBaseDistance <= 3;
  const retreatEligible =
    wasAliveAtBattleStart &&
    survivedPreviousBattle &&
    (attackedInPreviousBattle || wasTargetedInPreviousBattle) &&
    hasFriendlyBase &&
    isRetreatEligiblePosition(positionAtBattleStart) &&
    enemyBaseWithin3AtBattleStart;
  const retreatEligibilityReason = retreatEligible
    ? "eligible"
    : !wasAliveAtBattleStart
      ? "unit was not alive at battle start"
      : !survivedPreviousBattle
        ? "unit did not survive battle"
        : !attackedInPreviousBattle && !wasTargetedInPreviousBattle
          ? "unit was not recorded as a valid battle participant"
          : !hasFriendlyBase
            ? "unit has no friendly controlled base"
            : !isRetreatEligiblePosition(positionAtBattleStart)
              ? `battle-start position was ${positionAtBattleStart.kind}`
              : !enemyBaseWithin3AtBattleStart
                ? `nearest enemy-controlled base distance at battle was ${enemyBaseDistance}`
                : "not eligible";

  return {
    unitId: unit.id,
    battleTurnNumber,
    positionAtBattleStart,
    enemyBaseDistanceAtBattleStart: Number.isFinite(enemyBaseDistance)
      ? enemyBaseDistance
      : undefined,
    enemyBaseWithin3AtBattleStart,
    wasAliveAtBattleStart,
    survivedPreviousBattle,
    attackedInPreviousBattle,
    wasTargetedInPreviousBattle,
    retreatEligible,
    retreatEligibilityReason,
  };
}

export function getRetreatMoveEffect(
  state: GameState,
  unit: Unit,
  from: UnitPosition,
  to: UnitPosition,
): RetreatMoveEffect {
  if (!isAlive(unit)) return "none";
  const toFriendlyBase =
    to.kind === "base" &&
    getControlledFriendlyBases(state, unit.ownerTeamId).some(
      (base) => base.id === to.baseId,
    );
  if (toFriendlyBase) return isRetreating(unit) ? "complete" : "none";
  if (to.kind === "water" || to.kind === "removed")
    return isRetreating(unit) ? "release" : "none";

  const before = getNearestFriendlyBaseDistance(state, unit.ownerTeamId, from);
  const after = getNearestFriendlyBaseDistance(state, unit.ownerTeamId, to);
  if (!before || !after) return "none";

  if (isRetreating(unit)) {
    return after.distance <= before.distance ? "maintain" : "release";
  }

  if (isUnitRetreatEligible(state, unit) && after.distance < before.distance)
    return "start";
  return "none";
}
