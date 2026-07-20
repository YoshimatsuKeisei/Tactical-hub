import type {
  Base,
  GameState,
  Unit,
  UnitPosition,
  UnitStatus,
  UnitTurnFlags,
} from "../types";
import { chebyshevDistance } from "../utils/distance";
import { getTile, positionKey } from "../utils/position";
import { canMoveBetweenGroundPositions, getBaseConnectedRoadSectionIds, getBridgePositionAt, getPositionCoord, getRoadSectionIdForPosition } from "../utils/roadTopology";

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

export function getRetreatTargetBaseId(unit: Unit) {
  return unit.statuses.find((status) => status.kind === "retreating")?.retreatTargetBaseId;
}

export function withRetreatingStatus(unit: Unit, retreating: boolean, retreatTargetBaseId?: string): Unit {
  const statuses = unit.statuses.filter(
    (status) => status.kind !== "retreating",
  );
  return retreating
    ? {
        ...unit,
        statuses: [...statuses, { kind: "retreating", retreatTargetBaseId: retreatTargetBaseId ?? getRetreatTargetBaseId(unit) ?? "" } satisfies UnitStatus],
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

/** Shortest legal ground route to a currently controlled friendly base.
 * Enemy and neutral bases are deliberately not graph nodes, so they cannot be
 * crossed merely because they are geometrically close to the unit. */
export function getLegalRetreatRouteDistance(
  state: GameState,
  teamId: string,
  position: UnitPosition,
  targetBaseId?: string,
): DistanceResult | undefined {
  const friendlyBases = getControlledFriendlyBases(state, teamId).filter((base) => !targetBaseId || base.id === targetBaseId);
  if (!friendlyBases.length || position.kind === "removed" || position.kind === "water") return undefined;
  if (position.kind === "base") {
    return friendlyBases.some((base) => base.id === position.baseId)
      ? { distance: 0, baseIds: [position.baseId] }
      : undefined;
  }

  const startSection = position.kind === "tile" ? getRoadSectionIdForPosition(state, position) : undefined;
  if (position.kind === "tile" && !startSection) return undefined;
  const friendlyBySection = new Map<string, Base[]>();
  for (const base of friendlyBases) {
    for (const section of getBaseConnectedRoadSectionIds(state, base.id)) {
      const entries = friendlyBySection.get(section) ?? [];
      entries.push(base);
      friendlyBySection.set(section, entries);
    }
  }

  type GroundPosition = Extract<UnitPosition, { kind: "tile" | "bridge" }>;
  type Node = { position: GroundPosition; distance: number };
  const queue: Node[] = [{ position, distance: 0 }];
  const seen = new Set([positionKey(position)]);
  let best = Number.POSITIVE_INFINITY;
  const baseIds = new Set<string>();
  const directions = [-1, 0, 1].flatMap((dx) => [-1, 0, 1].map((dy) => ({ dx, dy }))).filter(({ dx, dy }) => dx || dy);

  while (queue.length) {
    const current = queue.shift()!;
    if (current.distance >= best) continue;
    const section = current.position.kind === "tile" ? getRoadSectionIdForPosition(state, current.position) : undefined;
    const currentCoord = getPositionCoord(state, current.position);
    for (const base of section ? friendlyBySection.get(section) ?? [] : []) {
      const adjacent = Boolean(currentCoord && base.coords.some((cell) => chebyshevDistance(cell, currentCoord) === 1));
      if (!adjacent) continue;
      const distance = current.distance + 1;
      if (distance < best) { best = distance; baseIds.clear(); }
      if (distance === best) baseIds.add(base.id);
    }
    for (const { dx, dy } of directions) {
      if (!currentCoord) continue;
      const x = currentCoord.x + dx, y = currentCoord.y + dy;
      const bridge = getBridgePositionAt(state, x, y);
      const candidate: GroundPosition = bridge ?? { kind: "tile", x, y };
      const tile = getTile(state.map.tiles, x, y);
      if (!bridge && (!tile || !["road", "baseGate", "reorganize"].includes(tile.terrain))) continue;
      if (state.constructions.some((entry) => entry.active && entry.kind === "obstacle" && entry.tiles.some((cell) => cell.x === x && cell.y === y))) continue;
      if (!canMoveBetweenGroundPositions(state, current.position, candidate)) continue;
      const key = positionKey(candidate);
      if (seen.has(key)) continue;
      seen.add(key);
      queue.push({ position: candidate, distance: current.distance + 1 });
    }
  }
  return Number.isFinite(best) ? { distance: best, baseIds: [...baseIds].sort() } : undefined;
}

export function getRetreatTargetBaseIdForMove(
  state: GameState,
  unit: Unit,
  from: UnitPosition,
  to: UnitPosition,
) {
  return getControlledFriendlyBases(state, unit.ownerTeamId)
    .map((base) => ({
      baseId: base.id,
      before: getLegalRetreatRouteDistance(state, unit.ownerTeamId, from, base.id)?.distance,
      after: getLegalRetreatRouteDistance(state, unit.ownerTeamId, to, base.id)?.distance,
    }))
    .filter((entry): entry is { baseId: string; before: number; after: number } => entry.before !== undefined && entry.after !== undefined && entry.after < entry.before)
    .sort((a, b) => a.after - b.after || a.before - b.before || a.baseId.localeCompare(b.baseId))[0]?.baseId;
}

export function clearInvalidRetreatTargets(state: GameState) {
  const invalidUnitIds = new Set<string>();
  state.units = state.units.map((unit) => {
    const flag = getUnitTurnFlags(state, unit.id);
    const contextValid = !flag?.retreatEligible || isRetreatContextValid(state, unit);
    if (!contextValid) invalidUnitIds.add(unit.id);
    if (!isRetreating(unit)) return unit;
    const targetBaseId = getRetreatTargetBaseId(unit);
    const team = state.teams.find((candidate) => candidate.id === unit.ownerTeamId);
    const valid = Boolean(
      contextValid &&
      targetBaseId &&
      isAlive(unit) &&
      team?.status === "active" &&
      state.bases.some((base) => base.id === targetBaseId) &&
      getBaseControllerTeamId(state, state.bases.find((base) => base.id === targetBaseId)!) === unit.ownerTeamId &&
      getLegalRetreatRouteDistance(state, unit.ownerTeamId, unit.position, targetBaseId),
    );
    if (valid) return unit;
    invalidUnitIds.add(unit.id);
    return withRetreatingStatus(unit, false);
  });
  if (invalidUnitIds.size) {
    state.unitTurnFlags = state.unitTurnFlags.map((flag) => invalidUnitIds.has(flag.unitId)
      ? { ...flag, retreatEligible: false, retreatEligibilityReason: "retreat target is no longer valid" }
      : flag);
  }
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
  const eligibilityFlag = getUnitTurnFlags(state, unit.id);
  if (eligibilityFlag?.retreatEligible && !isRetreatContextValid(state, unit)) return [];
  if (!retreating && !isUnitRetreatEligible(state, unit)) return [];

  const flag = eligibilityFlag;
  const retreatTargetBaseId = retreating ? getRetreatTargetBaseId(unit) : flag?.retreatFriendlyBaseIdsAtEligibility?.[0];
  const friendly = retreatTargetBaseId
    ? getLegalRetreatRouteDistance(state, unit.ownerTeamId, unit.position, retreatTargetBaseId)
    : getLegalRetreatRouteDistance(state, unit.ownerTeamId, unit.position);
  const hostile = flag?.retreatHostileBaseIdsAtEligibility?.length
    ? { distance: 0, baseIds: flag.retreatHostileBaseIdsAtEligibility }
    : getNearestEnemyBaseDistance(state, unit.ownerTeamId, unit.position);

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
  return Boolean(getUnitTurnFlags(state, unit.id)?.retreatEligible && isRetreatContextValid(state, unit));
}

export function isRetreatContextValid(state: GameState, unit: Unit) {
  const flag = getUnitTurnFlags(state, unit.id);
  if (!flag?.retreatEligible || !isAlive(unit) || state.teams.find((team) => team.id === unit.ownerTeamId)?.status !== "active") return false;
  const friendlyIds = flag.retreatFriendlyBaseIdsAtEligibility;
  const hostileIds = flag.retreatHostileBaseIdsAtEligibility;
  if (!friendlyIds?.length || !hostileIds?.length) return true;
  const friendlyStillFriendly = friendlyIds.some((baseId) => {
    const base = state.bases.find((entry) => entry.id === baseId);
    return Boolean(base && getBaseControllerTeamId(state, base) === unit.ownerTeamId && getLegalRetreatRouteDistance(state, unit.ownerTeamId, unit.position, base.id));
  });
  const hostileStillHostile = hostileIds.some((baseId) => {
    const base = state.bases.find((entry) => entry.id === baseId);
    const controller = base ? getBaseControllerTeamId(state, base) : undefined;
    return Boolean(controller && controller !== unit.ownerTeamId && state.teams.find((team) => team.id === controller)?.status !== "eliminated");
  });
  return friendlyStillFriendly && hostileStillHostile;
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
  const friendlyAtEligibility = getLegalRetreatRouteDistance(state, unit.ownerTeamId, positionAtBattleStart)?.baseIds ?? [];
  const hostileAtEligibility = getNearestEnemyBaseDistance(state, unit.ownerTeamId, positionAtBattleStart)?.baseIds ?? [];
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
    retreatFriendlyBaseIdsAtEligibility: friendlyAtEligibility,
    retreatHostileBaseIdsAtEligibility: hostileAtEligibility,
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
  if (toFriendlyBase) return isRetreating(unit)
    ? to.kind === "base" && to.baseId === getRetreatTargetBaseId(unit) ? "complete" : "release"
    : "none";
  if (to.kind === "water" || to.kind === "removed")
    return isRetreating(unit) ? "release" : "none";

  const before = getLegalRetreatRouteDistance(state, unit.ownerTeamId, from);
  const after = getLegalRetreatRouteDistance(state, unit.ownerTeamId, to);
  if (!before || !after) return isRetreating(unit) ? "release" : "none";

  if (isRetreating(unit)) {
    const targetBaseId = getRetreatTargetBaseId(unit);
    if (!targetBaseId) return "release";
    const targetBefore = getLegalRetreatRouteDistance(state, unit.ownerTeamId, from, targetBaseId);
    const targetAfter = getLegalRetreatRouteDistance(state, unit.ownerTeamId, to, targetBaseId);
    return targetBefore && targetAfter && targetAfter.distance <= targetBefore.distance ? "maintain" : "release";
  }

  if (isUnitRetreatEligible(state, unit) && getRetreatTargetBaseIdForMove(state, unit, from, to))
    return "start";
  return "none";
}
