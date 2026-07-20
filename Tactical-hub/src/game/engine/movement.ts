import { UNIT_STATS } from "../constants";
import type { GameState, MovementIntent, Unit, UnitPosition } from "../types";
import {
  getBaseAtTile,
  getTile,
  getUnitAtBoardCell,
  positionKey,
  samePosition,
  tileKey,
} from "../utils/position";
import {
  canMoveBetweenGroundPositions,
  getBridgePositionAt,
  getPositionCoord,
  isGroundPositionConnectedToBase,
} from "../utils/roadTopology";
import {
  getBaseControllerTeamId,
  clearInvalidRetreatTargets,
  getRetreatMoveEffect,
  getRetreatTargetBaseIdForMove,
  isRetreating,
  withRetreatingStatus,
} from "./retreat";
import { completeSiegeCapture, transferBaseOwnership } from "./capture";
import { getSiegeState, resetInactiveSieges } from "./siege";
import { defeatTeamsWithoutBases } from "./defeat";
import { resolveTeamTeleports } from "./teleport";
import { isTeamProductionPending } from "./productionSchedule";

export type MovementStep =
  | { kind: "ground"; from: UnitPosition; to: UnitPosition }
  | { kind: "enter-base"; from: UnitPosition; to: UnitPosition; baseId: string }
  | {
      kind: "leave-base";
      from: UnitPosition;
      to: UnitPosition;
      baseId: string;
    };

export type MovementPath = {
  destination: UnitPosition;
  cost: number;
  steps: MovementStep[];
};

export type MovementValidationResult =
  | { valid: true; path: MovementPath }
  | { valid: false; reason: string };

const directions = [-1, 0, 1]
  .flatMap((dx) => [-1, 0, 1].map((dy) => ({ dx, dy })))
  .filter(({ dx, dy }) => dx || dy);

function getSeatOrder(state: GameState) {
  return state.movementSeatOrderTeamIds.length
    ? state.movementSeatOrderTeamIds
    : state.teams.filter((team) => !team.isNeutral).map((team) => team.id).sort((a, b) => a.localeCompare(b));
}

function getRotatedActiveMovementOrder(state: GameState, startIndex = state.movementOrderStartIndex) {
  const seats = getSeatOrder(state);
  if (!seats.length) return [];
  const normalizedStart = ((startIndex % seats.length) + seats.length) % seats.length;
  const active = new Set(state.teams.filter((team) => team.status === "active").map((team) => team.id));
  return [...seats.slice(normalizedStart), ...seats.slice(0, normalizedStart)].filter((teamId) => active.has(teamId));
}

export function beginMovementPhase(state: GameState): GameState {
  const next = structuredClone(state) as GameState;
  next.movementOrderTeamIds = getRotatedActiveMovementOrder(next);
  next.movementCompletedTeamIds = [];
  next.productionCompletedTeamIdsThisTurn = [];
  next.currentMovementTeamId = next.movementOrderTeamIds[0];
  next.teleportIntents = [];
  next.movedUnitIdsThisMovementPhase = [];
  next.phase = next.turnState.phase = "movement_input";
  return next;
}

export function getNextMovementTeamId(state: GameState) {
  const currentIndex = state.currentMovementTeamId
    ? state.movementOrderTeamIds.indexOf(state.currentMovementTeamId)
    : -1;
  return state.movementOrderTeamIds.slice(currentIndex + 1).find((teamId) =>
    state.teams.some((team) => team.id === teamId && team.status === "active") &&
    !state.movementCompletedTeamIds.includes(teamId),
  );
}

function canEnterWater(unit: Unit) {
  return unit.type === "ninja";
}

function baseCanBeEntered(state: GameState, baseId: string, unit: Unit) {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  if (!base) return false;
  const hasEnemy = base.slots.some((slot) => {
    if (!slot.unitId) return false;
    const occupant = state.units.find(
      (candidate) => candidate.id === slot.unitId,
    );
    return occupant?.ownerTeamId !== unit.ownerTeamId;
  });
  return !hasEnemy && base.slots.some((slot) => !slot.unitId);
}

function emptyBasePositions(state: GameState, baseId: string): UnitPosition[] {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  return (
    base?.slots
      .filter((slot) => !slot.unitId)
      .map((slot) => ({ kind: "base" as const, baseId, slotId: slot.id })) ?? []
  );
}

function isFriendlyControlledBase(
  state: GameState,
  unit: Unit,
  baseId: string,
) {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  return Boolean(
    base && getBaseControllerTeamId(state, base) === unit.ownerTeamId,
  );
}

function isGroundPosition(
  position: UnitPosition,
): position is Extract<UnitPosition, { kind: "tile" | "water" | "bridge" }> {
  return position.kind === "tile" || position.kind === "water" || position.kind === "bridge";
}

function positionForTile(
  state: GameState,
  unit: Unit,
  x: number,
  y: number,
): UnitPosition | undefined {
  const tile = getTile(state.map.tiles, x, y);
  if (!tile || tile.terrain === "outside" || tile.terrain === "base")
    return undefined;
  const bridge = getBridgePositionAt(state, x, y);
  const position: UnitPosition = bridge ?? (tile.terrain === "lake" ? { kind: "water", x, y } : { kind: "tile", x, y });
  return isLegalDestination(state, unit, position) ? position : undefined;
}

export function isLegalDestination(
  state: GameState,
  unit: Unit,
  destination: UnitPosition,
): boolean {
  if (destination.kind === "bridge") {
    const bridge = state.constructions.find((entry) => entry.active && entry.kind === "bridge" && entry.id === destination.bridgeId);
    const cell = bridge?.tiles[destination.cellIndex];
    return Boolean(cell && !getUnitAtBoardCell(state, cell.x, cell.y) && !state.constructions.some((entry) => entry.active && entry.kind === "obstacle" && entry.tiles.some((tile) => tile.x === cell.x && tile.y === cell.y)));
  }
  if (destination.kind === "tile" || destination.kind === "water") {
    const tile = getTile(state.map.tiles, destination.x, destination.y);
    if (!tile || tile.terrain === "outside" || tile.terrain === "base")
      return false;
    if (getUnitAtBoardCell(state, destination.x, destination.y)) return false;
    if (state.constructions.some((entry) => entry.active && entry.kind === "obstacle" && entry.tiles.some((cell) => cell.x === destination.x && cell.y === destination.y))) return false;
    if (destination.kind === "water")
      return tile.terrain === "lake" && canEnterWater(unit);
    return ["road", "baseGate", "reorganize"].includes(tile.terrain);
  }

  if (destination.kind === "base")
    return baseCanBeEntered(state, destination.baseId, unit);
  return false;
}

function nextGroundPositionsFromBase(
  state: GameState,
  unit: Unit,
  baseId: string,
) {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  if (!base) return [];
  const positions = new Map<string, UnitPosition>();
  for (const coord of base.coords) {
    for (const { dx, dy } of directions) {
      const x = coord.x + dx;
      const y = coord.y + dy;
      if (getBaseAtTile(state.bases, x, y)) continue;
      const position = positionForTile(state, unit, x, y);

      if (!position) {
        continue;
      }

      // Base-to-lake movement is intentionally undefined. Ninjas enter and
      // leave water only through an adjacent normal ground tile.
      if (position.kind === "water") continue;

      /*
       * 地上へ退城する場合は、
       * この拠点へ接続した道区間だけを許可する。
       *
       * 水上忍者の既存挙動は変更しない。
       */
      if (
        position.kind === "tile" &&
        !isGroundPositionConnectedToBase(state, position, baseId)
      ) {
        continue;
      }

      positions.set(positionKey(position), position);
    }
  }
  return [...positions.values()];
}

export function getMovementPaths(
  state: GameState,
  unitId: string,
): MovementPath[] {
  const unit = state.units.find((candidate) => candidate.id === unitId);
  if (
    !unit ||
    unit.position.kind === "removed" ||
    false
  )
    return [];

  const maxMove = UNIT_STATS[unit.type].move;
  const results = new Map<string, MovementPath>();
  const queue: {
    position: UnitPosition;
    cost: number;
    steps: MovementStep[];
  }[] = [{ position: unit.position, cost: 0, steps: [] }];
  const visited = new Set<string>([`${positionKey(unit.position)}:0`]);

  while (queue.length) {
    const current = queue.shift()!;
    if (current.cost >= maxMove) continue;

    if (current.position.kind === "base") {
      const nextCost = current.cost + 1;

      for (const destination of nextGroundPositionsFromBase(
        state,
        unit,
        current.position.baseId,
      )) {
        const path: MovementPath = {
          destination,
          cost: nextCost,
          steps: [
            ...current.steps,
            {
              kind: "leave-base",
              from: current.position,
              to: destination,
              baseId: current.position.baseId,
            },
          ],
        };

        results.set(positionKey(destination), path);

        /*
         * 騎兵など、退城後にも移動力が残る駒は、
         * 退城先から残りの移動探索を続ける。
         *
         * 移動力1の駒は nextCost === maxMove となるため、
         * 退城した地点で終了する。
         */
        if (nextCost < maxMove) {
          const visitedKey = `${positionKey(destination)}:${nextCost}`;

          if (!visited.has(visitedKey)) {
            visited.add(visitedKey);

            queue.push({
              position: destination,
              cost: nextCost,
              steps: path.steps,
            });
          }
        }
      }

      continue;
    }

    if (!isGroundPosition(current.position)) continue;

    const currentCoord = getPositionCoord(state, current.position);
    if (!currentCoord) continue;

    for (const { dx, dy } of directions) {
      const x = currentCoord.x + dx;
      const y = currentCoord.y + dy;
      const key = tileKey(x, y);
      const nextCost = current.cost + 1;

      const base = getBaseAtTile(state.bases, x, y);

      if (base) {
        // Lake-to-base movement is intentionally undefined.
        if (current.position.kind === "water") continue;
        /*
         * 地上から入城する場合、
         * 現在の道区間がその拠点へ接続している必要がある。
         *
         * 水上忍者の既存挙動はここでは変更しない。
         */
        if (
          current.position.kind === "tile" &&
          !isGroundPositionConnectedToBase(state, current.position, base.id)
        ) {
          continue;
        }

        for (const basePosition of emptyBasePositions(state, base.id)) {
          if (!isLegalDestination(state, unit, basePosition)) continue;
          const enterPath: MovementPath = {
            destination: basePosition,
            cost: nextCost,
            steps: [
              ...current.steps,
              {
                kind: "enter-base",
                from: current.position,
                to: basePosition,
                baseId: base.id,
              },
            ],
          };
          results.set(positionKey(basePosition), enterPath);
          if (
            nextCost < maxMove &&
            isFriendlyControlledBase(state, unit, base.id)
          ) {
            const visitedKey = `${positionKey(basePosition)}:${nextCost}`;
            if (!visited.has(visitedKey)) {
              visited.add(visitedKey);
              queue.push({
                position: basePosition,
                cost: nextCost,
                steps: enterPath.steps,
              });
            }
          }
        }
        continue;
      }

      const destination = positionForTile(state, unit, x, y);

      if (!destination) {
        continue;
      }

      /*
       * 通常の地上移動は同一道区間内だけ許可する。
       *
       * 異なる道区間への移動は、
       * 自軍拠点をenter-base → leave-baseとして
       * 正式に経由した場合のみ成立する。
       */
      if (
        !canMoveBetweenGroundPositions(state, current.position, destination)
      ) {
        continue;
      }

      const path: MovementPath = {
        destination,
        cost: nextCost,
        steps: [
          ...current.steps,
          { kind: "ground", from: current.position, to: destination },
        ],
      };
      results.set(positionKey(destination), path);
      const visitedKey = `${key}:${nextCost}`;
      if (!visited.has(visitedKey)) {
        visited.add(visitedKey);
        queue.push({
          position: destination,
          cost: nextCost,
          steps: path.steps,
        });
      }
    }
  }

  return [...results.values()];
}

export function getMovementCandidates(
  state: GameState,
  unitId: string,
): UnitPosition[] {
  const unit = state.units.find((candidate) => candidate.id === unitId);
  if (state.phase === "movement_input" && unit?.ownerTeamId !== state.currentMovementTeamId) return [];
  if (state.movedUnitIdsThisMovementPhase.includes(unitId) || state.teleportIntents.some((intent) => intent.targetUnitId === unitId)) return [];
  if (state.phase !== "movement_input" || !unit) return getMovementPaths(state, unitId).map((path) => path.destination);
  const planningState = structuredClone(state) as GameState;
  const teammatePlans = planningState.turnState.actionIntents
    .find((intent) => intent.teamId === unit.ownerTeamId)
    ?.movementIntents.filter((intent) => intent.unitId !== unitId && !intent.stay) ?? [];
  for (const planned of teammatePlans) applyPosition(planningState, planned.unitId, planned.to);
  const teleportDestinations = new Set(state.teleportIntents.map((intent) => positionKey(intent.to)));
  return getMovementPaths(planningState, unitId).map((path) => path.destination).filter((position) => !teleportDestinations.has(positionKey(position)));
}

export function getTeamMovementCandidates(state: GameState, teamId: string) {
  if (state.phase !== "movement_input" || state.currentMovementTeamId !== teamId) return [];
  return state.units
    .filter((unit) => unit.ownerTeamId === teamId && unit.hp > 0 && unit.position.kind !== "removed")
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((unit) => ({
      unitId: unit.id,
      destinations: getMovementCandidates(state, unit.id).sort((left, right) => positionKey(left).localeCompare(positionKey(right))),
    }));
}

export function validateMovementPath(
  state: GameState,
  unit: Unit,
  from: UnitPosition,
  to: UnitPosition,
): MovementValidationResult {
  if (!samePosition(unit.position, from))
    return { valid: false, reason: "unit is no longer at the source position" };
  const path = getMovementPaths(state, unit.id).find((candidate) =>
    samePosition(candidate.destination, to),
  );
  return path
    ? { valid: true, path }
    : { valid: false, reason: "no legal movement path" };
}

export function saveMovementIntent(
  state: GameState,
  intent: MovementIntent,
): GameState {
  const unit = state.units.find((candidate) => candidate.id === intent.unitId);
  if (
    state.phase !== "movement_input" ||
    intent.teamId !== state.currentMovementTeamId ||
    unit?.ownerTeamId !== intent.teamId ||
    state.movementCompletedTeamIds.includes(intent.teamId)
  ) return state;
  if (state.movedUnitIdsThisMovementPhase.includes(intent.unitId) || state.teleportIntents.some((entry) => entry.targetUnitId === intent.unitId || (!intent.stay && samePosition(entry.to, intent.to)))) return state;
  const existing = state.turnState.actionIntents.find(
    (candidate) => candidate.teamId === intent.teamId,
  );
  if (
    !intent.stay &&
    existing?.movementIntents.some((movement) =>
      movement.unitId !== intent.unitId && !movement.stay && samePosition(movement.to, intent.to),
    )
  ) return state;
  const actionIntents = existing
    ? state.turnState.actionIntents.map((candidate) =>
        candidate.teamId === intent.teamId
          ? {
              ...candidate,
              movementIntents: [
                ...candidate.movementIntents.filter(
                  (movement) => movement.unitId !== intent.unitId,
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
          movementIntents: [intent],
          attackIntents: [],
        },
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
  state.units = state.units.map((unit) =>
    unit.id === unitId ? { ...unit, position: to } : unit,
  );
}

function applyRetreatStatus(
  state: GameState,
  unitId: string,
  retreating: boolean,
  retreatTargetBaseId?: string,
) {
  state.units = state.units.map((unit) =>
    unit.id === unitId ? withRetreatingStatus(unit, retreating, retreatTargetBaseId) : unit,
  );
}

function resolveCurrentTeamMovement(state: GameState, teamId: string): GameState {
  const next = structuredClone(state) as GameState;
  resolveTeamTeleports(next, teamId);
  resetInactiveSieges(next);
  const defendingCountsAtStart = new Map(next.bases.map((base) => [base.id, next.units.filter((unit) => unit.hp > 0 && unit.position.kind === "base" && unit.position.baseId === base.id && unit.ownerTeamId === base.ownerTeamId).length]));
  const intents = next.turnState.actionIntents
    .filter((intent) => intent.teamId === teamId)
    .flatMap((intent) => intent.movementIntents.filter((movement) => movement.teamId === teamId))
    .sort((a, b) => {
      const aUnit = next.units.find((unit) => unit.id === a.unitId);
      const bUnit = next.units.find((unit) => unit.id === b.unitId);
      const priority =
        (aUnit ? UNIT_STATS[aUnit.type].priority : 99) -
        (bUnit ? UNIT_STATS[bUnit.type].priority : 99);
      return priority || a.unitId.localeCompare(b.unitId);
    });

  for (const intent of intents) {
    const unit = next.units.find((candidate) => candidate.id === intent.unitId);
    if (!unit) {
      next.logs.push({
        id: `log-move-stay-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "movement",
        message: `${intent.unitId} stayed.`,
      });
      continue;
    }

    if (intent.stay) {
      const wasRetreating = isRetreating(unit);
      if (wasRetreating) applyRetreatStatus(next, unit.id, false);
      next.logs.push({
        id: `log-move-stay-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "movement",
        message: wasRetreating
          ? `${intent.unitId} stayed and ended retreat.`
          : `${intent.unitId} stayed.`,
        relatedIds: [intent.unitId],
      });
      continue;
    }

    const movement = validateMovementPath(next, unit, intent.from, intent.to);
    if (movement.valid) {
      const retreatEffect = getRetreatMoveEffect(
        next,
        unit,
        intent.from,
        intent.to,
      );
      const retreatTargetBaseId = retreatEffect === "start"
        ? getRetreatTargetBaseIdForMove(next, unit, intent.from, intent.to)
        : undefined;
      applyPosition(next, unit.id, intent.to);
      next.movedUnitIdsThisMovementPhase.push(unit.id);
      if (retreatEffect === "start" || retreatEffect === "maintain")
        applyRetreatStatus(next, unit.id, true, retreatTargetBaseId);
      if (retreatEffect === "release" || retreatEffect === "complete")
        applyRetreatStatus(next, unit.id, false);

      const retreatMessage =
        retreatEffect === "start"
          ? " Retreat started."
          : retreatEffect === "maintain"
            ? " Retreat maintained."
            : retreatEffect === "release"
              ? " Retreat ended."
              : retreatEffect === "complete"
                ? " Retreat completed at a friendly base."
                : "";
      next.logs.push({
        id: `log-move-ok-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "movement",
        message: `${unit.id} moved to ${positionKey(intent.to)}.${retreatMessage}`,
        relatedIds: [unit.id],
      });
    } else {
      next.logs.push({
        id: `log-move-fail-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "movement",
        message: `${intent.unitId} failed to move to ${positionKey(intent.to)}: ${movement.reason}.`,
        relatedIds: [intent.unitId],
      });
    }
  }

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) =>
    intent.teamId === teamId ? { ...intent, movementIntents: [] } : intent,
  );
  const combatAbandonmentBases = new Set<string>();
  for (const base of [...next.bases]) {
    if (!(defendingCountsAtStart.get(base.id) ?? 0)) continue;
    const defendersRemain = next.units.some((unit) => unit.hp > 0 && unit.position.kind === "base" && unit.position.baseId === base.id && unit.ownerTeamId === base.ownerTeamId);
    if (defendersRemain) continue;
    const siege = getSiegeState(next, base.id);
    if (siege?.active && siege.defenderLossOccurred) {
      const candidates = siege.teamRecords.filter((record) => record.teamId !== siege.defendingTeamId && (record.defenderKills > 0 || record.effectiveAttackTurns > 0)).map((record) => record.teamId);
      completeSiegeCapture(next, siege, candidates, "combat_abandonment");
      combatAbandonmentBases.add(base.id);
    }
  }
  for (const base of [...next.bases]) {
    if (combatAbandonmentBases.has(base.id)) continue;
    const occupyingEnemy = next.units.find((unit) => unit.hp > 0 && unit.position.kind === "base" && unit.position.baseId === base.id && unit.ownerTeamId !== base.ownerTeamId);
    const ownerDefender = next.units.some((unit) => unit.hp > 0 && unit.position.kind === "base" && unit.position.baseId === base.id && unit.ownerTeamId === base.ownerTeamId);
    if (occupyingEnemy && !ownerDefender) {
      transferBaseOwnership(next, base.id, occupyingEnemy.ownerTeamId);
      next.logs.push({ id: `log-simple-capture-${next.logs.length}`, turnNumber: next.turnNumber, type: "capture", message: `単純放棄された拠点への入城占領: ${base.id} → ${occupyingEnemy.ownerTeamId}`, relatedIds: [base.id, occupyingEnemy.ownerTeamId] });
    }
  }
  defeatTeamsWithoutBases(next);
  clearInvalidRetreatTargets(next);
  next.movementCompletedTeamIds = [...new Set([...next.movementCompletedTeamIds, teamId])];
  next.movementOrderTeamIds = next.movementOrderTeamIds.filter((orderedTeamId) =>
    next.teams.some((team) => team.id === orderedTeamId && team.status === "active"),
  );
  const nextTeamId = next.movementOrderTeamIds.find((orderedTeamId) =>
    !next.movementCompletedTeamIds.includes(orderedTeamId),
  );
  if (nextTeamId) {
    next.currentMovementTeamId = nextTeamId;
    next.phase = next.turnState.phase = "movement_input";
    return next;
  }

  next.currentMovementTeamId = undefined;
  next.unitTurnFlags = [];
  next.turnNumber += 1;
  next.turnState.turnNumber = next.turnNumber;
  const seats = getSeatOrder(next);
  next.movementOrderStartIndex = seats.length ? (next.movementOrderStartIndex + 1) % seats.length : 0;
  next.movementOrderTeamIds = getRotatedActiveMovementOrder(next);
  next.movementCompletedTeamIds = [];
  next.teleportIntents = [];
  next.movedUnitIdsThisMovementPhase = [];
  if (next.rewardPlacementRequests.some((request) => !request.completed && !request.expired)) {
    next.phaseAfterRewards = "attack_input";
    next.phase = "reward_placement";
  } else next.phase = "attack_input";
  next.turnState.phase = next.phase;
  return next;
}

export function submitMovement(state: GameState, teamId: string): GameState {
  if (
    state.phase !== "movement_input" ||
    state.currentMovementTeamId !== teamId ||
    state.teams.find((team) => team.id === teamId)?.status !== "active" ||
    state.movementCompletedTeamIds.includes(teamId)
  ) return state;
  if (!isTeamProductionPending(state, teamId)) return resolveCurrentTeamMovement(state, teamId);
  const hasSavedProduction = state.turnState.actionIntents.some(
    (intent) => intent.teamId === teamId && intent.productionChoices.length > 0,
  );
  if (hasSavedProduction) return state;
  const skipped = {
    ...state,
    productionCompletedTeamIdsThisTurn: [...new Set([...state.productionCompletedTeamIdsThisTurn, teamId])],
  };
  return resolveCurrentTeamMovement(skipped, teamId);
}

export function resolveMovement(state: GameState): GameState {
  return state.currentMovementTeamId ? submitMovement(state, state.currentMovementTeamId) : state;
}
