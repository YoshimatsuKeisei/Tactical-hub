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
  isGroundPositionConnectedToBase,
} from "../utils/roadTopology";
import {
  getBaseControllerTeamId,
  getRetreatMoveEffect,
  isRetreating,
  withRetreatingStatus,
} from "./retreat";

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
): position is Extract<UnitPosition, { kind: "tile" | "water" }> {
  return position.kind === "tile" || position.kind === "water";
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
  const position: UnitPosition =
    tile.terrain === "lake" ? { kind: "water", x, y } : { kind: "tile", x, y };
  return isLegalDestination(state, unit, position) ? position : undefined;
}

export function isLegalDestination(
  state: GameState,
  unit: Unit,
  destination: UnitPosition,
): boolean {
  if (destination.kind === "tile" || destination.kind === "water") {
    const tile = getTile(state.map.tiles, destination.x, destination.y);
    if (!tile || tile.terrain === "outside" || tile.terrain === "base")
      return false;
    if (getUnitAtBoardCell(state, destination.x, destination.y)) return false;
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
    unit.position.kind === "bridge"
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

    for (const { dx, dy } of directions) {
      const x = current.position.x + dx;
      const y = current.position.y + dy;
      const key = tileKey(x, y);
      const nextCost = current.cost + 1;

      const base = getBaseAtTile(state.bases, x, y);

      if (base) {
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
  return getMovementPaths(state, unitId).map((path) => path.destination);
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
  const existing = state.turnState.actionIntents.find(
    (candidate) => candidate.teamId === intent.teamId,
  );
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
) {
  state.units = state.units.map((unit) =>
    unit.id === unitId ? withRetreatingStatus(unit, retreating) : unit,
  );
}

export function resolveMovement(state: GameState): GameState {
  const next = structuredClone(state) as GameState;
  const intents = next.turnState.actionIntents
    .flatMap((intent) => intent.movementIntents)
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
      applyPosition(next, unit.id, intent.to);
      if (retreatEffect === "start" || retreatEffect === "maintain")
        applyRetreatStatus(next, unit.id, true);
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

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) => ({
    ...intent,
    movementIntents: [],
  }));
  next.unitTurnFlags = [];
  next.turnNumber += 1;
  next.phase = "movement_input";
  next.turnState.turnNumber = next.turnNumber;
  next.turnState.phase = "movement_input";
  return next;
}
