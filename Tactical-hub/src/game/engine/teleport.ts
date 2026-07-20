import type { GameState, TeleportIntent, Unit, UnitPosition } from "../types";
import { getBridgePositionAt } from "../utils/roadTopology";
import { getTile, getUnitAtBoardCell, positionKey } from "../utils/position";
import { getOperationalAreaTiles } from "./construction";
import { isWithinEncourageRange } from "./encouragement";
import { getRetreatMoveEffect, getRetreatTargetBaseIdForMove, withRetreatingStatus } from "./retreat";

const alive = (unit: Unit) => unit.hp > 0 && unit.position.kind !== "removed";
export const isTeleportAvailable = (state: GameState, strategistId: string) =>
  state.turnNumber >= (state.teleportCooldowns.find((entry) => entry.strategistUnitId === strategistId)?.availableFromTurn ?? 0);

export function getTeleportStrategists(state: GameState, teamId: string) {
  return state.units.filter((unit) => unit.ownerTeamId === teamId && unit.type === "strategist" && unit.role === "teleporter" && alive(unit));
}

function ownReservations(state: GameState, teamId: string, exceptStrategistId?: string) {
  const moves = state.turnState.actionIntents.find((entry) => entry.teamId === teamId)?.movementIntents ?? [];
  const teleports = state.teleportIntents.filter((entry) => entry.teamId === teamId && entry.strategistUnitId !== exceptStrategistId);
  return { moves, teleports };
}

export function getTeleportTargetCandidates(state: GameState, strategistId: string) {
  const strategist = getTeleportStrategists(state, state.currentMovementTeamId ?? "").find((unit) => unit.id === strategistId);
  if (!strategist || state.phase !== "movement_input" || !isTeleportAvailable(state, strategist.id)) return [];
  const reserved = ownReservations(state, strategist.ownerTeamId, strategist.id);
  const reservedIds = new Set([...reserved.moves.map((entry) => entry.unitId), ...reserved.teleports.map((entry) => entry.targetUnitId), ...state.movedUnitIdsThisMovementPhase]);
  return state.units.filter((unit) =>
    alive(unit) && unit.ownerTeamId === strategist.ownerTeamId && unit.id !== strategist.id &&
    unit.type !== "king" && unit.type !== "engineer" &&
    unit.type !== "strategist" &&
    state.teams.find((team) => team.id === unit.ownerTeamId)?.status === "active" &&
    !reservedIds.has(unit.id) && isWithinEncourageRange(state, unit, strategist),
  ).sort((a, b) => a.id.localeCompare(b.id));
}

export function getTeleportDestinationCandidates(state: GameState, strategistId: string): UnitPosition[] {
  const strategist = getTeleportStrategists(state, state.currentMovementTeamId ?? "").find((unit) => unit.id === strategistId);
  if (!strategist || state.phase !== "movement_input" || !isTeleportAvailable(state, strategist.id)) return [];
  const reserved = ownReservations(state, strategist.ownerTeamId, strategist.id);
  const reservedKeys = new Set([...reserved.moves.filter((entry) => !entry.stay).map((entry) => positionKey(entry.to)), ...reserved.teleports.map((entry) => positionKey(entry.to))]);
  const destinations: UnitPosition[] = [];
  for (const cell of getOperationalAreaTiles(state, strategist.ownerTeamId)) {
    if (getUnitAtBoardCell(state, cell.x, cell.y)) continue;
    if (state.constructions.some((entry) => entry.active && entry.kind === "obstacle" && entry.tiles.some((tile) => tile.x === cell.x && tile.y === cell.y))) continue;
    const bridge = getBridgePositionAt(state, cell.x, cell.y);
    const tile = getTile(state.map.tiles, cell.x, cell.y);
    const position = bridge ?? (tile?.terrain === "road" ? { kind: "tile" as const, x: cell.x, y: cell.y } : undefined);
    if (position && !reservedKeys.has(positionKey(position))) destinations.push(position);
  }
  for (const base of state.bases.filter((entry) => entry.ownerTeamId === strategist.ownerTeamId))
    for (const slot of base.slots.filter((entry) => !entry.unitId)) {
      const position: UnitPosition = { kind: "base", baseId: base.id, slotId: slot.id };
      if (!reservedKeys.has(positionKey(position))) destinations.push(position);
    }
  return destinations.sort((a, b) => positionKey(a).localeCompare(positionKey(b)));
}

export function saveTeleportIntent(state: GameState, intent: TeleportIntent) {
  if (intent.teamId !== state.currentMovementTeamId) return state;
  if (!getTeleportTargetCandidates(state, intent.strategistUnitId).some((unit) => unit.id === intent.targetUnitId)) return state;
  if (!getTeleportDestinationCandidates(state, intent.strategistUnitId).some((position) => positionKey(position) === positionKey(intent.to))) return state;
  return { ...state, teleportIntents: [...state.teleportIntents.filter((entry) => entry.strategistUnitId !== intent.strategistUnitId), intent] };
}

export function cancelTeleportIntent(state: GameState, strategistId: string) {
  if (state.currentMovementTeamId !== state.units.find((unit) => unit.id === strategistId)?.ownerTeamId) return state;
  return { ...state, teleportIntents: state.teleportIntents.filter((entry) => entry.strategistUnitId !== strategistId) };
}

function place(state: GameState, unit: Unit, to: UnitPosition) {
  if (unit.position.kind === "base") {
    const { baseId, slotId } = unit.position;
    const slot = state.bases.find((base) => base.id === baseId)?.slots.find((entry) => entry.id === slotId);
    if (slot) slot.unitId = undefined;
  }
  if (to.kind === "base") {
    const slot = state.bases.find((base) => base.id === to.baseId)?.slots.find((entry) => entry.id === to.slotId);
    if (slot) slot.unitId = unit.id;
  }
  unit.position = to;
}

export function resolveTeamTeleports(state: GameState, teamId: string) {
  const intents = state.teleportIntents.filter((entry) => entry.teamId === teamId).sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId));
  state.teleportIntents = state.teleportIntents.filter((entry) => entry.teamId !== teamId);
  for (const intent of intents) {
    const target = getTeleportTargetCandidates(state, intent.strategistUnitId).find((unit) => unit.id === intent.targetUnitId);
    const destination = getTeleportDestinationCandidates(state, intent.strategistUnitId).find((position) => positionKey(position) === positionKey(intent.to));
    if (!target || !destination) {
      state.logs.push({ id: `log-teleport-fail-${state.logs.length}`, turnNumber: state.turnNumber, type: "movement", message: `${intent.strategistUnitId} teleport failed revalidation.`, relatedIds: [intent.strategistUnitId, intent.targetUnitId] });
      continue;
    }
    const effect = getRetreatMoveEffect(state, target, target.position, destination);
    const retreatBaseId = effect === "start" ? getRetreatTargetBaseIdForMove(state, target, target.position, destination) : undefined;
    place(state, target, destination);
    if (effect === "start" || effect === "maintain") Object.assign(target, withRetreatingStatus(target, true, retreatBaseId));
    if (effect === "release" || effect === "complete") Object.assign(target, withRetreatingStatus(target, false));
    state.movedUnitIdsThisMovementPhase.push(target.id);
    state.teleportCooldowns = [...state.teleportCooldowns.filter((entry) => entry.strategistUnitId !== intent.strategistUnitId), { strategistUnitId: intent.strategistUnitId, availableFromTurn: state.turnNumber + 5 }];
    state.logs.push({ id: `log-teleport-ok-${state.logs.length}`, turnNumber: state.turnNumber, type: "movement", message: `${intent.strategistUnitId} teleported ${target.id} to ${positionKey(destination)}.`, relatedIds: [intent.strategistUnitId, target.id] });
  }
}
