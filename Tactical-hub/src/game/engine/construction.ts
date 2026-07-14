import type { BoardCoord, Construction, GameState, StrategistActionIntent, Unit } from "../types";
import { getBaseConnectedRoadSectionIds, getRoadSectionIdAtTile } from "../utils/roadTopology";
import { getTile, getUnitAtBoardCell, tileKey } from "../utils/position";

const ORTHOGONAL = [{ dx: 1, dy: 0 }, { dx: -1, dy: 0 }, { dx: 0, dy: 1 }, { dx: 0, dy: -1 }];
const key = (cell: BoardCoord) => tileKey(cell.x, cell.y);
const bridgeKey = (cells: BoardCoord[]) => {
  const forward = cells.map(key).join("|");
  const reverse = [...cells].reverse().map(key).join("|");
  return forward < reverse ? forward : reverse;
};
const active = (construction: Construction) => construction.active;

export function getConstructionAt(state: GameState, x: number, y: number, kind?: Construction["kind"]) {
  return state.constructions.find((entry) => active(entry) && (!kind || entry.kind === kind) && entry.tiles.some((cell) => cell.x === x && cell.y === y));
}

export function getBuilderUnits(state: GameState, teamId?: string) {
  return state.units.filter((unit) => unit.type === "strategist" && unit.role === "builder" && unit.hp > 0 && unit.position.kind !== "removed" && (!teamId || unit.ownerTeamId === teamId) && state.teams.find((team) => team.id === unit.ownerTeamId)?.status === "active");
}

export function countLivingStrategists(state: GameState, teamId: string) {
  return state.units.filter((unit) => unit.ownerTeamId === teamId && unit.type === "strategist" && unit.hp > 0 && unit.position.kind !== "removed").length;
}

export function getOperationalRoadSectionIds(state: GameState, teamId: string) {
  const team = state.teams.find((candidate) => candidate.id === teamId);
  if (team?.status !== "active") return new Set<string>();
  const controlled = new Set(team.controlledBaseIds);
  const sections = new Set<string>();
  for (const base of state.bases.filter((entry) => controlled.has(entry.id) || entry.ownerTeamId === teamId)) {
    for (const section of getBaseConnectedRoadSectionIds(state, base.id)) sections.add(section);
  }
  return sections;
}

export function getOperationalRoadTiles(state: GameState, teamId: string) {
  const sections = getOperationalRoadSectionIds(state, teamId);
  return state.map.tiles.filter((tile) => tile.roadSectionId && sections.has(tile.roadSectionId));
}

function ownPending(state: GameState, teamId: string, exceptUnitId?: string) {
  return state.strategistActionIntents.filter((intent) => intent.teamId === teamId && intent.strategistUnitId !== exceptUnitId);
}

export function getObstacleCandidates(state: GameState, strategistUnitId: string) {
  const strategist = getBuilderUnits(state).find((unit) => unit.id === strategistUnitId);
  if (!strategist || getManagedConstruction(state, strategist.id, "obstacle") || !isAvailable(state, strategist.id, "obstacle")) return [];
  const operational = new Set(getOperationalRoadTiles(state, strategist.ownerTeamId).map((tile) => tileKey(tile.x, tile.y)));
  const connectedBridgeCells = state.constructions.filter((entry) => entry.active && entry.kind === "bridge").flatMap((bridge) => bridge.tiles.filter((cell) => ORTHOGONAL.some(({ dx, dy }) => operational.has(tileKey(cell.x + dx, cell.y + dy)))));
  const reserved = new Set(ownPending(state, strategist.ownerTeamId, strategist.id).filter((intent) => intent.action === "place_obstacle").flatMap((intent) => intent.tiles ?? []).map(key));
  return [...getOperationalRoadTiles(state, strategist.ownerTeamId).map(({ x, y }) => ({ x, y })), ...connectedBridgeCells]
    .filter((cell, index, all) => all.findIndex((other) => key(other) === key(cell)) === index)
    .filter((cell) => !reserved.has(key(cell)) && !getConstructionAt(state, cell.x, cell.y, "obstacle") && !getUnitAtBoardCell(state, cell.x, cell.y));
}

export function getBridgeCandidates(state: GameState, strategistUnitId: string) {
  const strategist = getBuilderUnits(state).find((unit) => unit.id === strategistUnitId);
  if (!strategist || getManagedConstruction(state, strategist.id, "bridge") || !isAvailable(state, strategist.id, "bridge")) return [];
  const operational = getOperationalRoadTiles(state, strategist.ownerTeamId);
  const occupied = new Set(state.constructions.filter((entry) => entry.active && entry.kind === "bridge").flatMap((entry) => entry.tiles).map(key));
  for (const intent of ownPending(state, strategist.ownerTeamId, strategist.id).filter((entry) => entry.action === "place_bridge")) for (const cell of intent.tiles ?? []) occupied.add(key(cell));
  const candidates = new Map<string, BoardCoord[]>();
  for (const start of operational) for (const { dx, dy } of ORTHOGONAL) {
    const cells: BoardCoord[] = [];
    let x = start.x + dx, y = start.y + dy;
    while (getTile(state.map.tiles, x, y)?.terrain === "lake") { cells.push({ x, y }); x += dx; y += dy; }
    const opposite = getTile(state.map.tiles, x, y);
    if (!cells.length || !opposite?.roadSectionId || cells.some((cell) => occupied.has(key(cell)))) continue;
    candidates.set(bridgeKey(cells), cells);
  }
  return [...candidates.values()].sort((a, b) => a.map(key).join().localeCompare(b.map(key).join()));
}

export function getManagedConstruction(state: GameState, managerUnitId: string, kind: Construction["kind"]) {
  return state.constructions.find((entry) => entry.active && entry.managerUnitId === managerUnitId && entry.kind === kind);
}
function canSafelyReset(state: GameState, construction: Construction) {
  return construction.kind !== "bridge" || !state.units.some((unit) => unit.position.kind === "bridge" && unit.position.bridgeId === construction.id);
}
export function isAvailable(state: GameState, unitId: string, kind: Construction["kind"]) {
  return state.turnNumber >= (state.strategistCooldowns.find((entry) => entry.strategistUnitId === unitId && entry.kind === kind)?.availableFromTurn ?? 0);
}

export function saveStrategistActionIntent(state: GameState, intent: StrategistActionIntent) {
  const strategist = getBuilderUnits(state, intent.teamId).find((unit) => unit.id === intent.strategistUnitId);
  if (!strategist || state.strategistSubmittedTeamIds.includes(intent.teamId)) return state;
  const legal = intent.action === "pass" ||
    (intent.action === "place_bridge" && getBridgeCandidates(state, strategist.id).some((candidate) => bridgeKey(candidate) === bridgeKey(intent.tiles ?? []))) ||
    (intent.action === "place_obstacle" && intent.tiles?.length === 1 && getObstacleCandidates(state, strategist.id).some((cell) => key(cell) === key(intent.tiles![0]))) ||
    (intent.action === "reset_bridge" && getManagedConstruction(state, strategist.id, "bridge")?.id === intent.constructionId && canSafelyReset(state, getManagedConstruction(state, strategist.id, "bridge")!)) ||
    (intent.action === "reset_obstacle" && getManagedConstruction(state, strategist.id, "obstacle")?.id === intent.constructionId);
  if (!legal) return state;
  return { ...state, strategistActionIntents: [...state.strategistActionIntents.filter((entry) => entry.strategistUnitId !== strategist.id), intent] };
}

export function submitStrategistActions(state: GameState, teamId: string) {
  if (state.phase !== "strategist_action_input" || state.teams.find((team) => team.id === teamId)?.status !== "active") return state;
  const next = structuredClone(state) as GameState;
  for (const unit of getBuilderUnits(next, teamId)) if (!next.strategistActionIntents.some((intent) => intent.strategistUnitId === unit.id)) next.strategistActionIntents.push({ teamId, strategistUnitId: unit.id, action: "pass" });
  next.strategistSubmittedTeamIds = [...new Set([...next.strategistSubmittedTeamIds, teamId])];
  next.logs.push({ id: `log-strategist-submit-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${teamId} submitted strategist actions.`, relatedIds: [teamId] });
  if (next.teams.filter((team) => team.status === "active").every((team) => next.strategistSubmittedTeamIds.includes(team.id))) next.phase = next.turnState.phase = "strategist_action_resolution";
  return next;
}

function conflictGroups(intents: StrategistActionIntent[]) {
  const conflicts = new Set<string>();
  for (let a = 0; a < intents.length; a++) for (let b = a + 1; b < intents.length; b++) {
    if (intents[a].teamId === intents[b].teamId || !intents[a].action.startsWith("place_") || intents[a].action !== intents[b].action) continue;
    if ((intents[a].tiles ?? []).some((cell) => (intents[b].tiles ?? []).some((other) => key(cell) === key(other)))) { conflicts.add(intents[a].strategistUnitId); conflicts.add(intents[b].strategistUnitId); }
  }
  return conflicts;
}

export function resolveStrategistActions(state: GameState) {
  if (state.phase !== "strategist_action_resolution") return state;
  const next = structuredClone(state) as GameState;
  const intents = [...next.strategistActionIntents].sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId));
  for (const intent of intents.filter((entry) => entry.action.startsWith("reset_"))) {
    const construction = next.constructions.find((entry) => entry.id === intent.constructionId && entry.managerUnitId === intent.strategistUnitId && entry.active);
    if (!construction || !canSafelyReset(next, construction)) continue;
    construction.active = false;
    if (construction.kind === "bridge") for (const obstacle of next.constructions.filter((entry) => entry.active && entry.kind === "obstacle" && entry.tiles.some((cell) => construction.tiles.some((bridgeCell) => key(cell) === key(bridgeCell))))) obstacle.active = false;
    next.strategistCooldowns = [...next.strategistCooldowns.filter((entry) => entry.strategistUnitId !== intent.strategistUnitId || entry.kind !== construction.kind), { strategistUnitId: intent.strategistUnitId, kind: construction.kind, availableFromTurn: next.turnNumber + 5 }];
    next.logs.push({ id: `log-construction-reset-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${intent.teamId} ${intent.strategistUnitId} reset ${construction.kind}; available turn ${next.turnNumber + 5}.`, relatedIds: [construction.id, intent.strategistUnitId] });
  }
  const placements = intents.filter((entry) => entry.action.startsWith("place_"));
  const conflicts = conflictGroups(placements);
  for (const intent of placements) {
    if (conflicts.has(intent.strategistUnitId)) { next.logs.push({ id: `log-construction-conflict-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${intent.teamId} ${intent.strategistUnitId} placement failed by simultaneous conflict.`, relatedIds: [intent.strategistUnitId] }); continue; }
    const snapshotIntent = { ...intent, tiles: intent.tiles?.map((cell) => ({ ...cell })) };
    const legal = snapshotIntent.action === "place_bridge" ? getBridgeCandidates(next, snapshotIntent.strategistUnitId).some((candidate) => bridgeKey(candidate) === bridgeKey(snapshotIntent.tiles ?? [])) : snapshotIntent.action === "place_obstacle" && snapshotIntent.tiles?.length === 1 && getObstacleCandidates(next, snapshotIntent.strategistUnitId).some((cell) => key(cell) === key(snapshotIntent.tiles![0]));
    if (!legal) { next.logs.push({ id: `log-construction-prerequisite-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${intent.teamId} ${intent.strategistUnitId} placement prerequisite disappeared.`, relatedIds: [intent.strategistUnitId] }); continue; }
    const kind = intent.action === "place_bridge" ? "bridge" : "obstacle";
    next.constructions.push({ id: `${kind}-${next.turnNumber}-${next.constructions.length}`, kind, ownerTeamId: intent.teamId, managerUnitId: intent.strategistUnitId, tiles: intent.tiles!, placedTurn: next.turnNumber, active: true });
    next.logs.push({ id: `log-construction-place-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${intent.teamId} ${intent.strategistUnitId} placed ${kind} at ${intent.tiles!.map(key).join(" / ")}.`, relatedIds: [intent.strategistUnitId] });
  }
  next.strategistActionIntents = []; next.strategistSubmittedTeamIds = [];
  next.phase = next.turnState.phase = "movement_input";
  return next;
}

export function getOwnStrategistPreview(state: GameState, teamId: string) {
  return state.strategistActionIntents.filter((intent) => intent.teamId === teamId && intent.action.startsWith("place_")).flatMap((intent) => (intent.tiles ?? []).map((cell) => ({ ...cell, valid: true, kind: intent.action === "place_bridge" ? "bridge" as const : "obstacle" as const })));
}

export function beginStrategistActionPhase(state: GameState) {
  const next = structuredClone(state) as GameState;
  next.phase = next.turnState.phase = "strategist_action_input";
  next.strategistActionIntents = [];
  next.strategistSubmittedTeamIds = [];
  next.logs.push({ id: `log-strategist-start-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: "Strategist action phase started." });
  for (const cooldown of next.strategistCooldowns.filter((entry) => entry.availableFromTurn === next.turnNumber)) {
    next.logs.push({ id: `log-cooldown-ready-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${cooldown.strategistUnitId} may place ${cooldown.kind} again.`, relatedIds: [cooldown.strategistUnitId] });
  }
  return next;
}
