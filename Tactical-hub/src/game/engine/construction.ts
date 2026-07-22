import type { BoardCoord, Construction, GameState, StrategistActionIntent, Unit } from "../types";
import { getBaseConnectedRoadSectionIds, getRoadSectionIdAtTile } from "../utils/roadTopology";
import { getTile, getUnitAtBoardCell, tileKey } from "../utils/position";
import { chebyshevDistance } from "../utils/distance";
import { getKingCampaign, recordKingDamage } from "./kingCampaign";
import { resolveKingDefeats, type DefeatedKingPlan } from "./defeat";
import { beginMovementPhase } from "./movement";

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

export function getOperationalAreaTiles(state: GameState, teamId: string): BoardCoord[] {
  const roads = getOperationalRoadTiles(state, teamId);
  const operationalRoads = new Set(roads.map((tile) => tileKey(tile.x, tile.y)));
  const connectedBridgeCells = state.constructions
    .filter(
      (entry) =>
        entry.active &&
        entry.kind === "bridge" &&
        entry.tiles.some((cell) =>
          [-1, 0, 1].some((dx) => [-1, 0, 1].some((dy) => (dx !== 0 || dy !== 0) &&
            operationalRoads.has(tileKey(cell.x + dx, cell.y + dy)),
          )),
        ),
    )
    .flatMap((bridge) => bridge.tiles);
  const cells = new Map<string, BoardCoord>();
  for (const cell of [...roads, ...connectedBridgeCells])
    cells.set(key(cell), { x: cell.x, y: cell.y });
  return [...cells.values()].sort((left, right) => left.x - right.x || left.y - right.y);
}

function ownPending(state: GameState, teamId: string, exceptUnitId?: string) {
  return state.strategistActionIntents.filter((intent) => intent.teamId === teamId && intent.strategistUnitId !== exceptUnitId);
}

export function getObstacleCandidates(state: GameState, strategistUnitId: string) {
  const strategist = getBuilderUnits(state).find((unit) => unit.id === strategistUnitId);
  if (!strategist || getManagedConstructions(state, strategist.id, "obstacle").length >= getConstructionManagementLimit(state, strategist.id, "obstacle") || !isAvailable(state, strategist.id, "obstacle")) return [];
  const reserved = new Set(ownPending(state, strategist.ownerTeamId, strategist.id).filter((intent) => intent.action === "place_obstacle").flatMap((intent) => intent.tiles ?? []).map(key));
  return getOperationalAreaTiles(state, strategist.ownerTeamId)
    .filter((cell) => !reserved.has(key(cell)) && !getConstructionAt(state, cell.x, cell.y, "obstacle") && !getUnitAtBoardCell(state, cell.x, cell.y));
}

export function getBridgeCandidates(state: GameState, strategistUnitId: string) {
  const strategist = getBuilderUnits(state).find((unit) => unit.id === strategistUnitId);
  if (!strategist || getManagedConstructions(state, strategist.id, "bridge").length >= getConstructionManagementLimit(state, strategist.id, "bridge") || !isAvailable(state, strategist.id, "bridge")) return [];
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

export function getManagedConstructions(state: GameState, managerUnitId: string, kind: Construction["kind"]) {
  return state.constructions.filter((entry) => entry.active && entry.managerUnitId === managerUnitId && entry.kind === kind);
}

export function getConstructionManagementLimit(state: GameState, strategistUnitId: string, kind: Construction["kind"]) {
  const strategist = getBuilderUnits(state).find((unit) => unit.id === strategistUnitId);
  if (!strategist) return 0;
  const team = state.teams.find((candidate) => candidate.id === strategist.ownerTeamId);
  const conquests = team?.conqueredTeamIds?.length ?? 0;
  if (conquests >= 2) return 2;
  if (conquests === 1 && team?.constructionCapacityBonusStrategistId === strategist.id) return 2;
  return 1;
}

export function assignConstructionCapacityBonus(state: GameState, teamId: string, strategistUnitId: string): GameState {
  const team = state.teams.find((candidate) => candidate.id === teamId);
  const strategist = getBuilderUnits(state, teamId).find((unit) => unit.id === strategistUnitId);
  if (!team || (team.conqueredTeamIds?.length ?? 0) !== 1 || !strategist) return state;
  return {
    ...state,
    teams: state.teams.map((candidate) =>
      candidate.id === teamId
        ? { ...candidate, constructionCapacityBonusStrategistId: strategistUnitId }
        : candidate,
    ),
  };
}

export function assignConstructionManager(state: GameState, constructionId: string, strategistUnitId: string): GameState {
  const construction = state.constructions.find((entry) => entry.id === constructionId && entry.active && !entry.managerUnitId && entry.ownerTeamId);
  const strategist = getBuilderUnits(state).find((unit) => unit.id === strategistUnitId);
  if (!construction || !strategist || strategist.ownerTeamId !== construction.ownerTeamId) return state;
  if (getManagedConstructions(state, strategist.id, construction.kind).length >= getConstructionManagementLimit(state, strategist.id, construction.kind)) return state;
  return {
    ...state,
    constructions: state.constructions.map((entry) =>
      entry.id === constructionId ? { ...entry, managerUnitId: strategistUnitId } : entry,
    ),
  };
}

export function clearDeadConstructionManagers(state: GameState) {
  const livingBuilders = new Set(getBuilderUnits(state).map((unit) => unit.id));
  state.constructions = state.constructions.map((construction) =>
    construction.managerUnitId && !livingBuilders.has(construction.managerUnitId)
      ? { ...construction, managerUnitId: undefined }
      : construction,
  );
  state.teams = state.teams.map((team) =>
    team.constructionCapacityBonusStrategistId && !livingBuilders.has(team.constructionCapacityBonusStrategistId)
      ? { ...team, constructionCapacityBonusStrategistId: undefined }
      : team,
  );
}
function canSafelyReset(state: GameState, construction: Construction, resettingTeamId: string) {
  if (construction.kind !== "bridge") return true;
  return !state.units.some(
    (unit) =>
      unit.position.kind === "bridge" &&
      unit.position.bridgeId === construction.id &&
      (unit.id === construction.managerUnitId ||
        (unit.type === "king" && unit.ownerTeamId === resettingTeamId)),
  );
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
    (intent.action === "reset_bridge" && getManagedConstructions(state, strategist.id, "bridge").some((entry) => entry.id === intent.constructionId && canSafelyReset(state, entry, intent.teamId))) ||
    (intent.action === "reset_obstacle" && getManagedConstructions(state, strategist.id, "obstacle").some((entry) => entry.id === intent.constructionId));
  if (!legal) return state;
  return { ...state, strategistActionIntents: [...state.strategistActionIntents.filter((entry) => entry.strategistUnitId !== strategist.id), intent] };
}

export function getStrategistActionCandidates(state: GameState, teamId: string): StrategistActionIntent[] {
  if (
    state.phase !== "strategist_action_input" ||
    state.teams.find((team) => team.id === teamId)?.status !== "active" ||
    state.strategistSubmittedTeamIds.includes(teamId)
  ) return [];
  return getBuilderUnits(state, teamId)
    .slice()
    .sort((left, right) => left.id.localeCompare(right.id))
    .flatMap((builder) => getStrategistActionCandidatesForUnit(state, teamId, builder.id));
}

export function getStrategistActionCandidatesForUnit(state: GameState, teamId: string, strategistUnitId: string): StrategistActionIntent[] {
  if (
    state.phase !== "strategist_action_input" ||
    state.teams.find((team) => team.id === teamId)?.status !== "active" ||
    state.strategistSubmittedTeamIds.includes(teamId)
  ) return [];
  const builder = getBuilderUnits(state, teamId).find((unit) => unit.id === strategistUnitId);
  if (!builder) return [];
  const placements: StrategistActionIntent[] = [
    ...getBridgeCandidates(state, builder.id).map((tiles) => ({ teamId, strategistUnitId: builder.id, action: "place_bridge" as const, tiles })),
    ...getObstacleCandidates(state, builder.id).map((cell) => ({ teamId, strategistUnitId: builder.id, action: "place_obstacle" as const, tiles: [cell] })),
  ];
  const resets = (["bridge", "obstacle"] as const).flatMap((kind) =>
    getManagedConstructions(state, builder.id, kind)
      .slice()
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((construction) => ({
        teamId,
        strategistUnitId: builder.id,
        action: (kind === "bridge" ? "reset_bridge" : "reset_obstacle") as StrategistActionIntent["action"],
        constructionId: construction.id,
      })),
  );
  return [...placements, ...resets, { teamId, strategistUnitId: builder.id, action: "pass" as const }]
    .filter((candidate) => saveStrategistActionIntent(state, candidate) !== state);
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

function setCooldown(state: GameState, strategistUnitId: string | undefined, kind: Construction["kind"]) {
  if (!strategistUnitId) return;
  state.strategistCooldowns = [
    ...state.strategistCooldowns.filter(
      (entry) => entry.strategistUnitId !== strategistUnitId || entry.kind !== kind,
    ),
    { strategistUnitId, kind, availableFromTurn: state.turnNumber + 5 },
  ];
}

function randomized<T>(values: T[], rng: () => number) {
  return values
    .map((value) => ({ value, order: rng() }))
    .sort((left, right) => left.order - right.order)
    .map((entry) => entry.value);
}

function recordFloodKill(state: GameState, resettingTeamId: string, victim: Unit) {
  if (victim.ownerTeamId === resettingTeamId) return;
  state.teams = state.teams.map((team) =>
    team.id === resettingTeamId
      ? { ...team, defeatedUnitCount: (team.defeatedUnitCount ?? 0) + 1 }
      : team,
  );
}

function removeFloodVictim(state: GameState, unit: Unit, resettingTeamId: string) {
  recordFloodKill(state, resettingTeamId, unit);
  unit.hp = 0;
  unit.position = { kind: "removed", reason: "water_trap" };
  unit.statuses = [];
}

type FloodSnapshot = {
  unit: Unit;
  origin: BoardCoord;
  resettingTeamId: string;
};

function resolveBridgeFloods(
  state: GameState,
  resets: { construction: Construction; teamId: string }[],
  rng: () => number,
) {
  if (!resets.length) return;
  const resetByBridgeId = new Map(resets.map((entry) => [entry.construction.id, entry]));
  const snapshots = new Map<string, FloodSnapshot>();
  for (const unit of state.units) {
    if (unit.position.kind !== "bridge") continue;
    const reset = resetByBridgeId.get(unit.position.bridgeId);
    const origin = reset?.construction.tiles[unit.position.cellIndex];
    if (reset && origin && !snapshots.has(unit.id))
      snapshots.set(unit.id, { unit, origin: { ...origin }, resettingTeamId: reset.teamId });
  }

  const resetTileKeys = new Set(resets.flatMap((entry) => entry.construction.tiles.map(key)));
  for (const reset of resets) {
    reset.construction.active = false;
    setCooldown(state, reset.construction.managerUnitId, "bridge");
  }
  for (const obstacle of state.constructions.filter(
    (entry) =>
      entry.active &&
      entry.kind === "obstacle" &&
      entry.tiles.some((cell) => resetTileKeys.has(key(cell))),
  )) {
    obstacle.active = false;
    setCooldown(state, obstacle.managerUnitId, "obstacle");
  }

  const survivingKings: FloodSnapshot[] = [];
  const defeatedKingData: { unit: Unit; resettingTeamId: string }[] = [];
  const orderedSnapshots = [...snapshots.values()].sort((left, right) =>
    left.unit.id.localeCompare(right.unit.id),
  );
  for (const snapshot of orderedSnapshots) {
    const { unit, origin, resettingTeamId } = snapshot;
    if (unit.type === "ninja") {
      unit.position = { kind: "water", ...origin };
      continue;
    }
    if (unit.type !== "king") {
      removeFloodVictim(state, unit, resettingTeamId);
      continue;
    }
    unit.hp -= 1;
    if (unit.ownerTeamId !== resettingTeamId)
      recordKingDamage(state, unit.id, unit.ownerTeamId, resettingTeamId, 1);
    if (unit.hp <= 0) {
      unit.position = { kind: "removed", reason: "water_trap" };
      unit.statuses = [];
      recordFloodKill(state, resettingTeamId, unit);
      defeatedKingData.push({ unit, resettingTeamId });
    } else survivingKings.push(snapshot);
  }

  const roadAssigned = new Set<string>();
  const unmatched: typeof survivingKings = [];
  const orderedSurvivingKings = [...survivingKings].sort((left, right) =>
    left.unit.id.localeCompare(right.unit.id),
  );
  for (const snapshot of randomized(orderedSurvivingKings, rng)) {
    const occupied = new Set(
      state.units.flatMap((unit) =>
        unit.position.kind === "tile" ? [tileKey(unit.position.x, unit.position.y)] : [],
      ),
    );
    const activeBridgeTiles = new Set(
      state.constructions
        .filter((entry) => entry.active && entry.kind === "bridge")
        .flatMap((entry) => entry.tiles)
        .map(key),
    );
    const candidates = state.map.tiles
      .filter((tile) => tile.terrain === "road")
      .filter((tile) => !occupied.has(tileKey(tile.x, tile.y)))
      .filter((tile) => !roadAssigned.has(tileKey(tile.x, tile.y)))
      .filter((tile) => !getConstructionAt(state, tile.x, tile.y, "obstacle"))
      .filter((tile) => !activeBridgeTiles.has(tileKey(tile.x, tile.y)))
      .sort((left, right) => left.x - right.x || left.y - right.y);
    if (!candidates.length) {
      unmatched.push(snapshot);
      continue;
    }
    const minimum = Math.min(...candidates.map((tile) => chebyshevDistance(snapshot.origin, tile)));
    const nearest = randomized(
      candidates
        .filter((tile) => chebyshevDistance(snapshot.origin, tile) === minimum)
        .sort((left, right) => left.x - right.x || left.y - right.y),
      rng,
    )[0];
    roadAssigned.add(tileKey(nearest.x, nearest.y));
    snapshot.unit.position = { kind: "tile", x: nearest.x, y: nearest.y };
  }

  const noBase: typeof unmatched = [];
  const orderedUnmatched = [...unmatched].sort((left, right) =>
    left.unit.id.localeCompare(right.unit.id),
  );
  for (const snapshot of randomized(orderedUnmatched, rng)) {
    const team = state.teams.find((candidate) => candidate.id === snapshot.unit.ownerTeamId);
    const bases = state.bases
      .filter(
        (base) => base.ownerTeamId === snapshot.unit.ownerTeamId && base.slots.some((slot) => !slot.unitId),
      )
      .sort((left, right) => left.id.localeCompare(right.id));
    if (!bases.length) {
      noBase.push(snapshot);
      continue;
    }
    const distance = (base: (typeof bases)[number]) =>
      Math.min(...base.coords.map((cell) => chebyshevDistance(snapshot.origin, cell)));
    const minimum = Math.min(...bases.map(distance));
    const nearest = bases
      .filter((base) => distance(base) === minimum)
      .sort((left, right) => left.id.localeCompare(right.id));
    const home = nearest.find((base) => base.id === team?.homeBaseId);
    const base = home ?? randomized(nearest, rng)[0];
    const slot = [...base.slots]
      .sort((left, right) => left.id.localeCompare(right.id))
      .find((candidate) => !candidate.unitId)!;
    slot.unitId = snapshot.unit.id;
    snapshot.unit.position = { kind: "base", baseId: base.id, slotId: slot.id };
  }

  for (const snapshot of noBase) {
    removeFloodVictim(state, snapshot.unit, snapshot.resettingTeamId);
    defeatedKingData.push({ unit: snapshot.unit, resettingTeamId: snapshot.resettingTeamId });
  }

  const defeatedKings: DefeatedKingPlan[] = defeatedKingData.flatMap(({ unit, resettingTeamId }) => {
    const campaign = getKingCampaign(state, unit.id);
    return campaign
      ? [{ kingUnitId: unit.id, kingTeamId: unit.ownerTeamId, candidateTeamIds: [resettingTeamId], campaign }]
      : [];
  });
  if (defeatedKings.length) resolveKingDefeats(state, defeatedKings, [], rng);
  clearDeadConstructionManagers(state);
}

export function resolveStrategistActions(state: GameState, rng: () => number = Math.random) {
  if (state.phase !== "strategist_action_resolution") return state;
  const next = structuredClone(state) as GameState;
  const intents = [...next.strategistActionIntents].sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId));
  const bridgeResets: { construction: Construction; teamId: string }[] = [];
  for (const intent of intents.filter((entry) => entry.action.startsWith("reset_"))) {
    const construction = next.constructions.find((entry) => entry.id === intent.constructionId && entry.managerUnitId === intent.strategistUnitId && entry.active);
    if (!construction || !canSafelyReset(next, construction, intent.teamId)) continue;
    if (construction.kind === "bridge") bridgeResets.push({ construction, teamId: intent.teamId });
    else {
      construction.active = false;
      setCooldown(next, construction.managerUnitId, "obstacle");
    }
    next.logs.push({ id: `log-construction-reset-${next.logs.length}`, turnNumber: next.turnNumber, type: "construction", message: `${intent.teamId} ${intent.strategistUnitId} reset ${construction.kind}; available turn ${next.turnNumber + 5}.`, relatedIds: [construction.id, intent.strategistUnitId] });
  }
  resolveBridgeFloods(next, bridgeResets, rng);
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
  const movementReady = beginMovementPhase(next);
  if (next.rewardPlacementRequests.some((request) => !request.completed && !request.expired)) {
    movementReady.phaseAfterRewards = "movement_input";
    movementReady.phase = movementReady.turnState.phase = "reward_placement";
  }
  return movementReady;
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
