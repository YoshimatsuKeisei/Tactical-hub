import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  assignConstructionCapacityBonus,
  assignConstructionManager,
  beginStrategistActionPhase,
  clearDeadConstructionManagers,
  getBridgeCandidates,
  getConstructionManagementLimit,
  getObstacleCandidates,
  getOperationalRoadTiles,
  isAvailable,
  resolveStrategistActions,
  saveStrategistActionIntent,
  submitStrategistActions,
} from "../engine/construction";
import { resolveBattle, saveAttackIntent } from "../engine/battle";
import { resolveKingDefeats } from "../engine/defeat";
import { getMovementCandidates } from "../engine/movement";
import { resolveProduction, saveProductionChoice } from "../engine/production";
import { createInitialGameState } from "../initialState";
import { areRoadSectionsDynamicallyConnected } from "../utils/roadTopology";
import type { BoardCoord, Construction, GameState, KingCampaignState, Unit, UnitPosition } from "../types";

const RESET_TEAM = "team-1";
const MANAGER_ID = "home-1-strategist";
const BRIDGE_ID = "phase4b-bridge";

function makeState() {
  const state = createInitialGameState();
  state.units.find((unit) => unit.id === MANAGER_ID)!.role = "builder";
  return state;
}

function clearPreviousSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((candidate) => candidate.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function relocate(state: GameState, unit: Unit, position: UnitPosition) {
  clearPreviousSlot(state, unit.position);
  unit.position = position;
  if (position.kind === "base") {
    const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((candidate) => candidate.id === position.slotId);
    if (slot) slot.unitId = unit.id;
  }
}

function addUnit(state: GameState, id: string, teamId: string, type: Unit["type"], position: UnitPosition, hp = UNIT_STATS[type].hp) {
  const unit: Unit = { id, ownerTeamId: teamId, type, hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function addBridge(state: GameState, input: Partial<Construction> = {}) {
  const construction: Construction = {
    id: input.id ?? BRIDGE_ID,
    kind: "bridge",
    ownerTeamId: input.ownerTeamId ?? RESET_TEAM,
    managerUnitId: input.managerUnitId === undefined ? MANAGER_ID : input.managerUnitId,
    tiles: input.tiles ?? [{ x: 4, y: 2 }, { x: 4, y: 3 }],
    placedTurn: input.placedTurn ?? 1,
    active: input.active ?? true,
  };
  state.constructions.push(construction);
  return construction;
}

function bridgePosition(bridgeId = BRIDGE_ID, cellIndex = 0): UnitPosition {
  return { kind: "bridge", bridgeId, cellIndex };
}

function saveReset(state: GameState, bridgeId = BRIDGE_ID, managerId = MANAGER_ID) {
  return saveStrategistActionIntent(state, {
    teamId: RESET_TEAM,
    strategistUnitId: managerId,
    action: "reset_bridge",
    constructionId: bridgeId,
  });
}

function resolveSavedReset(state: GameState, rng: () => number = () => 0) {
  const saved = saveReset(state);
  saved.phase = saved.turnState.phase = "strategist_action_resolution";
  return resolveStrategistActions(saved, rng);
}

function blockRoads(state: GameState, teamId: string) {
  for (const tile of state.map.tiles.filter((candidate) => candidate.terrain === "road")) {
    state.constructions.push({
      id: `road-block-${teamId}-${tile.x}-${tile.y}`,
      kind: "obstacle",
      ownerTeamId: RESET_TEAM,
      tiles: [{ x: tile.x, y: tile.y }],
      placedTurn: 1,
      active: true,
    });
  }
}

function fillOwnedBaseSlots(state: GameState, teamId: string) {
  for (const base of state.bases.filter((candidate) => candidate.ownerTeamId === teamId)) {
    for (const slot of base.slots.filter((candidate) => !candidate.unitId)) {
      const unit = addUnit(state, `slot-fill-${base.id}-${slot.id}`, teamId, "infantry", { kind: "base", baseId: base.id, slotId: slot.id });
      slot.unitId = unit.id;
    }
  }
}

function sequenceRng(values: number[]) {
  let index = 0;
  return () => values[Math.min(index++, values.length - 1)] ?? 0;
}

function submitAllActiveTeams(state: GameState) {
  let next = state;
  for (const team of state.teams.filter((candidate) => candidate.status === "active"))
    next = submitStrategistActions(next, team.id);
  return next;
}

function resultSummary(state: GameState) {
  return {
    constructions: state.constructions
      .map((entry) => ({ id: entry.id, active: entry.active, ownerTeamId: entry.ownerTeamId, managerUnitId: entry.managerUnitId }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    units: state.units
      .map((unit) => ({ id: unit.id, hp: unit.hp, position: unit.position }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    defeatedUnitCount: state.teams
      .map((team) => ({ id: team.id, count: team.defeatedUnitCount ?? 0 }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    cooldowns: [...state.strategistCooldowns].sort(
      (left, right) =>
        left.strategistUnitId.localeCompare(right.strategistUnitId) ||
        left.kind.localeCompare(right.kind),
    ),
    roadConnected: areRoadSectionsDynamicallyConnected(
      state,
      "road-home-1-neutral-north",
      "road-home-1-neutral-center",
    ),
  };
}

function conquestCampaign(kingTeamId = "team-2", conqueror = RESET_TEAM): KingCampaignState {
  return {
    kingUnitId: `home-${kingTeamId.slice(-1)}-king`,
    kingTeamId,
    contributions: [{ teamId: conqueror, cumulativeDamage: 1, effectiveAttackTurns: 1 }],
  };
}

describe("Phase 4-B black-box decision tables", () => {
  describe("A. manual bridge reset eligibility", () => {
    it.each([
      ["nobody", undefined, undefined, true],
      ["managing builder", MANAGER_ID, RESET_TEAM, false],
      ["resetting team's king", "home-1-king", RESET_TEAM, false],
      ["friendly infantry", "friendly-infantry", RESET_TEAM, true],
      ["enemy infantry", "enemy-infantry", "team-2", true],
      ["enemy king", "home-2-king", "team-2", true],
      ["another friendly strategist", "other-friendly-strategist", RESET_TEAM, true],
      ["enemy strategist", "enemy-strategist", "team-2", true],
    ])("occupant=%s => reset accepted=%s", (_, occupantId, teamId, expected) => {
      const state = makeState();
      addBridge(state);
      if (occupantId) {
        let occupant = state.units.find((unit) => unit.id === occupantId);
        if (!occupant) occupant = addUnit(state, occupantId, teamId!, occupantId.includes("strategist") ? "strategist" : "infantry", bridgePosition());
        else relocate(state, occupant, bridgePosition());
      }
      expect(saveReset(state).strategistActionIntents.some((intent) => intent.action === "reset_bridge")).toBe(expected);
    });

    it("rejects manual reset for an unmanaged bridge", () => {
      const state = makeState();
      const construction = addBridge(state);
      construction.managerUnitId = undefined;
      expect(saveReset(state).strategistActionIntents).toEqual([]);
    });
  });

  describe("B. water-tactic occupant outcomes", () => {
    it.each([
      ["friendly ninja", RESET_TEAM, "ninja", "water", 0],
      ["enemy ninja", "team-2", "ninja", "water", 0],
      ["friendly regular", RESET_TEAM, "cavalry", "removed", 0],
      ["enemy regular", "team-2", "cavalry", "removed", 1],
    ] as const)("%s", (_, teamId, type, positionKind, killDelta) => {
      const state = makeState(); addBridge(state);
      const before = state.teams.find((team) => team.id === RESET_TEAM)!.defeatedUnitCount ?? 0;
      const victim = addUnit(state, `victim-${teamId}-${type}`, teamId, type, bridgePosition());
      const resolved = resolveSavedReset(state);
      const result = resolved.units.find((unit) => unit.id === victim.id)!;
      expect(result.position.kind).toBe(positionKind);
      expect(result.hp).toBe(type === "ninja" ? UNIT_STATS.ninja.hp : 0);
      expect(resolved.teams.find((team) => team.id === RESET_TEAM)!.defeatedUnitCount).toBe(before + killDelta);
    });

    it.each([
      [1, 0, "removed"],
      [2, 1, "tile"],
      [3, 2, "tile"],
    ] as const)("enemy king HP%s -> HP%s and %s", (initialHp, expectedHp, expectedPosition) => {
      const state = makeState(); addBridge(state);
      const king = state.units.find((unit) => unit.id === "home-2-king")!;
      king.hp = initialHp;
      relocate(state, king, bridgePosition());
      const resolved = resolveSavedReset(state);
      const result = resolved.units.find((unit) => unit.id === king.id)!;
      expect(result.hp).toBe(expectedHp);
      expect(result.position.kind).toBe(expectedPosition);
      const contribution = resolved.kingCampaignStates.find((campaign) => campaign.kingUnitId === king.id)?.contributions.find((entry) => entry.teamId === RESET_TEAM);
      if (initialHp === 1) {
        expect(resolved.logs.some((log) => log.id.startsWith("log-king-damage-") && log.relatedIds?.includes(king.id))).toBe(true);
        expect(resolved.logs.some((log) => log.id.startsWith("log-king-attack-") && log.relatedIds?.includes(king.id))).toBe(false);
        expect(resolved.teams.find((team) => team.id === "team-2")?.status).toBe("defeated");
      } else expect(contribution).toMatchObject({ cumulativeDamage: 1, effectiveAttackTurns: 0 });
    });
  });

  describe("D. ownership and management", () => {
    it.each(["bridge", "obstacle"] as const)("a dead manager leaves an active unmanaged %s with the same owner", (kind) => {
      const state = makeState();
      state.constructions.push({ id: `managed-${kind}`, kind, ownerTeamId: RESET_TEAM, managerUnitId: MANAGER_ID, tiles: [{ x: 4, y: kind === "bridge" ? 2 : 1 }], placedTurn: 1, active: true });
      const manager = state.units.find((unit) => unit.id === MANAGER_ID)!;
      manager.hp = 0; manager.position = { kind: "removed", reason: "defeated" };
      clearDeadConstructionManagers(state);
      expect(state.constructions[0]).toMatchObject({ kind, ownerTeamId: RESET_TEAM, active: true, managerUnitId: undefined });
    });

    it("keeps unmanaged bridge and obstacle board effects while preventing operation", () => {
      const state = makeState();
      addBridge(state).managerUnitId = undefined;
      state.constructions.push({ id: "unmanaged-obstacle", kind: "obstacle", ownerTeamId: RESET_TEAM, tiles: [{ x: 4, y: 1 }], placedTurn: 1, active: true });
      const mover = addUnit(state, "effect-mover", RESET_TEAM, "cavalry", { kind: "tile", x: 3, y: 1 });
      expect(getMovementCandidates(state, mover.id)).not.toContainEqual({ kind: "tile", x: 4, y: 1 });
      expect(getMovementCandidates(state, mover.id)).toContainEqual({ kind: "bridge", bridgeId: BRIDGE_ID, cellIndex: 0 });
      expect(saveReset(state).strategistActionIntents).toEqual([]);
      expect(assignConstructionManager(state, BRIDGE_ID, "home-2-strategist")).toBe(state);
    });
  });
});

describe("Phase 4-B simultaneous bridge reset ordering", () => {
  function makeTwoBridgeResolutionState() {
    let state = makeState();
    state.phase = state.turnState.phase = "strategist_action_input";

    const secondManagerId = "team-1-second-builder";
    addUnit(state, secondManagerId, RESET_TEAM, "strategist", { kind: "tile", x: 9, y: 1 }).role = "builder";
    addBridge(state, {
      id: "public-bridge-a",
      managerUnitId: MANAGER_ID,
      tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }],
    });
    addBridge(state, {
      id: "public-bridge-b",
      managerUnitId: secondManagerId,
      tiles: [{ x: 7, y: 2 }, { x: 7, y: 3 }, { x: 7, y: 4 }, { x: 7, y: 5 }, { x: 7, y: 6 }],
    });

    addUnit(state, "public-victim-a", "team-2", "infantry", bridgePosition("public-bridge-a"));
    addUnit(state, "public-victim-b", "team-3", "infantry", bridgePosition("public-bridge-b"));
    state.units.find((unit) => unit.id === "home-2-strategist")!.role = "builder";
    state.units.find((unit) => unit.id === "home-3-strategist")!.role = "builder";
    state.constructions.push(
      {
        id: "public-obstacle-a",
        kind: "obstacle",
        ownerTeamId: "team-2",
        managerUnitId: "home-2-strategist",
        tiles: [{ x: 4, y: 2 }],
        placedTurn: 1,
        active: true,
      },
      {
        id: "public-obstacle-b",
        kind: "obstacle",
        ownerTeamId: "team-3",
        managerUnitId: "home-3-strategist",
        tiles: [{ x: 7, y: 3 }],
        placedTurn: 1,
        active: true,
      },
    );

    state = saveStrategistActionIntent(state, {
      teamId: RESET_TEAM,
      strategistUnitId: MANAGER_ID,
      action: "reset_bridge",
      constructionId: "public-bridge-a",
    });
    state = saveStrategistActionIntent(state, {
      teamId: RESET_TEAM,
      strategistUnitId: secondManagerId,
      action: "reset_bridge",
      constructionId: "public-bridge-b",
    });
    return { state: submitAllActiveTeams(state), secondManagerId };
  }

  it("resets two distinct bridges in one public strategist-action resolution", () => {
    const fixture = makeTwoBridgeResolutionState();
    expect(fixture.state.phase).toBe("strategist_action_resolution");
    expect(fixture.state.strategistActionIntents.filter((intent) => intent.action === "reset_bridge")).toHaveLength(2);
    expect(areRoadSectionsDynamicallyConnected(
      fixture.state,
      "road-home-1-neutral-north",
      "road-home-1-neutral-center",
    )).toBe(true);
    const beforeKills = fixture.state.teams.find((team) => team.id === RESET_TEAM)!.defeatedUnitCount ?? 0;

    const resolved = resolveStrategistActions(fixture.state, sequenceRng([0.8, 0.2, 0.6, 0.4]));
    for (const id of ["public-bridge-a", "public-bridge-b", "public-obstacle-a", "public-obstacle-b"])
      expect(resolved.constructions.find((entry) => entry.id === id)?.active).toBe(false);
    for (const id of ["public-victim-a", "public-victim-b"])
      expect(resolved.units.find((unit) => unit.id === id)).toMatchObject({ hp: 0, position: { kind: "removed" } });
    expect(resolved.teams.find((team) => team.id === RESET_TEAM)?.defeatedUnitCount).toBe(beforeKills + 2);
    expect(resolved.strategistCooldowns).toEqual(expect.arrayContaining([
      expect.objectContaining({ strategistUnitId: MANAGER_ID, kind: "bridge", availableFromTurn: resolved.turnNumber + 5 }),
      expect.objectContaining({ strategistUnitId: fixture.secondManagerId, kind: "bridge", availableFromTurn: resolved.turnNumber + 5 }),
      expect.objectContaining({ strategistUnitId: "home-2-strategist", kind: "obstacle", availableFromTurn: resolved.turnNumber + 5 }),
      expect.objectContaining({ strategistUnitId: "home-3-strategist", kind: "obstacle", availableFromTurn: resolved.turnNumber + 5 }),
    ]));
    expect(areRoadSectionsDynamicallyConnected(
      resolved,
      "road-home-1-neutral-north",
      "road-home-1-neutral-center",
    )).toBe(false);
  });

  it("keeps reset and competing-king retreat results independent of intent, construction, and unit array order", () => {
    const fixture = makeTwoBridgeResolutionState();
    let state = fixture.state;
    const firstKing = state.units.find((unit) => unit.id === "home-2-king")!;
    firstKing.hp = 2;
    relocate(state, firstKing, bridgePosition("public-bridge-a"));
    const secondKing = addUnit(
      state,
      "team-2-secondary-king",
      "team-2",
      "king",
      bridgePosition("public-bridge-b"),
      2,
    );
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push("home-1");
    blockRoads(state, "team-2");
    state.constructions.find((entry) => entry.id === "road-block-team-2-4-1")!.active = false;

    const canonical = structuredClone(state);
    const reordered = structuredClone(state);
    reordered.strategistActionIntents.reverse();
    reordered.constructions.reverse();
    reordered.units.reverse();
    const rngValues = [0.75, 0.25, 0.6, 0.4, 0.9, 0.1, 0.8, 0.2];
    const canonicalResult = resolveStrategistActions(canonical, sequenceRng(rngValues));
    const reorderedResult = resolveStrategistActions(reordered, sequenceRng(rngValues));

    expect(resultSummary(reorderedResult)).toEqual(resultSummary(canonicalResult));
    const kingPositions = [firstKing.id, secondKing.id].map(
      (id) => canonicalResult.units.find((unit) => unit.id === id)!.position.kind,
    );
    expect(kingPositions.sort()).toEqual(["base", "tile"]);
  });
});

describe("Phase 4-B boundary values", () => {
  it("enforces bridge and forced-obstacle cooldown at T+4 and releases both at T+5", () => {
    const state = makeState(); state.turnNumber = 10; addBridge(state);
    const obstacleManager = state.units.find((unit) => unit.id === "home-2-strategist")!; obstacleManager.role = "builder";
    state.constructions.push({ id: "forced-obstacle", kind: "obstacle", ownerTeamId: "team-2", managerUnitId: obstacleManager.id, tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    const resolved = resolveSavedReset(state);
    for (const [turn, expected] of [[14, false], [15, true]] as const) {
      resolved.turnNumber = turn;
      expect(isAvailable(resolved, MANAGER_ID, "bridge")).toBe(expected);
      expect(isAvailable(resolved, obstacleManager.id, "obstacle")).toBe(expected);
      expect(getBridgeCandidates(resolved, MANAGER_ID).length > 0).toBe(expected);
      expect(getObstacleCandidates(resolved, obstacleManager.id).length > 0).toBe(expected);
    }
  });

  it.each([
    [0, undefined, 1, 1],
    [1, "second-builder", 1, 2],
    [2, undefined, 2, 2],
    [3, undefined, 2, 2],
  ] as const)("conquests=%s caps first=%s second=%s", (count, bonusId, firstLimit, secondLimit) => {
    let state = makeState();
    const second = addUnit(state, "second-builder", RESET_TEAM, "strategist", { kind: "tile", x: 5, y: 1 }); second.role = "builder";
    state.teams.find((team) => team.id === RESET_TEAM)!.conqueredTeamIds = ["team-2", "team-3", "team-4"].slice(0, count);
    if (bonusId) state = assignConstructionCapacityBonus(state, RESET_TEAM, bonusId);
    for (const kind of ["bridge", "obstacle"] as const) {
      expect(getConstructionManagementLimit(state, MANAGER_ID, kind)).toBe(firstLimit);
      expect(getConstructionManagementLimit(state, second.id, kind)).toBe(secondLimit);
    }
  });

  it.each([
    [1, 0, 0],
    [1, 1, 1],
    [2, 1, 0],
    [2, 2, 2],
  ] as const)("base retreat targets=%s empty slots=%s places=%s without overlap", (targetCount, emptySlots, expectedPlaced) => {
    const state = makeState(); addBridge(state, { tiles: [{ x: 15, y: 3 }, { x: 15, y: 4 }] });
    blockRoads(state, "team-2");
    const original = state.units.find((unit) => unit.id === "home-2-king")!; original.hp = 2; relocate(state, original, bridgePosition(BRIDGE_ID, 0));
    const kings = [original];
    if (targetCount === 2) kings.push(addUnit(state, "second-team-2-king", "team-2", "king", bridgePosition(BRIDGE_ID, 1), 2));
    const home = state.bases.find((base) => base.id === "home-2")!;
    const currentlyEmpty = home.slots.filter((slot) => !slot.unitId);
    for (const slot of currentlyEmpty.slice(emptySlots)) {
      const filler = addUnit(state, `base-capacity-fill-${slot.id}`, "team-2", "infantry", { kind: "base", baseId: home.id, slotId: slot.id });
      slot.unitId = filler.id;
    }
    const resolved = resolveSavedReset(state);
    const basePositions = kings.map((king) => resolved.units.find((unit) => unit.id === king.id)!.position).filter((position): position is Extract<UnitPosition, { kind: "base" }> => position.kind === "base");
    expect(basePositions).toHaveLength(expectedPlaced);
    expect(new Set(basePositions.map((position) => `${position.baseId}:${position.slotId}`)).size).toBe(basePositions.length);
    if (targetCount === 2 && emptySlots === 1)
      expect(resolved.teams.find((team) => team.id === "team-2")?.status).toBe("defeated");
  });

  it.each([
    [1, 0, 1],
    [1, 1, 0],
    [2, 1, 1],
    [2, 2, 0],
  ] as const)("capacity=%s managed=%s leaves %s assignable slots", (capacity, managedCount, remaining) => {
    let state = makeState();
    if (capacity === 2) {
      state.teams.find((team) => team.id === RESET_TEAM)!.conqueredTeamIds = ["team-2", "team-3"];
    }
    for (let index = 0; index < managedCount; index += 1)
      state.constructions.push({ id: `managed-${index}`, kind: "bridge", ownerTeamId: RESET_TEAM, managerUnitId: MANAGER_ID, tiles: [{ x: 4 + index, y: 2 }], placedTurn: 1, active: true });
    expect(getConstructionManagementLimit(state, MANAGER_ID, "bridge") - state.constructions.filter((entry) => entry.managerUnitId === MANAGER_ID && entry.kind === "bridge").length).toBe(remaining);
    expect(getBridgeCandidates(state, MANAGER_ID).length > 0).toBe(remaining > 0);
  });

  it("does not delete inherited equipment above capacity and leaves overflow unmanaged", () => {
    const state = makeState();
    state.constructions.push(
      { id: "orphan-1", kind: "bridge", ownerTeamId: RESET_TEAM, tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true },
      { id: "orphan-2", kind: "bridge", ownerTeamId: RESET_TEAM, tiles: [{ x: 5, y: 2 }], placedTurn: 1, active: true },
    );
    const first = assignConstructionManager(state, "orphan-1", MANAGER_ID);
    const overflow = assignConstructionManager(first, "orphan-2", MANAGER_ID);
    expect(overflow.constructions).toHaveLength(2);
    expect(overflow.constructions.find((entry) => entry.id === "orphan-1")?.managerUnitId).toBe(MANAGER_ID);
    expect(overflow.constructions.find((entry) => entry.id === "orphan-2")?.managerUnitId).toBeUndefined();
  });

  it.each([
    [1, 0, 0],
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 2],
  ] as const)("retreat targets=%s road capacity=%s places=%s without overlap", (targetCount, roadCapacity, expectedPlaced) => {
    const state = makeState();
    addBridge(state, { tiles: [{ x: 15, y: 3 }, { x: 15, y: 4 }] });
    blockRoads(state, "retreat-capacity");
    const availableRoads: BoardCoord[] = [];
    for (const obstacle of state.constructions.filter((entry) => entry.id.startsWith("road-block-retreat-capacity-")).slice(0, roadCapacity)) {
      obstacle.active = false;
      availableRoads.push(obstacle.tiles[0]);
    }
    const kings = ["home-2-king", "home-3-king"].slice(0, targetCount).map((id, index) => {
      const king = state.units.find((unit) => unit.id === id)!;
      relocate(state, king, bridgePosition(BRIDGE_ID, index));
      return king;
    });
    if (roadCapacity === 0) {
      for (const king of kings) fillOwnedBaseSlots(state, king.ownerTeamId);
    }
    const resolved = resolveSavedReset(state);
    const roadPositions = kings.map((king) => resolved.units.find((unit) => unit.id === king.id)!.position).filter((position): position is Extract<UnitPosition, { kind: "tile" }> => position.kind === "tile");
    expect(roadPositions).toHaveLength(expectedPlaced);
    expect(new Set(roadPositions.map((position) => `${position.x},${position.y}`)).size).toBe(roadPositions.length);
  });
});

describe("Phase 4-B state transitions and scenarios", () => {
  it("transitions managed -> orphaned -> produced successor -> explicit assignment -> resettable", () => {
    let state = makeState(); addBridge(state);
    state.constructions.push({ id: "managed-obstacle", kind: "obstacle", ownerTeamId: RESET_TEAM, managerUnitId: MANAGER_ID, tiles: [{ x: 5, y: 1 }], placedTurn: 1, active: true });
    relocate(state, state.units.find((unit) => unit.id === MANAGER_ID)!, { kind: "tile", x: 4, y: 1 });
    const killer = addUnit(state, "builder-killer", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    state = resolveBattle(saveAttackIntent(state, { teamId: "team-2", attackerUnitId: killer.id, target: { kind: "unit", unitId: MANAGER_ID }, pass: false }), () => 0);
    expect(state.constructions.every((entry) => entry.ownerTeamId === RESET_TEAM && !entry.managerUnitId && entry.active)).toBe(true);

    state = saveProductionChoice(state, { teamId: RESET_TEAM, baseId: "home-1", unitType: "strategist" });
    state = resolveProduction(state);
    const successor = state.units.find((unit) => unit.ownerTeamId === RESET_TEAM && unit.type === "strategist" && unit.position.kind !== "removed")!;
    successor.role = "builder";
    expect(state.constructions[0].managerUnitId).toBeUndefined();
    expect(saveStrategistActionIntent(state, { teamId: RESET_TEAM, strategistUnitId: successor.id, action: "reset_bridge", constructionId: BRIDGE_ID }).strategistActionIntents).toEqual([]);

    state = assignConstructionManager(state, BRIDGE_ID, successor.id);
    state = assignConstructionManager(state, "managed-obstacle", successor.id);
    expect(state.constructions.every((entry) => entry.managerUnitId === successor.id)).toBe(true);
    expect(saveStrategistActionIntent(state, { teamId: RESET_TEAM, strategistUnitId: successor.id, action: "reset_bridge", constructionId: BRIDGE_ID }).strategistActionIntents).toHaveLength(1);
    expect(saveStrategistActionIntent(state, { teamId: RESET_TEAM, strategistUnitId: successor.id, action: "reset_obstacle", constructionId: "managed-obstacle" }).strategistActionIntents).toHaveLength(1);

    const obstacleReset = saveStrategistActionIntent(state, { teamId: RESET_TEAM, strategistUnitId: successor.id, action: "reset_obstacle", constructionId: "managed-obstacle" });
    obstacleReset.phase = obstacleReset.turnState.phase = "strategist_action_resolution";
    const operated = resolveStrategistActions(obstacleReset, () => 0);
    expect(operated.constructions.find((entry) => entry.id === "managed-obstacle")?.active).toBe(false);
    expect(operated.strategistCooldowns).toContainEqual({ strategistUnitId: successor.id, kind: "obstacle", availableFromTurn: state.turnNumber + 5 });
  });

  it("revalidates a saved bridge reset and cancels it if the manager becomes a forbidden occupant", () => {
    const state = makeState(); addBridge(state);
    const saved = saveReset(state);
    relocate(saved, saved.units.find((unit) => unit.id === MANAGER_ID)!, bridgePosition());
    saved.phase = saved.turnState.phase = "strategist_action_resolution";
    const resolved = resolveStrategistActions(saved, () => 0);
    expect(resolved.constructions.find((entry) => entry.id === BRIDGE_ID)?.active).toBe(true);
    expect(resolved.strategistCooldowns).toEqual([]);
  });

  it("releases a conquest bonus when its builder dies without auto-assigning equipment", () => {
    let state = makeState();
    state.teams.find((team) => team.id === RESET_TEAM)!.conqueredTeamIds = ["team-2"];
    state = assignConstructionCapacityBonus(state, RESET_TEAM, MANAGER_ID);
    state.constructions.push({ id: "bonus-orphan", kind: "bridge", ownerTeamId: RESET_TEAM, managerUnitId: MANAGER_ID, tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    const manager = state.units.find((unit) => unit.id === MANAGER_ID)!; manager.hp = 0; manager.position = { kind: "removed", reason: "defeated" };
    const successor = addUnit(state, "bonus-successor", RESET_TEAM, "strategist", { kind: "tile", x: 5, y: 1 }); successor.role = "builder";
    clearDeadConstructionManagers(state);
    expect(state.teams.find((team) => team.id === RESET_TEAM)?.constructionCapacityBonusStrategistId).toBeUndefined();
    expect(state.constructions.find((entry) => entry.id === "bonus-orphan")?.managerUnitId).toBeUndefined();
    expect(getConstructionManagementLimit(state, successor.id, "bridge")).toBe(1);
  });

  it("announces a reset slot as available when the T+5 strategist phase begins", () => {
    const state = makeState(); state.turnNumber = 15;
    state.strategistCooldowns.push({ strategistUnitId: MANAGER_ID, kind: "bridge", availableFromTurn: 15 });
    const begun = beginStrategistActionPhase(state);
    expect(begun.logs.some((log) => log.id.startsWith("log-cooldown-ready-") && log.relatedIds?.includes(MANAGER_ID))).toBe(true);
  });

  it("scenario 1: resolves mixed occupants, foreign obstacle, cooldowns, and removes bridge connectivity", () => {
    const state = makeState(); state.turnNumber = 20;
    addBridge(state, { tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }, { x: 4, y: 4 }] });
    const obstacleManager = state.units.find((unit) => unit.id === "home-3-strategist")!; obstacleManager.role = "builder";
    state.constructions.push({ id: "foreign-obstacle", kind: "obstacle", ownerTeamId: "team-3", managerUnitId: obstacleManager.id, tiles: [{ x: 4, y: 3 }], placedTurn: 1, active: true });
    const friendly = addUnit(state, "mixed-friendly", RESET_TEAM, "infantry", bridgePosition(BRIDGE_ID, 0));
    const enemy = addUnit(state, "mixed-enemy", "team-2", "cavalry", bridgePosition(BRIDGE_ID, 1));
    const ninja = addUnit(state, "mixed-ninja", "team-3", "ninja", bridgePosition(BRIDGE_ID, 2));
    const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition(BRIDGE_ID, 2));
    const beforeKills = state.teams.find((team) => team.id === RESET_TEAM)!.defeatedUnitCount ?? 0;
    expect(areRoadSectionsDynamicallyConnected(state, "road-home-1-neutral-north", "road-home-1-neutral-center")).toBe(true);
    const resolved = resolveSavedReset(state);
    expect(resolved.units.find((unit) => unit.id === friendly.id)?.position.kind).toBe("removed");
    expect(resolved.units.find((unit) => unit.id === enemy.id)?.position.kind).toBe("removed");
    expect(resolved.units.find((unit) => unit.id === ninja.id)?.position).toEqual({ kind: "water", x: 4, y: 4 });
    expect(resolved.units.find((unit) => unit.id === king.id)).toMatchObject({ hp: 1 });
    expect(resolved.teams.find((team) => team.id === RESET_TEAM)!.defeatedUnitCount).toBe(beforeKills + 1);
    expect(resolved.constructions.find((entry) => entry.id === BRIDGE_ID)?.active).toBe(false);
    expect(resolved.constructions.find((entry) => entry.id === "foreign-obstacle")?.active).toBe(false);
    expect(areRoadSectionsDynamicallyConnected(resolved, "road-home-1-neutral-north", "road-home-1-neutral-center")).toBe(false);
    expect(resolved.strategistCooldowns).toEqual(expect.arrayContaining([
      { strategistUnitId: MANAGER_ID, kind: "bridge", availableFromTurn: 25 },
      { strategistUnitId: obstacleManager.id, kind: "obstacle", availableFromTurn: 25 },
    ]));
  });

  it("force-removes an unmanaged bridge obstacle without creating an ownerless cooldown", () => {
    const state = makeState(); state.turnNumber = 9; addBridge(state);
    state.constructions.push({ id: "unmanaged-bridge-obstacle", kind: "obstacle", ownerTeamId: "team-2", tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    const resolved = resolveSavedReset(state);
    expect(resolved.constructions.find((entry) => entry.id === "unmanaged-bridge-obstacle")?.active).toBe(false);
    expect(resolved.strategistCooldowns).not.toContainEqual(expect.objectContaining({ kind: "obstacle" }));
  });

  it("scenario 3: conquest transfers equipment, raises count, and permits explicit assignment", () => {
    let state = makeState();
    state.teams.find((team) => team.id === "team-2")!.constructionCapacityBonusStrategistId = "home-2-strategist";
    state.constructions.push({ id: "team-2-bridge", kind: "bridge", ownerTeamId: "team-2", managerUnitId: "home-2-strategist", tiles: [{ x: 15, y: 2 }], placedTurn: 1, active: true });
    const campaign = conquestCampaign();
    resolveKingDefeats(state, [{ kingUnitId: campaign.kingUnitId, kingTeamId: "team-2", candidateTeamIds: [RESET_TEAM], campaign }], [], () => 0);
    expect(state.constructions[0]).toMatchObject({ ownerTeamId: RESET_TEAM, managerUnitId: undefined, active: true });
    expect(state.teams.find((team) => team.id === RESET_TEAM)?.conqueredTeamIds).toEqual(["team-2"]);
    expect(state.teams.find((team) => team.id === "team-2")?.constructionCapacityBonusStrategistId).toBeUndefined();
    state = assignConstructionCapacityBonus(state, RESET_TEAM, MANAGER_ID);
    expect(getConstructionManagementLimit(state, MANAGER_ID, "bridge")).toBe(2);
    state = assignConstructionManager(state, "team-2-bridge", MANAGER_ID);
    expect(state.constructions[0].managerUnitId).toBe(MANAGER_ID);
  });

  it("scenario 4: simultaneous king defeat neutralizes bases/equipment without capacity gain or assignment", () => {
    const state = makeState();
    state.constructions.push(
      { id: "team-2-remnant", kind: "bridge", ownerTeamId: "team-2", managerUnitId: "home-2-strategist", tiles: [{ x: 15, y: 2 }], placedTurn: 1, active: true },
      { id: "team-3-remnant", kind: "obstacle", ownerTeamId: "team-3", managerUnitId: "home-3-strategist", tiles: [{ x: 16, y: 16 }], placedTurn: 1, active: true },
    );
    const campaigns = [conquestCampaign("team-2"), conquestCampaign("team-3")];
    resolveKingDefeats(state, campaigns.map((campaign) => ({ kingUnitId: campaign.kingUnitId, kingTeamId: campaign.kingTeamId, candidateTeamIds: [RESET_TEAM], campaign })), [], () => 0);
    expect(state.bases.filter((base) => ["home-2", "home-3"].includes(base.id)).every((base) => base.ownerTeamId === "neutral")).toBe(true);
    expect(state.constructions.every((entry) => entry.active && !entry.ownerTeamId && !entry.managerUnitId)).toBe(true);
    expect(state.teams.find((team) => team.id === RESET_TEAM)?.conqueredTeamIds ?? []).toEqual([]);
    expect(getConstructionManagementLimit(state, MANAGER_ID, "bridge")).toBe(1);
    expect(assignConstructionManager(state, "team-2-remnant", MANAGER_ID)).toBe(state);
  });

  it("scenario 5: retreat selection degrades from road to home-base to death", () => {
    const makeKingState = () => {
      const state = makeState(); addBridge(state, { tiles: [{ x: 15, y: 3 }] });
      const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());
      const relay = state.bases.find((base) => base.id === "neutral-north")!; relay.ownerTeamId = "team-2";
      state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push(relay.id);
      return { state, kingId: king.id };
    };

    const roadCase = makeKingState();
    expect(resolveSavedReset(roadCase.state).units.find((unit) => unit.id === roadCase.kingId)?.position.kind).toBe("tile");

    const baseCase = makeKingState(); blockRoads(baseCase.state, "team-2");
    expect(resolveSavedReset(baseCase.state).units.find((unit) => unit.id === baseCase.kingId)?.position).toMatchObject({ kind: "base", baseId: "home-2" });

    const deathCase = makeKingState(); blockRoads(deathCase.state, "team-2"); fillOwnedBaseSlots(deathCase.state, "team-2");
    expect(resolveSavedReset(deathCase.state).units.find((unit) => unit.id === deathCase.kingId)?.position.kind).toBe("removed");
  });

  it("chooses the nearest owned non-home base when roads are unavailable", () => {
    const state = makeState(); addBridge(state, { tiles: [{ x: 11, y: 3 }] });
    const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());
    const relay = state.bases.find((base) => base.id === "neutral-north")!; relay.ownerTeamId = "team-2";
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push(relay.id);
    blockRoads(state, "team-2");
    expect(resolveSavedReset(state).units.find((unit) => unit.id === king.id)?.position).toMatchObject({ kind: "base", baseId: relay.id });
  });

  it("uses injected RNG between equidistant non-home bases", () => {
    const run = (rng: () => number) => {
      const state = makeState(); addBridge(state, { tiles: [{ x: 11, y: 6 }] });
      const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());
      for (const baseId of ["neutral-north", "neutral-center"]) {
        const base = state.bases.find((candidate) => candidate.id === baseId)!; base.ownerTeamId = "team-2";
        state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push(baseId);
      }
      blockRoads(state, "team-2");
      return resolveSavedReset(state, rng).units.find((unit) => unit.id === king.id)!.position;
    };
    const first = run(sequenceRng([0, 0, 0.1, 0.9]));
    const second = run(sequenceRng([0, 0, 0.9, 0.1]));
    expect(first).not.toEqual(second);
    expect([first, second].every((position) => position.kind === "base" && ["neutral-north", "neutral-center"].includes(position.baseId))).toBe(true);
  });

  it("uses injected RNG to choose between equidistant roads", () => {
    const findFixture = () => {
      const state = makeState();
      const roads = state.map.tiles.filter((candidate) => candidate.terrain === "road");
      for (const tile of state.map.tiles.filter((candidate) => candidate.terrain === "lake")) {
        const distances = roads.map((road) => ({ road, distance: Math.max(Math.abs(tile.x - road.x), Math.abs(tile.y - road.y)) }));
        const minimum = Math.min(...distances.map((entry) => entry.distance));
        const nearest = distances.filter((entry) => entry.distance === minimum).map((entry) => entry.road);
        if (nearest.length === 2) return { tile, nearest };
      }
      throw new Error("No equidistant-road fixture found");
    };
    const fixture = findFixture();
    const run = (rngValues: number[]) => {
      const state = makeState(); addBridge(state, { tiles: [{ x: fixture.tile.x, y: fixture.tile.y }] });
      const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());
      return resolveSavedReset(state, sequenceRng(rngValues)).units.find((unit) => unit.id === king.id)!.position;
    };
    const first = run([0, 0.1, 0.9]);
    const second = run([0, 0.9, 0.1]);
    expect(first).not.toEqual(second);
    expect([first, second].every((position) => position.kind === "tile" && fixture.nearest.some((road) => road.x === position.x && road.y === position.y))).toBe(true);
  });

  it("prefers a nearer road outside the unit team's operational area over a farther operational road", () => {
    const state = makeState();
    const operational = new Set(getOperationalRoadTiles(state, "team-2").map((tile) => `${tile.x},${tile.y}`));
    const roads = state.map.tiles.filter((tile) => tile.terrain === "road");
    const origin = state.map.tiles.find((tile) => {
      if (tile.terrain !== "lake") return false;
      const distances = roads.map((road) => ({ road, distance: Math.max(Math.abs(tile.x - road.x), Math.abs(tile.y - road.y)) }));
      const minimum = Math.min(...distances.map((entry) => entry.distance));
      const operationalMinimum = Math.min(...distances.filter((entry) => operational.has(`${entry.road.x},${entry.road.y}`)).map((entry) => entry.distance));
      return minimum < operationalMinimum && distances.some((entry) => entry.distance === minimum && !operational.has(`${entry.road.x},${entry.road.y}`));
    });
    if (!origin) throw new Error("No outside-operational-area retreat fixture found");
    addBridge(state, { tiles: [{ x: origin.x, y: origin.y }] });
    const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());

    const resolved = resolveSavedReset(state, () => 0);
    const position = resolved.units.find((unit) => unit.id === king.id)!.position;
    expect(position.kind).toBe("tile");
    if (position.kind === "tile") expect(operational.has(`${position.x},${position.y}`)).toBe(false);
  });

  it("allows the nearest road on the enemy-home side regardless of ownership or road section", () => {
    const state = makeState(); addBridge(state, { tiles: [{ x: 4, y: 2 }] });
    const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());
    const team1Roads = new Set(getOperationalRoadTiles(state, RESET_TEAM).map((tile) => `${tile.x},${tile.y}`));
    const team2Roads = new Set(getOperationalRoadTiles(state, "team-2").map((tile) => `${tile.x},${tile.y}`));

    const position = resolveSavedReset(state, () => 0).units.find((unit) => unit.id === king.id)!.position;
    expect(position.kind).toBe("tile");
    if (position.kind === "tile") {
      expect(team1Roads.has(`${position.x},${position.y}`)).toBe(true);
      expect(team2Roads.has(`${position.x},${position.y}`)).toBe(false);
    }
  });

  it("excludes occupied roads, obstructed roads, and active bridge cells from retreat", () => {
    const state = makeState(); addBridge(state, { tiles: [{ x: 4, y: 2 }] });
    const origin = { x: 4, y: 2 };
    const roads = state.map.tiles
      .filter((tile) => tile.terrain === "road")
      .sort((left, right) =>
        Math.max(Math.abs(origin.x - left.x), Math.abs(origin.y - left.y)) - Math.max(Math.abs(origin.x - right.x), Math.abs(origin.y - right.y)) ||
        left.x - right.x || left.y - right.y,
      );
    const occupied = roads[0], obstructed = roads[1], bridged = roads[2];
    addUnit(state, "retreat-road-occupant", RESET_TEAM, "infantry", { kind: "tile", x: occupied.x, y: occupied.y });
    state.constructions.push(
      { id: "retreat-road-obstacle", kind: "obstacle", ownerTeamId: RESET_TEAM, tiles: [{ x: obstructed.x, y: obstructed.y }], placedTurn: 1, active: true },
      { id: "retreat-road-active-bridge", kind: "bridge", ownerTeamId: RESET_TEAM, tiles: [{ x: bridged.x, y: bridged.y }], placedTurn: 1, active: true },
    );
    const king = state.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(state, king, bridgePosition());

    const position = resolveSavedReset(state, () => 0).units.find((unit) => unit.id === king.id)!.position;
    expect(position).not.toEqual(expect.objectContaining({ kind: "tile", x: occupied.x, y: occupied.y }));
    expect(position).not.toEqual(expect.objectContaining({ kind: "tile", x: obstructed.x, y: obstructed.y }));
    expect(position).not.toEqual(expect.objectContaining({ kind: "tile", x: bridged.x, y: bridged.y }));
  });

  it("keeps equidistant-road RNG results stable when map tile storage order is reversed", () => {
    const base = makeState();
    const roads = base.map.tiles.filter((tile) => tile.terrain === "road");
    const origin = base.map.tiles.find((tile) => {
      if (tile.terrain !== "lake") return false;
      const distances = roads.map((road) => Math.max(Math.abs(tile.x - road.x), Math.abs(tile.y - road.y)));
      const minimum = Math.min(...distances);
      return distances.filter((distance) => distance === minimum).length >= 2;
    });
    if (!origin) throw new Error("No equidistant-road ordering fixture found");
    addBridge(base, { tiles: [{ x: origin.x, y: origin.y }] });
    const king = base.units.find((unit) => unit.id === "home-2-king")!; king.hp = 2; relocate(base, king, bridgePosition());
    const reversed = structuredClone(base); reversed.map.tiles.reverse();

    const first = resolveSavedReset(base, sequenceRng([0.7, 0.2, 0.9, 0.4]));
    const second = resolveSavedReset(reversed, sequenceRng([0.7, 0.2, 0.9, 0.4]));
    expect(second.units.find((unit) => unit.id === king.id)?.position).toEqual(first.units.find((unit) => unit.id === king.id)?.position);
  });
});
