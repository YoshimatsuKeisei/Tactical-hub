import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { getAttackCandidates } from "../engine/battle";
import { getMovementCandidates, saveMovementIntent, submitMovement } from "../engine/movement";
import { getTeleportDestinationCandidates, saveTeleportIntent } from "../engine/teleport";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";
import { positionKey } from "../utils/position";

function addUnit(state: GameState, id: string, ownerTeamId: string, type: Unit["type"], position: UnitPosition, role?: Unit["role"]) {
  const unit: Unit = { id, ownerTeamId, type, role, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function addBridge(state: GameState) {
  state.constructions.push({
    id: "ninja-water-bridge",
    kind: "bridge",
    ownerTeamId: "team-1",
    tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }],
    placedTurn: 1,
    active: true,
  });
}

function keys(state: GameState, unitId: string) {
  return getMovementCandidates(state, unitId).map(positionKey);
}

describe("Phase 4-D ninja water movement", () => {
  it.each([
    ["orthogonal", { kind: "tile", x: 4, y: 1 }, "4,2"],
    ["diagonal", { kind: "tile", x: 4, y: 1 }, "5,2"],
  ] as const)("enters adjacent water from a road: %s", (_, from, expected) => {
    const state = createInitialGameState();
    const ninja = addUnit(state, `enter-${expected}`, "team-1", "ninja", from);
    expect(keys(state, ninja.id)).toContain(expected);
  });

  it.each([
    ["orthogonal", { kind: "water", x: 4, y: 2 }, "5,2"],
    ["diagonal", { kind: "water", x: 4, y: 2 }, "5,3"],
  ] as const)("moves between adjacent lake cells: %s", (_, from, expected) => {
    const state = createInitialGameState();
    const ninja = addUnit(state, `swim-${expected}`, "team-1", "ninja", from);
    expect(keys(state, ninja.id)).toContain(expected);
  });

  it.each([
    ["orthogonal", { kind: "water", x: 4, y: 2 }, "4,1"],
    ["diagonal", { kind: "water", x: 5, y: 2 }, "4,1"],
  ] as const)("lands on an adjacent road: %s", (_, from, expected) => {
    const state = createInitialGameState();
    const ninja = addUnit(state, `land-${expected}-${from.x}`, "team-1", "ninja", from);
    expect(keys(state, ninja.id)).toContain(expected);
  });

  it("allows one-cell water movement but not a two-cell destination", () => {
    const state = createInitialGameState();
    const ninja = addUnit(state, "water-boundary", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    expect(keys(state, ninja.id)).toContain("5,3");
    expect(keys(state, ninja.id)).not.toContain("6,4");
  });

  it("uses active bridges from roads but never directly between a bridge and the lake", () => {
    const roadState = createInitialGameState();
    addBridge(roadState);
    const roadNinja = addUnit(roadState, "road-bridge-ninja", "team-1", "ninja", { kind: "tile", x: 4, y: 1 });
    expect(getMovementCandidates(roadState, roadNinja.id)).toContainEqual({ kind: "bridge", bridgeId: "ninja-water-bridge", cellIndex: 0 });

    const bridgeState = createInitialGameState();
    addBridge(bridgeState);
    const bridgeNinja = addUnit(bridgeState, "bridge-road-ninja", "team-1", "ninja", { kind: "bridge", bridgeId: "ninja-water-bridge", cellIndex: 0 });
    expect(getMovementCandidates(bridgeState, bridgeNinja.id)).toContainEqual({ kind: "tile", x: 4, y: 1 });
    expect(keys(bridgeState, bridgeNinja.id)).not.toContain("5,2");

    const waterState = createInitialGameState();
    addBridge(waterState);
    const waterNinja = addUnit(waterState, "water-bridge-ninja", "team-1", "ninja", { kind: "water", x: 5, y: 2 });
    expect(getMovementCandidates(waterState, waterNinja.id)).not.toContainEqual({ kind: "bridge", bridgeId: "ninja-water-bridge", cellIndex: 0 });
  });

  it("does not allow lake-to-bridge movement when an existing lake position is stored as tile", () => {
    const state = createInitialGameState();
    addBridge(state);
    const existingStateNinja = addUnit(state, "legacy-lake-position-ninja", "team-1", "ninja", { kind: "tile", x: 5, y: 2 });
    const candidates = getMovementCandidates(state, existingStateNinja.id);
    expect(candidates).not.toContainEqual({ kind: "bridge", bridgeId: "ninja-water-bridge", cellIndex: 0 });
    expect(candidates).not.toContainEqual({ kind: "bridge", bridgeId: "ninja-water-bridge", cellIndex: 1 });
  });

  it("keeps non-ninjas out of lake cells", () => {
    const state = createInitialGameState();
    const infantry = addUnit(state, "water-blocked-infantry", "team-1", "infantry", { kind: "tile", x: 4, y: 1 });
    expect(keys(state, infantry.id)).not.toEqual(expect.arrayContaining(["4,2", "5,2"]));
  });

  it("allows attacks only between enemy ninjas whose current positions are both water", () => {
    const waterState = createInitialGameState();
    const first = addUnit(waterState, "water-ninja-a", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    const second = addUnit(waterState, "water-ninja-b", "team-2", "ninja", { kind: "water", x: 5, y: 3 });
    expect(getAttackCandidates(waterState, first.id).map((target) => target.unitId)).toContain(second.id);
    expect(getAttackCandidates(waterState, second.id).map((target) => target.unitId)).toContain(first.id);

    const mixedState = createInitialGameState();
    addBridge(mixedState);
    const water = addUnit(mixedState, "mixed-water-ninja", "team-1", "ninja", { kind: "water", x: 5, y: 2 });
    const road = addUnit(mixedState, "mixed-road-ninja", "team-2", "ninja", { kind: "tile", x: 4, y: 1 });
    const bridge = addUnit(mixedState, "mixed-bridge-ninja", "team-2", "ninja", { kind: "bridge", bridgeId: "ninja-water-bridge", cellIndex: 0 });
    const infantry = addUnit(mixedState, "mixed-water-infantry", "team-2", "infantry", { kind: "water", x: 5, y: 3 });
    expect(getAttackCandidates(mixedState, water.id).map((target) => target.unitId)).not.toEqual(expect.arrayContaining([road.id, bridge.id, infantry.id]));
    expect(getAttackCandidates(mixedState, road.id).map((target) => target.unitId)).not.toContain(water.id);
    expect(getAttackCandidates(mixedState, bridge.id).map((target) => target.unitId)).not.toContain(water.id);
    expect(getAttackCandidates(mixedState, infantry.id).map((target) => target.unitId)).not.toContain(water.id);
  });

  it("can attack normally after landing and is marked moved in sequential movement", () => {
    let state = createInitialGameState();
    const ninja = addUnit(state, "landing-attacker", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    const enemy = addUnit(state, "landing-target", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    state = saveMovementIntent(state, { teamId: "team-1", unitId: ninja.id, from: ninja.position, to: { kind: "tile", x: 4, y: 1 }, stay: false });
    state = submitMovement(state, "team-1");
    expect(state.movedUnitIdsThisMovementPhase).toContain(ninja.id);
    expect(state.currentMovementTeamId).toBe("team-2");
    while (state.phase === "movement_input" && state.currentMovementTeamId) state = submitMovement(state, state.currentMovementTeamId);
    expect(state.phase).toBe("attack_input");
    expect(getAttackCandidates(state, ninja.id).map((target) => target.unitId)).toContain(enemy.id);
  });

  it("shares destination reservations across normal, ninja-water, and teleport movement", () => {
    let ninjaFirst = createInitialGameState();
    const ninja = addUnit(ninjaFirst, "reservation-ninja", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    const infantry = addUnit(ninjaFirst, "reservation-infantry", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    ninjaFirst = saveMovementIntent(ninjaFirst, { teamId: "team-1", unitId: ninja.id, from: ninja.position, to: { kind: "tile", x: 4, y: 1 }, stay: false });
    expect(keys(ninjaFirst, infantry.id)).not.toContain("4,1");

    let normalFirst = createInitialGameState();
    const secondNinja = addUnit(normalFirst, "reservation-ninja-2", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    const secondInfantry = addUnit(normalFirst, "reservation-infantry-2", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    normalFirst = saveMovementIntent(normalFirst, { teamId: "team-1", unitId: secondInfantry.id, from: secondInfantry.position, to: { kind: "tile", x: 4, y: 1 }, stay: false });
    expect(keys(normalFirst, secondNinja.id)).not.toContain("4,1");

    let teleportFirst = createInitialGameState();
    const thirdNinja = addUnit(teleportFirst, "reservation-ninja-3", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    const teleporter = addUnit(teleportFirst, "reservation-teleporter", "team-1", "strategist", { kind: "tile", x: 3, y: 1 }, "teleporter");
    const target = addUnit(teleportFirst, "reservation-target", "team-1", "infantry", { kind: "tile", x: 3, y: 2 });
    expect(getTeleportDestinationCandidates(teleportFirst, teleporter.id).map(positionKey)).toContain("4,1");
    teleportFirst = saveTeleportIntent(teleportFirst, { teamId: "team-1", strategistUnitId: teleporter.id, targetUnitId: target.id, to: { kind: "tile", x: 4, y: 1 } });
    expect(keys(teleportFirst, thirdNinja.id)).not.toContain("4,1");
  });

  it("clears retreat state immediately when a retreating ninja enters water", () => {
    let state = createInitialGameState();
    const ninja = addUnit(state, "retreating-water-ninja", "team-1", "ninja", { kind: "tile", x: 4, y: 1 });
    ninja.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1", remainingTurns: 2, sourceId: "test" });
    state = saveMovementIntent(state, { teamId: "team-1", unitId: ninja.id, from: ninja.position, to: { kind: "water", x: 4, y: 2 }, stay: false });
    state = submitMovement(state, "team-1");
    const resolved = state.units.find((unit) => unit.id === ninja.id)!;
    expect(resolved.position).toEqual({ kind: "water", x: 4, y: 2 });
    expect(resolved.statuses.some((status) => status.kind === "retreating")).toBe(false);
  });
});
