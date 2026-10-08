import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { createBcInferenceRequest } from "../cpu/browserBcPolicy";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { buildRlObservation, enumerateRlDecisions } from "../cpu/rlEnvironment";
import { createRlFeatureSpec, createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { commitUnitMovement, getMovementCandidates } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { Construction, GameState, Unit, UnitPosition } from "../types";

const jumpDestination: UnitPosition = { kind: "tile", x: 5, y: 1 };

function addUnit(state: GameState, id: string, type: Unit["type"], position: UnitPosition, ownerTeamId = "team-1") {
  const unit: Unit = { id, ownerTeamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function addObstacle(state: GameState, id: string, x: number, y: number, active = true): Construction {
  const obstacle: Construction = {
    id,
    kind: "obstacle",
    ownerTeamId: "team-2",
    managerUnitId: "enemy-builder",
    tiles: [{ x, y }],
    placedTurn: 1,
    active,
  };
  state.constructions.push(obstacle);
  return obstacle;
}

function jumpState(type: Unit["type"] = "ninja") {
  const state = createInitialGameState();
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  const unit = addUnit(state, "a-obstacle-jumper", type, { kind: "tile", x: 3, y: 1 });
  const obstacle = addObstacle(state, "jump-obstacle", 4, 1);
  return { state, unit, obstacle };
}

function hasJump(state: GameState, unitId: string) {
  return getMovementCandidates(state, unitId).some((position) =>
    position.kind === "tile" && position.x === 5 && position.y === 1,
  );
}

describe("ninja obstacle traversal", () => {
  it("offers the cell directly behind one adjacent active obstacle only to a ninja", () => {
    const ninja = jumpState();
    expect(hasJump(ninja.state, ninja.unit.id)).toBe(true);
    expect(getMovementCandidates(ninja.state, ninja.unit.id)).not.toContainEqual({ kind: "tile", x: 4, y: 1 });

    const infantry = jumpState("infantry");
    expect(hasJump(infantry.state, infantry.unit.id)).toBe(false);
    expect(getMovementCandidates(infantry.state, infantry.unit.id)).not.toContainEqual({ kind: "tile", x: 4, y: 1 });
  });

  it("rejects an occupied destination, a second obstacle, and a unit on the obstacle cell", () => {
    const occupied = jumpState();
    addUnit(occupied.state, "destination-occupant", "infantry", jumpDestination, "team-2");
    expect(hasJump(occupied.state, occupied.unit.id)).toBe(false);

    const doubled = jumpState();
    addObstacle(doubled.state, "second-obstacle", 5, 1);
    expect(hasJump(doubled.state, doubled.unit.id)).toBe(false);

    const blockedByUnit = jumpState();
    addUnit(blockedByUnit.state, "obstacle-cell-occupant", "infantry", { kind: "tile", x: 4, y: 1 }, "team-2");
    expect(hasJump(blockedByUnit.state, blockedByUnit.unit.id)).toBe(false);

    const inactive = jumpState();
    inactive.obstacle.active = false;
    expect(hasJump(inactive.state, inactive.unit.id)).toBe(false);
  });

  it("commits as one immediate movement without removing or disabling the obstacle", () => {
    const { state, unit, obstacle } = jumpState();
    const beforeIntents = structuredClone(state.turnState.actionIntents);
    const moved = commitUnitMovement(state, {
      teamId: "team-1",
      unitId: unit.id,
      from: unit.position,
      to: jumpDestination,
      stay: false,
    });

    expect(moved.units.find((candidate) => candidate.id === unit.id)?.position).toEqual(jumpDestination);
    expect(moved.movedUnitIdsThisMovementPhase).toContain(unit.id);
    expect(moved.turnState.actionIntents).toEqual(beforeIntents);
    expect(moved.constructions.find((construction) => construction.id === obstacle.id)).toEqual(obstacle);
    expect(UNIT_STATS.ninja.move).toBe(1);
  });

  it("preserves existing water movement and does not jump an obstacle from water", () => {
    const waterState = createInitialGameState();
    const waterNinja = addUnit(waterState, "water-ninja", "ninja", { kind: "water", x: 4, y: 2 });
    expect(getMovementCandidates(waterState, waterNinja.id)).toContainEqual({ kind: "tile", x: 4, y: 1 });
    expect(getMovementCandidates(waterState, waterNinja.id)).toContainEqual({ kind: "water", x: 5, y: 3 });

    const separatedLakeState = createInitialGameState();
    const separatedLakeNinja = addUnit(separatedLakeState, "separated-lake-ninja", "ninja", { kind: "water", x: 9, y: 3 });
    addObstacle(separatedLakeState, "shore-obstacle", 10, 3);
    expect(getMovementCandidates(separatedLakeState, separatedLakeNinja.id)).not.toContainEqual({ kind: "water", x: 11, y: 3 });
  });

  it("exposes the shared destination to random, heuristic, BC, and RL legal enumeration", () => {
    const { state } = jumpState();
    const rlKeys = enumerateRlDecisions(state, createCpuRuntime(7), (teamId) => teamId === "team-1")
      .map((entry) => entry.action.actionKey);
    expect(rlKeys).toContain("movement:team-1:a-obstacle-jumper:5,1");

    const randomSettings: CpuTeamSettings = { "team-1": "random_cpu", "team-2": "human", "team-3": "human", "team-4": "human" };
    const randomDecisions = Array.from({ length: 256 }, (_, index) =>
      getRandomCpuDecision(state, createCpuRuntime((index * 0x01000000) >>> 0), randomSettings),
    );
    expect(randomDecisions.some((decision) =>
      decision?.kind === "movement" && decision.to?.kind === "tile" && decision.to.x === 5 && decision.to.y === 1,
    )).toBe(true);

    const heuristic = createHeuristicCpuPolicy();
    heuristic.setDecisionDiagnosticsEnabled(true);
    const heuristicSettings: CpuTeamSettings = { ...randomSettings, "team-1": "heuristic_cpu" };
    expect(() => heuristic(state, createCpuRuntime(7), heuristicSettings)).not.toThrow();
    expect(heuristic.getLastDecisionDiagnostics()?.legalActionKeys).toContain("movement:team-1:a-obstacle-jumper:5,1");

    const bcSettings: CpuTeamSettings = { ...randomSettings, "team-1": "bc_cpu" };
    const bc = createBcInferenceRequest(state, createCpuRuntime(7), bcSettings);
    expect(bc?.request?.legalActions.actionKeys).toContain("movement:team-1:a-obstacle-jumper:5,1");
  });

  it("keeps the existing observation/action schema versions and feature widths", () => {
    const state = createInitialGameState();
    const observation = buildRlObservation(state, "team-1", "team-1");
    expect(createRlFeatureSpec(observation)).toMatchObject({
      schemaVersion: 1,
      observationSchemaVersion: 1,
      actionSchemaVersion: 1,
      globalWidth: 15,
      teamWidth: 15,
      unitWidth: 57,
      mapTileWidth: 38,
      baseWidth: 87,
      constructionWidth: 22,
      strategicGlobalWidth: 36,
      actionFeatureWidth: 3403,
    });
    expect(createRlFeatureSpecV2(observation)).toMatchObject({
      schemaVersion: 2,
      observationSchemaVersion: 2,
      actionSchemaVersion: 2,
      globalWidth: 15,
      teamWidth: 15,
      unitWidth: 58,
      mapTileWidth: 38,
      baseWidth: 87,
      constructionWidth: 22,
      strategicGlobalWidth: 36,
      actionFeatureWidth: 3955,
    });
  });
});
