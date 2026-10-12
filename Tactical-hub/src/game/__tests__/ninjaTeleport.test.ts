import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { createBcInferenceRequest } from "../cpu/browserBcPolicy";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { enumerateRlDecisions, RlEnvironment } from "../cpu/rlEnvironment";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { getAttackCandidates } from "../engine/battle";
import { commitUnitMovement, getMovementCandidates } from "../engine/movement";
import { getNinjaTeleportDestinations } from "../engine/ninjaTeleport";
import { createInitialGameState } from "../initialState";
import type { Construction, GameState, Unit, UnitPosition, UnitType } from "../types";
import { positionKey } from "../utils/position";

function addUnit(state: GameState, id: string, ownerTeamId: string, type: UnitType, position: UnitPosition) {
  const unit: Unit = { id, ownerTeamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function addObstacle(state: GameState, x: number, y: number) {
  const obstacle: Construction = {
    id: `obstacle-${x}-${y}`, kind: "obstacle", ownerTeamId: "team-2",
    tiles: [{ x, y }], placedTurn: 1, active: true,
  };
  state.constructions.push(obstacle);
}

function teleportState() {
  const state = createInitialGameState();
  state.phase = state.turnState.phase = "movement_input";
  state.currentMovementTeamId = "team-1";
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  const ninja = addUnit(state, "a-ninja-teleporter", "team-1", "ninja", { kind: "tile", x: 3, y: 1 });
  const enemy = addUnit(state, "enemy-in-operation", "team-2", "infantry", { kind: "tile", x: 6, y: 1 });
  return { state, ninja, enemy };
}

const keys = (positions: UnitPosition[]) => positions.map(positionKey).sort();

describe("one-use ninja teleport", () => {
  it("adds legal cells in all eight adjacent directions to ordinary movement candidates", () => {
    const { state, ninja } = teleportState();

    expect(keys(getNinjaTeleportDestinations(state, ninja.id))).toEqual(["5,1", "6,2", "7,1"]);
    const movementKeys = keys(getMovementCandidates(state, ninja.id));
    expect(movementKeys).toContain("4,1");
    expect(movementKeys).toContain("5,1");
    expect(movementKeys).toContain("6,2");
    expect(movementKeys).toContain("7,1");
  });

  it("ignores enemies and landing cells outside the acting team's operational area", () => {
    const { state, ninja, enemy } = teleportState();
    enemy.position = { kind: "tile", x: 14, y: 1 };

    expect(getNinjaTeleportDestinations(state, ninja.id)).toEqual([]);
  });

  it("requires a ground source and rejects water, obstacles, and occupied landing cells", () => {
    const blocked = teleportState();
    addObstacle(blocked.state, 5, 1);
    addUnit(blocked.state, "landing-occupant", "team-1", "infantry", { kind: "tile", x: 7, y: 1 });
    expect(keys(getNinjaTeleportDestinations(blocked.state, blocked.ninja.id))).toEqual(["6,2"]);

    const water = teleportState();
    water.ninja.position = { kind: "water", x: 4, y: 2 };
    expect(getNinjaTeleportDestinations(water.state, water.ninja.id)).toEqual([]);
    expect(getMovementCandidates(water.state, water.ninja.id).some((position) => position.kind === "water")).toBe(true);
  });

  it("does not require a continuous path and consumes the ability only after success", () => {
    const { state, ninja } = teleportState();
    addObstacle(state, 4, 1);
    const destination: UnitPosition = { kind: "tile", x: 7, y: 1 };
    expect(getMovementCandidates(state, ninja.id)).toContainEqual(destination);

    const moved = commitUnitMovement(state, {
      teamId: "team-1", unitId: ninja.id, from: ninja.position, to: destination, stay: false,
    });
    const movedNinja = moved.units.find((unit) => unit.id === ninja.id)!;
    expect(movedNinja.position).toEqual(destination);
    expect(moved.ninjaTeleportUsedUnitIds).toContain(ninja.id);
    expect(moved.movedUnitIdsThisMovementPhase).toContain(ninja.id);
    expect(moved.logs.at(-1)?.message).toContain("used ninja teleport");
    moved.phase = moved.turnState.phase = "attack_input";
    expect(getAttackCandidates(moved, ninja.id).map((target) => target.unitId)).toContain("enemy-in-operation");

    const restored = JSON.parse(JSON.stringify(moved)) as GameState;
    restored.phase = restored.turnState.phase = "movement_input";
    restored.currentMovementTeamId = "team-1";
    restored.movedUnitIdsThisMovementPhase = [];
    expect(restored.ninjaTeleportUsedUnitIds).toContain(ninja.id);
    expect(getNinjaTeleportDestinations(restored, ninja.id)).toEqual([]);
  });

  it("does not consume the ability on an illegal or newly occupied destination", () => {
    const invalid = teleportState();
    const rejected = commitUnitMovement(invalid.state, {
      teamId: "team-1", unitId: invalid.ninja.id, from: invalid.ninja.position,
      to: { kind: "tile", x: 8, y: 1 }, stay: false,
    });
    expect(rejected).toBe(invalid.state);
    expect(invalid.state.ninjaTeleportUsedUnitIds ?? []).not.toContain(invalid.ninja.id);

    const conflict = teleportState();
    const destination: UnitPosition = { kind: "tile", x: 7, y: 1 };
    expect(getNinjaTeleportDestinations(conflict.state, conflict.ninja.id)).toContainEqual(destination);
    addUnit(conflict.state, "late-occupant", "team-1", "infantry", destination);
    const conflicted = commitUnitMovement(conflict.state, {
      teamId: "team-1", unitId: conflict.ninja.id, from: conflict.ninja.position, to: destination, stay: false,
    });
    expect(conflicted).toBe(conflict.state);
    expect(conflict.state.ninjaTeleportUsedUnitIds ?? []).not.toContain(conflict.ninja.id);
  });

  it("prefers ordinary movement for an overlapping destination and prevents a second move", () => {
    const { state, ninja, enemy } = teleportState();
    enemy.position = { kind: "tile", x: 5, y: 1 };
    const ordinaryDestination: UnitPosition = { kind: "tile", x: 4, y: 1 };
    expect(getNinjaTeleportDestinations(state, ninja.id)).toContainEqual(ordinaryDestination);

    const moved = commitUnitMovement(state, {
      teamId: "team-1", unitId: ninja.id, from: ninja.position, to: ordinaryDestination, stay: false,
    });
    expect(moved.ninjaTeleportUsedUnitIds ?? []).not.toContain(ninja.id);
    expect(getMovementCandidates(moved, ninja.id)).toEqual([]);
  });

  it("exposes the same destination to Random, Heuristic, BC, and RL action enumeration", () => {
    const { state } = teleportState();
    const expectedKey = "movement:team-1:a-ninja-teleporter:7,1";
    const rlKeys = enumerateRlDecisions(state, createCpuRuntime(7), (teamId) => teamId === "team-1")
      .map((entry) => entry.action.actionKey);
    expect(rlKeys).toContain(expectedKey);

    const randomSettings: CpuTeamSettings = { "team-1": "random_cpu", "team-2": "human", "team-3": "human", "team-4": "human" };
    const randomDecisions = Array.from({ length: 256 }, (_, index) =>
      getRandomCpuDecision(state, createCpuRuntime((index * 0x01000000) >>> 0), randomSettings),
    );
    expect(randomDecisions.some((decision) =>
      decision?.kind === "movement" && decision.to?.kind === "tile" && decision.to.x === 7 && decision.to.y === 1,
    )).toBe(true);

    const heuristic = createHeuristicCpuPolicy();
    heuristic.setDecisionDiagnosticsEnabled(true);
    const heuristicSettings: CpuTeamSettings = { ...randomSettings, "team-1": "heuristic_cpu" };
    heuristic(state, createCpuRuntime(7), heuristicSettings);
    expect(heuristic.getLastDecisionDiagnostics()?.legalActionKeys).toContain(expectedKey);

    const bcSettings: CpuTeamSettings = { ...randomSettings, "team-1": "bc_cpu" };
    expect(createBcInferenceRequest(state, createCpuRuntime(7), bcSettings)?.request?.legalActions.actionKeys).toContain(expectedKey);
  });

  it("consumes the ability through the prevalidated in-place RL execution path", () => {
    const { state, ninja } = teleportState();
    const actionKey = "movement:team-1:a-ninja-teleporter:7,1";
    const environment = new RlEnvironment(undefined, 1, "current", true, {
      cpuStep: { rlPrevalidatedMovement: true },
    });
    environment.reset(11, 4, state);
    expect(environment.getLegalActions("team-1").map((action) => action.actionKey)).toContain(actionKey);

    environment.stepWithoutObservation(actionKey);
    const moved = environment.getStateForValidation().units.find((unit) => unit.id === ninja.id)!;
    expect(moved.position).toEqual({ kind: "tile", x: 7, y: 1 });
    expect(environment.getStateForValidation().ninjaTeleportUsedUnitIds).toContain(ninja.id);
  });
});
