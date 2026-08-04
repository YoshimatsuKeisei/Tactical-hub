import { describe, expect, it } from "vitest";
import { getAttackCandidates } from "../engine/battle";
import { beginMovementPhase, getMovementCandidates, saveMovementIntent, submitMovement } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit } from "../types";
import { createTeamVisibleState, isUnitVisibleToTeam } from "../visibility";
import { createCpuRuntime } from "../cpu/types";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { enumerateRlDecisions, RlEnvironment } from "../cpu/rlEnvironment";
import { RL_ACTION_TYPES, RL_ACTION_ENCODER_VERSION } from "../cpu/rlActionEncoder";
import { RL_OBSERVATION_ENCODER_VERSION, RL_UNIT_TYPES, getRlObservationFeatureSpec } from "../cpu/rlObservationEncoder";

function waterNinja(id: string, teamId: string, x: number, y: number): Unit {
  return { id, ownerTeamId: teamId, type: "ninja", hp: 1, position: { kind: "water", x, y }, statuses: [] };
}

function collisionState() {
  const state = createInitialGameState();
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  const mover = waterNinja("water-a", "team-1", 4, 2);
  const waiting = waterNinja("water-b", "team-2", 5, 2);
  state.units.push(mover, waiting);
  return { state, mover, waiting };
}

describe("team-scoped water ninja visibility", () => {
  it("shows own water ninja, hides it from enemies, and keeps ground/bridge ninja visible", () => {
    const state = createInitialGameState();
    const water = waterNinja("hidden-water", "team-1", 4, 2);
    const ground: Unit = { ...waterNinja("ground", "team-1", 4, 2), position: { kind: "tile", x: 4, y: 1 } };
    state.constructions.push({ id: "visible-bridge", kind: "bridge", tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    const bridge: Unit = { ...waterNinja("bridge", "team-1", 4, 2), position: { kind: "bridge", bridgeId: "visible-bridge", cellIndex: 0 } };
    state.units.push(water, ground, bridge);
    expect(createTeamVisibleState(state, "team-1").units.map((unit) => unit.id)).toContain(water.id);
    expect(createTeamVisibleState(state, "team-2").units.map((unit) => unit.id)).not.toContain(water.id);
    expect(createTeamVisibleState(state, "team-3").units.map((unit) => unit.id)).not.toContain(water.id);
    expect(createTeamVisibleState(state, "team-2").units.map((unit) => unit.id)).toEqual(expect.arrayContaining([ground.id, bridge.id]));
  });

  it("keeps the hidden occupied water tile selectable and detects collision only at resolution", () => {
    let { state, mover, waiting } = collisionState();
    state.phase = state.turnState.phase = "attack_input";
    expect(getAttackCandidates(state, mover.id).map((target) => target.unitId)).toEqual([waiting.id]);
    expect(getAttackCandidates(state, waiting.id).map((target) => target.unitId)).toEqual([mover.id]);
    state.phase = state.turnState.phase = "movement_input";
    expect(getMovementCandidates(state, mover.id)).toContainEqual(waiting.position);
    const saved = saveMovementIntent(state, { teamId: "team-1", unitId: mover.id, from: mover.position, to: waiting.position, stay: false });
    expect(saved).not.toBe(state);
    state = submitMovement(saved, "team-1", () => 0);
    expect(state.units.find((unit) => unit.id === mover.id)?.position).toEqual(mover.position);
    expect(state.units.find((unit) => unit.id === waiting.id)?.position).toEqual(waiting.position);
    expect(state.movedUnitIdsThisMovementPhase).toEqual(expect.arrayContaining([mover.id, waiting.id]));
    expect(isUnitVisibleToTeam(state, waiting, "team-1")).toBe(true);
    expect(isUnitVisibleToTeam(state, mover, "team-2")).toBe(true);
    expect(isUnitVisibleToTeam(state, mover, "team-3")).toBe(false);
    expect(isUnitVisibleToTeam(state, waiting, "team-3")).toBe(false);
    state.phase = state.turnState.phase = "attack_input";
    expect(getAttackCandidates(state, mover.id).map((target) => target.unitId)).toEqual([waiting.id]);
    expect(getAttackCandidates(state, waiting.id).map((target) => target.unitId)).toEqual([mover.id]);
    expect(beginMovementPhase(state).ninjaRevealStates).toEqual([]);
  });

  it("does not reveal same-team collisions and blocks already-visible occupants", () => {
    const { state, mover, waiting } = collisionState();
    waiting.ownerTeamId = mover.ownerTeamId;
    expect(getMovementCandidates(createTeamVisibleState(state, "team-1"), mover.id)).not.toContainEqual(waiting.position);
    waiting.ownerTeamId = "team-2";
    state.ninjaRevealStates = [{ ninjaUnitId: waiting.id, visibleToTeamIds: ["team-1"] }];
    expect(getMovementCandidates(createTeamVisibleState(state, "team-1"), mover.id)).not.toContainEqual(waiting.position);
  });

  it("keeps enemy water ninjas hidden from observations while exposing ninja-vs-ninja attack actions", () => {
    const { state, mover, waiting } = collisionState();
    state.phase = state.turnState.phase = "attack_input";
    state.currentMovementTeamId = undefined;
    state.units = [mover, waiting];
    const randomRuntime = createCpuRuntime(3);
    const random = getRandomCpuDecision(state, randomRuntime, { "team-1": "random_cpu", "team-2": "human" });
    expect(random?.kind).toBe("attack");
    const heuristic = createHeuristicCpuPolicy()(state, createCpuRuntime(3), { "team-1": "heuristic_cpu", "team-2": "human" });
    expect(heuristic?.kind).toBe("attack");
    const decisions = enumerateRlDecisions(state, createCpuRuntime(3), (teamId) => teamId === "team-1");
    expect(decisions.flatMap((entry) => entry.decision.kind === "attack" ? [entry.decision.intent.target?.unitId] : [])).toContain(waiting.id);
    const environment = new RlEnvironment();
    environment.reset(3, 4, state);
    expect(environment.getObservation("team-1").units.map((unit) => unit.id)).toContain(mover.id);
    expect(environment.getObservation("team-1").units.map((unit) => unit.id)).not.toContain(waiting.id);
  });

  it("keeps legacy RL schemas free of merge and heavy-infantry action/type additions", () => {
    expect(RL_ACTION_TYPES.some((value) => /merge|combine|heavy/i.test(value))).toBe(false);
    expect(RL_UNIT_TYPES).not.toContain("heavy_infantry");
    expect(RL_ACTION_ENCODER_VERSION).toBe(1);
    expect(RL_OBSERVATION_ENCODER_VERSION).toBe(1);
    const environment = new RlEnvironment();
    const observation = environment.reset(1, 4);
    const featureSpec = getRlObservationFeatureSpec(observation);
    expect(Object.values(featureSpec).every((value) => typeof value !== "number" || Number.isFinite(value))).toBe(true);
  });
});
