import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { getTeamAttackCandidates, saveAttackIntent } from "../engine/battle";
import { getStrategistActionCandidates, saveStrategistActionIntent } from "../engine/construction";
import { getTeamMovementCandidates, saveMovementIntent, submitMovement, validateMovementPath } from "../engine/movement";
import { getProductionCandidates, resolveProduction, saveProductionChoice } from "../engine/production";
import { getRewardPlacementCandidates, placeRewardUnit } from "../engine/reward";
import { getTeamTeleportCandidates, getTeleportDestinationCandidates, getTeleportStrategists, getTeleportTargetCandidates } from "../engine/teleport";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";
import { positionKey } from "../utils/position";

function addUnit(state: GameState, id: string, teamId: string, type: Unit["type"], position: UnitPosition, role?: Unit["role"]) {
  const unit: Unit = { id, ownerTeamId: teamId, type, role, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function setPhase(state: GameState, phase: GameState["phase"]) {
  state.phase = state.turnState.phase = phase;
  return state;
}

describe("Phase 5-A CPU candidate access", () => {
  it("exposes existing legal candidates for every major input phase without UI state", () => {
    const production = setPhase(createInitialGameState(), "production");
    expect(getProductionCandidates(production, "team-1").length).toBeGreaterThan(0);

    const movement = createInitialGameState();
    const ninja = addUnit(movement, "cpu-water-ninja", "team-1", "ninja", { kind: "tile", x: 4, y: 1 });
    expect(getTeamMovementCandidates(movement, "team-1").find((entry) => entry.unitId === ninja.id)?.destinations.map(positionKey)).toContain("4,2");

    const teleporter = addUnit(movement, "cpu-teleporter", "team-1", "strategist", { kind: "tile", x: 3, y: 1 }, "teleporter");
    addUnit(movement, "cpu-teleport-target", "team-1", "infantry", { kind: "tile", x: 3, y: 2 });
    expect(getTeleportStrategists(movement, "team-1").map((unit) => unit.id)).toContain(teleporter.id);
    expect(getTeleportTargetCandidates(movement, teleporter.id).length).toBeGreaterThan(0);
    expect(getTeleportDestinationCandidates(movement, teleporter.id).length).toBeGreaterThan(0);
    expect(getTeamTeleportCandidates(movement, "team-1").some((entry) => entry.strategistUnitId === teleporter.id)).toBe(true);

    const attack = setPhase(createInitialGameState(), "attack_input");
    const attacker = addUnit(attack, "cpu-attacker", "team-1", "infantry", { kind: "tile", x: 4, y: 1 });
    const defender = addUnit(attack, "cpu-defender", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    expect(getTeamAttackCandidates(attack, "team-1").find((entry) => entry.attackerUnitId === attacker.id)?.targets.map((target) => target.unitId)).toContain(defender.id);

    const reward = setPhase(createInitialGameState(), "reward_placement");
    reward.rewardPlacementRequests.push({ id: "cpu-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    expect(getRewardPlacementCandidates(reward, "team-1").length).toBeGreaterThan(0);

    const construction = setPhase(createInitialGameState(), "strategist_action_input");
    construction.units.find((unit) => unit.id === "home-1-strategist")!.role = "builder";
    expect(getStrategistActionCandidates(construction, "team-1").some((candidate) => candidate.action === "pass")).toBe(true);
  });

  it("returns no candidates to teams without the current phase operation right", () => {
    const production = setPhase(createInitialGameState(), "production");
    production.teams.find((team) => team.id === "team-2")!.status = "defeated";
    expect(getProductionCandidates(production, "team-2")).toEqual([]);

    const movement = createInitialGameState();
    expect(getTeamMovementCandidates(movement, "team-2")).toEqual([]);
    expect(getTeamTeleportCandidates(movement, "team-2")).toEqual([]);

    const attack = setPhase(createInitialGameState(), "movement_input");
    expect(getTeamAttackCandidates(attack, "team-1")).toEqual([]);

    const reward = setPhase(createInitialGameState(), "reward_placement");
    reward.rewardPlacementRequests.push({ id: "owned-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    expect(getRewardPlacementCandidates(reward, "team-2")).toEqual([]);

    const construction = setPhase(createInitialGameState(), "strategist_action_input");
    construction.units.find((unit) => unit.id === "home-1-strategist")!.role = "builder";
    construction.strategistSubmittedTeamIds.push("team-1");
    expect(getStrategistActionCandidates(construction, "team-1")).toEqual([]);
  });

  it("reflects saved friendly reservations and the latest board after sequential movement", () => {
    let state = createInitialGameState();
    const first = addUnit(state, "cpu-first-mover", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    const ninja = addUnit(state, "cpu-reserved-ninja", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    state = saveMovementIntent(state, { teamId: "team-1", unitId: first.id, from: first.position, to: { kind: "tile", x: 4, y: 1 }, stay: false });
    expect(getTeamMovementCandidates(state, "team-1").find((entry) => entry.unitId === ninja.id)?.destinations.map(positionKey)).not.toContain("4,1");

    state = submitMovement(state, "team-1");
    const later = addUnit(state, "cpu-later-mover", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    expect(getTeamMovementCandidates(state, "team-2").find((entry) => entry.unitId === later.id)?.destinations.map(positionKey)).not.toContain("4,1");
  });

  it("returns candidates accepted by the existing validators and intent entry points", () => {
    const movement = createInitialGameState();
    const mover = addUnit(movement, "cpu-validated-mover", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    const destination = getTeamMovementCandidates(movement, "team-1").find((entry) => entry.unitId === mover.id)!.destinations[0];
    expect(validateMovementPath(movement, mover, mover.position, destination).valid).toBe(true);

    const production = setPhase(createInitialGameState(), "production");
    const choice = getProductionCandidates(production, "team-1")[0];
    expect(resolveProduction(saveProductionChoice(production, choice)).units.length).toBeGreaterThan(production.units.length);

    const attack = setPhase(createInitialGameState(), "attack_input");
    const attacker = addUnit(attack, "cpu-valid-attacker", "team-1", "infantry", { kind: "tile", x: 4, y: 1 });
    addUnit(attack, "cpu-valid-defender", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    const target = getTeamAttackCandidates(attack, "team-1").find((entry) => entry.attackerUnitId === attacker.id)!.targets[0];
    expect(saveAttackIntent(attack, { teamId: "team-1", attackerUnitId: attacker.id, target, pass: false })).not.toBe(attack);

    const reward = setPhase(createInitialGameState(), "reward_placement");
    reward.rewardPlacementRequests.push({ id: "valid-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    const placement = getRewardPlacementCandidates(reward, "team-1")[0];
    expect(placeRewardUnit(reward, placement.requestId, placement.baseId, placement.unitType)).not.toBe(reward);

    const construction = setPhase(createInitialGameState(), "strategist_action_input");
    construction.units.find((unit) => unit.id === "home-1-strategist")!.role = "builder";
    const action = getStrategistActionCandidates(construction, "team-1")[0];
    expect(saveStrategistActionIntent(construction, action)).not.toBe(construction);
  });

  it("enumerates purely and in stable order independent of backing unit order", () => {
    const state = createInitialGameState();
    addUnit(state, "cpu-order-b", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    addUnit(state, "cpu-order-a", "team-1", "ninja", { kind: "water", x: 4, y: 2 });
    const before = structuredClone(state);
    const first = getTeamMovementCandidates(state, "team-1");
    const second = getTeamMovementCandidates(state, "team-1");
    expect(state).toEqual(before);
    expect(second).toEqual(first);

    const reordered = structuredClone(state);
    reordered.units.reverse();
    expect(getTeamMovementCandidates(reordered, "team-1")).toEqual(first);
  });
});
