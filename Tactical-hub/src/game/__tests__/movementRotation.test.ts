import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { saveAttackIntent } from "../engine/battle";
import {
  beginMovementPhase,
  getMovementCandidates,
  saveMovementIntent,
  submitMovement,
} from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";
import { positionKey } from "../utils/position";

function clearSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((candidate) => candidate.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function relocate(state: GameState, unitId: string, position: UnitPosition) {
  const unit = state.units.find((candidate) => candidate.id === unitId)!;
  clearSlot(state, unit.position);
  unit.position = position;
  if (position.kind === "base") {
    const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((candidate) => candidate.id === position.slotId);
    if (slot) slot.unitId = unit.id;
  }
  return unit;
}

function addUnit(state: GameState, id: string, teamId: string, type: Unit["type"], position: UnitPosition) {
  const unit: Unit = { id, ownerTeamId: teamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function finishMovementRound(state: GameState) {
  let next = state;
  while (next.phase === "movement_input" && next.currentMovementTeamId)
    next = submitMovement(next, next.currentMovementTeamId);
  return next;
}

describe("rotating sequential team movement", () => {
  it("rotates the stable four-seat order every turn and removes inactive teams", () => {
    const first = createInitialGameState();
    expect(first.movementOrderTeamIds).toEqual(["team-1", "team-2", "team-3", "team-4"]);
    const afterFirst = finishMovementRound(first);
    expect(afterFirst.phase).toBe("attack_input");
    expect(afterFirst.movementOrderTeamIds).toEqual(["team-2", "team-3", "team-4", "team-1"]);

    const second = beginMovementPhase(afterFirst);
    expect(second.currentMovementTeamId).toBe("team-2");
    const afterSecond = finishMovementRound(second);
    expect(afterSecond.movementOrderTeamIds).toEqual(["team-3", "team-4", "team-1", "team-2"]);

    const third = beginMovementPhase(afterSecond);
    third.teams.find((team) => team.id === "team-4")!.status = "defeated";
    const filtered = beginMovementPhase(third);
    expect(filtered.movementOrderTeamIds).toEqual(["team-3", "team-1", "team-2"]);
  });

  it("advances on pass and enters attack input only after every active team completes", () => {
    let state = createInitialGameState();
    state = submitMovement(state, "team-1");
    expect(state).toMatchObject({ phase: "movement_input", currentMovementTeamId: "team-2", movementCompletedTeamIds: ["team-1"] });
    state = submitMovement(state, "team-2");
    state = submitMovement(state, "team-3");
    expect(state.phase).toBe("movement_input");
    state = submitMovement(state, "team-4");
    expect(state.phase).toBe("attack_input");
    expect(state.currentMovementTeamId).toBeUndefined();
  });

  it("accepts movement input only from the current team", () => {
    const state = createInitialGameState();
    const enemy = state.units.find((unit) => unit.id === "home-2-strategist")!;
    expect(getMovementCandidates(state, enemy.id)).toEqual([]);
    const rejected = saveMovementIntent(state, { teamId: "team-2", unitId: enemy.id, from: enemy.position, to: enemy.position, stay: true });
    expect(rejected).toBe(state);
    expect(submitMovement(state, "team-2")).toBe(state);
  });

  it("recomputes later-team candidates from the updated occupied and vacated board", () => {
    let occupied = createInitialGameState();
    relocate(occupied, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    relocate(occupied, "home-2-strategist", { kind: "tile", x: 6, y: 1 });
    occupied = saveMovementIntent(occupied, {
      teamId: "team-1", unitId: "home-1-strategist",
      from: { kind: "tile", x: 4, y: 1 }, to: { kind: "tile", x: 5, y: 1 }, stay: false,
    });
    occupied = submitMovement(occupied, "team-1");
    expect(occupied.units.find((unit) => unit.id === "home-1-strategist")?.position).toEqual({ kind: "tile", x: 5, y: 1 });
    expect(getMovementCandidates(occupied, "home-2-strategist").map(positionKey)).not.toContain("5,1");

    let vacated = createInitialGameState();
    relocate(vacated, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    relocate(vacated, "home-2-strategist", { kind: "tile", x: 5, y: 1 });
    vacated = saveMovementIntent(vacated, {
      teamId: "team-1", unitId: "home-1-strategist",
      from: { kind: "tile", x: 4, y: 1 }, to: { kind: "tile", x: 3, y: 1 }, stay: false,
    });
    vacated = submitMovement(vacated, "team-1");
    expect(getMovementCandidates(vacated, "home-2-strategist").map(positionKey)).toContain("4,1");
  });

  it("prevents cavalry swaps because the later cavalry sees the resolved board", () => {
    let state = createInitialGameState();
    const first = addUnit(state, "team-1-sequential-cavalry", "team-1", "cavalry", { kind: "tile", x: 4, y: 1 });
    const second = addUnit(state, "team-2-sequential-cavalry", "team-2", "cavalry", { kind: "tile", x: 6, y: 1 });
    state = saveMovementIntent(state, { teamId: "team-1", unitId: first.id, from: first.position, to: { kind: "tile", x: 5, y: 1 }, stay: false });
    state = submitMovement(state, "team-1");
    expect(getMovementCandidates(state, second.id).map(positionKey)).not.toContain("5,1");
  });

  it("updates an ownerless base before the later team acts without creating a reward unit", () => {
    let state = createInitialGameState();
    const base = state.bases.find((candidate) => candidate.id === "neutral-north")!;
    for (const slot of base.slots) {
      if (slot.unitId) {
        const defender = state.units.find((unit) => unit.id === slot.unitId);
        if (defender) defender.position = { kind: "removed", reason: "defeated" };
      }
      slot.unitId = undefined;
    }
    base.ownerTeamId = "neutral";
    const mover = state.units.find((unit) => unit.id === "home-1-strategist")!;
    clearSlot(state, mover.position);
    let destination: UnitPosition | undefined;
    let source: UnitPosition | undefined;
    for (const tile of state.map.tiles.filter((candidate) => ["road", "baseGate", "reorganize"].includes(candidate.terrain))) {
      mover.position = { kind: "tile", x: tile.x, y: tile.y };
      destination = getMovementCandidates(state, mover.id).find((candidate) => candidate.kind === "base" && candidate.baseId === base.id);
      if (destination) { source = mover.position; break; }
    }
    if (!destination || !source) throw new Error("No neutral-base entry fixture found");
    const rewardsBefore = state.rewardPlacementRequests.length;
    state = saveMovementIntent(state, { teamId: "team-1", unitId: mover.id, from: source, to: destination, stay: false });
    state = submitMovement(state, "team-1");
    expect(state.bases.find((candidate) => candidate.id === base.id)?.ownerTeamId).toBe("team-1");
    expect(state.rewardPlacementRequests).toHaveLength(rewardsBefore);

    const later = relocate(state, "home-2-strategist", source);
    expect(getMovementCandidates(state, later.id).some((candidate) => candidate.kind === "base" && candidate.baseId === base.id)).toBe(false);
  });

  it("keeps same-team destination reservations and legal chained movement", () => {
    let state = createInitialGameState();
    const front = addUnit(state, "a-front", "team-1", "infantry", { kind: "tile", x: 4, y: 1 });
    const back = addUnit(state, "b-back", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    state = saveMovementIntent(state, { teamId: "team-1", unitId: front.id, from: front.position, to: { kind: "tile", x: 5, y: 1 }, stay: false });
    expect(getMovementCandidates(state, back.id).map(positionKey)).toContain("4,1");
    state = saveMovementIntent(state, { teamId: "team-1", unitId: back.id, from: back.position, to: { kind: "tile", x: 4, y: 1 }, stay: false });
    const duplicate = addUnit(state, "c-duplicate", "team-1", "infantry", { kind: "tile", x: 6, y: 1 });
    const rejected = saveMovementIntent(state, { teamId: "team-1", unitId: duplicate.id, from: duplicate.position, to: { kind: "tile", x: 5, y: 1 }, stay: false });
    expect(rejected).toBe(state);

    const resolved = submitMovement(state, "team-1");
    expect(resolved.units.find((unit) => unit.id === front.id)?.position).toEqual({ kind: "tile", x: 5, y: 1 });
    expect(resolved.units.find((unit) => unit.id === back.id)?.position).toEqual({ kind: "tile", x: 4, y: 1 });
  });

  it("opens simultaneous attack input for every team only after movement finishes", () => {
    let state = finishMovementRound(createInitialGameState());
    expect(state.phase).toBe("attack_input");
    state = saveAttackIntent(state, { teamId: "team-1", attackerUnitId: "home-1-king", pass: true });
    state = saveAttackIntent(state, { teamId: "team-2", attackerUnitId: "home-2-king", pass: true });
    expect(state.turnState.actionIntents.flatMap((intent) => intent.attackIntents).map((intent) => intent.teamId)).toEqual(expect.arrayContaining(["team-1", "team-2"]));
  });
});
