import { describe, expect, it } from "vitest";
import { getAttackCandidates, resolveBattle } from "../engine/battle";
import { getHeavyInfantryMergeCandidates, mergeHeavyInfantry } from "../engine/heavyInfantry";
import { getMovementCandidates } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";

function addToBase(state: GameState, id: string, slotId: string) {
  const base = state.bases.find((candidate) => candidate.id === "home-1")!;
  const slot = base.slots.find((candidate) => candidate.id === slotId)!;
  const unit: Unit = { id, ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "base", baseId: base.id, slotId }, statuses: [] };
  slot.unitId = id;
  state.units.push(unit);
  return unit;
}

function attackFixture(targetType: UnitType, encouraged = false, heavyTarget = false) {
  const state = createInitialGameState();
  state.phase = state.turnState.phase = "attack_input";
  state.units.forEach((unit) => { unit.hp = 0; unit.position = { kind: "removed", reason: "defeated" }; });
  const attacker: Unit = { id: "heavy", ownerTeamId: "team-1", type: "infantry", formation: "heavy", hp: 2, position: { kind: "tile", x: 4, y: 1 }, statuses: [] };
  const target: Unit = { id: "target", ownerTeamId: "team-2", type: targetType, formation: heavyTarget ? "heavy" : undefined, hp: heavyTarget ? 2 : 1, position: { kind: "tile", x: 5, y: 1 }, statuses: [] };
  state.units.push(attacker, target);
  if (encouraged) state.units.push({ id: "encourager", ownerTeamId: "team-1", type: "strategist", role: "encourage", hp: 1, position: { kind: "tile", x: 3, y: 1 }, statuses: [] });
  return { state, attacker, target };
}

function positionedFixture(firstPosition: UnitPosition, secondPosition: UnitPosition) {
  const state = createInitialGameState();
  const removedIds = new Set(state.units.filter((unit) => unit.ownerTeamId === "team-1").map((unit) => unit.id));
  state.units = state.units.filter((unit) => !removedIds.has(unit.id));
  for (const base of state.bases) for (const slot of base.slots) if (slot.unitId && removedIds.has(slot.unitId)) slot.unitId = undefined;
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  for (const position of [firstPosition, secondPosition]) {
    if (position.kind !== "tile") continue;
    const tile = state.map.tiles.find((candidate) => candidate.x === position.x && candidate.y === position.y);
    if (tile) Object.assign(tile, { terrain: "road", symbol: "", roadSectionId: "merge-fixture", baseId: undefined });
  }
  const first: Unit = { id: "merge-a", ownerTeamId: "team-1", type: "infantry", hp: 1, position: firstPosition, statuses: [] };
  const second: Unit = { id: "merge-b", ownerTeamId: "team-1", type: "infantry", hp: 1, position: secondPosition, statuses: [] };
  state.units.push(first, second);
  return { state, first, second };
}

describe("heavy infantry", () => {
  it("merges two eligible infantry into one moved HP2 unit and frees one slot", () => {
    const state = createInitialGameState();
    const first = addToBase(state, "merge-a", "slot_0_1");
    const second = addToBase(state, "merge-b", "slot_1_1");
    expect(getHeavyInfantryMergeCandidates(state, first.id).map((unit) => unit.id)).toEqual([second.id]);
    const merged = mergeHeavyInfantry(state, first.id, second.id);
    expect(merged.units.find((unit) => unit.id === first.id)).toMatchObject({ type: "infantry", formation: "heavy", hp: 2 });
    expect(merged.units.find((unit) => unit.id === second.id)).toMatchObject({ hp: 0, position: { kind: "removed", reason: "merged" } });
    expect(merged.bases.find((base) => base.id === "home-1")?.slots.find((slot) => slot.id === "slot_1_1")?.unitId).toBeUndefined();
    expect(getMovementCandidates(merged, first.id)).toEqual([]);
  });

  it.each([
    [-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1],
  ])("allows all eight adjacent directions (%i,%i)", (dx, dy) => {
    const { state, first, second } = positionedFixture(
      { kind: "tile", x: 10, y: 10 },
      { kind: "tile", x: 10 + dx, y: 10 + dy },
    );
    expect(getHeavyInfantryMergeCandidates(state, first.id).map((unit) => unit.id)).toEqual([second.id]);
  });

  it("allows adjacent road and active-bridge infantry but rejects distance two", () => {
    const road = positionedFixture({ kind: "tile", x: 4, y: 1 }, { kind: "tile", x: 5, y: 1 });
    expect(getHeavyInfantryMergeCandidates(road.state, road.first.id).map((unit) => unit.id)).toEqual([road.second.id]);
    road.second.position = { kind: "tile", x: 6, y: 1 };
    expect(getHeavyInfantryMergeCandidates(road.state, road.first.id)).toEqual([]);

    const bridge = positionedFixture(
      { kind: "bridge", bridgeId: "merge-bridge", cellIndex: 0 },
      { kind: "bridge", bridgeId: "merge-bridge", cellIndex: 1 },
    );
    bridge.state.constructions.push({ id: "merge-bridge", kind: "bridge", tiles: [{ x: 4, y: 2 }, { x: 5, y: 2 }], placedTurn: 1, active: true });
    expect(getHeavyInfantryMergeCandidates(bridge.state, bridge.first.id).map((unit) => unit.id)).toEqual([bridge.second.id]);
  });

  it("allows moved infantry while retaining the existing special-state prohibitions", () => {
    const state = createInitialGameState();
    const first = addToBase(state, "merge-a", "slot_0_1");
    const second = addToBase(state, "merge-b", "slot_1_1");
    state.movedUnitIdsThisMovementPhase.push(second.id);
    expect(getHeavyInfantryMergeCandidates(state, first.id).map((unit) => unit.id)).toEqual([second.id]);
    state.movedUnitIdsThisMovementPhase.push(first.id);
    expect(getHeavyInfantryMergeCandidates(state, first.id).map((unit) => unit.id)).toEqual([second.id]);
    second.formation = "heavy";
    expect(getHeavyInfantryMergeCandidates(state, first.id)).toEqual([]);
  });

  it.each([
    ["enemy", (unit: Unit) => { unit.ownerTeamId = "team-2"; }],
    ["non-infantry", (unit: Unit) => { unit.type = "archer"; }],
    ["dead", (unit: Unit) => { unit.hp = 0; }],
    ["removed", (unit: Unit) => { unit.position = { kind: "removed", reason: "defeated" }; }],
    ["retreating", (unit: Unit) => { unit.statuses = [{ kind: "retreating", retreatTargetBaseId: "home-1" }]; }],
    ["heavy", (unit: Unit) => { unit.formation = "heavy"; }],
  ] as const)("rejects a %s partner", (_label, alter) => {
    const { state, first, second } = positionedFixture({ kind: "tile", x: 4, y: 1 }, { kind: "tile", x: 5, y: 1 });
    alter(second);
    expect(getHeavyInfantryMergeCandidates(state, first.id)).toEqual([]);
  });

  it("can attack once in the following attack phase after merging", () => {
    let state = createInitialGameState();
    const first = addToBase(state, "merge-a", "slot_0_1");
    const second = addToBase(state, "merge-b", "slot_1_1");
    state = mergeHeavyInfantry(state, first.id, second.id);
    const retained = state.units.find((unit) => unit.id === first.id)!;
    retained.position = { kind: "tile", x: 4, y: 1 };
    const enemy: Unit = { id: "merge-enemy", ownerTeamId: "team-2", type: "archer", hp: 1, position: { kind: "tile", x: 5, y: 1 }, statuses: [] };
    state.units.push(enemy);
    state.phase = state.turnState.phase = "attack_input";
    expect(getAttackCandidates(state, retained.id).map((target) => target.unitId)).toEqual([enemy.id]);
    expect(state.units.filter((unit) => unit.id === retained.id && unit.position.kind !== "removed")).toHaveLength(1);
  });

  it.each([
    ["infantry", false, 5], ["archer", false, 4], ["engineer", false, 4], ["ninja", false, 4],
    ["strategist", false, 4], ["cavalry", false, 6], ["king", false, 6], ["infantry", true, 6],
  ] as const)("uses the heavy attack denominator against %s (heavy=%s)", (targetType, heavyTarget, expected) => {
    const { state, attacker } = attackFixture(targetType, false, heavyTarget);
    expect(getAttackCandidates(state, attacker.id)[0]?.baseSuccessDenominator).toBe(expected);
  });

  it("applies encouragement after the heavy denominator and takes damage like infantry", () => {
    const encouraged = attackFixture("archer", true);
    expect(getAttackCandidates(encouraged.state, encouraged.attacker.id)[0]?.finalSuccessDenominator).toBe(3);
    const attacked = attackFixture("cavalry");
    attacked.state.turnState.actionIntents = [{ teamId: "team-2", productionChoices: [], movementIntents: [], attackIntents: [{ teamId: "team-2", attackerUnitId: attacked.target.id, target: { kind: "unit", unitId: attacked.attacker.id }, pass: false }] }];
    const resolved = resolveBattle(attacked.state, () => 0);
    expect(resolved.units.find((unit) => unit.id === attacked.attacker.id)).toMatchObject({ hp: 1, formation: "heavy" });
  });
});
