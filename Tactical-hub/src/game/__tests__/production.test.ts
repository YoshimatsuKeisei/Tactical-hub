import { describe, expect, it } from "vitest";
import { getAvailableProductionTypes, getProductionCandidates, resolveProduction, saveProductionChoice, submitTeamProduction } from "../engine/production";
import { getMovementCandidates, saveMovementIntent, submitMovement } from "../engine/movement";
import { isProductionTurn } from "../engine/productionSchedule";
import { createInitialGameState } from "../initialState";
import { UNIT_STATS } from "../constants";
import type { GameState, UnitType } from "../types";

function fillHomeBase(state: GameState) {
  const base = state.bases.find((candidate) => candidate.id === "home-1")!;
  base.slots
    .filter((slot) => !slot.unitId)
    .forEach((slot, index) => {
      const unitType: UnitType = index === 0 ? "infantry" : "archer";
      const unit = {
        id: `home-1-fill-${index}`,
        ownerTeamId: "team-1",
        type: unitType,
        hp: UNIT_STATS[unitType].hp,
        position: { kind: "base" as const, baseId: base.id, slotId: slot.id },
        statuses: [],
      };
      slot.unitId = unit.id;
      state.units.push(unit);
    });
  return state;
}

describe("production", () => {
  it.each([
    [1, true], [2, false], [5, false], [6, true], [10, false], [11, true],
  ])("offers the movement-phase production step on turn %i: %s", (turnNumber, expected) => {
    const state = createInitialGameState();
    state.turnNumber = state.turnState.turnNumber = turnNumber;
    expect(isProductionTurn(state)).toBe(expected);
    expect(getProductionCandidates(state, "team-1").length > 0).toBe(expected);
    expect(getProductionCandidates(state, "team-2")).toEqual([]);
  });

  it("produces before movement and lets the new unit move immediately", () => {
    let state = createInitialGameState();
    const unitIdsBefore = new Set(state.units.map((unit) => unit.id));
    expect(getMovementCandidates(state, "home-1-king").length).toBeGreaterThan(0);
    state = saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "infantry" });
    state = submitTeamProduction(state, "team-1");
    const produced = state.units.find((unit) => !unitIdsBefore.has(unit.id))!;
    const destination = getMovementCandidates(state, produced.id)[0];
    expect(destination).toBeDefined();
    state = saveMovementIntent(state, { teamId: "team-1", unitId: produced.id, from: produced.position, to: destination, stay: false });
    state = submitMovement(state, "team-1");
    expect(state.units.find((unit) => unit.id === produced.id)?.position).toEqual(destination);
  });

  it("allows an explicit production pass before movement", () => {
    const state = createInitialGameState();
    expect(submitMovement(state, "team-1").currentMovementTeamId).toBe("team-2");
    const passed = submitTeamProduction(state, "team-1");
    expect(passed.productionCompletedTeamIdsThisTurn).toContain("team-1");
    expect(submitMovement(passed, "team-1").currentMovementTeamId).toBe("team-2");
  });

  it("places produced units inside a base slot", () => {
    const state = createInitialGameState();
    const planned = saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "infantry" });
    const resolved = resolveProduction(planned);
    const produced = resolved.units.find((unit) => unit.id.startsWith("home-1-infantry-"));

    expect(produced?.position).toEqual({ kind: "base", baseId: "home-1", slotId: "slot_0_1" });
    expect(resolved.bases.find((base) => base.id === "home-1")?.slots.find((slot) => slot.id === "slot_0_1")?.unitId).toBe(
      produced?.id,
    );
  });

  it("keeps produced units as base positions", () => {
    const state = createInitialGameState();
    const resolved = resolveProduction(
      saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "cavalry" }),
    );
    const produced = resolved.units.find((unit) => unit.id.startsWith("home-1-cavalry-"));

    expect(produced?.position.kind).toBe("base");
  });

  it("does not allow production into a full base", () => {
    const state = fillHomeBase(createInitialGameState());
    expect(getAvailableProductionTypes(state, "team-1", "home-1")).toEqual([]);
    const resolved = resolveProduction(saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "infantry" }));

    expect(resolved.units).toHaveLength(state.units.length);
    expect(resolved.logs.at(-1)?.message).toContain("Production failed");
  });

  it("does not list kings or apprentice ninjas as production choices", () => {
    const state = createInitialGameState();

    expect(getAvailableProductionTypes(state, "team-1", "home-1")).not.toContain("king");
    expect(getAvailableProductionTypes(state, "team-1", "home-1")).not.toContain("apprentice_ninja");
  });

  it("produces replacement strategists as encourage strategists", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    const home = state.bases.find((base) => base.id === "home-1")!;
    const oldSlot = home.slots.find((slot) => slot.unitId === strategist.id)!;
    oldSlot.unitId = undefined;
    strategist.position = { kind: "removed", reason: "defeated" };
    strategist.hp = 0;

    const resolved = resolveProduction(saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "strategist" }));
    const produced = resolved.units.find((unit) => unit.id.startsWith("home-1-strategist-"));

    expect(produced?.role).toBe("encourage");
  });

  it("exposes each strategist role as a distinct production choice and preserves the selected role", () => {
    let state = createInitialGameState();
    const existing = state.units.find((unit) => unit.id === "home-1-strategist")!;
    const slot = state.bases.find((base) => base.id === "home-1")!.slots.find((candidate) => candidate.unitId === existing.id)!;
    slot.unitId = undefined;
    existing.hp = 0;
    existing.position = { kind: "removed", reason: "defeated" };
    const roles = getProductionCandidates(state, "team-1").filter((choice) => choice.unitType === "strategist").map((choice) => choice.strategistRole).sort();
    expect(roles).toEqual(["builder", "encourage", "teleporter"]);
    state = saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "strategist", strategistRole: "builder" });
    state = submitTeamProduction(state, "team-1");
    expect(state.units.find((unit) => unit.id.startsWith("home-1-strategist-") && unit.hp > 0)?.role).toBe("builder");
  });
});
