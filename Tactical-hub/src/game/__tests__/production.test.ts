import { describe, expect, it } from "vitest";
import { getAvailableProductionTypes, resolveProduction, saveProductionChoice } from "../engine/production";
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
});
