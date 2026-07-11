import { describe, expect, it } from "vitest";
import { resolveProduction, saveProductionChoice } from "../engine/production";
import { createInitialGameState } from "../initialState";

const expectedKingSlots = {
  "home-1": "slot_0_0",
  "home-2": "slot_0_1",
  "home-3": "slot_1_1",
  "home-4": "slot_1_0",
};

describe("initial base slots", () => {
  it("creates four local slots for every base", () => {
    const state = createInitialGameState();

    for (const base of state.bases) {
      expect(base.slots).toHaveLength(4);
      expect(new Set(base.slots.map((slot) => `${slot.localRow}/${slot.localCol}`))).toEqual(
        new Set(["0/0", "0/1", "1/0", "1/1"]),
      );
    }
  });

  it("starts each home base with only a king and strategist", () => {
    const state = createInitialGameState();

    for (const base of state.bases.filter((candidate) => candidate.type === "home")) {
      const units = state.units.filter((unit) => unit.position.kind === "base" && unit.position.baseId === base.id);
      expect(units.map((unit) => unit.type).sort()).toEqual(["king", "strategist"]);
    }
  });

  it("places each king in the outer safe slot and each strategist elsewhere", () => {
    const state = createInitialGameState();

    for (const [baseId, kingSlotId] of Object.entries(expectedKingSlots)) {
      const king = state.units.find((unit) => unit.id === `${baseId}-king`);
      const strategist = state.units.find((unit) => unit.id === `${baseId}-strategist`);

      expect(king?.position).toEqual({ kind: "base", baseId, slotId: kingSlotId });
      expect(strategist?.position.kind).toBe("base");
      expect(strategist?.position.kind === "base" ? strategist.position.slotId : undefined).not.toBe(kingSlotId);
    }
  });

  it("sets every initial strategist as an encourage strategist", () => {
    const state = createInitialGameState();

    for (const base of state.bases.filter((candidate) => candidate.type === "home")) {
      const strategist = state.units.find((unit) => unit.id === `${base.id}-strategist`);

      expect(strategist?.type).toBe("strategist");
      expect(strategist?.role).toBe("encourage");
    }
  });

  it("keeps exactly one unit id per occupied BaseSlot", () => {
    const state = createInitialGameState();
    const slotUnitIds = state.bases.flatMap((base) => base.slots.flatMap((slot) => (slot.unitId ? [slot.unitId] : [])));

    expect(new Set(slotUnitIds).size).toBe(slotUnitIds.length);
    expect(slotUnitIds.every((unitId) => state.units.some((unit) => unit.id === unitId))).toBe(true);
  });

  it("starts every home base with two empty slots", () => {
    const state = createInitialGameState();

    for (const base of state.bases.filter((candidate) => candidate.type === "home")) {
      expect(base.slots.filter((slot) => !slot.unitId)).toHaveLength(2);
    }
  });

  it("produces into empty BaseSlots without sharing a slot", () => {
    const state = createInitialGameState();
    const afterFirst = resolveProduction(
      saveProductionChoice(state, { teamId: "team-1", baseId: "home-1", unitType: "infantry" }),
    );
    const afterSecond = resolveProduction(
      saveProductionChoice(afterFirst, { teamId: "team-1", baseId: "home-1", unitType: "cavalry" }),
    );
    const home = afterSecond.bases.find((base) => base.id === "home-1")!;
    const homeUnits = afterSecond.units.filter((unit) => unit.position.kind === "base" && unit.position.baseId === "home-1");
    const slotIds = homeUnits.map((unit) => (unit.position.kind === "base" ? unit.position.slotId : ""));

    expect(home.slots.filter((slot) => !slot.unitId)).toHaveLength(0);
    expect(new Set(slotIds).size).toBe(homeUnits.length);
  });
});
