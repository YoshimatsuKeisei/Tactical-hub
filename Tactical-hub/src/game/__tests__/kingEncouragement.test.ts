import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  getEncourageAreaTileKeys,
  getEncourageRadius,
  getEncouragedUnitIds,
  isUnitEncouragedByStrategist,
} from "../engine/encouragement";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";

function move(state: GameState, unit: Unit, position: UnitPosition) {
  if (unit.position.kind === "base") {
    const { baseId, slotId } = unit.position;
    const slot = state.bases.find((base) => base.id === baseId)?.slots.find((candidate) => candidate.id === slotId);
    if (slot) slot.unitId = undefined;
  }
  unit.position = position;
}

function addInfantry(state: GameState, id: string, ownerTeamId: string, x: number, y: number) {
  const unit: Unit = {
    id, ownerTeamId, type: "infantry", hp: UNIT_STATS.infantry.hp,
    position: { kind: "tile", x, y }, statuses: [],
  };
  state.units.push(unit);
  return unit;
}

describe("king encouragement", () => {
  it("uses the strategist radius rule near and away from an owned base", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-1-king")!;

    expect(getEncourageRadius(state, king)).toBe(1);
    move(state, king, { kind: "tile", x: 4, y: 1 });
    expect(getEncourageRadius(state, king)).toBe(1);
    move(state, king, { kind: "tile", x: 5, y: 1 });
    expect(getEncourageRadius(state, king)).toBe(2);
    expect(getEncourageAreaTileKeys(state, king)).toContain("7,1");
  });

  it("encourages living friendly units but not enemies or itself", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-1-king")!;
    move(state, king, { kind: "tile", x: 4, y: 1 });
    const friendly = addInfantry(state, "king-friendly", "team-1", 5, 1);
    const enemy = addInfantry(state, "king-enemy", "team-2", 4, 2);

    expect(isUnitEncouragedByStrategist(state, friendly, king)).toBe(true);
    expect(isUnitEncouragedByStrategist(state, enemy, king)).toBe(false);
    expect(isUnitEncouragedByStrategist(state, king, king)).toBe(false);
  });

  it("stops encouraging after the king dies", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-1-king")!;
    move(state, king, { kind: "tile", x: 4, y: 1 });
    const friendly = addInfantry(state, "king-friendly", "team-1", 5, 1);
    expect(isUnitEncouragedByStrategist(state, friendly, king)).toBe(true);

    king.hp = 0;
    king.position = { kind: "removed", reason: "defeated" };
    expect(isUnitEncouragedByStrategist(state, friendly, king)).toBe(false);
  });

  it("keeps overlapping king and strategist encouragement boolean", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-1-king")!;
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    move(state, king, { kind: "tile", x: 4, y: 1 });
    move(state, strategist, { kind: "tile", x: 6, y: 1 });
    addInfantry(state, "overlap-friendly", "team-1", 5, 1);

    expect([...getEncouragedUnitIds(state)].filter((unitId) => unitId === "overlap-friendly")).toHaveLength(1);
  });
});
