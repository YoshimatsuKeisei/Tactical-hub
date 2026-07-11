import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  getEncourageRadius,
  getEncouragedUnitIds,
  getEncouragedUnitIdsByStrategist,
  isUnitEncouraged,
} from "../engine/encouragement";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";
import { chebyshevDistance } from "../utils/distance";

function clearPreviousSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const base = state.bases.find((candidate) => candidate.id === position.baseId)!;
  const slot = base.slots.find((candidate) => candidate.id === position.slotId)!;
  slot.unitId = undefined;
}

function putUnit(state: GameState, id: string, position: UnitPosition) {
  const unit = state.units.find((candidate) => candidate.id === id)!;
  clearPreviousSlot(state, unit.position);
  unit.position = position;
  if (position.kind === "base") {
    const base = state.bases.find((candidate) => candidate.id === position.baseId)!;
    const slot = base.slots.find((candidate) => candidate.id === position.slotId)!;
    slot.unitId = id;
  }
  return unit;
}

function addUnit(state: GameState, id: string, ownerTeamId: string, type: UnitType, position: UnitPosition): Unit {
  const unit: Unit = { id, ownerTeamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  if (position.kind === "base") {
    const base = state.bases.find((candidate) => candidate.id === position.baseId)!;
    const slot = base.slots.find((candidate) => candidate.id === position.slotId)!;
    slot.unitId = id;
  }
  return unit;
}

function minDistanceToAnyBase(state: GameState, position: { x: number; y: number }) {
  return Math.min(...state.bases.flatMap((base) => base.coords.map((coord) => chebyshevDistance(position, coord))));
}

function tileAtBaseDistance(state: GameState, distance: number) {
  const tile = state.map.tiles.find((candidate) => minDistanceToAnyBase(state, candidate) === distance);
  if (!tile) throw new Error(`No tile at base distance ${distance}`);
  return { kind: "tile" as const, x: tile.x, y: tile.y };
}

describe("encouragement", () => {
  it("uses radius 1 for encourage strategists inside a base", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;

    expect(getEncourageRadius(state, strategist)).toBe(1);
  });

  it("uses radius 1 for encourage strategists 1 or 2 cells from a base", () => {
    const state = createInitialGameState();
    const strategist = putUnit(state, "home-1-strategist", tileAtBaseDistance(state, 1));
    expect(getEncourageRadius(state, strategist)).toBe(1);

    strategist.position = tileAtBaseDistance(state, 2);
    expect(getEncourageRadius(state, strategist)).toBe(1);
  });

  it("uses radius 2 for encourage strategists at least 3 cells from every base", () => {
    const state = createInitialGameState();
    const strategist = putUnit(state, "home-1-strategist", tileAtBaseDistance(state, 3));

    expect(getEncourageRadius(state, strategist)).toBe(2);
  });

  it("encourages friendly ground units in range but not units outside range", () => {
    const state = createInitialGameState();
    const strategist = putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    strategist.role = "encourage";
    addUnit(state, "friendly-near", "team-1", "infantry", { kind: "tile", x: 5, y: 1 });
    addUnit(state, "friendly-far", "team-1", "infantry", { kind: "tile", x: 7, y: 1 });

    expect(isUnitEncouraged(state, state.units.find((unit) => unit.id === "friendly-near")!)).toBe(true);
    expect(isUnitEncouraged(state, state.units.find((unit) => unit.id === "friendly-far")!)).toBe(false);
  });

  it("does not encourage enemies, neutral units, water units, removed units, or the strategist itself", () => {
    const state = createInitialGameState();
    const strategist = putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    strategist.role = "encourage";
    addUnit(state, "enemy-near", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    addUnit(state, "neutral-near", "neutral", "infantry", { kind: "tile", x: 4, y: 2 });
    addUnit(state, "water-near", "team-1", "ninja", { kind: "water", x: 5, y: 2 });
    addUnit(state, "removed-near", "team-1", "infantry", { kind: "removed", reason: "defeated" });

    const encouragedIds = getEncouragedUnitIds(state);

    expect(encouragedIds.has("enemy-near")).toBe(false);
    expect(encouragedIds.has("neutral-near")).toBe(false);
    expect(encouragedIds.has("water-near")).toBe(false);
    expect(encouragedIds.has("removed-near")).toBe(false);
    expect(encouragedIds.has("home-1-strategist")).toBe(false);
  });

  it("encourages friendly units in the same base", () => {
    const state = createInitialGameState();

    expect(getEncouragedUnitIdsByStrategist(state, state.units.find((unit) => unit.id === "home-1-strategist")!)).toContain("home-1-king");
  });

  it("treats overlapping encourage ranges as a boolean effect", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    const second = addUnit(state, "team-1-strategist-extra", "team-1", "strategist", { kind: "tile", x: 6, y: 1 });
    second.role = "encourage";
    addUnit(state, "friendly-overlap", "team-1", "infantry", { kind: "tile", x: 5, y: 1 });

    expect([...getEncouragedUnitIds(state)].filter((unitId) => unitId === "friendly-overlap")).toHaveLength(1);
  });
});
