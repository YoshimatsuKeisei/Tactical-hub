import { describe, expect, it } from "vitest";
import { getMovementCandidates, isLegalDestination, resolveMovement, saveMovementIntent } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { GameState, UnitPosition } from "../types";
import { positionKey } from "../utils/position";

function putUnitOnTile(state: GameState, unitId: string, position: UnitPosition) {
  const unit = state.units.find((candidate) => candidate.id === unitId)!;
  const previous = unit.position;
  if (previous.kind === "base") {
    const base = state.bases.find((candidate) => candidate.id === previous.baseId)!;
    const slot = base.slots.find((candidate) => candidate.id === previous.slotId)!;
    slot.unitId = undefined;
  }
  unit.position = position;
}

describe("movement", () => {
  it("does not allow normal units to move onto lake cells", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });

    expect(getMovementCandidates(state, "home-1-strategist").map(positionKey)).not.toContain("4,2");
  });

  it("allows ninjas to move onto lake cells", () => {
    const state = createInitialGameState();
    const ninja = {
      id: "team-1-ninja-test",
      ownerTeamId: "team-1",
      type: "ninja" as const,
      hp: 1,
      position: { kind: "tile" as const, x: 4, y: 1 },
      statuses: [],
    };
    state.units.push(ninja);

    expect(getMovementCandidates(state, ninja.id).map(positionKey)).toContain("4,2");
  });

  it("does not allow moving onto another unit", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    putUnitOnTile(state, "home-1-king", { kind: "tile", x: 5, y: 1 });

    expect(getMovementCandidates(state, "home-1-strategist").map(positionKey)).not.toContain("5,1");
  });

  it("saving a movement intent does not mutate unit position", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    const before = state.units.find((unit) => unit.id === "home-1-strategist")!.position;
    const planned = saveMovementIntent(state, {
      teamId: "team-1",
      unitId: "home-1-strategist",
      from: before,
      to: { kind: "tile", x: 5, y: 1 },
      stay: false,
    });

    expect(planned.units.find((unit) => unit.id === "home-1-strategist")?.position).toEqual(before);
    expect(planned.turnState.actionIntents[0].movementIntents).toHaveLength(1);
  });

  it("movement resolution applies saved movement intents", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    const planned = saveMovementIntent(state, {
      teamId: "team-1",
      unitId: "home-1-strategist",
      from: { kind: "tile", x: 4, y: 1 },
      to: { kind: "tile", x: 5, y: 1 },
      stay: false,
    });
    const resolved = resolveMovement(planned);

    expect(resolved.units.find((unit) => unit.id === "home-1-strategist")?.position).toEqual({ kind: "tile", x: 5, y: 1 });
  });

  it("fails movement that becomes illegal at resolution time", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    const planned = saveMovementIntent(state, {
      teamId: "team-1",
      unitId: "home-1-strategist",
      from: { kind: "tile", x: 4, y: 1 },
      to: { kind: "tile", x: 5, y: 1 },
      stay: false,
    });
    putUnitOnTile(planned, "home-1-king", { kind: "tile", x: 5, y: 1 });
    const resolved = resolveMovement(planned);

    expect(resolved.units.find((unit) => unit.id === "home-1-strategist")?.position).toEqual({ kind: "tile", x: 4, y: 1 });
    expect(resolved.logs.at(-1)?.message).toContain("failed");
  });

  it("does not treat units inside bases as normal board coordinates", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-1-king")!;

    expect(king.position.kind).toBe("base");
    expect(getMovementCandidates(state, king.id).map(positionKey)).not.toContain("1,1");
  });

  it("does not allow moving onto outside cells", () => {
    const state = createInitialGameState();
    const unit = state.units.find((candidate) => candidate.id === "home-1-infantry")!;

    expect(isLegalDestination(state, unit, { kind: "tile", x: 0, y: 0 })).toBe(false);
  });

  it("treats P, d, and G cells as movable ground", () => {
    const state = createInitialGameState();
    const unit = state.units.find((candidate) => candidate.id === "home-1-infantry")!;

    expect(isLegalDestination(state, unit, { kind: "tile", x: 6, y: 1 })).toBe(true);
    expect(isLegalDestination(state, unit, { kind: "tile", x: 3, y: 3 })).toBe(true);
    expect(isLegalDestination(state, unit, { kind: "tile", x: 3, y: 1 })).toBe(true);
  });
});
