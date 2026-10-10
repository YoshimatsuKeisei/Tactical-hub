import { describe, expect, it } from "vitest";
import { createInitialGameState } from "../game/initialState";
import type { AttackIntent, Unit } from "../game/types";
import {
  createUnitVisualPresentation,
  getUnitSpriteKind,
} from "./unitVisualEvents";

function unit(overrides: Partial<Unit> = {}): Unit {
  return {
    id: "king",
    ownerTeamId: "team-1",
    type: "king",
    hp: 3,
    position: { kind: "tile", x: 3, y: 3 },
    statuses: [],
    ...overrides,
  };
}

describe("generic animated-unit routing", () => {
  it("maps king, every strategist role, and engineer without changing other units", () => {
    expect(getUnitSpriteKind(unit({ type: "king" }))).toBe("6Crusader");
    for (const role of ["builder", "encourage", "teleporter"] as const) {
      expect(getUnitSpriteKind(unit({ type: "strategist", role }))).toBe("10Caster");
    }
    expect(getUnitSpriteKind(unit({ type: "engineer" }))).toBe("catapult");
    expect(getUnitSpriteKind(unit({ type: "infantry" }))).toBeUndefined();
  });
});

describe("generic battle presentation events", () => {
  it("emits ordered attacks for king, strategist, and engineer only", () => {
    const before = createInitialGameState();
    const king = unit();
    const strategist = unit({ id: "strategist", type: "strategist", role: "builder", hp: 1, position: { kind: "tile", x: 4, y: 4 } });
    const engineer = unit({ id: "engineer", type: "engineer", hp: 1, position: { kind: "tile", x: 5, y: 5 } });
    const infantry = unit({ id: "infantry", type: "infantry", hp: 1, position: { kind: "tile", x: 6, y: 6 } });
    const target = unit({ id: "target", type: "cavalry", hp: 1, ownerTeamId: "team-2", position: { kind: "tile", x: 8, y: 3 } });
    before.units = [king, strategist, engineer, infantry, target];
    const intents: AttackIntent[] = [king, strategist, engineer, infantry].map((attacker) => ({
      teamId: attacker.ownerTeamId,
      attackerUnitId: attacker.id,
      target: { kind: "unit", unitId: target.id },
      pass: false,
    }));
    expect(createUnitVisualPresentation(before, structuredClone(before), intents).events.map((event) => event.unitId))
      .toEqual(["king", "strategist", "engineer"]);
  });

  it("emits hit for surviving HP loss on king and strategist", () => {
    const before = createInitialGameState();
    before.units = [unit(), unit({ id: "strategist", type: "strategist", role: "teleporter", hp: 2 })];
    const after = structuredClone(before);
    after.units[0].hp = 2;
    after.units[1].hp = 1;
    expect(createUnitVisualPresentation(before, after, []).events.map((event) => [event.unitId, event.kind]))
      .toEqual([["king", "hit"], ["strategist", "hit"]]);
  });

  it("creates presentation-only death overlays at the old position", () => {
    const before = createInitialGameState();
    before.units = [unit({ id: "strategist", type: "strategist", role: "encourage", hp: 1, position: { kind: "tile", x: 9, y: 7 } })];
    const after = structuredClone(before);
    after.units[0].hp = 0;
    after.units[0].position = { kind: "removed", reason: "defeated" };
    const presentation = createUnitVisualPresentation(before, after, []);
    expect(presentation.events).toMatchObject([{ unitId: "strategist", kind: "death" }]);
    expect(presentation.deathOverlays).toMatchObject([{
      unit: { id: "strategist", type: "strategist" },
      coord: { x: 9, y: 7 },
      events: [{ kind: "death" }],
    }]);
    expect(after.units[0].position.kind).toBe("removed");
  });

  it("creates an engineer death overlay that the Catapult renderer maps to break", () => {
    const before = createInitialGameState();
    before.units = [unit({ id: "engineer", type: "engineer", hp: 1 })];
    const after = structuredClone(before);
    after.units[0].position = { kind: "removed", reason: "defeated" };
    expect(createUnitVisualPresentation(before, after, []).deathOverlays).toMatchObject([{
      unit: { id: "engineer", type: "engineer" },
      events: [{ kind: "death" }],
    }]);
  });

  it("preserves attack before death for a unit defeated in the same battle", () => {
    const before = createInitialGameState();
    const king = unit();
    const target = unit({ id: "target", ownerTeamId: "team-2", position: { kind: "tile", x: 4, y: 3 } });
    before.units = [king, target];
    const after = structuredClone(before);
    after.units[0].position = { kind: "removed", reason: "defeated" };
    const presentation = createUnitVisualPresentation(before, after, [{
      teamId: "team-1", attackerUnitId: "king", target: { kind: "unit", unitId: "target" }, pass: false,
    }]);
    expect(presentation.events.filter((event) => event.unitId === "king").map((event) => event.kind))
      .toEqual(["attack", "death"]);
    expect(presentation.deathOverlays[0].events.map((event) => event.kind))
      .toEqual(["attack", "death"]);
  });

  it("does not mutate battle state or battle intents", () => {
    const before = createInitialGameState();
    before.units = [unit()];
    const after = structuredClone(before);
    after.units[0].hp = 2;
    const intents: AttackIntent[] = [];
    const beforeCopy = structuredClone(before);
    const afterCopy = structuredClone(after);
    createUnitVisualPresentation(before, after, intents);
    expect(before).toEqual(beforeCopy);
    expect(after).toEqual(afterCopy);
    expect(intents).toEqual([]);
  });
});
