import { describe, expect, it } from "vitest";
import { createInitialGameState } from "../game/initialState";
import type { AttackIntent, Unit } from "../game/types";
import {
  applyUnitVisualEventDirections,
  createUnitDirectionMap,
  createUnitVisualPresentation,
  getInitialUnitDirection,
  getUnitSpriteKind,
  updateUnitDirectionMap,
} from "./unitVisualEvents";
import type { BattlePresentationMetadata } from "../game/engine/battle";

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
  it("maps every configured sprite unit and leaves cavalry/apprentice ninja unchanged", () => {
    expect(getUnitSpriteKind(unit({ type: "king" }))).toBe("6Crusader");
    for (const role of ["builder", "encourage", "teleporter"] as const) {
      expect(getUnitSpriteKind(unit({ type: "strategist", role }))).toBe("10Caster");
    }
    expect(getUnitSpriteKind(unit({ type: "engineer" }))).toBe("catapult");
    expect(getUnitSpriteKind(unit({ type: "infantry" }))).toBe("1Knight");
    expect(getUnitSpriteKind(unit({ type: "infantry", formation: "heavy" }))).toBe("4Paladin");
    expect(getUnitSpriteKind(unit({ type: "archer" }))).toBe("2Archer");
    expect(getUnitSpriteKind(unit({ type: "ninja" }))).toBe("7DeathKnight");
    expect(getUnitSpriteKind(unit({ type: "cavalry" }))).toBeUndefined();
    expect(getUnitSpriteKind(unit({ type: "apprentice_ninja" }))).toBeUndefined();
  });
});

describe("generic battle presentation events", () => {
  it("emits ordered attacks for every sprite-backed attacker", () => {
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
      .toEqual(["king", "strategist", "engineer", "infantry"]);
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

describe("persistent eight-direction presentation", () => {
  it("faces each initial base toward board center, with neutral-center fixed down", () => {
    const state = createInitialGameState();
    const expectedByBaseId = new Map([
      ["home-1", 1], ["home-2", 7], ["home-3", 5], ["home-4", 3],
      ["neutral-north", 0], ["neutral-east", 6], ["neutral-south", 4], ["neutral-west", 2],
      ["neutral-center", 0],
    ]);
    for (const base of state.bases) {
      const directions = state.units
        .filter((entry) => entry.position.kind === "base" && entry.position.baseId === base.id)
        .map((entry) => getInitialUnitDirection(state, entry));
      expect(new Set(directions), base.id).toEqual(new Set([expectedByBaseId.get(base.id)]));
    }
  });

  it("uses the source base rather than owner or slot for newly produced units", () => {
    const before = createInitialGameState();
    const after = structuredClone(before);
    const center = after.bases.find((base) => base.id === "neutral-center")!;
    center.ownerTeamId = "team-3";
    const centerUnit = unit({ id: "center-produced", ownerTeamId: "team-3", position: { kind: "base", baseId: center.id, slotId: center.slots[3].id } });
    const east = after.bases.find((base) => base.id === "neutral-east")!;
    east.ownerTeamId = "team-1";
    const eastUnit = unit({ id: "east-produced", ownerTeamId: "team-1", position: { kind: "base", baseId: east.id, slotId: east.slots[1].id } });
    after.units.push(centerUnit, eastUnit);
    const directions = updateUnitDirectionMap(before, after, createUnitDirectionMap(before));
    expect(directions.get(centerUnit.id)).toBe(0);
    expect(directions.get(eastUnit.id)).toBe(6);
  });

  it.each([
    [-1, 1, 7], [0, 1, 0], [1, 1, 1], [-1, 0, 6],
    [1, 0, 2], [-1, -1, 5], [0, -1, 4], [1, -1, 3],
  ] as const)("keeps movement delta %s/%s as idle direction %s", (dx, dy, direction) => {
    const before = createInitialGameState();
    before.units = [unit()];
    const current = new Map([["king", 3 as const]]);
    const after = structuredClone(before);
    after.units[0].position = { kind: "tile", x: 3 + dx, y: 3 + dy };
    expect(updateUnitDirectionMap(before, after, current).get("king")).toBe(direction);
  });

  it("detects base entry and exit from display coordinates but preserves stay and ordinary clones", () => {
    const before = createInitialGameState();
    const moving = before.units.find((entry) => entry.position.kind === "base" && entry.position.baseId === "home-1")!;
    const current = new Map([[moving.id, 4 as const]]);
    const entered = structuredClone(before);
    const enteredUnit = entered.units.find((entry) => entry.id === moving.id)!;
    enteredUnit.position = { kind: "base", baseId: "neutral-center", slotId: "slot_0_0" };
    expect(updateUnitDirectionMap(before, entered, current).get(moving.id)).toBe(1);
    expect(updateUnitDirectionMap(before, structuredClone(before), current).get(moving.id)).toBe(4);

    const exited = structuredClone(before);
    exited.units.find((entry) => entry.id === moving.id)!.position = { kind: "tile", x: 5, y: 1 };
    expect(updateUnitDirectionMap(before, exited, current).get(moving.id)).toBe(2);
  });

  it("uses displayed tile, water, bridge, and teleport coordinates without changing movement rules", () => {
    const tile = createInitialGameState();
    tile.units = [unit({ position: { kind: "tile", x: 3, y: 3 } })];
    const water = structuredClone(tile);
    water.units[0].position = { kind: "water", x: 4, y: 2 };
    const afterWater = updateUnitDirectionMap(tile, water, new Map([["king", 0]]));
    expect(afterWater.get("king")).toBe(3);

    const bridge = structuredClone(water);
    bridge.constructions.push({
      id: "bridge-direction", kind: "bridge", ownerTeamId: "team-1",
      tiles: [{ x: 4, y: 2 }, { x: 5, y: 2 }], placedTurn: 1, active: true,
    });
    bridge.units[0].position = { kind: "bridge", bridgeId: "bridge-direction", cellIndex: 1 };
    expect(updateUnitDirectionMap(water, bridge, afterWater).get("king")).toBe(2);
  });

  it("preserves direction across phase, turn, and ownership-only changes", () => {
    const before = createInitialGameState();
    const tracked = before.units[0];
    const after = structuredClone(before);
    after.phase = after.turnState.phase = "attack_input";
    after.turnNumber = after.turnState.turnNumber = 2;
    after.units[0].ownerTeamId = "team-2";
    const position = after.units[0].position;
    if (position.kind === "base") after.bases.find((base) => base.id === position.baseId)!.ownerTeamId = "team-2";
    expect(updateUnitDirectionMap(before, after, new Map([[tracked.id, 7]])).get(tracked.id)).toBe(7);
  });

  it("resets presentation directions when a new initial game replaces a later game", () => {
    const later = createInitialGameState();
    later.turnNumber = later.turnState.turnNumber = 4;
    const fresh = createInitialGameState();
    const tracked = fresh.units.find((entry) => entry.position.kind === "base" && entry.position.baseId === "home-1")!;
    expect(updateUnitDirectionMap(later, fresh, new Map([[tracked.id, 6]])).get(tracked.id)).toBe(1);
  });

  it("uses every valid attack target direction, including a miss, then keeps the last event direction", () => {
    const before = createInitialGameState();
    before.units = [unit(), unit({ id: "target", ownerTeamId: "team-2", position: { kind: "tile", x: 2, y: 2 } })];
    const battle: BattlePresentationMetadata = {
      attacks: [{ eventId: "attack-1", attackerUnitId: "king", targetUnitId: "target" }],
      hits: [],
    };
    const presentation = createUnitVisualPresentation(before, structuredClone(before), {
      battle,
      directions: new Map([["king", 0]]),
    });
    expect(presentation.events).toMatchObject([{ unitId: "king", kind: "attack", direction: 5 }]);
    expect(applyUnitVisualEventDirections(new Map([["king", 0]]), presentation.events).get("king")).toBe(5);
  });

  it("uses event order so a later hit direction becomes the final idle direction", () => {
    expect(applyUnitVisualEventDirections(new Map([["king", 0]]), [
      { unitId: "king", eventId: "attack", kind: "attack", direction: 2 },
      { unitId: "king", eventId: "hit", kind: "hit", direction: 6 },
    ]).get("king")).toBe(6);
  });

  it("faces the nearest actual hit attacker, excludes misses, and breaks equal distances by attacker id", () => {
    const before = createInitialGameState();
    const target = unit({ id: "target", hp: 3, position: { kind: "tile", x: 5, y: 5 } });
    before.units = [
      target,
      unit({ id: "attacker-b", ownerTeamId: "team-2", position: { kind: "tile", x: 6, y: 5 } }),
      unit({ id: "attacker-a", ownerTeamId: "team-3", position: { kind: "tile", x: 4, y: 5 } }),
      unit({ id: "miss-nearest", ownerTeamId: "neutral", position: { kind: "tile", x: 5, y: 4 } }),
    ];
    const after = structuredClone(before);
    after.units[0].hp = 1;
    const attacks = before.units.slice(1).map((attacker) => ({
      eventId: `event-${attacker.id}`, attackerUnitId: attacker.id, targetUnitId: target.id,
    }));
    const presentation = createUnitVisualPresentation(before, after, {
      battle: { attacks, hits: attacks.filter((event) => event.attackerUnitId !== "miss-nearest") },
      directions: new Map([[target.id, 0]]),
    });
    expect(presentation.events.find((event) => event.unitId === target.id)).toMatchObject({ kind: "hit", direction: 6 });
  });

  it("keeps the previous direction when HP falls without usable hit-source metadata", () => {
    const before = createInitialGameState();
    before.units = [unit({ hp: 3 })];
    const after = structuredClone(before);
    after.units[0].hp = 2;
    const presentation = createUnitVisualPresentation(before, after, {
      battle: { attacks: [], hits: [] },
      directions: new Map([["king", 7]]),
    });
    expect(presentation.events).toMatchObject([{ unitId: "king", kind: "hit", direction: 7 }]);
  });

  it("passes the actual hit direction to a death overlay and preserves prior direction for non-combat removal", () => {
    const before = createInitialGameState();
    before.units = [
      unit({ id: "victim", hp: 1, position: { kind: "tile", x: 5, y: 5 } }),
      unit({ id: "neutral-attacker", ownerTeamId: "neutral", position: { kind: "tile", x: 5, y: 6 } }),
      unit({ id: "merged", hp: 1, position: { kind: "tile", x: 7, y: 7 } }),
    ];
    const after = structuredClone(before);
    after.units[0].hp = 0;
    after.units[0].position = { kind: "removed", reason: "defeated" };
    after.units[2].hp = 0;
    after.units[2].position = { kind: "removed", reason: "merged" };
    const hit = { eventId: "neutral-hit", attackerUnitId: "neutral-attacker", targetUnitId: "victim" };
    const presentation = createUnitVisualPresentation(before, after, {
      battle: { attacks: [hit], hits: [hit] },
      directions: new Map([["victim", 4], ["merged", 3]]),
    });
    expect(presentation.deathOverlays.find((overlay) => overlay.unit.id === "victim")?.direction).toBe(0);
    expect(presentation.deathOverlays.find((overlay) => overlay.unit.id === "merged")?.direction).toBe(3);
  });

  it("releases removed unit directions after their overlay has captured the direction", () => {
    const before = createInitialGameState();
    before.units = [unit()];
    const after = structuredClone(before);
    after.units[0].hp = 0;
    after.units[0].position = { kind: "removed", reason: "defeated" };
    expect(updateUnitDirectionMap(before, after, new Map([["king", 7]])).has("king")).toBe(false);
  });
});
