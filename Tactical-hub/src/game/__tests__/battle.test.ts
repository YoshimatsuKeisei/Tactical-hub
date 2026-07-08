import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { getAttackCandidates, resolveBattle, saveAttackIntent } from "../engine/battle";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";

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

function targetIds(state: GameState, attackerUnitId: string) {
  return getAttackCandidates(state, attackerUnitId).map((target) => target.unitId);
}

describe("battle", () => {
  it("lets range 1 units target adjacent enemies only", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    state.units.find((unit) => unit.id === "home-1-strategist")!.type = "infantry";
    addUnit(state, "enemy-adjacent", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    addUnit(state, "enemy-far", "team-2", "infantry", { kind: "tile", x: 8, y: 1 });

    expect(targetIds(state, "home-1-strategist")).toContain("enemy-adjacent");
    expect(targetIds(state, "home-1-strategist")).not.toContain("enemy-far");
  });

  it("lets archers target enemies at range 3", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-archer-test", "team-1", "archer", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-range-3", "team-2", "infantry", { kind: "tile", x: 7, y: 1 });

    expect(targetIds(state, "team-1-archer-test")).toContain("enemy-range-3");
  });

  it("does not give strategists attack candidates", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });

    expect(getAttackCandidates(state, "home-1-strategist")).toEqual([]);
  });

  it("does not mutate GameState when saving AttackIntent", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    const before = structuredClone(state.units);
    const planned = saveAttackIntent(state, {
      teamId: "team-1",
      attackerUnitId: "home-1-king",
      target: { kind: "unit", unitId: "enemy-adjacent" },
      pass: false,
    });

    expect(planned.units).toEqual(before);
    expect(planned.turnState.actionIntents[0].attackIntents).toHaveLength(1);
  });

  it("reduces target HP by 1 on hit and removes units at 0 HP", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    const planned = saveAttackIntent(state, {
      teamId: "team-1",
      attackerUnitId: "home-1-king",
      target: { kind: "unit", unitId: "enemy-adjacent" },
      pass: false,
    });
    const resolved = resolveBattle(planned, () => 0);
    const target = resolved.units.find((unit) => unit.id === "enemy-adjacent")!;

    expect(target.hp).toBe(0);
    expect(target.position).toEqual({ kind: "removed", reason: "defeated" });
  });

  it("keeps kings at 3 initial HP and defeats their team at 0 HP", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-2-king")!;
    expect(king.hp).toBe(3);
    king.hp = 1;
    putUnit(state, "home-1-king", { kind: "tile", x: 18, y: 1 });
    putUnit(state, "home-2-king", { kind: "tile", x: 19, y: 1 });
    const resolved = resolveBattle(
      saveAttackIntent(state, {
        teamId: "team-1",
        attackerUnitId: "home-1-king",
        target: { kind: "unit", unitId: "home-2-king" },
        pass: false,
      }),
      () => 0,
    );

    expect(resolved.units.find((unit) => unit.id === "home-2-king")?.position.kind).toBe("removed");
    expect(resolved.teams.find((team) => team.id === "team-2")?.status).toBe("defeated");
  });

  it("can target enemy units inside a base when the base is in range", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-archer-test", "team-1", "archer", { kind: "tile", x: 8, y: 1 });

    expect(targetIds(state, "team-1-archer-test")).toContain("neutral-north-infantry");
  });

  it("protects okuzashiki units while another friendly unit remains in the home base", () => {
    const state = createInitialGameState();
    addUnit(state, "team-2-archer-test", "team-2", "archer", { kind: "tile", x: 3, y: 1 });

    expect(targetIds(state, "team-2-archer-test")).not.toContain("home-1-king");
    expect(targetIds(state, "team-2-archer-test")).toContain("home-1-strategist");
  });

  it("allows targeting the okuzashiki unit when it is the only friendly unit in the home base", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    clearPreviousSlot(state, strategist.position);
    strategist.position = { kind: "removed", reason: "defeated" };
    strategist.hp = 0;
    addUnit(state, "team-2-archer-test", "team-2", "archer", { kind: "tile", x: 3, y: 1 });

    expect(targetIds(state, "team-2-archer-test")).toContain("home-1-king");
  });

  it("changes GameState only when battle is resolved", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "king", { kind: "tile", x: 5, y: 1 });
    const planned = saveAttackIntent(state, {
      teamId: "team-1",
      attackerUnitId: "home-1-king",
      target: { kind: "unit", unitId: "enemy-adjacent" },
      pass: false,
    });
    const beforeResolveHp = planned.units.find((unit) => unit.id === "enemy-adjacent")?.hp;
    const resolved = resolveBattle(planned, () => 0);

    expect(beforeResolveHp).toBe(3);
    expect(resolved.units.find((unit) => unit.id === "enemy-adjacent")?.hp).toBe(2);
  });
});
