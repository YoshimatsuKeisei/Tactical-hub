import { describe, expect, it } from "vitest";
import { resolveBattle } from "../engine/battle";
import { createInitialGameState } from "../initialState";
import { getProductionCandidates } from "../engine/production";
import { getTeamMovementCandidates } from "../engine/movement";
import type { GameState, Unit, UnitType } from "../types";

function add(state: GameState, id: string, teamId: string, type: UnitType, x: number, y: number): Unit {
  const unit: Unit = { id, ownerTeamId: teamId, type, hp: 1, position: { kind: "tile", x, y }, statuses: [] };
  state.units.push(unit);
  return unit;
}

function fixture(type: UnitType, distance = 1) {
  const state = createInitialGameState();
  state.phase = state.turnState.phase = "battle_resolution";
  state.units.forEach((unit) => { unit.hp = 0; unit.position = { kind: "removed", reason: "defeated" }; });
  const neutralId = state.teams.find((team) => team.status === "neutral")!.id;
  const guard = add(state, "neutral-guard", neutralId, type, 4, 1);
  const enemy = add(state, "player-unit", "team-1", "infantry", 4 + distance, 1);
  return { state, guard, enemy };
}

describe("neutral base defense attacks", () => {
  it.each(["infantry", "cavalry"] as const)("lets a neutral %s attack one adjacent enemy", (type) => {
    const { state, enemy } = fixture(type);
    const resolved = resolveBattle(state, () => 0);
    expect(resolved.units.find((unit) => unit.id === enemy.id)?.position.kind).toBe("removed");
    expect(resolved.logs.filter((log) => log.message.includes("neutral-guard -> player-unit"))).toHaveLength(1);
  });

  it("lets a neutral archer attack in range but not out of range", () => {
    const inRange = fixture("archer", 3);
    expect(resolveBattle(inRange.state, () => 0).units.find((unit) => unit.id === inRange.enemy.id)?.position.kind).toBe("removed");
    const outOfRange = fixture("archer", 4);
    expect(resolveBattle(outOfRange.state, () => 0).units.find((unit) => unit.id === outOfRange.enemy.id)?.position.kind).not.toBe("removed");
  });

  it("resolves a saved player attack and neutral counterattack simultaneously", () => {
    const { state, guard, enemy } = fixture("infantry");
    state.turnState.actionIntents = [{ teamId: "team-1", productionChoices: [], movementIntents: [], attackIntents: [{ teamId: "team-1", attackerUnitId: enemy.id, target: { kind: "unit", unitId: guard.id }, pass: false }] }];
    const resolved = resolveBattle(state, () => 0);
    expect(resolved.units.find((unit) => unit.id === guard.id)?.position.kind).toBe("removed");
    expect(resolved.units.find((unit) => unit.id === enemy.id)?.position.kind).toBe("removed");
  });

  it("does not expose neutral production or movement decisions", () => {
    const state = createInitialGameState();
    const neutralId = state.teams.find((team) => team.status === "neutral")!.id;
    expect(getProductionCandidates(state, neutralId)).toEqual([]);
    expect(getTeamMovementCandidates(state, neutralId)).toEqual([]);
  });
});
