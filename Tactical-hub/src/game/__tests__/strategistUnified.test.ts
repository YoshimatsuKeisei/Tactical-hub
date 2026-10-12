import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  clearDeadConstructionManagers, getBuilderUnits, isAvailable,
} from "../engine/construction";
import { isUnitEncouragedByStrategist } from "../engine/encouragement";
import { resolveProduction, saveProductionChoice } from "../engine/production";
import { getTeleportStrategists, isTeleportAvailable } from "../engine/teleport";
import { enumerateRlDecisions } from "../cpu/rlEnvironment";
import { createCpuRuntime } from "../cpu/types";
import { createInitialGameState } from "../initialState";
import type { GameState, StrategistRole, Unit } from "../types";

function moveStrategistToRoad(state: GameState, strategist: Unit) {
  if (strategist.position.kind === "base") {
    const { baseId, slotId } = strategist.position;
    const slot = state.bases.find((base) => base.id === baseId)
      ?.slots.find((candidate) => candidate.id === slotId);
    if (slot) slot.unitId = undefined;
  }
  strategist.position = { kind: "tile", x: 4, y: 1 };
}

function addFriendlyTarget(state: GameState) {
  const unit: Unit = {
    id: "unified-friendly", ownerTeamId: "team-1", type: "infantry",
    hp: UNIT_STATS.infantry.hp, position: { kind: "tile", x: 5, y: 1 }, statuses: [],
  };
  state.units.push(unit);
  return unit;
}

describe("unified strategist abilities", () => {
  it("gives every initial strategist encourage, build, and teleport capability", () => {
    const state = createInitialGameState();
    for (const strategist of state.units.filter((unit) => unit.type === "strategist")) {
      const king = state.units.find((unit) => unit.type === "king" && unit.ownerTeamId === strategist.ownerTeamId)!;
      expect(isUnitEncouragedByStrategist(state, king, strategist)).toBe(true);
      expect(getBuilderUnits(state, strategist.ownerTeamId)).toContain(strategist);
      expect(getTeleportStrategists(state, strategist.ownerTeamId)).toContain(strategist);
    }
  });

  it("produces one canonical strategist that has all three capabilities", () => {
    let state = createInitialGameState();
    state = saveProductionChoice(state, {
      teamId: "team-1", baseId: "home-1", unitType: "strategist", strategistRole: "encourage",
    });
    state = resolveProduction(state);
    const produced = state.units.find((unit) => unit.id === "home-1-strategist-2")!;
    const king = state.units.find((unit) => unit.id === "home-1-king")!;

    expect(produced.role).toBe("encourage");
    expect(isUnitEncouragedByStrategist(state, king, produced)).toBe(true);
    expect(getBuilderUnits(state, "team-1")).toContain(produced);
    expect(getTeleportStrategists(state, "team-1")).toContain(produced);
  });

  it.each(["encourage", "builder", "teleporter", undefined] as const)(
    "treats legacy role value %s as a unified strategist",
    (role) => {
      const state = createInitialGameState();
      const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
      if (role) strategist.role = role as StrategistRole;
      else delete strategist.role;
      moveStrategistToRoad(state, strategist);
      const target = addFriendlyTarget(state);

      expect(isUnitEncouragedByStrategist(state, target, strategist)).toBe(true);
      expect(getBuilderUnits(state, "team-1").map((unit) => unit.id)).toContain(strategist.id);
      expect(getTeleportStrategists(state, "team-1").map((unit) => unit.id)).toContain(strategist.id);
    },
  );

  it("keeps construction and teleport cooldowns independent on the same strategist", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    state.turnNumber = 8;
    state.strategistCooldowns = [
      { strategistUnitId: strategist.id, kind: "bridge", availableFromTurn: 10 },
      { strategistUnitId: strategist.id, kind: "obstacle", availableFromTurn: 12 },
    ];
    state.teleportCooldowns = [{ strategistUnitId: strategist.id, availableFromTurn: 11 }];

    expect(isAvailable(state, strategist.id, "bridge")).toBe(false);
    expect(isAvailable(state, strategist.id, "obstacle")).toBe(false);
    expect(isTeleportAvailable(state, strategist.id)).toBe(false);
    state.turnNumber = 10;
    expect(isAvailable(state, strategist.id, "bridge")).toBe(true);
    expect(isAvailable(state, strategist.id, "obstacle")).toBe(false);
    expect(isTeleportAvailable(state, strategist.id)).toBe(false);
    state.turnNumber = 12;
    expect(isAvailable(state, strategist.id, "obstacle")).toBe(true);
    expect(isTeleportAvailable(state, strategist.id)).toBe(true);
  });

  it("releases a unified strategist's construction management when it dies", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    strategist.role = "teleporter";
    state.constructions.push({
      id: "legacy-role-bridge", kind: "bridge", ownerTeamId: "team-1",
      managerUnitId: strategist.id, tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true,
    });
    strategist.hp = 0;
    strategist.position = { kind: "removed", reason: "defeated" };
    clearDeadConstructionManagers(state);

    expect(state.constructions[0]).toMatchObject({ active: true, managerUnitId: undefined });
  });

  it("exposes one strategist production choice and unified legacy strategists to RL", () => {
    const production = createInitialGameState();
    const productionActions = enumerateRlDecisions(
      production, createCpuRuntime(1), (teamId) => teamId === "team-1",
    ).filter((entry) => entry.action.unitType === "strategist");
    expect(productionActions).toHaveLength(1);
    expect(productionActions[0].action.strategistRole).toBe("encourage");

    const construction = createInitialGameState();
    const strategist = construction.units.find((unit) => unit.id === "home-1-strategist")!;
    strategist.role = "teleporter";
    construction.phase = construction.turnState.phase = "strategist_action_input";
    const constructionActions = enumerateRlDecisions(
      construction, createCpuRuntime(2), (teamId) => teamId === "team-1",
    );
    expect(constructionActions.some((entry) =>
      entry.decision.kind === "strategist" && entry.decision.intent.strategistUnitId === strategist.id,
    )).toBe(true);

    const movement = createInitialGameState();
    const legacyStrategist = movement.units.find((unit) => unit.id === "home-1-strategist")!;
    legacyStrategist.role = "builder";
    movement.phase = movement.turnState.phase = "movement_input";
    movement.currentMovementTeamId = "team-1";
    movement.productionCompletedTeamIdsThisTurn = ["team-1"];
    const runtime = createCpuRuntime(3);
    enumerateRlDecisions(movement, runtime, (teamId) => teamId === "team-1");
    runtime.processedKeys = movement.units
      .filter((unit) => unit.ownerTeamId === "team-1" && unit.hp > 0 && unit.position.kind !== "removed")
      .map((unit) => `movement:team-1:${unit.id}`);
    const teleportActions = enumerateRlDecisions(
      movement, runtime, (teamId) => teamId === "team-1",
    );
    expect(teleportActions.some((entry) =>
      entry.decision.kind === "teleport" && entry.decision.strategistUnitId === legacyStrategist.id,
    )).toBe(true);
  });
});
