import { describe, expect, it } from "vitest";
import {
  getMovementCandidates,
  getMovementPaths,
  isLegalDestination,
  resolveMovement,
  saveMovementIntent,
  validateMovementPath,
} from "../engine/movement";
import { createInitialGameState } from "../initialState";
import { UNIT_STATS } from "../constants";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";
import { positionKey } from "../utils/position";

function putUnitOnTile(
  state: GameState,
  unitId: string,
  position: UnitPosition,
) {
  const unit = state.units.find((candidate) => candidate.id === unitId)!;
  const previous = unit.position;
  if (previous.kind === "base") {
    const base = state.bases.find(
      (candidate) => candidate.id === previous.baseId,
    )!;
    const slot = base.slots.find(
      (candidate) => candidate.id === previous.slotId,
    )!;
    slot.unitId = undefined;
  }
  unit.position = position;
}

function addUnit(
  state: GameState,
  id: string,
  ownerTeamId: string,
  type: UnitType,
  position: UnitPosition,
): Unit {
  const unit: Unit = {
    id,
    ownerTeamId,
    type,
    hp: UNIT_STATS[type].hp,
    position,
    statuses: [],
  };
  state.units.push(unit);
  if (position.kind === "base") {
    const base = state.bases.find(
      (candidate) => candidate.id === position.baseId,
    )!;
    const slot = base.slots.find(
      (candidate) => candidate.id === position.slotId,
    )!;
    slot.unitId = id;
  }
  return unit;
}

function clearBase(state: GameState, baseId: string) {
  const base = state.bases.find((candidate) => candidate.id === baseId)!;
  for (const slot of base.slots) {
    if (slot.unitId) {
      const unit = state.units.find(
        (candidate) => candidate.id === slot.unitId,
      );
      if (unit) unit.position = { kind: "removed", reason: "defeated" };
    }
    slot.unitId = undefined;
  }
}

function makeBaseFriendly(state: GameState, baseId: string) {
  clearBase(state, baseId);
  state.bases.find((base) => base.id === baseId)!.ownerTeamId = "team-1";
  const team = state.teams.find((candidate) => candidate.id === "team-1")!;
  if (!team.controlledBaseIds.includes(baseId))
    team.controlledBaseIds.push(baseId);
}

describe("movement", () => {
  function addActiveBridge(
    state: GameState,
    ownerTeamId: string,
    id = "remote-bridge",
    tiles = [
      { x: 7, y: 2 },
      { x: 7, y: 3 },
      { x: 7, y: 4 },
      { x: 7, y: 5 },
      { x: 7, y: 6 },
    ],
  ) {
    state.constructions.push({
      id,
      kind: "bridge",
      ownerTeamId,
      managerUnitId: `${ownerTeamId}-builder`,
      tiles,
      placedTurn: 1,
      active: true,
    });
  }

  it("does not allow normal units to move onto lake cells", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });

    expect(
      getMovementCandidates(state, "home-1-strategist").map(positionKey),
    ).not.toContain("4,2");
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

    expect(getMovementCandidates(state, ninja.id).map(positionKey)).toContain(
      "4,2",
    );
  });

  it("does not allow moving onto another unit", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    putUnitOnTile(state, "home-1-king", { kind: "tile", x: 5, y: 1 });

    expect(
      getMovementCandidates(state, "home-1-strategist").map(positionKey),
    ).not.toContain("5,1");
  });

  it("saving a movement intent does not mutate unit position", () => {
    const state = createInitialGameState();
    putUnitOnTile(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    const before = state.units.find(
      (unit) => unit.id === "home-1-strategist",
    )!.position;
    const planned = saveMovementIntent(state, {
      teamId: "team-1",
      unitId: "home-1-strategist",
      from: before,
      to: { kind: "tile", x: 5, y: 1 },
      stay: false,
    });

    expect(
      planned.units.find((unit) => unit.id === "home-1-strategist")?.position,
    ).toEqual(before);
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

    expect(
      resolved.units.find((unit) => unit.id === "home-1-strategist")?.position,
    ).toEqual({ kind: "tile", x: 5, y: 1 });
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

    expect(
      resolved.units.find((unit) => unit.id === "home-1-strategist")?.position,
    ).toEqual({ kind: "tile", x: 4, y: 1 });
    expect(resolved.logs.at(-1)?.message).toContain("failed");
  });

  it("does not treat units inside bases as normal board coordinates", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-1-king")!;

    expect(king.position.kind).toBe("base");
    expect(
      getMovementCandidates(state, king.id).map(positionKey),
    ).not.toContain("1,1");
  });

  it("does not allow moving onto outside cells", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "home-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    expect(isLegalDestination(state, unit, { kind: "tile", x: 0, y: 0 })).toBe(
      false,
    );
  });

  it("allows movement within the same road section", () => {
    const state = createInitialGameState();

    const unit = addUnit(
      state,
      "team-1-infantry-road-test",
      "team-1",
      "infantry",
      {
        kind: "tile",
        x: 4,
        y: 1,
      },
    );

    expect(getMovementCandidates(state, unit.id).map(positionKey)).toContain(
      "5,1",
    );
  });

  it("treats every active bridge cell as a neutral, traversable movement node", () => {
    const state = createInitialGameState();
    addActiveBridge(state, "team-2", "enemy-bridge", [
      { x: 4, y: 2 },
      { x: 4, y: 3 },
    ]);
    const infantry = addUnit(state, "bridge-infantry", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    expect(getMovementCandidates(state, infantry.id)).toContainEqual({
      kind: "bridge",
      bridgeId: "enemy-bridge",
      cellIndex: 0,
    });

    const diagonalInfantry = addUnit(
      state,
      "diagonal-bridge-infantry",
      "team-1",
      "infantry",
      { kind: "tile", x: 3, y: 3 },
    );
    expect(getMovementCandidates(state, diagonalInfantry.id)).toContainEqual({
      kind: "bridge",
      bridgeId: "enemy-bridge",
      cellIndex: 0,
    });

    infantry.position = { kind: "bridge", bridgeId: "enemy-bridge", cellIndex: 0 };
    expect(getMovementCandidates(state, infantry.id)).toContainEqual({
      kind: "bridge",
      bridgeId: "enemy-bridge",
      cellIndex: 1,
    });

    const cavalry = addUnit(state, "bridge-cavalry", "team-1", "cavalry", {
      kind: "bridge",
      bridgeId: "enemy-bridge",
      cellIndex: 0,
    });
    expect(getMovementCandidates(state, cavalry.id)).toContainEqual({
      kind: "tile",
      x: 4,
      y: 4,
    });
  });

  it("blocks only bridge cells carrying an active obstacle", () => {
    const state = createInitialGameState();
    addActiveBridge(state, "team-2", "blocked-bridge", [
      { x: 4, y: 2 },
      { x: 4, y: 3 },
    ]);
    state.constructions.push({
      id: "bridge-obstacle",
      kind: "obstacle",
      ownerTeamId: "team-1",
      managerUnitId: "team-1-builder",
      tiles: [{ x: 4, y: 3 }],
      placedTurn: 1,
      active: true,
    });
    state.constructions.push({
      id: "inactive-bridge-obstacle",
      kind: "obstacle",
      ownerTeamId: "team-2",
      managerUnitId: "team-2-builder",
      tiles: [{ x: 4, y: 2 }],
      placedTurn: 0,
      active: false,
    });
    const cavalry = addUnit(state, "blocked-cavalry", "team-1", "cavalry", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    const candidates = getMovementCandidates(state, cavalry.id);
    expect(candidates).toContainEqual({
      kind: "bridge",
      bridgeId: "blocked-bridge",
      cellIndex: 0,
    });
    expect(candidates).not.toContainEqual({
      kind: "bridge",
      bridgeId: "blocked-bridge",
      cellIndex: 1,
    });
    expect(candidates).not.toContainEqual({ kind: "tile", x: 4, y: 4 });
  });

  it.each(["team-1", "team-2"])(
    "does not shorten friendly-base travel when a remote %s bridge exists",
    (bridgeOwner) => {
      const withoutBridge = createInitialGameState();
      const withBridge = createInitialGameState();
      addUnit(withoutBridge, "base-infantry", "team-1", "infantry", { kind: "tile", x: 3, y: 2 });
      addUnit(withBridge, "base-infantry", "team-1", "infantry", { kind: "tile", x: 3, y: 2 });
      addActiveBridge(withBridge, bridgeOwner);

      const infantryWithout = getMovementCandidates(withoutBridge, "base-infantry").map(positionKey).sort();
      const infantryWith = getMovementCandidates(withBridge, "base-infantry").map(positionKey).sort();
      expect(infantryWith).toEqual(infantryWithout);
      expect(infantryWith).not.toContain("3,3");

      withoutBridge.units = withoutBridge.units.filter((unit) => unit.id !== "base-infantry");
      withBridge.units = withBridge.units.filter((unit) => unit.id !== "base-infantry");
      addUnit(withoutBridge, "base-cavalry", "team-1", "cavalry", { kind: "tile", x: 3, y: 2 });
      addUnit(withBridge, "base-cavalry", "team-1", "cavalry", { kind: "tile", x: 3, y: 2 });
      const cavalryWithout = getMovementCandidates(withoutBridge, "base-cavalry").map(positionKey).sort();
      const cavalryWith = getMovementCandidates(withBridge, "base-cavalry").map(positionKey).sort();
      expect(cavalryWith).toEqual(cavalryWithout);
      expect(cavalryWith).not.toContain("4,4");
    },
  );

  it("keeps bridge and base movement independent of state array ordering", () => {
    const state = createInitialGameState();
    addActiveBridge(state, "team-1");
    addUnit(state, "ordered-cavalry", "team-1", "cavalry", {
      kind: "tile",
      x: 3,
      y: 2,
    });
    state.turnState.actionIntents.push({
      teamId: "team-1",
      productionChoices: [],
      attackIntents: [],
      movementIntents: [{
        teamId: "team-1",
        unitId: "ordered-cavalry",
        from: { kind: "tile", x: 3, y: 2 },
        to: { kind: "tile", x: 3, y: 1 },
        stay: false,
      }],
    });
    const reordered = structuredClone(state);
    reordered.units.reverse();
    reordered.constructions.reverse();
    reordered.turnState.actionIntents.reverse();
    for (const intent of reordered.turnState.actionIntents) {
      intent.movementIntents.reverse();
    }

    expect(
      getMovementCandidates(reordered, "ordered-cavalry").map(positionKey).sort(),
    ).toEqual(
      getMovementCandidates(state, "ordered-cavalry").map(positionKey).sort(),
    );
  });

  it("blocks direct movement between different road sections", () => {
    const state = createInitialGameState();

    const unit = addUnit(
      state,
      "team-1-infantry-section-test",
      "team-1",
      "infantry",
      {
        kind: "tile",
        x: 3,
        y: 2,
      },
    );

    /*
     * 3,2はhome-1↔north、
     * 3,3はhome-1↔center。
     *
     * 座標上は隣接しているが別の道区間。
     */
    expect(
      getMovementCandidates(state, unit.id).map(positionKey),
    ).not.toContain("3,3");
  });

  it("allows cavalry to change road sections through a friendly base", () => {
    const state = createInitialGameState();

    const cavalry = addUnit(
      state,
      "team-1-cavalry-friendly-base-test",
      "team-1",
      "cavalry",
      {
        kind: "tile",
        x: 3,
        y: 1,
      },
    );

    const destination: UnitPosition = {
      kind: "tile",
      x: 2,
      y: 3,
    };

    const path = getMovementPaths(state, cavalry.id).find(
      (candidate) =>
        positionKey(candidate.destination) === positionKey(destination),
    );

    expect(path).toBeDefined();

    expect(path?.steps.map((step) => step.kind)).toEqual([
      "enter-base",
      "leave-base",
    ]);
  });

  it("does not let move-1 units move directly over a base to the road beyond", () => {
    const state = createInitialGameState();
    makeBaseFriendly(state, "neutral-north");
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 9,
      y: 1,
    });

    expect(
      getMovementCandidates(state, "team-1-infantry-test").map(positionKey),
    ).not.toContain("12,1");
  });

  it("allows cavalry to pass through an empty friendly base as enter-base then leave-base", () => {
    const state = createInitialGameState();
    makeBaseFriendly(state, "neutral-north");
    const cavalry = addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    const destination: UnitPosition = { kind: "tile", x: 12, y: 1 };

    expect(getMovementCandidates(state, cavalry.id).map(positionKey)).toContain(
      positionKey(destination),
    );
    const validation = validateMovementPath(
      state,
      cavalry,
      cavalry.position,
      destination,
    );
    expect(validation.valid).toBe(true);
    if (validation.valid) {
      expect(validation.path.steps.map((step) => step.kind)).toEqual([
        "enter-base",
        "leave-base",
      ]);
    }
  });

  it("blocks cavalry base-through when the friendly base is full", () => {
    const state = createInitialGameState();
    makeBaseFriendly(state, "neutral-north");
    const base = state.bases.find(
      (candidate) => candidate.id === "neutral-north",
    )!;
    base.slots.forEach((slot, index) =>
      addUnit(state, `team-1-blocker-${index}`, "team-1", "infantry", {
        kind: "base",
        baseId: base.id,
        slotId: slot.id,
      }),
    );
    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });

    expect(
      getMovementCandidates(state, "team-1-cavalry-test").map(positionKey),
    ).not.toContain("12,1");
  });

  it("does not use enemy or neutral bases as same-turn intermediates", () => {
    const enemy = createInitialGameState();
    clearBase(enemy, "neutral-north");
    enemy.bases.find((base) => base.id === "neutral-north")!.ownerTeamId =
      "team-2";
    enemy.teams
      .find((team) => team.id === "team-2")!
      .controlledBaseIds.push("neutral-north");
    addUnit(enemy, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    expect(
      getMovementCandidates(enemy, "team-1-cavalry-test").map(positionKey),
    ).not.toContain("12,1");

    const neutral = createInitialGameState();
    clearBase(neutral, "neutral-north");
    addUnit(neutral, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    expect(
      getMovementCandidates(neutral, "team-1-cavalry-test").map(positionKey),
    ).not.toContain("12,1");
  });

  it("keeps base entry as a base position and validates saved intents with the same path rules", () => {
    const state = createInitialGameState();
    makeBaseFriendly(state, "neutral-north");
    const cavalry = addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    const baseDestination = getMovementCandidates(state, cavalry.id).find(
      (candidate) =>
        candidate.kind === "base" && candidate.baseId === "neutral-north",
    )!;

    const resolved = resolveMovement(
      saveMovementIntent(state, {
        teamId: "team-1",
        unitId: cavalry.id,
        from: cavalry.position,
        to: baseDestination,
        stay: false,
      }),
    );

    expect(
      resolved.units.find((unit) => unit.id === cavalry.id)?.position,
    ).toEqual(baseDestination);
  });

  it("rejects direct saved movement intents that have no legal path", () => {
    const state = createInitialGameState();
    clearBase(state, "neutral-north");
    const cavalry = addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    const destination: UnitPosition = { kind: "tile", x: 12, y: 1 };

    expect(
      getMovementPaths(state, cavalry.id).map((path) =>
        positionKey(path.destination),
      ),
    ).not.toContain(positionKey(destination));

    const resolved = resolveMovement(
      saveMovementIntent(state, {
        teamId: "team-1",
        unitId: cavalry.id,
        from: cavalry.position,
        to: destination,
        stay: false,
      }),
    );

    expect(
      resolved.units.find((unit) => unit.id === cavalry.id)?.position,
    ).toEqual({ kind: "tile", x: 9, y: 1 });
    expect(resolved.logs.at(-1)?.message).toContain("no legal movement path");
  });

  it("allows cavalry inside a base to leave and use its second movement step", () => {
    const state = createInitialGameState();

    makeBaseFriendly(state, "neutral-north");

    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "base",
      baseId: "neutral-north",
      slotId: "slot_0_0",
    });

    const paths = getMovementPaths(state, "team-1-cavalry-test");

    const candidates = paths.map((path) => positionKey(path.destination));

    // 1歩目：拠点から道路へ退城
    expect(candidates).toContain("9,1");

    // 2歩目：退城先からさらに道路へ移動
    expect(candidates).toContain("8,1");

    const twoStepPath = paths.find(
      (path) => positionKey(path.destination) === "8,1",
    );

    expect(twoStepPath).toBeDefined();

    expect(twoStepPath?.steps.map((step) => step.kind)).toEqual([
      "leave-base",
      "ground",
    ]);
  });
});
