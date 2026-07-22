import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  applyEncouragementToDenominator,
  getAttackCandidates,
  getAttackEnumerationContext,
  getBaseAttackDenominator,
  getFinalAttackDenominator,
  getTeamAttackCandidates,
  resolveBattle,
  saveAttackIntent,
} from "../engine/battle";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";
import { getRoadAttackDistance } from "../utils/roadTopology";

function clearPreviousSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const base = state.bases.find(
    (candidate) => candidate.id === position.baseId,
  )!;
  const slot = base.slots.find(
    (candidate) => candidate.id === position.slotId,
  )!;
  slot.unitId = undefined;
}

function putUnit(state: GameState, id: string, position: UnitPosition) {
  const unit = state.units.find((candidate) => candidate.id === id)!;
  clearPreviousSlot(state, unit.position);
  unit.position = position;
  if (position.kind === "base") {
    const base = state.bases.find(
      (candidate) => candidate.id === position.baseId,
    )!;
    const slot = base.slots.find(
      (candidate) => candidate.id === position.slotId,
    )!;
    slot.unitId = id;
  }
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

function targetIds(state: GameState, attackerUnitId: string) {
  return getAttackCandidates(state, attackerUnitId).map(
    (target) => target.unitId,
  );
}

function targetDenominator(
  state: GameState,
  attackerUnitId: string,
  targetUnitId: string,
) {
  return getAttackCandidates(state, attackerUnitId).find(
    (target) => target.unitId === targetUnitId,
  )?.finalSuccessDenominator;
}

function saveAttack(
  state: GameState,
  teamId: string,
  attackerUnitId: string,
  targetUnitId: string,
) {
  return saveAttackIntent(state, {
    teamId,
    attackerUnitId,
    target: { kind: "unit", unitId: targetUnitId },
    pass: false,
  });
}

function saveAttacks(
  state: GameState,
  attacks: Array<[string, string, string]>,
) {
  return attacks.reduce(
    (next, [teamId, attackerUnitId, targetUnitId]) =>
      saveAttack(next, teamId, attackerUnitId, targetUnitId),
    state,
  );
}

describe("battle", () => {
  it.each([3, 4] as const)("keeps whole-team and target-unit attack candidates identical in a %i-player state", (players) => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    if (players === 3) state.teams.find((team) => team.id === "team-4")!.status = "defeated";
    const archer = addUnit(state, `equivalence-archer-${players}`, "team-1", "archer", { kind: "tile", x: 4, y: 1 });
    addUnit(state, `equivalence-near-${players}`, "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
    addUnit(state, `equivalence-far-${players}`, "team-2", "infantry", { kind: "tile", x: 8, y: 1 });
    addUnit(state, `equivalence-water-${players}`, "team-2", "ninja", { kind: "water", x: 4, y: 2 });
    state.constructions.push({ id: `equivalence-obstacle-${players}`, kind: "obstacle", ownerTeamId: "team-3", tiles: [{ x: 6, y: 1 }], placedTurn: 1, active: true });
    const whole = getTeamAttackCandidates(state, "team-1");
    const context = getAttackEnumerationContext(state);
    expect(getAttackEnumerationContext(state)).toBe(context);
    for (const entry of whole) expect(getAttackCandidates(state, entry.attackerUnitId, context)).toEqual(entry.targets);
    expect(getAttackCandidates(state, archer.id, context)).toEqual(whole.find((entry) => entry.attackerUnitId === archer.id)?.targets);

    const changed = structuredClone(state);
    changed.units.find((unit) => unit.id === archer.id)!.position = { kind: "tile", x: 7, y: 1 };
    expect(getAttackEnumerationContext(changed)).not.toBe(context);
  });
  it("lets range 1 units target adjacent enemies only", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    state.units.find((unit) => unit.id === "home-1-strategist")!.type =
      "infantry";
    addUnit(state, "enemy-adjacent", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    addUnit(state, "enemy-far", "team-2", "infantry", {
      kind: "tile",
      x: 8,
      y: 1,
    });

    expect(targetIds(state, "home-1-strategist")).toContain("enemy-adjacent");
    expect(targetIds(state, "home-1-strategist")).not.toContain("enemy-far");
  });

  it("allows attacks within the same road section", () => {
    const state = createInitialGameState();

    addUnit(state, "team-1-archer-same-road", "team-1", "archer", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    addUnit(state, "team-2-infantry-same-road", "team-2", "infantry", {
      kind: "tile",
      x: 7,
      y: 1,
    });

    expect(targetIds(state, "team-1-archer-same-road")).toContain(
      "team-2-infantry-same-road",
    );
  });

  it("blocks attacks between different road sections even when within range", () => {
    const state = createInitialGameState();

    addUnit(state, "team-1-archer-different-road", "team-1", "archer", {
      kind: "tile",
      x: 3,
      y: 2,
    });

    addUnit(state, "team-2-infantry-different-road", "team-2", "infantry", {
      kind: "tile",
      x: 3,
      y: 3,
    });

    expect(targetIds(state, "team-1-archer-different-road")).not.toContain(
      "team-2-infantry-different-road",
    );
  });

  it("allows an archer inside a base to attack a connected road section", () => {
    const state = createInitialGameState();

    const homeBase = state.bases.find((base) => base.id === "home-1")!;

    const emptySlot = homeBase.slots.find((slot) => !slot.unitId)!;

    addUnit(state, "team-1-archer-inside-home", "team-1", "archer", {
      kind: "base",
      baseId: homeBase.id,
      slotId: emptySlot.id,
    });

    addUnit(state, "team-2-infantry-connected-road", "team-2", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    expect(targetIds(state, "team-1-archer-inside-home")).toContain(
      "team-2-infantry-connected-road",
    );
  });

  it("blocks attacks through a base regardless of base ownership", () => {
    const ownershipCases = ["team-1", "team-2", "neutral-team"];

    for (const ownerTeamId of ownershipCases) {
      const state = createInitialGameState();

      const blockingBase = state.bases.find(
        (base) => base.id === "neutral-north",
      )!;

      blockingBase.ownerTeamId = ownerTeamId;

      addUnit(state, `team-1-archer-${ownerTeamId}`, "team-1", "archer", {
        kind: "tile",
        x: 9,
        y: 1,
      });

      addUnit(state, `team-2-infantry-${ownerTeamId}`, "team-2", "infantry", {
        kind: "tile",
        x: 12,
        y: 1,
      });

      expect(targetIds(state, `team-1-archer-${ownerTeamId}`)).not.toContain(
        `team-2-infantry-${ownerTeamId}`,
      );
    }
  });

  it("allows a unit inside a base to attack outward beyond that base", () => {
    const state = createInitialGameState();

    const base = state.bases.find(
      (candidate) => candidate.id === "neutral-north",
    )!;

    const emptySlot = base.slots.find((slot) => !slot.unitId)!;

    addUnit(state, "team-1-archer-inside-base", "team-1", "archer", {
      kind: "base",
      baseId: base.id,
      slotId: emptySlot.id,
    });

    addUnit(state, "team-2-infantry-outside-base", "team-2", "infantry", {
      kind: "tile",
      x: 13,
      y: 1,
    });

    expect(targetIds(state, "team-1-archer-inside-base")).toContain(
      "team-2-infantry-outside-base",
    );
  });

  it("allows attacks in both directions between a bridge and a connected base", () => {
    const state = createInitialGameState();
    const base = state.bases.find(
      (candidate) => candidate.id === "neutral-north",
    )!;
    const emptySlot = base.slots.find((slot) => !slot.unitId)!;

    state.constructions.push({
      id: "north-base-bridge",
      kind: "bridge",
      ownerTeamId: "team-1",
      managerUnitId: "team-1-builder",
      tiles: [
        { x: 10, y: 3 },
        { x: 10, y: 4 },
      ],
      placedTurn: 1,
      active: true,
    });
    addUnit(state, "team-1-archer-on-bridge", "team-1", "archer", {
      kind: "bridge",
      bridgeId: "north-base-bridge",
      cellIndex: 0,
    });
    addUnit(state, "team-2-archer-inside-base", "team-2", "archer", {
      kind: "base",
      baseId: base.id,
      slotId: emptySlot.id,
    });

    expect(targetIds(state, "team-1-archer-on-bridge")).toContain(
      "team-2-archer-inside-base",
    );
    expect(targetIds(state, "team-2-archer-inside-base")).toContain(
      "team-1-archer-on-bridge",
    );
  });

  it("measures archer range along the bridge and roads instead of across lake cells", () => {
    const state = createInitialGameState();
    state.constructions.push({
      id: "long-bridge",
      kind: "bridge",
      ownerTeamId: "team-1",
      managerUnitId: "team-1-builder",
      tiles: [2, 3, 4, 5, 6].map((y) => ({ x: 7, y })),
      placedTurn: 1,
      active: true,
    });
    addUnit(state, "team-1-archer-mid-bridge", "team-1", "archer", {
      kind: "bridge",
      bridgeId: "long-bridge",
      cellIndex: 2,
    });
    addUnit(state, "team-2-infantry-across-lake", "team-2", "infantry", {
      kind: "tile",
      x: 9,
      y: 1,
    });

    expect(targetIds(state, "team-1-archer-mid-bridge")).not.toContain(
      "team-2-infantry-across-lake",
    );
  });

  it("measures engineer range along the bridge and roads", () => {
    const state = createInitialGameState();
    state.constructions.push({
      id: "engineer-bridge",
      kind: "bridge",
      ownerTeamId: "team-1",
      managerUnitId: "team-1-builder",
      tiles: [2, 3, 4, 5, 6].map((y) => ({ x: 7, y })),
      placedTurn: 1,
      active: true,
    });
    const engineer = addUnit(state, "team-2-engineer-on-bridge", "team-2", "engineer", {
      kind: "bridge",
      bridgeId: "engineer-bridge",
      cellIndex: 2,
    });

    const strategist = state.units.find(
      (unit) => unit.id === "home-1-strategist",
    )!;
    expect(
      getRoadAttackDistance(state, engineer.position, strategist.position),
    ).toBe(6);

    expect(targetIds(state, "team-2-engineer-on-bridge")).not.toContain(
      "home-1-strategist",
    );
  });

  it.each([
    ["diagonally", [{ x: 7, y: 2 }, { x: 7, y: 3 }], [{ x: 8, y: 4 }, { x: 8, y: 5 }]],
    ["in parallel", [{ x: 7, y: 2 }, { x: 7, y: 3 }], [{ x: 8, y: 2 }, { x: 8, y: 3 }]],
  ])("measures attack range across different bridges touching %s", (_, firstTiles, secondTiles) => {
    const state = createInitialGameState();
    state.constructions.push(
      {
        id: "first-touching-bridge",
        kind: "bridge",
        ownerTeamId: "team-1",
        managerUnitId: "builder-1",
        tiles: firstTiles,
        placedTurn: 1,
        active: true,
      },
      {
        id: "second-touching-bridge",
        kind: "bridge",
        ownerTeamId: "team-2",
        managerUnitId: "builder-2",
        tiles: secondTiles,
        placedTurn: 1,
        active: true,
      },
    );
    addUnit(state, "archer-on-first-bridge", "team-1", "archer", {
      kind: "bridge",
      bridgeId: "first-touching-bridge",
      cellIndex: 1,
    });
    addUnit(state, "cavalry-on-second-bridge", "team-2", "cavalry", {
      kind: "bridge",
      bridgeId: "second-touching-bridge",
      cellIndex: 0,
    });

    expect(targetIds(state, "archer-on-first-bridge")).toContain(
      "cavalry-on-second-bridge",
    );
  });

  it("does not give strategists attack candidates", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });

    expect(getAttackCandidates(state, "home-1-strategist")).toEqual([]);
  });

  it("does not mutate GameState when saving AttackIntent", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    const before = structuredClone(state.units);
    const planned = saveAttack(
      state,
      "team-1",
      "home-1-king",
      "enemy-adjacent",
    );

    expect(planned.units).toEqual(before);
    expect(planned.turnState.actionIntents[0].attackIntents).toHaveLength(1);
  });

  it("reduces target HP by 1 on success and removes units at 0 HP", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    const resolved = resolveBattle(
      saveAttack(state, "team-1", "home-1-king", "enemy-adjacent"),
      () => 0,
    );
    const target = resolved.units.find((unit) => unit.id === "enemy-adjacent")!;

    expect(target.hp).toBe(0);
    expect(target.position).toEqual({ kind: "removed", reason: "defeated" });
  });

  it("keeps kings at 3 initial HP and defeats their team at 0 HP", () => {
    const state = createInitialGameState();
    const king = state.units.find((unit) => unit.id === "home-2-king")!;
    expect(king.hp).toBe(3);
    king.hp = 1;
    putUnit(state, "home-1-king", { kind: "tile", x: 17, y: 1 });
    putUnit(state, "home-2-king", { kind: "tile", x: 18, y: 1 });
    const resolved = resolveBattle(
      saveAttack(state, "team-1", "home-1-king", "home-2-king"),
      () => 0,
    );

    expect(
      resolved.units.find((unit) => unit.id === "home-2-king")?.position.kind,
    ).toBe("removed");
    expect(resolved.teams.find((team) => team.id === "team-2")?.status).toBe(
      "defeated",
    );
  });

  it("can target enemy units inside a base when the base is in range", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-archer-test", "team-1", "archer", {
      kind: "tile",
      x: 8,
      y: 1,
    });

    expect(targetIds(state, "team-1-archer-test")).toContain(
      "neutral-north-infantry",
    );
  });

  it("protects okuzashiki units while another friendly unit remains in the home base", () => {
    const state = createInitialGameState();
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 3,
      y: 1,
    });

    expect(targetIds(state, "team-2-archer-test")).not.toContain("home-1-king");
    expect(targetIds(state, "team-2-archer-test")).toContain(
      "home-1-strategist",
    );
  });

  it("allows targeting the okuzashiki unit when it is the only friendly unit in the home base", () => {
    const state = createInitialGameState();
    const strategist = state.units.find(
      (unit) => unit.id === "home-1-strategist",
    )!;
    clearPreviousSlot(state, strategist.position);
    strategist.position = { kind: "removed", reason: "defeated" };
    strategist.hp = 0;
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 3,
      y: 1,
    });

    expect(targetIds(state, "team-2-archer-test")).toContain("home-1-king");
  });

  it("changes GameState only when battle is resolved", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "enemy-adjacent", "team-2", "king", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    const planned = saveAttack(
      state,
      "team-1",
      "home-1-king",
      "enemy-adjacent",
    );
    const beforeResolveHp = planned.units.find(
      (unit) => unit.id === "enemy-adjacent",
    )?.hp;
    const resolved = resolveBattle(planned, () => 0);

    expect(beforeResolveHp).toBe(3);
    expect(
      resolved.units.find((unit) => unit.id === "enemy-adjacent")?.hp,
    ).toBe(2);
  });

  it("resolves mutual attacks fully simultaneously", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });

    const planned = saveAttacks(state, [
      ["team-1", "team-1-infantry-test", "team-2-infantry-test"],
      ["team-2", "team-2-infantry-test", "team-1-infantry-test"],
    ]);
    const resolved = resolveBattle(planned, () => 0);

    expect(
      resolved.units.find((unit) => unit.id === "team-1-infantry-test")
        ?.position.kind,
    ).toBe("removed");
    expect(
      resolved.units.find((unit) => unit.id === "team-2-infantry-test")
        ?.position.kind,
    ).toBe("removed");
  });

  it("resolves 3-unit cyclic attacks even when every attacker is defeated", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 3,
      y: 1,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    addUnit(state, "team-3-infantry-test", "team-3", "infantry", {
      kind: "tile",
      x: 3,
      y: 2,
    });

    const planned = saveAttacks(state, [
      ["team-1", "team-1-infantry-test", "team-2-infantry-test"],
      ["team-2", "team-2-infantry-test", "team-3-infantry-test"],
      ["team-3", "team-3-infantry-test", "team-1-infantry-test"],
    ]);
    const resolved = resolveBattle(planned, () => 0);

    expect(
      resolved.units.find((unit) => unit.id === "team-1-infantry-test")
        ?.position.kind,
    ).toBe("removed");
    expect(
      resolved.units.find((unit) => unit.id === "team-2-infantry-test")
        ?.position.kind,
    ).toBe("removed");
    expect(
      resolved.units.find((unit) => unit.id === "team-3-infantry-test")
        ?.position.kind,
    ).toBe("removed");
  });

  it("keeps an attack if the attacker is defeated in the same battle", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-king", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });

    const planned = saveAttacks(state, [
      ["team-1", "home-1-king", "team-2-infantry-test"],
      ["team-2", "team-2-infantry-test", "home-1-king"],
    ]);
    const resolved = resolveBattle(planned, () => 0);

    expect(
      resolved.units.find((unit) => unit.id === "team-2-infantry-test")
        ?.position.kind,
    ).toBe("removed");
    expect(resolved.units.find((unit) => unit.id === "home-1-king")?.hp).toBe(
      2,
    );
  });

  it("combines multiple successful attacks into target damage", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-a", "team-1", "infantry", {
      kind: "tile",
      x: 3,
      y: 1,
    });
    addUnit(state, "team-1-infantry-b", "team-1", "infantry", {
      kind: "tile",
      x: 3,
      y: 2,
    });
    addUnit(state, "enemy-king", "team-2", "king", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    const planned = saveAttacks(state, [
      ["team-1", "team-1-infantry-a", "enemy-king"],
      ["team-1", "team-1-infantry-b", "enemy-king"],
    ]);
    const resolved = resolveBattle(planned, () => 0);

    expect(resolved.units.find((unit) => unit.id === "enemy-king")?.hp).toBe(1);
  });

  it("does not change simultaneous battle results when attack intent order changes", () => {
    const first = createInitialGameState();
    addUnit(first, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    addUnit(first, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    const second = structuredClone(first) as GameState;

    const firstResolved = resolveBattle(
      saveAttacks(first, [
        ["team-1", "team-1-infantry-test", "team-2-infantry-test"],
        ["team-2", "team-2-infantry-test", "team-1-infantry-test"],
      ]),
      () => 0,
    );
    const secondResolved = resolveBattle(
      saveAttacks(second, [
        ["team-2", "team-2-infantry-test", "team-1-infantry-test"],
        ["team-1", "team-1-infantry-test", "team-2-infantry-test"],
      ]),
      () => 0,
    );

    expect(
      firstResolved.units.map((unit) => [unit.id, unit.hp, unit.position.kind]),
    ).toEqual(
      secondResolved.units.map((unit) => [
        unit.id,
        unit.hp,
        unit.position.kind,
      ]),
    );
  });

  it("does not depend on unitId ordering for simultaneous outcomes", () => {
    const state = createInitialGameState();
    addUnit(state, "z-attacker", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    addUnit(state, "a-defender", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttacks(state, [
        ["team-1", "z-attacker", "a-defender"],
        ["team-2", "a-defender", "z-attacker"],
      ]),
      () => 0,
    );

    expect(
      resolved.units.find((unit) => unit.id === "z-attacker")?.position.kind,
    ).toBe("removed");
    expect(
      resolved.units.find((unit) => unit.id === "a-defender")?.position.kind,
    ).toBe("removed");
  });

  it("uses the unit-type attack success table", () => {
    const normal = { targetInBase: false };

    expect(getBaseAttackDenominator("infantry", "archer", normal)).toBe(5);
    expect(getBaseAttackDenominator("infantry", "cavalry", normal)).toBe(7);
    expect(getBaseAttackDenominator("infantry", "infantry", normal)).toBe(6);
    expect(getBaseAttackDenominator("archer", "cavalry", normal)).toBe(5);
    expect(getBaseAttackDenominator("cavalry", "infantry", normal)).toBe(5);
    expect(getBaseAttackDenominator("ninja", "infantry", normal)).toBe(7);
    expect(getBaseAttackDenominator("ninja", "strategist", normal)).toBe(5);
    expect(getBaseAttackDenominator("king", "infantry", normal)).toBe(5);
    expect(getBaseAttackDenominator("king", "king", normal)).toBe(6);
    expect(
      getBaseAttackDenominator("strategist", "infantry", normal),
    ).toBeNull();
  });

  it("limits engineers to siege attacks and allows kings inside bases", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-engineer-test", "team-1", "engineer", {
      kind: "tile",
      x: 8,
      y: 1,
    });
    addUnit(state, "enemy-ground-infantry", "team-2", "infantry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    addUnit(state, "enemy-base-king", "team-2", "king", {
      kind: "base",
      baseId: "neutral-north",
      slotId: "slot_1_1",
    });

    const ids = targetIds(state, "team-1-engineer-test");
    expect(ids).toContain("neutral-north-infantry");
    expect(ids).toContain("neutral-north-cavalry");
    expect(ids).toContain("enemy-base-king");
    expect(ids).not.toContain("enemy-ground-infantry");
    expect(
      getBaseAttackDenominator("engineer", "infantry", { targetInBase: true }),
    ).toBe(5);
    expect(
      getBaseAttackDenominator("engineer", "engineer", { targetInBase: true }),
    ).toBe(6);
    expect(
      getBaseAttackDenominator("engineer", "king", { targetInBase: true }),
    ).toBe(5);
    expect(
      getBaseAttackDenominator("engineer", "infantry", { targetInBase: false }),
    ).toBeNull();
  });

  it("applies water ninja combat restrictions", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    addUnit(state, "team-1-ground-ninja", "team-1", "ninja", {
      kind: "tile",
      x: 4,
      y: 3,
    });
    addUnit(state, "team-1-water-ninja", "team-1", "ninja", {
      kind: "water",
      x: 4,
      y: 2,
    });
    addUnit(state, "team-2-water-ninja", "team-2", "ninja", {
      kind: "water",
      x: 5,
      y: 2,
    });
    addUnit(state, "team-2-ground-infantry", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 3,
    });

    expect(targetIds(state, "team-1-infantry-test")).not.toContain(
      "team-2-water-ninja",
    );
    expect(targetIds(state, "team-1-water-ninja")).not.toContain(
      "team-2-ground-infantry",
    );
    expect(targetIds(state, "team-1-water-ninja")).toContain(
      "team-2-water-ninja",
    );
    expect(targetIds(state, "team-1-ground-ninja")).not.toContain(
      "team-2-water-ninja",
    );
  });

  it("supports encouragement as a non-stacking denominator reduction", () => {
    expect(applyEncouragementToDenominator(5, true)).toBe(4);
    expect(applyEncouragementToDenominator(6, true)).toBe(5);
    expect(applyEncouragementToDenominator(7, true)).toBe(6);
    expect(
      getFinalAttackDenominator("infantry", "archer", {
        targetInBase: false,
        encouraged: true,
      }),
    ).toBe(4);
    expect(
      applyEncouragementToDenominator(
        applyEncouragementToDenominator(5, true),
        false,
      ),
    ).toBe(4);
  });

  it("does not change attack success when only the defender is encouraged", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 6,
      y: 1,
    });

    expect(
      targetDenominator(state, "team-2-infantry-test", "team-1-infantry-test"),
    ).toBe(6);
  });

  it("does not encourage water attackers even if they carry an encouraged status", () => {
    const state = createInitialGameState();
    const attacker = addUnit(state, "team-1-water-ninja", "team-1", "ninja", {
      kind: "water",
      x: 4,
      y: 2,
    });
    attacker.statuses.push({ kind: "encouraged" }, { kind: "encouraged" });
    addUnit(state, "team-2-water-ninja", "team-2", "ninja", {
      kind: "water",
      x: 5,
      y: 2,
    });

    expect(
      targetDenominator(state, "team-1-water-ninja", "team-2-water-ninja"),
    ).toBe(6);
  });

  it("keeps battle-start encouragement even if the strategist is defeated in that battle", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    putUnit(state, "home-1-strategist", { kind: "tile", x: 5, y: 1 });
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 3,
      y: 2,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 6,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttacks(state, [
        ["team-1", "team-1-infantry-test", "team-2-archer-test"],
        ["team-2", "team-2-infantry-test", "home-1-strategist"],
      ]),
      () => 0,
    );

    expect(
      resolved.units.find((unit) => unit.id === "home-1-strategist")?.position
        .kind,
    ).toBe("removed");
    expect(
      resolved.logs.some(
        (log) =>
          log.message.includes("team-1-infantry-test") &&
          log.message.includes("final 1/4"),
      ),
    ).toBe(true);
  });

  it("keeps fully simultaneous resolution with an encouraged mutual attack", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 6,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttacks(state, [
        ["team-1", "team-1-infantry-test", "team-2-infantry-test"],
        ["team-2", "team-2-infantry-test", "team-1-infantry-test"],
      ]),
      () => 0,
    );

    expect(
      resolved.units.find((unit) => unit.id === "team-1-infantry-test")
        ?.position.kind,
    ).toBe("removed");
    expect(
      resolved.units.find((unit) => unit.id === "team-2-infantry-test")
        ?.position.kind,
    ).toBe("removed");
    expect(
      resolved.logs.some(
        (log) =>
          log.message.includes("team-1-infantry-test") &&
          log.message.includes("final 1/5"),
      ),
    ).toBe(true);
  });

  it("sorts attack candidates by provisional deterministic success priority", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    addUnit(state, "enemy-normal", "team-2", "infantry", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    addUnit(state, "enemy-favorable", "team-2", "archer", {
      kind: "tile",
      x: 3,
      y: 1,
    });
    addUnit(state, "enemy-king", "team-2", "king", {
      kind: "tile",
      x: 3,
      y: 2,
    });

    const ordered = targetIds(state, "team-1-infantry-test");
    expect(ordered.slice(0, 3)).toEqual([
      "enemy-king",
      "enemy-favorable",
      "enemy-normal",
    ]);
    expect(targetIds(state, "team-1-infantry-test")).toEqual(ordered);
  });

  it("sorts wounded enemies before full-HP enemies under otherwise equal conditions", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-archer-test", "team-1", "archer", {
      kind: "tile",
      x: 4,
      y: 1,
    });
    const wounded = addUnit(state, "enemy-king-wounded", "team-2", "king", {
      kind: "tile",
      x: 5,
      y: 1,
    });
    addUnit(state, "enemy-king-full", "team-2", "king", {
      kind: "tile",
      x: 6,
      y: 1,
    });
    wounded.hp = 2;

    expect(targetIds(state, "team-1-archer-test").slice(0, 2)).toEqual([
      "enemy-king-wounded",
      "enemy-king-full",
    ]);
  });
});
