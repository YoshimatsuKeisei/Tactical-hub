import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  getAttackCandidates,
  resolveBattle,
  saveAttackIntent,
} from "../engine/battle";
import { resolveMovement, saveMovementIntent } from "../engine/movement";
import {
  getBaseControllerTeamId,
  getEnemyControlledBases,
  getNearestFriendlyBaseDistance,
  getRetreatDirectionIndicators,
  getRetreatDebugInfo,
  getRetreatMoveEffect,
  getUnitTurnFlags,
  isRetreating,
  isUnitRetreatEligible,
} from "../engine/retreat";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";

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

function markRetreatEligible(state: GameState, unitId: string) {
  state.unitTurnFlags = [
    {
      unitId,
      battleTurnNumber: state.turnNumber,
      wasAliveAtBattleStart: true,
      survivedPreviousBattle: true,
      attackedInPreviousBattle: true,
      wasTargetedInPreviousBattle: false,
      retreatEligible: true,
    },
  ];
}

describe("retreat", () => {
  it("marks a surviving battle participant near an enemy base as retreat eligible", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttack(
        state,
        "team-1",
        "team-1-infantry-test",
        "team-2-infantry-test",
      ),
      () => 1,
    );

    expect(getUnitTurnFlags(resolved, "team-1-infantry-test")).toMatchObject({
      attackedInPreviousBattle: true,
      survivedPreviousBattle: true,
      retreatEligible: true,
    });
    expect(
      isUnitRetreatEligible(
        resolved,
        resolved.units.find((unit) => unit.id === "team-1-infantry-test")!,
      ),
    ).toBe(true);
  });

  it("marks a surviving target as retreat eligible even when the incoming attack misses", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttack(state, "team-2", "team-2-archer-test", "team-1-cavalry-test"),
      () => 1,
    );

    expect(
      resolved.units.find((unit) => unit.id === "team-1-cavalry-test")?.hp,
    ).toBe(1);
    expect(getUnitTurnFlags(resolved, "team-1-cavalry-test")).toMatchObject({
      attackedInPreviousBattle: false,
      wasTargetedInPreviousBattle: true,
      survivedPreviousBattle: true,
      enemyBaseWithin3AtBattleStart: true,
      enemyBaseDistanceAtBattleStart: 1,
      retreatEligible: true,
    });
  });

  it("marks a surviving target as retreat eligible when the incoming attack hits but does not defeat it", () => {
    const state = createInitialGameState();
    const cavalry = addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    cavalry.hp = 2;
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttack(state, "team-2", "team-2-archer-test", "team-1-cavalry-test"),
      () => 0,
    );

    expect(
      resolved.units.find((unit) => unit.id === "team-1-cavalry-test")?.hp,
    ).toBe(1);
    expect(
      getUnitTurnFlags(resolved, "team-1-cavalry-test")?.retreatEligible,
    ).toBe(true);
  });

  it("keeps retreat eligibility in the final resolveBattle state after intent clearing and phase update", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const finalState = resolveBattle(
      saveAttack(state, "team-2", "team-2-archer-test", "team-1-cavalry-test"),
      () => 1,
    );
    const finalUnit = finalState.units.find(
      (unit) => unit.id === "team-1-cavalry-test",
    )!;

    expect(finalState.phase).toBe("attack_input");
    expect(
      finalState.turnState.actionIntents.flatMap(
        (intent) => intent.attackIntents ?? [],
      ),
    ).toEqual([]);
    expect(getUnitTurnFlags(finalState, finalUnit.id)?.retreatEligible).toBe(
      true,
    );
    expect(isUnitRetreatEligible(finalState, finalUnit)).toBe(true);
    expect(getRetreatDebugInfo(finalState, finalUnit.id)).toMatchObject({
      eligible: true,
      wasTargeted: true,
      participatedInBattle: true,
      survivedBattle: true,
      battlePositionKind: "tile",
      nearestHostileBaseId: "home-2",
      nearestHostileBaseController: "team-2",
      withinHostileBaseRangeAtBattle: true,
    });
  });

  it("marks the attacker as retreat eligible when it also satisfies the hostile-base range condition", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-archer-test", "team-1", "archer", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    addUnit(state, "team-2-cavalry-test", "team-2", "cavalry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttack(state, "team-1", "team-1-archer-test", "team-2-cavalry-test"),
      () => 1,
    );

    expect(getUnitTurnFlags(resolved, "team-1-archer-test")).toMatchObject({
      attackedInPreviousBattle: true,
      retreatEligible: true,
    });
  });

  it("uses the real initial map to treat team-2 home as a hostile controlled base for team-1", () => {
    const state = createInitialGameState();
    const home2 = state.bases.find((base) => base.id === "home-2")!;

    expect(getBaseControllerTeamId(state, home2)).toBe("team-2");
    expect(
      getEnemyControlledBases(state, "team-1").map((base) => base.id),
    ).toContain("home-2");
    expect(
      getBaseControllerTeamId(
        {
          ...state,
          bases: state.bases.map((base) =>
            base.id === "home-2"
              ? { ...base, ownerTeamId: undefined as unknown as string }
              : base,
          ),
        },
        home2,
      ),
    ).toBe("team-2");
  });

  it("grants eligibility at battle-time hostile home distance 3 and denies it at distance 4", () => {
    const atThree = createInitialGameState();
    addUnit(atThree, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 16,
      y: 1,
    });
    addUnit(atThree, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 15,
      y: 1,
    });
    const resolvedAtThree = resolveBattle(
      saveAttack(
        atThree,
        "team-2",
        "team-2-archer-test",
        "team-1-cavalry-test",
      ),
      () => 1,
    );
    expect(
      getUnitTurnFlags(resolvedAtThree, "team-1-cavalry-test"),
    ).toMatchObject({
      enemyBaseDistanceAtBattleStart: 3,
      retreatEligible: true,
    });

    const atFour = createInitialGameState();
    addUnit(atFour, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 15,
      y: 1,
    });
    addUnit(atFour, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 14,
      y: 1,
    });
    const resolvedAtFour = resolveBattle(
      saveAttack(atFour, "team-2", "team-2-archer-test", "team-1-cavalry-test"),
      () => 1,
    );
    expect(
      getUnitTurnFlags(resolvedAtFour, "team-1-cavalry-test"),
    ).toMatchObject({
      enemyBaseDistanceAtBattleStart: 4,
      retreatEligible: false,
    });
  });

  it("uses enemy controlledBaseIds, not only base ownerTeamId, for battle-time retreat eligibility", () => {
    const state = createInitialGameState();
    state.teams
      .find((team) => team.id === "team-2")!
      .controlledBaseIds.push("neutral-north");
    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 9,
      y: 1,
    });
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 8,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttack(state, "team-2", "team-2-archer-test", "team-1-cavalry-test"),
      () => 1,
    );

    expect(getUnitTurnFlags(resolved, "team-1-cavalry-test")).toMatchObject({
      wasTargetedInPreviousBattle: true,
      enemyBaseDistanceAtBattleStart: 1,
      retreatEligible: true,
    });
  });

  it("does not mark distant, defeated, water, or non-participant units as retreat eligible", () => {
    const distant = createInitialGameState();

    addUnit(distant, "team-1-archer-test", "team-1", "archer", {
      kind: "tile",
      x: 4,
      y: 1,
    });

    addUnit(distant, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 7,
      y: 1,
    });

    const distantResolved = resolveBattle(
      saveAttack(
        distant,
        "team-1",
        "team-1-archer-test",
        "team-2-infantry-test",
      ),
      () => 1,
    );

    expect(
      getUnitTurnFlags(distantResolved, "team-1-archer-test")?.retreatEligible,
    ).toBe(false);
    expect(
      getUnitTurnFlags(distantResolved, "team-1-archer-test")
        ?.retreatEligibilityReason,
    ).toContain("distance");

    const defeated = createInitialGameState();
    addUnit(defeated, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    addUnit(defeated, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });
    const defeatedResolved = resolveBattle(
      saveAttack(
        defeated,
        "team-2",
        "team-2-infantry-test",
        "team-1-infantry-test",
      ),
      () => 0,
    );
    expect(
      getUnitTurnFlags(defeatedResolved, "team-1-infantry-test")
        ?.retreatEligible,
    ).toBe(false);

    const water = createInitialGameState();
    addUnit(water, "team-1-water-ninja", "team-1", "ninja", {
      kind: "water",
      x: 17,
      y: 2,
    });
    addUnit(water, "team-2-water-ninja", "team-2", "ninja", {
      kind: "water",
      x: 16,
      y: 2,
    });
    const waterResolved = resolveBattle(
      saveAttack(water, "team-1", "team-1-water-ninja", "team-2-water-ninja"),
      () => 1,
    );
    expect(
      getUnitTurnFlags(waterResolved, "team-1-water-ninja")?.retreatEligible,
    ).toBe(false);
    expect(getUnitTurnFlags(waterResolved, "home-1-king")).toBeUndefined();
  });

  it("uses the nearest controlled friendly base as the retreat destination", () => {
    const state = createInitialGameState();
    state.teams
      .find((team) => team.id === "team-1")!
      .controlledBaseIds.push("neutral-north");
    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId =
      "team-1";

    const distance = getNearestFriendlyBaseDistance(state, "team-1", {
      kind: "tile",
      x: 9,
      y: 1,
    });

    expect(distance).toEqual({ distance: 1, baseIds: ["neutral-north"] });
  });

  it("does not create retreat direction indicators for units without retreat state", () => {
    const state = createInitialGameState();
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });

    expect(
      getRetreatDirectionIndicators(state, "team-1-infantry-test"),
    ).toEqual([]);
  });

  it("labels retreat direction indicators for eligible units", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    markRetreatEligible(state, unit.id);

    expect(
      getRetreatDirectionIndicators(state, unit.id).map(
        (indicator) => indicator.label,
      ),
    ).toEqual(["撤退する", "継戦する"]);
  });

  it("labels retreat direction indicators for retreating units", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    unit.statuses.push({ kind: "retreating" });

    expect(
      getRetreatDirectionIndicators(state, unit.id).map(
        (indicator) => indicator.label,
      ),
    ).toEqual(["撤退を継続する", "撤退を解除する"]);
  });

  it("starts retreat only when an eligible unit moves closer to its nearest friendly base", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    markRetreatEligible(state, unit.id);

    expect(
      getRetreatMoveEffect(state, unit, unit.position, {
        kind: "tile",
        x: 17,
        y: 1,
      }),
    ).toBe("start");

    const resolved = resolveMovement(
      saveMovementIntent(state, {
        teamId: "team-1",
        unitId: unit.id,
        from: unit.position,
        to: { kind: "tile", x: 17, y: 1 },
        stay: false,
      }),
    );

    expect(
      isRetreating(
        resolved.units.find((candidate) => candidate.id === unit.id)!,
      ),
    ).toBe(true);
    expect(resolved.unitTurnFlags).toEqual([]);
  });

  it("does not start retreat after a failed movement resolution", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 18,
      y: 1,
    });
    addUnit(state, "team-1-blocker", "team-1", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });
    markRetreatEligible(state, unit.id);

    const resolved = resolveMovement(
      saveMovementIntent(state, {
        teamId: "team-1",
        unitId: unit.id,
        from: unit.position,
        to: { kind: "tile", x: 17, y: 1 },
        stay: false,
      }),
    );

    expect(
      isRetreating(
        resolved.units.find((candidate) => candidate.id === unit.id)!,
      ),
    ).toBe(false);
  });

  it("maintains retreat while moving no farther from a friendly base and releases it when moving farther", () => {
    const maintain = createInitialGameState();
    const maintainingUnit = addUnit(
      maintain,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 17, y: 1 },
    );
    maintainingUnit.statuses.push({ kind: "retreating" });
    const maintained = resolveMovement(
      saveMovementIntent(maintain, {
        teamId: "team-1",
        unitId: maintainingUnit.id,
        from: maintainingUnit.position,
        to: { kind: "tile", x: 16, y: 1 },
        stay: false,
      }),
    );
    expect(
      isRetreating(
        maintained.units.find((unit) => unit.id === maintainingUnit.id)!,
      ),
    ).toBe(true);

    const release = createInitialGameState();
    const releasingUnit = addUnit(
      release,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 17, y: 1 },
    );
    releasingUnit.statuses.push({ kind: "retreating" });
    const released = resolveMovement(
      saveMovementIntent(release, {
        teamId: "team-1",
        unitId: releasingUnit.id,
        from: releasingUnit.position,
        to: { kind: "tile", x: 18, y: 1 },
        stay: false,
      }),
    );
    expect(
      isRetreating(
        released.units.find((unit) => unit.id === releasingUnit.id)!,
      ),
    ).toBe(false);
  });

  it("ends retreat by explicit stay or by entering a friendly base", () => {
    const stay = createInitialGameState();
    const stayingUnit = addUnit(
      stay,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 17, y: 1 },
    );
    stayingUnit.statuses.push({ kind: "retreating" });
    const stayed = resolveMovement(
      saveMovementIntent(stay, {
        teamId: "team-1",
        unitId: stayingUnit.id,
        from: stayingUnit.position,
        to: stayingUnit.position,
        stay: true,
      }),
    );
    expect(
      isRetreating(stayed.units.find((unit) => unit.id === stayingUnit.id)!),
    ).toBe(false);

    const baseEntry = createInitialGameState();
    const enteringUnit = addUnit(
      baseEntry,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 3, y: 1 },
    );
    enteringUnit.statuses.push({ kind: "retreating" });
    const entered = resolveMovement(
      saveMovementIntent(baseEntry, {
        teamId: "team-1",
        unitId: enteringUnit.id,
        from: enteringUnit.position,
        to: { kind: "base", baseId: "home-1", slotId: "slot_1_0" },
        stay: false,
      }),
    );
    expect(
      isRetreating(entered.units.find((unit) => unit.id === enteringUnit.id)!),
    ).toBe(false);
    expect(
      entered.units.find((unit) => unit.id === enteringUnit.id)?.position.kind,
    ).toBe("base");
  });

  it("prevents retreating attackers while still allowing enemies to target retreating units normally", () => {
    const state = createInitialGameState();
    const retreating = addUnit(
      state,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 18, y: 1 },
    );
    retreating.statuses.push({ kind: "retreating" });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    expect(getAttackCandidates(state, retreating.id)).toEqual([]);
    expect(
      getAttackCandidates(state, "team-2-infantry-test").map(
        (target) => target.unitId,
      ),
    ).toContain(retreating.id);

    const resolved = resolveBattle(
      saveAttack(state, "team-1", retreating.id, "team-2-infantry-test"),
      () => 0,
    );

    expect(
      resolved.units.find((unit) => unit.id === "team-2-infantry-test")?.hp,
    ).toBe(1);
    expect(
      resolved.logs.some((log) => log.message.includes("retreating")),
    ).toBe(true);
  });

  it("does not apply a retreat defense probability change", () => {
    const state = createInitialGameState();
    const retreating = addUnit(
      state,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 18, y: 1 },
    );
    retreating.statuses.push({ kind: "retreating" });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const denominator = getAttackCandidates(state, "team-2-infantry-test").find(
      (target) => target.unitId === retreating.id,
    )?.finalSuccessDenominator;

    expect(denominator).toBe(6);
  });

  it("allows a unit to attack again after retreat is explicitly ended by stay", () => {
    const state = createInitialGameState();
    const retreating = addUnit(
      state,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 18, y: 1 },
    );
    retreating.statuses.push({ kind: "retreating" });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const afterStay = resolveMovement(
      saveMovementIntent(state, {
        teamId: "team-1",
        unitId: retreating.id,
        from: retreating.position,
        to: retreating.position,
        stay: true,
      }),
    );

    expect(
      isRetreating(afterStay.units.find((unit) => unit.id === retreating.id)!),
    ).toBe(false);
    expect(
      getAttackCandidates(afterStay, retreating.id).map(
        (target) => target.unitId,
      ),
    ).toContain("team-2-infantry-test");
  });

  it("removes retreating status when a retreating unit is defeated", () => {
    const state = createInitialGameState();
    const retreating = addUnit(
      state,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 18, y: 1 },
    );
    retreating.statuses.push({ kind: "retreating" });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const resolved = resolveBattle(
      saveAttack(state, "team-2", "team-2-infantry-test", retreating.id),
      () => 0,
    );

    const defeated = resolved.units.find((unit) => unit.id === retreating.id)!;
    expect(defeated.position.kind).toBe("removed");
    expect(isRetreating(defeated)).toBe(false);
  });

  it("treats same-base friendly units as distance 0 for retreat destination calculations", () => {
    const state = createInitialGameState();
    putUnit(state, "home-1-strategist", {
      kind: "base",
      baseId: "home-1",
      slotId: "slot_1_1",
    });

    expect(
      getNearestFriendlyBaseDistance(state, "team-1", {
        kind: "base",
        baseId: "home-1",
        slotId: "slot_1_1",
      })?.distance,
    ).toBe(0);
  });
});
