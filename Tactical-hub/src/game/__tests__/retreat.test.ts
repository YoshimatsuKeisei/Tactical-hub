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
  clearInvalidRetreatTargets,
  getEnemyControlledBases,
  getNearestFriendlyBaseDistance,
  getLegalRetreatRouteDistance,
  getRetreatDirectionIndicators,
  getRetreatTargetBaseId,
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

function makeNorthRelayHostile(state: GameState) {
  state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-2";
  state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push("neutral-north");
}

describe("retreat", () => {
  it("uses legal road routes and treats neutral or enemy bases as blockers", () => {
    const neutralBlocked = createInitialGameState();
    expect(getLegalRetreatRouteDistance(neutralBlocked, "team-1", { kind: "tile", x: 18, y: 1 })).toBeUndefined();

    const enemyBlocked = createInitialGameState();
    makeNorthRelayHostile(enemyBlocked);
    expect(getLegalRetreatRouteDistance(enemyBlocked, "team-1", { kind: "tile", x: 18, y: 1 })).toBeUndefined();

    const reachable = createInitialGameState();
    expect(getLegalRetreatRouteDistance(reachable, "team-1", { kind: "tile", x: 8, y: 1 })).toMatchObject({ baseIds: ["home-1"] });

    reachable.bases.find((base) => base.id === "home-2")!.ownerTeamId = "team-1";
    reachable.teams.find((team) => team.id === "team-1")!.controlledBaseIds.push("home-2");
    expect(getLegalRetreatRouteDistance(reachable, "team-1", { kind: "tile", x: 18, y: 1 })).toMatchObject({ baseIds: ["home-2"] });
  });

  it("marks a surviving battle participant near an enemy base as retreat eligible", () => {
    const state = createInitialGameState();
    makeNorthRelayHostile(state);
    addUnit(state, "team-1-infantry-test", "team-1", "infantry", {
      kind: "tile",
      x: 8,
      y: 1,
    });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 9,
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
    makeNorthRelayHostile(state);
    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 8,
      y: 1,
    });
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 9,
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
      enemyBaseDistanceAtBattleStart: 2,
      retreatEligible: true,
    });
  });

  it("marks a surviving target as retreat eligible when the incoming attack hits but does not defeat it", () => {
    const state = createInitialGameState();
    makeNorthRelayHostile(state);
    const cavalry = addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 8,
      y: 1,
    });
    cavalry.hp = 2;
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 9,
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
    makeNorthRelayHostile(state);
    addUnit(state, "team-1-cavalry-test", "team-1", "cavalry", {
      kind: "tile",
      x: 8,
      y: 1,
    });
    addUnit(state, "team-2-archer-test", "team-2", "archer", {
      kind: "tile",
      x: 9,
      y: 1,
    });

    const finalState = resolveBattle(
      saveAttack(state, "team-2", "team-2-archer-test", "team-1-cavalry-test"),
      () => 1,
    );
    const finalUnit = finalState.units.find(
      (unit) => unit.id === "team-1-cavalry-test",
    )!;

    expect(finalState.phase).toBe("strategist_action_input");
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
      nearestHostileBaseId: "neutral-north",
      nearestHostileBaseController: "team-2",
      withinHostileBaseRangeAtBattle: true,
    });
  });

  it("marks the attacker as retreat eligible when it also satisfies the hostile-base range condition", () => {
    const state = createInitialGameState();
    makeNorthRelayHostile(state);
    addUnit(state, "team-1-archer-test", "team-1", "archer", {
      kind: "tile",
      x: 8,
      y: 1,
    });
    addUnit(state, "team-2-cavalry-test", "team-2", "cavalry", {
      kind: "tile",
      x: 9,
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
      retreatEligible: false,
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
      retreatEligible: false,
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
      x: 8,
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
      x: 8,
      y: 1,
    });
    unit.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });

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
      x: 8,
      y: 1,
    });
    markRetreatEligible(state, unit.id);

    expect(
      getRetreatMoveEffect(state, unit, unit.position, {
        kind: "tile",
        x: 7,
        y: 1,
      }),
    ).toBe("start");

    const resolved = resolveMovement(
      saveMovementIntent(state, {
        teamId: "team-1",
        unitId: unit.id,
        from: unit.position,
        to: { kind: "tile", x: 7, y: 1 },
        stay: false,
      }),
    );

    expect(
      isRetreating(
        resolved.units.find((candidate) => candidate.id === unit.id)!,
      ),
    ).toBe(true);
    expect(getRetreatTargetBaseId(resolved.units.find((candidate) => candidate.id === unit.id)!)).toBe("home-1");
    expect(resolved.unitTurnFlags).toContainEqual(expect.objectContaining({ unitId: unit.id, retreatEligible: true }));
  });

  it("clears an invalid retreat target without automatically switching to another friendly base", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "team-1-retreater", "team-1", "infantry", { kind: "tile", x: 8, y: 1 });
    addUnit(state, "team-2-nearby", "team-2", "infantry", { kind: "tile", x: 9, y: 1 });
    unit.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-1";
    state.teams.find((team) => team.id === "team-1")!.controlledBaseIds.push("neutral-north");
    state.bases.find((base) => base.id === "home-1")!.ownerTeamId = "team-2";
    state.teams.find((team) => team.id === "team-1")!.controlledBaseIds = ["neutral-north"];
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push("home-1");

    expect(getLegalRetreatRouteDistance(state, "team-1", unit.position, "neutral-north")).toBeDefined();
    clearInvalidRetreatTargets(state);

    const cleared = state.units.find((candidate) => candidate.id === unit.id)!;
    expect(isRetreating(cleared)).toBe(false);
    expect(getRetreatTargetBaseId(cleared)).toBeUndefined();
    expect(getAttackCandidates(state, unit.id).map((target) => target.unitId)).toContain("team-2-nearby");
  });

  it("uses the battle-start retreat snapshot, then clears an invalid target before the next input phase", () => {
    const state = createInitialGameState();
    const defender = addUnit(state, "team-1-retreater", "team-1", "infantry", { kind: "tile", x: 8, y: 1 });
    defender.hp = 2;
    defender.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-2" });
    addUnit(state, "team-2-attacker", "team-2", "infantry", { kind: "tile", x: 9, y: 1 });
    const resolved = resolveBattle(saveAttack(state, "team-2", "team-2-attacker", defender.id), () => 0.08);

    expect(resolved.logs.some((log) => log.message.includes("final 1/12") && log.message.includes("result: success"))).toBe(true);
    const finalDefender = resolved.units.find((unit) => unit.id === defender.id)!;
    expect(finalDefender.hp).toBe(1);
    expect(isRetreating(finalDefender)).toBe(false);
    expect(getRetreatTargetBaseId(finalDefender)).toBeUndefined();
    expect(resolved.phase).toBe("strategist_action_input");
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
      { kind: "tile", x: 8, y: 1 },
    );
    maintainingUnit.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
    const maintained = resolveMovement(
      saveMovementIntent(maintain, {
        teamId: "team-1",
        unitId: maintainingUnit.id,
        from: maintainingUnit.position,
        to: { kind: "tile", x: 7, y: 1 },
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
      { kind: "tile", x: 7, y: 1 },
    );
    releasingUnit.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
    const released = resolveMovement(
      saveMovementIntent(release, {
        teamId: "team-1",
        unitId: releasingUnit.id,
        from: releasingUnit.position,
        to: { kind: "tile", x: 8, y: 1 },
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
    stayingUnit.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
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
    enteringUnit.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
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
    retreating.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
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

    const saved = saveAttack(state, "team-1", retreating.id, "team-2-infantry-test");
    expect(saved.turnState.actionIntents.flatMap((entry) => entry.attackIntents)).toEqual([]);
    const resolved = resolveBattle(saved, () => 0);

    expect(
      resolved.units.find((unit) => unit.id === "team-2-infantry-test")?.hp,
    ).toBe(1);
  });

  it("halves the final hit probability against formally retreating infantry", () => {
    const state = createInitialGameState();
    const retreating = addUnit(
      state,
      "team-1-infantry-test",
      "team-1",
      "infantry",
      { kind: "tile", x: 18, y: 1 },
    );
    retreating.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
    addUnit(state, "team-2-infantry-test", "team-2", "infantry", {
      kind: "tile",
      x: 17,
      y: 1,
    });

    const denominator = getAttackCandidates(state, "team-2-infantry-test").find(
      (target) => target.unitId === retreating.id,
    )?.finalSuccessDenominator;

    expect(denominator).toBe(12);
  });

  it("preserves matchup and encouragement before halving retreating infantry hit probability", () => {
    const denominatorFor = (attackerType: UnitType, encouraged = false) => {
      const state = createInitialGameState();
      const defender = addUnit(state, `defender-${attackerType}`, "team-1", "infantry", { kind: "tile", x: 5, y: 1 });
      defender.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
      addUnit(state, `attacker-${attackerType}`, "team-2", attackerType, { kind: "tile", x: 6, y: 1 });
      if (encouraged) {
        const strategist = addUnit(state, "team-2-strategist-test", "team-2", "strategist", { kind: "tile", x: 7, y: 1 });
        strategist.role = "encourage";
      }
      return getAttackCandidates(state, `attacker-${attackerType}`).find((target) => target.unitId === defender.id)?.finalSuccessDenominator;
    };
    expect(denominatorFor("cavalry")).toBe(10);
    expect(denominatorFor("infantry")).toBe(12);
    expect(denominatorFor("archer")).toBe(14);
    expect(denominatorFor("cavalry", true)).toBe(8);
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
    retreating.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
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
    retreating.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1" });
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
