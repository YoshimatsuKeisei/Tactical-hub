import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { getAttackCandidates, resolveBattle, saveAttackIntent } from "../engine/battle";
import { commitUnitMovement, getMovementCandidates } from "../engine/movement";
import { buildUnitTurnFlag, clearLegacyRetreatState } from "../engine/retreat";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition, UnitType } from "../types";

function clearBaseSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((candidate) => candidate.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function addUnit(state: GameState, id: string, ownerTeamId: string, type: UnitType, position: UnitPosition) {
  const unit: Unit = { id, ownerTeamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function putOnTile(state: GameState, unit: Unit, x: number, y: number) {
  clearBaseSlot(state, unit.position);
  unit.position = { kind: "tile", x, y };
}

function attackFixture() {
  const state = createInitialGameState();
  state.phase = state.turnState.phase = "attack_input";
  const attacker = addUnit(state, "legacy-retreat-attacker", "team-1", "archer", { kind: "tile", x: 4, y: 1 });
  const target = addUnit(state, "legacy-retreat-target", "team-2", "infantry", { kind: "tile", x: 5, y: 1 });
  attacker.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-1", remainingTurns: 9 });
  target.statuses.push({ kind: "retreating", retreatTargetBaseId: "home-2", remainingTurns: 9 });
  return { state, attacker, target };
}

describe("retreat mode removal", () => {
  it("keeps independent battle history while retreat eligibility is always disabled", () => {
    const state = createInitialGameState();
    const unit = addUnit(state, "battle-history-unit", "team-1", "infantry", { kind: "tile", x: 8, y: 1 });
    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-2";
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push("neutral-north");
    const flag = buildUnitTurnFlag(state, unit, state.turnNumber, true, true, false);
    expect(flag).toMatchObject({
      unitId: unit.id,
      wasAliveAtBattleStart: true,
      survivedPreviousBattle: true,
      attackedInPreviousBattle: true,
      wasTargetedInPreviousBattle: false,
      retreatEligible: false,
      retreatEligibilityReason: "retreat mode removed",
    });
    expect(flag.enemyBaseDistanceAtBattleStart).toBeTypeOf("number");
  });

  it("does not create retreat state when moving toward a friendly base", () => {
    const state = createInitialGameState();
    const unit = state.units.find((candidate) => candidate.id === "home-1-strategist")!;
    putOnTile(state, unit, 4, 1);
    state.unitTurnFlags = [{
      unitId: unit.id,
      battleTurnNumber: state.turnNumber,
      wasAliveAtBattleStart: true,
      survivedPreviousBattle: true,
      attackedInPreviousBattle: true,
      wasTargetedInPreviousBattle: false,
      retreatEligible: true,
      retreatFriendlyBaseIdsAtEligibility: ["home-1"],
      retreatHostileBaseIdsAtEligibility: ["neutral-north"],
    }];
    const destination = getMovementCandidates(state, unit.id)[0];
    expect(destination).toBeDefined();
    const moved = commitUnitMovement(state, { teamId: unit.ownerTeamId, unitId: unit.id, from: unit.position, to: destination, stay: false });
    expect(moved.units.find((candidate) => candidate.id === unit.id)?.statuses).not.toContainEqual(expect.objectContaining({ kind: "retreating" }));
    expect(moved.unitTurnFlags.find((flag) => flag.unitId === unit.id)).toMatchObject({ retreatEligible: false, retreatEligibilityReason: "retreat mode removed" });
  });

  it("allows a legacy retreating unit to attack with the normal hit denominator", () => {
    const { state, attacker, target } = attackFixture();
    const withLegacyStatus = getAttackCandidates(state, attacker.id).find((candidate) => candidate.unitId === target.id);
    const withoutLegacyStatus = structuredClone(state) as GameState;
    withoutLegacyStatus.units.find((unit) => unit.id === attacker.id)!.statuses = [];
    withoutLegacyStatus.units.find((unit) => unit.id === target.id)!.statuses = [];
    const normal = getAttackCandidates(withoutLegacyStatus, attacker.id).find((candidate) => candidate.unitId === target.id);
    expect(withLegacyStatus).toBeDefined();
    expect(withLegacyStatus?.finalSuccessDenominator).toBe(normal?.finalSuccessDenominator);
    const saved = saveAttackIntent(state, { teamId: attacker.ownerTeamId, attackerUnitId: attacker.id, target: { kind: "unit", unitId: target.id }, pass: false });
    expect(saved).not.toBe(state);
  });

  it("never grants retreat state after battle resolution", () => {
    const { state, attacker, target } = attackFixture();
    const saved = saveAttackIntent(state, { teamId: attacker.ownerTeamId, attackerUnitId: attacker.id, target: { kind: "unit", unitId: target.id }, pass: false });
    const resolved = resolveBattle(saved, () => 0.99);
    expect(resolved.units.every((unit) => unit.statuses.every((status) => status.kind !== "retreating"))).toBe(true);
    expect(resolved.unitTurnFlags).toEqual(expect.arrayContaining([
      expect.objectContaining({ unitId: attacker.id, attackedInPreviousBattle: true, retreatEligible: false }),
      expect.objectContaining({ unitId: target.id, wasTargetedInPreviousBattle: true, retreatEligible: false }),
    ]));
  });

  it("sanitizes only legacy retreat data from old saves", () => {
    const state = createInitialGameState();
    const unit = state.units[0];
    unit.statuses = [
      { kind: "retreating", retreatTargetBaseId: "home-1", remainingTurns: 7 },
      { kind: "encouraged", remainingTurns: 2, sourceId: "strategist" },
    ];
    state.unitTurnFlags = [{
      unitId: unit.id,
      battleTurnNumber: 4,
      positionAtBattleStart: structuredClone(unit.position),
      enemyBaseDistanceAtBattleStart: 2,
      enemyBaseWithin3AtBattleStart: true,
      retreatFriendlyBaseIdsAtEligibility: ["home-1"],
      retreatHostileBaseIdsAtEligibility: ["home-2"],
      wasAliveAtBattleStart: true,
      survivedPreviousBattle: true,
      attackedInPreviousBattle: false,
      wasTargetedInPreviousBattle: true,
      retreatEligible: true,
      retreatEligibilityReason: "eligible",
    }];
    const positionBefore = structuredClone(unit.position);
    clearLegacyRetreatState(state);
    expect(state.units[0].position).toEqual(positionBefore);
    expect(state.units[0].statuses).toEqual([{ kind: "encouraged", remainingTurns: 2, sourceId: "strategist" }]);
    expect(state.unitTurnFlags[0]).toMatchObject({
      battleTurnNumber: 4,
      enemyBaseDistanceAtBattleStart: 2,
      enemyBaseWithin3AtBattleStart: true,
      wasTargetedInPreviousBattle: true,
      retreatEligible: false,
      retreatEligibilityReason: "retreat mode removed",
    });
  });
});
