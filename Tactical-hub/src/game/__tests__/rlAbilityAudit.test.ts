import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { aggregatePpoAbilityAudits, PpoAbilityAuditTracker } from "../cpu/rlAbilityAudit";
import { RlEnvironmentV2, type RlLegalAction } from "../cpu/rlEnvironment";
import type { CpuDecision } from "../cpu/types";
import { commitUnitMovement } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { Construction, GameState, StrategistActionIntent, Unit, UnitPosition } from "../types";

function addUnit(state: GameState, id: string, ownerTeamId: string, type: Unit["type"], position: UnitPosition) {
  const unit: Unit = { id, ownerTeamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function movementDecision(unit: Unit, to: UnitPosition): CpuDecision {
  return { kind: "movement", teamId: unit.ownerTeamId, actorKey: `movement:${unit.ownerTeamId}:${unit.id}`, unitId: unit.id, to };
}

describe("PPO ability audit", () => {
  it("counts normal production without mixing reward placement and aggregates shares", () => {
    const state = createInitialGameState();
    const first = new PpoAbilityAuditTracker(state);
    first.beforeApply(state, { kind: "production", teamId: "team-1", actorKey: "p1", choice: { teamId: "team-1", baseId: "home-1", unitType: "ninja" } });
    first.beforeApply(state, { kind: "reward", teamId: "team-1", requestId: "r1", baseId: "home-1", unitType: "engineer" });
    const firstAudit = first.finalize(state);
    expect(firstAudit.production).toMatchObject({ selectionCount: 1, specialUnitCount: 1, specialUnitShare: 1 });
    expect(firstAudit.production.counts).toMatchObject({ ninja: 1, engineer: 0 });

    const second = new PpoAbilityAuditTracker(state);
    second.beforeApply(state, { kind: "production", teamId: "team-1", actorKey: "p2", choice: { teamId: "team-1", baseId: "home-1", unitType: "infantry" } });
    const aggregate = aggregatePpoAbilityAudits([
      { environmentIndex: 0, seed: 417, audit: firstAudit },
      { environmentIndex: 1, seed: 418, audit: second.finalize(state) },
    ]);
    expect(aggregate.production.selectionCount).toBe(2);
    expect(aggregate.production.counts).toMatchObject({ infantry: 1, ninja: 1 });
    expect(aggregate.production.averagePerMatch).toMatchObject({ infantry: 0.5, ninja: 0.5 });
  });

  it("separates construction instructions, successes, failures, bridge objects, and bridge tiles without double counting", () => {
    const state = createInitialGameState();
    const tracker = new PpoAbilityAuditTracker(state);
    const bridgeTiles = [{ x: 4, y: 2 }, { x: 4, y: 3 }];
    const bridgeIntent: StrategistActionIntent = { teamId: "team-1", strategistUnitId: "home-1-strategist", action: "place_bridge", tiles: bridgeTiles };
    tracker.beforeApply(state, { kind: "strategist", teamId: "team-1", actorKey: "bridge", intent: bridgeIntent });
    const bridge: Construction = { id: "bridge-audit", kind: "bridge", ownerTeamId: "team-1", managerUnitId: bridgeIntent.strategistUnitId, tiles: bridgeTiles.map((cell) => ({ ...cell })), placedTurn: 1, active: true };
    state.constructions.push(bridge);
    state.logs.push({ id: "audit-bridge-success", turnNumber: 1, type: "construction", message: "placed bridge", relatedIds: [bridgeIntent.strategistUnitId] });
    tracker.afterApply(state, { kind: "resolve_strategists", teamId: "all" });
    tracker.afterApply(state, { kind: "resolve_strategists", teamId: "all" });

    const obstacleIntent: StrategistActionIntent = { teamId: "team-1", strategistUnitId: "home-1-strategist", action: "place_obstacle", tiles: [{ x: 4, y: 1 }] };
    tracker.beforeApply(state, { kind: "strategist", teamId: "team-1", actorKey: "obstacle", intent: obstacleIntent });
    state.logs.push({ id: "audit-obstacle-fail", turnNumber: 1, type: "construction", message: "placement failed by simultaneous conflict.", relatedIds: [obstacleIntent.strategistUnitId] });
    tracker.afterApply(state, { kind: "resolve_strategists", teamId: "all" });

    const audit = tracker.finalize(state);
    expect(audit.strategist.bridge).toEqual({ instructionCount: 1, successCount: 1, failureCount: 0, placedCount: 1, placedTileCount: 2 });
    expect(audit.strategist.obstacle).toEqual({ instructionCount: 1, successCount: 0, failureCount: 1, placedCount: 0 });
  });

  it("counts encouraged attacks once while preserving strategist, king, and overlap attribution", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    state.units.find((unit) => unit.id === "home-1-king")!.position = { kind: "tile", x: 4, y: 1 };
    state.units.find((unit) => unit.id === "home-1-strategist")!.position = { kind: "tile", x: 4, y: 1 };
    const attacker = addUnit(state, "audit-encouraged-attacker", "team-1", "infantry", { kind: "tile", x: 5, y: 1 });
    const target = addUnit(state, "audit-encouraged-target", "team-2", "infantry", { kind: "tile", x: 6, y: 1 });
    const tracker = new PpoAbilityAuditTracker(state);
    tracker.beforeApply(state, {
      kind: "attack",
      teamId: "team-1",
      actorKey: `attack:team-1:${attacker.id}`,
      intent: { teamId: "team-1", attackerUnitId: attacker.id, target: { kind: "unit", unitId: target.id }, pass: false },
    });
    state.logs.push({
      id: "audit-battle-overlap",
      turnNumber: state.turnNumber,
      type: "battle",
      message: `${attacker.id} -> ${target.id} base 1/6, encouraged: yes, final 1/5, result: success, damage: 1.`,
      relatedIds: [attacker.id, target.id],
    });
    tracker.afterApply(state, { kind: "resolve_battle", teamId: "all" });
    tracker.afterApply(state, { kind: "resolve_battle", teamId: "all" });
    const audit = tracker.finalize(state);
    expect(audit.encouragement).toMatchObject({
      totalAttackCount: 1,
      appliedAttackCount: 1,
      hitCount: 1,
      missCount: 0,
      strategistAppliedAttackCount: 1,
      kingAppliedAttackCount: 1,
      bothAppliedAttackCount: 1,
    });
  });

  it("tracks completed, repeated, incomplete, and forced-exposure submersion cycles", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "movement_input";
    state.currentMovementTeamId = "team-1";
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    const ninja = addUnit(state, "audit-water-ninja", "team-1", "ninja", { kind: "tile", x: 4, y: 1 });
    const tracker = new PpoAbilityAuditTracker(state);
    const water = { kind: "water", x: 4, y: 2 } as const;
    tracker.beforeApply(state, movementDecision(ninja, water));
    ninja.position = water;
    tracker.afterApply(state, movementDecision(ninja, water));
    const shore = { kind: "tile", x: 4, y: 1 } as const;
    tracker.beforeApply(state, movementDecision(ninja, shore));
    ninja.position = shore;
    tracker.afterApply(state, movementDecision(ninja, shore));

    tracker.beforeApply(state, movementDecision(ninja, water));
    ninja.position = water;
    tracker.afterApply(state, movementDecision(ninja, water));
    tracker.beforeApply(state, { kind: "resolve_battle", teamId: "all" });
    ninja.position = { kind: "removed", reason: "defeated" };
    tracker.afterApply(state, { kind: "resolve_battle", teamId: "all" });

    const forced = addUnit(state, "audit-forced-ninja", "team-2", "ninja", { kind: "water", x: 5, y: 2 });
    const forcedTracker = new PpoAbilityAuditTracker(state);
    forcedTracker.beforeApply(state, { kind: "resolve_strategists", teamId: "all" });
    forced.position = { kind: "bridge", bridgeId: "audit-bridge", cellIndex: 0 };
    forcedTracker.afterApply(state, { kind: "resolve_strategists", teamId: "all" });

    expect(tracker.finalize(state).ninja).toMatchObject({ submersionStartCount: 2, submersionCompleteCount: 1, submersionIncompleteCount: 1 });
    expect(forcedTracker.finalize(state).ninja).toMatchObject({ forcedExposureCount: 1, submersionCompleteCount: 0 });
  });

  it("distinguishes successful ninja teleport and obstacle jump from ordinary movement", () => {
    const teleportState = createInitialGameState();
    teleportState.phase = teleportState.turnState.phase = "movement_input";
    teleportState.currentMovementTeamId = "team-1";
    teleportState.productionCompletedTeamIdsThisTurn = ["team-1"];
    const teleporter = addUnit(teleportState, "audit-teleporter", "team-1", "ninja", { kind: "tile", x: 3, y: 1 });
    addUnit(teleportState, "audit-enemy", "team-2", "infantry", { kind: "tile", x: 6, y: 1 });
    const teleportTo = { kind: "tile", x: 7, y: 1 } as const;
    const teleportDecision = movementDecision(teleporter, teleportTo);
    const teleportTracker = new PpoAbilityAuditTracker(teleportState);
    const teleportLegal: RlLegalAction[] = [{ actionKey: "movement:team-1:audit-teleporter:7,1", actionType: "movement", actorTeamId: "team-1", isPass: false, unitId: teleporter.id, tileId: "7,1" }];
    teleportTracker.observeDecisionPoint(teleportState, teleportLegal);
    teleportTracker.beforeApply(teleportState, teleportDecision);
    const teleported = commitUnitMovement(teleportState, { teamId: "team-1", unitId: teleporter.id, from: teleporter.position, to: teleportTo, stay: false });
    teleportTracker.afterApply(teleported, teleportDecision);
    expect(teleportTracker.finalize(teleported).ninja).toMatchObject({ teleportOpportunityCount: 1, teleportSuccessCount: 1, teleportUserCount: 1 });

    const jumpState = createInitialGameState();
    jumpState.phase = jumpState.turnState.phase = "movement_input";
    jumpState.currentMovementTeamId = "team-1";
    jumpState.productionCompletedTeamIdsThisTurn = ["team-1"];
    const jumper = addUnit(jumpState, "audit-jumper", "team-1", "ninja", { kind: "tile", x: 3, y: 1 });
    jumpState.ninjaTeleportUsedUnitIds = [jumper.id];
    jumpState.constructions.push({ id: "audit-obstacle", kind: "obstacle", ownerTeamId: "team-2", tiles: [{ x: 4, y: 1 }], placedTurn: 1, active: true });
    const jumpTo = { kind: "tile", x: 5, y: 1 } as const;
    const jumpDecision = movementDecision(jumper, jumpTo);
    const jumpTracker = new PpoAbilityAuditTracker(jumpState);
    const jumpLegal: RlLegalAction[] = [{ actionKey: "movement:team-1:audit-jumper:5,1", actionType: "movement", actorTeamId: "team-1", isPass: false, unitId: jumper.id, tileId: "5,1" }];
    jumpTracker.observeDecisionPoint(jumpState, jumpLegal);
    jumpTracker.beforeApply(jumpState, jumpDecision);
    const jumped = commitUnitMovement(jumpState, { teamId: "team-1", unitId: jumper.id, from: jumper.position, to: jumpTo, stay: false });
    jumpTracker.afterApply(jumped, jumpDecision);
    expect(jumpTracker.finalize(jumped).ninja).toMatchObject({ obstacleJumpOpportunityCount: 1, obstacleJumpSuccessCount: 1, obstacleJumpUserCount: 1 });
  });

  it("does not change action order, RNG consumption, or state hashes when enabled", () => {
    let tracker: PpoAbilityAuditTracker | undefined;
    const audited = new RlEnvironmentV2(undefined, true, {
      cpuStep: { rlInPlacePhaseTransitions: true, rlPrevalidatedMovement: true, rlInPlaceProduction: true },
      onBeforeApply: (state, decision) => tracker?.beforeApply(state, decision),
      onAfterApply: (state, decision) => tracker?.afterApply(state, decision),
    });
    const plain = new RlEnvironmentV2(undefined, true, {
      cpuStep: { rlInPlacePhaseTransitions: true, rlPrevalidatedMovement: true, rlInPlaceProduction: true },
    });
    audited.reset(417, 4);
    plain.reset(417, 4);
    tracker = new PpoAbilityAuditTracker(audited.getStateForValidation());
    for (let index = 0; index < 200 && !plain.isTerminal(); index += 1) {
      const actor = plain.getCurrentActorTeamId();
      expect(audited.getCurrentActorTeamId()).toBe(actor);
      if (!actor) break;
      const action = plain.getLegalActions(actor)[0];
      tracker.observeDecisionPoint(audited.getStateForValidation(), audited.getLegalActions(actor));
      plain.stepWithoutObservation(action.actionKey);
      audited.stepWithoutObservation(action.actionKey);
      expect(audited.getStateHash()).toBe(plain.getStateHash());
    }
  });
});
