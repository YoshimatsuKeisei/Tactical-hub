import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { aggregatePpoAbilityAudits, PpoAbilityAuditTracker } from "../cpu/rlAbilityAudit";
import { RlEnvironmentV2, type RlLegalAction } from "../cpu/rlEnvironment";
import type { CpuDecision } from "../cpu/types";
import { commitUnitMovement } from "../engine/movement";
import { resolveBattle, saveAttackIntent } from "../engine/battle";
import { resolveProduction, saveProductionChoice } from "../engine/production";
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
    expect(aggregate.production.selectionCount).toBe(aggregate.production.successCount + aggregate.production.failureCount);
  });

  it.each(["infantry", "cavalry", "archer", "engineer", "ninja", "strategist"] as const)(
    "separates %s selection from actual successful production",
    (unitType) => {
      let state = createInitialGameState();
      const tracker = new PpoAbilityAuditTracker(state);
      const choice = { teamId: "team-1", baseId: "home-1", unitType, ...(unitType === "strategist" ? { strategistRole: "encourage" as const } : {}) };
      const selection: CpuDecision = { kind: "production", teamId: "team-1", actorKey: "p", choice };
      tracker.beforeApply(state, selection);
      state = saveProductionChoice(state, choice);
      tracker.afterApply(state, selection);
      const resolution: CpuDecision = { kind: "resolve_production", teamId: "all" };
      tracker.beforeApply(state, resolution);
      state = resolveProduction(state);
      tracker.afterApply(state, resolution);
      const production = tracker.finalize(state).production;
      expect(production.counts[unitType]).toBe(1);
      expect(production.successCounts[unitType]).toBe(1);
      expect(production.failureCounts[unitType]).toBe(0);
      expect(production.selectionCount).toBe(production.successCount + production.failureCount);
    },
  );

  it("counts overwritten and engine-rejected production as failures without counting reward units", () => {
    let state = createInitialGameState();
    const tracker = new PpoAbilityAuditTracker(state);
    const firstChoice = { teamId: "team-1", baseId: "home-1", unitType: "infantry" as const };
    const secondChoice = { teamId: "team-1", baseId: "home-1", unitType: "ninja" as const };
    for (const [actorKey, choice] of [["p1", firstChoice], ["p2", secondChoice]] as const) {
      const decision: CpuDecision = { kind: "production", teamId: "team-1", actorKey, choice };
      tracker.beforeApply(state, decision);
      state = saveProductionChoice(state, choice);
      tracker.afterApply(state, decision);
    }
    for (const slot of state.bases.find((base) => base.id === "home-1")!.slots) {
      if (!slot.unitId) slot.unitId = `blocked-${slot.id}`;
    }
    const resolution: CpuDecision = { kind: "resolve_production", teamId: "all" };
    tracker.beforeApply(state, resolution);
    state = resolveProduction(state);
    tracker.afterApply(state, resolution);
    tracker.beforeApply(state, { kind: "reward", teamId: "team-1", requestId: "r", baseId: "home-1", unitType: "engineer" });
    const production = tracker.finalize(state).production;
    expect(production.counts).toMatchObject({ infantry: 1, ninja: 1, engineer: 0 });
    expect(production.successCount).toBe(0);
    expect(production.failureCounts).toMatchObject({ infantry: 1, ninja: 1 });
    expect(production.selectionCount).toBe(production.successCount + production.failureCount);
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

  it("handles a rebuilt log array and reused log id without rescanning or suppressing the new event", () => {
    const state = createInitialGameState();
    const tracker = new PpoAbilityAuditTracker(state);
    const intent: StrategistActionIntent = { teamId: "team-1", strategistUnitId: "home-1-strategist", action: "place_obstacle", tiles: [{ x: 4, y: 1 }] };
    tracker.beforeApply(state, { kind: "strategist", teamId: "team-1", actorKey: "obstacle", intent });
    state.logs = [];
    tracker.afterApply(state, { kind: "strategist", teamId: "team-1", actorKey: "obstacle", intent });
    tracker.beforeApply(state, { kind: "resolve_strategists", teamId: "all" });
    state.logs = [{ id: "log-setup", turnNumber: 1, type: "construction", message: "placement failed by simultaneous conflict.", relatedIds: [intent.strategistUnitId] }];
    tracker.afterApply(state, { kind: "resolve_strategists", teamId: "all" });
    tracker.afterApply(state, { kind: "resolve_strategists", teamId: "all" });
    expect(tracker.finalize(state).strategist.obstacle.failureCount).toBe(1);
  });

  it("counts resolved encouraged attacks once while preserving strategist, king, overlap, hit, and log consistency", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    state.units.find((unit) => unit.id === "home-1-king")!.position = { kind: "tile", x: 4, y: 1 };
    state.units.find((unit) => unit.id === "home-1-strategist")!.position = { kind: "tile", x: 4, y: 1 };
    const attacker = addUnit(state, "audit-encouraged-attacker", "team-1", "infantry", { kind: "tile", x: 5, y: 1 });
    const target = addUnit(state, "audit-encouraged-target", "team-2", "infantry", { kind: "tile", x: 6, y: 1 });
    const tracker = new PpoAbilityAuditTracker(state);
    const planned = saveAttackIntent(state, { teamId: "team-1", attackerUnitId: attacker.id, target: { kind: "unit", unitId: target.id }, pass: false });
    let observed = [] as Parameters<PpoAbilityAuditTracker["observeBattleResolution"]>[0];
    const resolved = resolveBattle(planned, () => 0, (events) => {
      observed = events;
      tracker.observeBattleResolution(events);
    });
    tracker.observeBattleResolution(observed);
    const battleLog = resolved.logs.find((log) => log.message.startsWith(`${attacker.id} -> ${target.id}`));
    expect(battleLog?.message).toContain("encouraged: yes");
    const audit = tracker.finalize(resolved);
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

  it("observes neutral attacks from the actual battle resolution rather than PPO attack selections", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    state.units = state.units.filter((unit) => unit.ownerTeamId !== "neutral");
    const attacker = addUnit(state, "neutral-audit-attacker", "neutral", "infantry", { kind: "tile", x: 5, y: 1 });
    addUnit(state, "neutral-audit-strategist", "neutral", "strategist", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "neutral-audit-king", "neutral", "king", { kind: "tile", x: 4, y: 2 });
    const target = addUnit(state, "neutral-audit-target", "team-1", "infantry", { kind: "tile", x: 6, y: 1 });
    const tracker = new PpoAbilityAuditTracker(state);
    let events = [] as Parameters<PpoAbilityAuditTracker["observeBattleResolution"]>[0];
    const resolved = resolveBattle(state, () => 0.99, (value) => { events = value; tracker.observeBattleResolution(value); });
    expect(events.some((event) => event.attackerUnitId === attacker.id && event.targetUnitId === target.id)).toBe(true);
    const audit = tracker.finalize(resolved).encouragement;
    expect(audit.totalAttackCount).toBeGreaterThanOrEqual(1);
    expect(audit.bothAppliedAttackCount).toBeGreaterThanOrEqual(1);
    expect(audit.missCount).toBeGreaterThanOrEqual(1);
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
    teleportTracker.observeDecisionPoint(teleportState, teleportLegal, [{ actionKey: teleportLegal[0].actionKey, unitId: teleporter.id, destinationKey: "7,1", ninjaTeleport: true, obstacleJump: false }]);
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
    jumpTracker.observeDecisionPoint(jumpState, jumpLegal, [{ actionKey: jumpLegal[0].actionKey, unitId: jumper.id, destinationKey: "5,1", ninjaTeleport: false, obstacleJump: true }]);
    jumpTracker.beforeApply(jumpState, jumpDecision);
    const jumped = commitUnitMovement(jumpState, { teamId: "team-1", unitId: jumper.id, from: jumper.position, to: jumpTo, stay: false });
    jumpTracker.afterApply(jumped, jumpDecision);
    expect(jumpTracker.finalize(jumped).ninja).toMatchObject({ obstacleJumpOpportunityCount: 1, obstacleJumpSuccessCount: 1, obstacleJumpUserCount: 1 });
  });

  it("counts overlapping ninja opportunities independently from the one action actually selected", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "movement_input";
    state.currentMovementTeamId = "team-1";
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    const ninja = addUnit(state, "audit-overlap-ninja", "team-1", "ninja", { kind: "tile", x: 3, y: 1 });
    const tracker = new PpoAbilityAuditTracker(state);
    const legal: RlLegalAction[] = [
      { actionKey: "teleport", actionType: "movement", actorTeamId: "team-1", isPass: false, unitId: ninja.id, tileId: "7,1" },
      { actionKey: "jump", actionType: "movement", actorTeamId: "team-1", isPass: false, unitId: ninja.id, tileId: "5,1" },
    ];
    tracker.observeDecisionPoint(state, legal, [
      { actionKey: "teleport", unitId: ninja.id, destinationKey: "7,1", ninjaTeleport: true, obstacleJump: false },
      { actionKey: "jump", unitId: ninja.id, destinationKey: "5,1", ninjaTeleport: false, obstacleJump: true },
    ]);
    const jumpTo = { kind: "tile", x: 5, y: 1 } as const;
    const decision = movementDecision(ninja, jumpTo);
    tracker.beforeApply(state, decision);
    ninja.position = jumpTo;
    tracker.afterApply(state, decision);
    const audit = tracker.finalize(state).ninja;
    expect(audit).toMatchObject({
      teleportOpportunityCount: 1,
      teleportSuccessCount: 0,
      obstacleJumpOpportunityCount: 1,
      obstacleJumpSuccessCount: 1,
    });
  });

  it("does not change action order, RNG consumption, or state hashes when enabled", () => {
    let tracker: PpoAbilityAuditTracker | undefined;
    const audited = new RlEnvironmentV2(undefined, true, {
      cpuStep: {
        rlInPlacePhaseTransitions: true,
        rlPrevalidatedMovement: true,
        rlInPlaceProduction: true,
        onBattleResolved: (events) => tracker?.observeBattleResolution(events),
      },
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
      tracker.observeDecisionPoint(
        audited.getStateForValidation(),
        audited.getLegalActions(actor),
        audited.getDecisionAuditHintsForValidation(actor),
      );
      plain.stepWithoutObservation(action.actionKey);
      audited.stepWithoutObservation(action.actionKey);
      expect(audited.getStateHash()).toBe(plain.getStateHash());
    }
  });
});
