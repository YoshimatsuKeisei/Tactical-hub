import { describe, expect, it } from "vitest";
import { describeRlDecision, RlEnvironment } from "../cpu/rlEnvironment";
import type { CpuDecision } from "../cpu/types";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";

function advanceDeterministically(environment: RlEnvironment, steps: number) {
  const phases = new Set<string>();
  for (let index = 0; index < steps && !environment.isTerminal(); index += 1) {
    const actor = environment.getCurrentActorTeamId();
    if (!actor) break;
    const actions = environment.getLegalActions(actor);
    expect(actions.length).toBeGreaterThan(0);
    phases.add(environment.getObservation(actor).phase);
    environment.step(actions[index % actions.length].actionKey);
  }
  return phases;
}

describe("Phase 5-D RL Environment", () => {
  it("preserves strategist production roles as distinct Action meaning", () => {
    const make = (strategistRole: "builder" | "teleporter"): CpuDecision => ({
      kind: "production",
      teamId: "team-1",
      actorKey: "production:team-1:home-1",
      choice: { teamId: "team-1", baseId: "home-1", unitType: "strategist", strategistRole },
    });
    const builder = describeRlDecision(make("builder"));
    const teleporter = describeRlDecision(make("teleporter"));

    expect(builder).toMatchObject({ unitType: "strategist", strategistRole: "builder", isPass: false });
    expect(teleporter).toMatchObject({ unitType: "strategist", strategistRole: "teleporter", isPass: false });
    expect(builder.actionKey).not.toBe(teleporter.actionKey);
  });

  it("preserves strategist action kind and the ordered multi-tile target", () => {
    const make = (action: "place_bridge" | "place_obstacle"): CpuDecision => ({
      kind: "strategist",
      teamId: "team-1",
      actorKey: "strategist:team-1:s-1",
      intent: { teamId: "team-1", strategistUnitId: "s-1", action, tiles: [{ x: 2, y: 3 }, { x: 2, y: 4 }] },
    });
    const bridge = describeRlDecision(make("place_bridge"));
    const obstacle = describeRlDecision(make("place_obstacle"));

    expect(bridge).toMatchObject({ strategistActionKind: "place_bridge", tileIds: ["2,3", "2,4"], isPass: false });
    expect(obstacle.strategistActionKind).toBe("place_obstacle");
    expect(bridge.actionKey).not.toBe(obstacle.actionKey);
  });

  it("distinguishes pass decisions, normal decisions, and phase-control decisions", () => {
    const movementPass = describeRlDecision({ kind: "movement", teamId: "team-1", actorKey: "movement:team-1:u-1", unitId: "u-1" });
    const movement = describeRlDecision({ kind: "movement", teamId: "team-1", actorKey: "movement:team-1:u-1", unitId: "u-1", to: { kind: "tile", x: 4, y: 5 } });
    const attackPass = describeRlDecision({ kind: "attack", teamId: "team-1", actorKey: "attack:team-1:u-1", intent: { teamId: "team-1", attackerUnitId: "u-1", pass: true } });
    const submit = describeRlDecision({ kind: "submit_movement", teamId: "team-1" });

    expect(movementPass).toMatchObject({ actionType: "movement", isPass: true, tileId: undefined });
    expect(movement).toMatchObject({ actionType: "movement", isPass: false, tileId: "4,5" });
    expect(attackPass).toMatchObject({ actionType: "attack", isPass: true });
    expect(submit).toMatchObject({ actionType: "submit_movement", isPass: false });
    expect(new Set([movementPass.actionKey, movement.actionKey, attackPass.actionKey, submit.actionKey]).size).toBe(4);
  });

  it("retains target and request references needed by a later Action Encoder", () => {
    const attack = describeRlDecision({
      kind: "attack",
      teamId: "team-1",
      actorKey: "attack:team-1:u-1",
      intent: { teamId: "team-1", attackerUnitId: "u-1", target: { kind: "unit", unitId: "u-2", baseId: "base-2", slotId: "slot-2" }, pass: false },
    });
    const teleport = describeRlDecision({
      kind: "teleport",
      teamId: "team-1",
      actorKey: "teleport:team-1:s-1",
      strategistUnitId: "s-1",
      intent: { teamId: "team-1", strategistUnitId: "s-1", targetUnitId: "u-3", to: { kind: "bridge", bridgeId: "bridge-1", cellIndex: 2 } },
    });
    const reward = describeRlDecision({ kind: "reward", teamId: "team-1", requestId: "reward-1", baseId: "base-1", unitType: "archer" });

    expect(attack).toMatchObject({ unitId: "u-1", targetId: "u-2", baseId: "base-2", slotId: "slot-2" });
    expect(teleport).toMatchObject({ unitId: "s-1", targetId: "u-3", tileId: "bridge:bridge-1:2" });
    expect(reward).toMatchObject({ requestId: "reward-1", baseId: "base-1", unitType: "archer" });
  });

  it("keeps actionKey unique within every enumerated legal action set", () => {
    const environment = new RlEnvironment();
    environment.reset(188, 4);
    for (let index = 0; index < 500 && !environment.isTerminal(); index += 1) {
      const actor = environment.getCurrentActorTeamId();
      if (!actor) break;
      const actions = environment.getLegalActions(actor);
      expect(new Set(actions.map((action) => action.actionKey)).size).toBe(actions.length);
      environment.step(actions[index % actions.length].actionKey);
    }
  });

  it("exposes the structured board and decision state needed for learning", () => {
    const state = createHeadlessInitialState(4);
    state.unitTurnFlags.push({
      unitId: state.units[0].id,
      battleTurnNumber: state.turnNumber,
      wasAliveAtBattleStart: true,
      survivedPreviousBattle: true,
      attackedInPreviousBattle: false,
      wasTargetedInPreviousBattle: true,
      retreatEligible: true,
    });
    state.siegeStates.push({
      baseId: state.bases[0].id,
      defendingTeamId: "team-1",
      teamRecords: [{ teamId: "team-2", defenderKills: 1, effectiveAttackTurns: 2 }],
      active: true,
      defenderLossOccurred: true,
      fallCandidateTeamIds: ["team-2"],
    });
    const king = state.units.find((unit) => unit.type === "king")!;
    state.kingCampaignStates.push({
      kingUnitId: king.id,
      kingTeamId: king.ownerTeamId,
      contributions: [{ teamId: "team-2", cumulativeDamage: 1, effectiveAttackTurns: 0 }],
    });
    state.rewardPlacementRequests.push({
      id: "observation-reward",
      teamId: "team-2",
      rewardType: "contribution_compensation",
      sourceBaseId: state.bases[0].id,
      destinationKind: "selectable",
      eligibleBaseIds: [state.bases[1].id],
      completed: false,
      expired: false,
    });
    state.strategistCooldowns.push({ strategistUnitId: state.units.find((unit) => unit.type === "strategist")!.id, kind: "bridge", availableFromTurn: 6 });
    state.teleportCooldowns.push({ strategistUnitId: state.units.find((unit) => unit.type === "strategist")!.id, availableFromTurn: 6 });

    const environment = new RlEnvironment();
    const observation = environment.reset(123, 4, state);

    expect(observation.map).toMatchObject({
      id: state.map.id,
      width: state.map.width,
      height: state.map.height,
      tiles: expect.arrayContaining([
        expect.objectContaining({ terrain: expect.any(String) }),
      ]),
    });
    expect(observation.map.tiles.some((tile) => tile.roadSectionId !== undefined)).toBe(true);
    expect(observation.map.bases).toEqual(state.map.bases);
    expect(observation.unitTurnFlags).toEqual(state.unitTurnFlags);
    expect(observation.siegeStates).toEqual(state.siegeStates);
    expect(observation.kingCampaignStates).toEqual(state.kingCampaignStates);
    expect(observation.rewardPlacementRequests).toEqual(state.rewardPlacementRequests);
    expect(observation.pendingRewardRequestIds).toContain("observation-reward");
    expect(observation.strategistCooldowns).toEqual(state.strategistCooldowns);
    expect(observation.teleportCooldowns).toEqual(state.teleportCooldowns);
  });

  it("returns an isolated snapshot and does not expose RNG, runtime, or logs", () => {
    const environment = new RlEnvironment();
    const observation = environment.reset(4321, 4);
    const originalMapName = observation.map.name;
    const originalUnitHp = observation.units[0].hp;

    observation.map.name = "mutated observation map";
    observation.units[0].hp = -999;
    observation.movementOrderTeamIds.reverse();

    const fresh = environment.getObservation(observation.observingTeamId);
    expect(fresh.map.name).toBe(originalMapName);
    expect(fresh.units[0].hp).toBe(originalUnitHp);
    expect(fresh.movementOrderTeamIds).not.toEqual(observation.movementOrderTeamIds);
    expect(observation).not.toHaveProperty("logs");
    expect(observation).not.toHaveProperty("seed");
    expect(observation).not.toHaveProperty("rngState");
    expect(observation).not.toHaveProperty("runtime");
    expect(observation).not.toHaveProperty("processedKeys");
    expect(observation).not.toHaveProperty("hiddenAttackIntents");
  });

  it("keeps other teams' unresolved intents private while exposing the observing team's own intents", () => {
    const state = createHeadlessInitialState(4);
    const ownUnit = state.units.find((unit) => unit.ownerTeamId === "team-1")!;
    const enemyUnit = state.units.find((unit) => unit.ownerTeamId === "team-2")!;
    state.phase = state.turnState.phase = "attack_input";
    state.turnState.actionIntents = [
      {
        teamId: "team-1",
        productionChoices: [],
        movementIntents: [],
        attackIntents: [{ teamId: "team-1", attackerUnitId: ownUnit.id, pass: true }],
      },
      {
        teamId: "team-2",
        productionChoices: [],
        movementIntents: [],
        attackIntents: [{ teamId: "team-2", attackerUnitId: enemyUnit.id, target: { kind: "unit", unitId: ownUnit.id }, pass: false }],
      },
    ];
    state.strategistActionIntents = [
      { teamId: "team-1", strategistUnitId: ownUnit.id, action: "pass" },
      { teamId: "team-2", strategistUnitId: enemyUnit.id, action: "place_obstacle", tiles: [{ x: 3, y: 3 }] },
    ];
    state.teleportIntents = [
      { teamId: "team-1", strategistUnitId: ownUnit.id, targetUnitId: ownUnit.id, to: ownUnit.position },
      { teamId: "team-2", strategistUnitId: enemyUnit.id, targetUnitId: enemyUnit.id, to: enemyUnit.position },
    ];

    const environment = new RlEnvironment();
    environment.reset(77, 4, state);

    const teamOne = environment.getObservation("team-1");
    expect(teamOne.actionIntents.map((intent) => intent.teamId)).toEqual(["team-1"]);
    expect(teamOne.actionIntents[0].attackIntents).toEqual(state.turnState.actionIntents[0].attackIntents);
    expect(teamOne.strategistActionIntents).toEqual([state.strategistActionIntents[0]]);
    expect(teamOne.teleportIntents).toEqual([state.teleportIntents[0]]);

    const teamTwo = environment.getObservation("team-2");
    expect(teamTwo.actionIntents.map((intent) => intent.teamId)).toEqual(["team-2"]);
    expect(teamTwo.strategistActionIntents).toEqual([state.strategistActionIntents[1]]);
    expect(teamTwo.teleportIntents).toEqual([state.teleportIntents[1]]);
  });

  it("resets reproducibly and returns JSON-serializable observations and legal actions", () => {
    const first = new RlEnvironment();
    const second = new RlEnvironment();
    const firstObservation = first.reset(1234, 4);
    const secondObservation = second.reset(1234, 4);
    expect(first.getStateHash()).toBe(second.getStateHash());
    expect(JSON.parse(JSON.stringify(firstObservation))).toEqual(firstObservation);
    const actor = first.getCurrentActorTeamId()!;
    expect(JSON.parse(JSON.stringify(first.getLegalActions(actor)))).toEqual(first.getLegalActions(actor));
    expect(secondObservation).toEqual(firstObservation);
  });

  it("produces the same hashes for the same seed and action-key sequence", () => {
    const first = new RlEnvironment(); first.reset(55, 4);
    const second = new RlEnvironment(); second.reset(55, 4);
    for (let index = 0; index < 120; index += 1) {
      const firstActor = first.getCurrentActorTeamId();
      const secondActor = second.getCurrentActorTeamId();
      expect(secondActor).toBe(firstActor);
      if (!firstActor) break;
      const firstActions = first.getLegalActions(firstActor);
      const secondActions = second.getLegalActions(secondActor!);
      expect(secondActions).toEqual(firstActions);
      const selected = firstActions[index % firstActions.length].actionKey;
      first.step(selected); second.step(selected);
      expect(second.getStateHash()).toBe(first.getStateHash());
    }
  });

  it("rejects illegal or stale keys and never exposes an inactive actor", () => {
    const environment = new RlEnvironment(); environment.reset(7, 4);
    expect(() => environment.step("not-a-legal-action")).toThrow(/Illegal or stale/);
    for (let index = 0; index < 100 && !environment.isTerminal(); index += 1) {
      const actor = environment.getCurrentActorTeamId()!;
      const observation = environment.getObservation(actor);
      expect(observation.teams.find((team) => team.id === actor)?.status).toBe("active");
      environment.step(environment.getLegalActions(actor)[0].actionKey);
    }
  });

  it("treats reward placement as a mandatory decision and returns to its saved movement point", () => {
    const state = createHeadlessInitialState(4);
    state.phase = state.turnState.phase = "reward_placement";
    state.phaseAfterRewards = "movement_input";
    state.currentMovementTeamId = "team-2";
    state.movementCompletedTeamIds = ["team-1"];
    state.rewardPlacementRequests.push({ id: "rl-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    const environment = new RlEnvironment(); environment.reset(9, 4, state);
    expect(environment.getCurrentActorTeamId()).toBe("team-1");
    const reward = environment.getLegalActions("team-1").find((action) => action.actionType === "reward")!;
    environment.step(reward.actionKey);
    expect(environment.getObservation("team-2")).toMatchObject({ phase: "movement_input", currentMovementTeamId: "team-2", movementCompletedTeamIds: ["team-1"] });
    expect(environment.getCurrentActorTeamId()).toBe("team-2");
  });

  it("crosses production, movement, attack and strategist decision phases without UI", () => {
    const environment = new RlEnvironment(); environment.reset(18, 4);
    const phases = advanceDeterministically(environment, 700);
    expect(phases.has("movement_input")).toBe(true);
    expect(phases.has("attack_input")).toBe(true);
    expect(phases.has("strategist_action_input")).toBe(true);
  });

  it("can be driven by the existing Random CPU Policy", () => {
    const first = new RlEnvironment(); first.reset(81, 4);
    const second = new RlEnvironment(); second.reset(81, 4);
    for (let index = 0; index < 250 && !first.isTerminal(); index += 1) {
      first.stepWithPolicy(); second.stepWithPolicy();
      expect(second.getStateHash()).toBe(first.getStateHash());
    }
    expect(first.getResult().actionCount).toBeGreaterThan(0);
  });

  it("reports terminal winner, losers, end reason and default terminal rewards", () => {
    const state = createHeadlessInitialState(4);
    for (const team of state.teams) if (!team.isNeutral && team.id !== "team-2") team.status = "defeated";
    const environment = new RlEnvironment(); environment.reset(3, 4, state);
    expect(environment.isTerminal()).toBe(true);
    expect(environment.getResult()).toMatchObject({ terminal: true, winnerTeamId: "team-2", endReason: "victory", rewards: { "team-1": -1, "team-2": 1, "team-3": -1, "team-4": -1 } });
  });
});
