import { describe, expect, it } from "vitest";
import { RlEnvironment } from "../cpu/rlEnvironment";
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
