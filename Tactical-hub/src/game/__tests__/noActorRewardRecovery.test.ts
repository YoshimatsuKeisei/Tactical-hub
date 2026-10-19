import { describe, expect, it } from "vitest";
import { advanceCpuOneStep } from "../cpu/cpuStep";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createCpuRuntime } from "../cpu/types";

function chooseIndex(step: number, seed: number, length: number) {
  const mixed = (Math.imul(step + 1, 1103515245) + Math.imul(seed, 12345)) >>> 0;
  return mixed % length;
}

describe("RL no_actor reward recovery", () => {
  it("lets the headless CPU expire an invalid reward without invoking a policy", () => {
    const state = createHeadlessInitialState(4);
    state.phase = state.turnState.phase = "reward_placement";
    state.phaseAfterRewards = "attack_input";
    state.rewardPlacementRequests.push({
      id: "cpu-invalid-neutral",
      teamId: "neutral",
      rewardType: "king_conquest_reward",
      sourceBaseId: "home-2",
      destinationKind: "fixed",
      fixedBaseId: "home-2",
      eligibleBaseIds: ["home-2"],
      completed: false,
      expired: false,
    });
    let policyCalled = false;

    const result = advanceCpuOneStep(state, createCpuRuntime(417), {}, () => {
      policyCalled = true;
      return undefined;
    });

    expect(result.applied).toBe(true);
    expect(policyCalled).toBe(false);
    expect(result.state.phase).toBe("attack_input");
    expect(result.state.rewardPlacementRequests[0]).toMatchObject({ expired: true, expirationReason: "recipient_not_active_player" });
  });

  it("expires an invalid neutral legacy reward and resumes before actor enumeration", () => {
    const state = createHeadlessInitialState(4);
    state.phase = state.turnState.phase = "reward_placement";
    state.phaseAfterRewards = "strategist_action_input";
    state.rewardPlacementRequests.push({
      id: "legacy-neutral-reward",
      teamId: "neutral",
      rewardType: "king_conquest_reward",
      sourceBaseId: "home-2",
      sourceKingUnitId: "home-2-king",
      destinationKind: "fixed",
      fixedBaseId: "home-2",
      eligibleBaseIds: ["home-2"],
      completed: false,
      expired: false,
    });

    const environment = new RlEnvironmentV2();
    environment.reset(417, 4, state);

    expect(environment.getStateForValidation()).toMatchObject({
      phase: "strategist_action_input",
      turnState: { phase: "strategist_action_input" },
      phaseAfterRewards: undefined,
      rewardPlacementRequests: [expect.objectContaining({
        id: "legacy-neutral-reward",
        expired: true,
        expirationReason: "recipient_not_active_player",
      })],
    });
    expect(environment.getCurrentActorTeamId()).toBe("team-1");
  });

  it("does not reproduce the former seed 417 no_actor boundary", () => {
    const seed = 417;
    const environment = new RlEnvironmentV2(undefined, true, {
      cpuStep: {
        rlInPlacePhaseTransitions: true,
        rlPrevalidatedMovement: true,
        rlInPlaceProduction: true,
      },
    });
    environment.reset(seed, 4);

    let applied = 0;
    for (; applied < 2_600 && !environment.isTerminal(); applied += 1) {
      const actor = environment.getCurrentActorTeamId();
      expect(actor, `nonterminal no_actor at deterministic action ${applied}`).toBeDefined();
      const legal = environment.getLegalActionsForEncoding(actor!);
      expect(legal.length).toBeGreaterThan(0);
      environment.stepWithoutObservation(legal[chooseIndex(applied, seed, legal.length)].actionKey);
    }

    expect(applied >= 2_443 || environment.isTerminal()).toBe(true);
    if (!environment.isTerminal()) expect(environment.getCurrentActorTeamId()).toBeDefined();
  });
});
