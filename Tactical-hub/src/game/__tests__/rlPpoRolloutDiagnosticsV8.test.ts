import { describe, expect, it } from "vitest";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";
import {
  createPpoRolloutEnvironmentDiagnosticV8,
  PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS,
} from "../cpu/rlPpoRolloutDiagnosticsV8";

describe("PPO V8 rollout diagnostics", () => {
  it("summarizes only the actual natural-victory predicate inputs and standard safety comparisons", () => {
    const state = createHeadlessInitialState(4);
    state.turnNumber = 1_001;
    state.phase = state.turnState.phase = "attack_input";
    state.teams.find((team) => team.id === "team-4")!.status =
      "defeated";

    const diagnostic = createPpoRolloutEnvironmentDiagnosticV8({
      environmentIndex: 3,
      currentEpisodeSeed: 27,
      generation: 2,
      episodeDecisionCount: 100_000,
      currentActorTeamId: "team-2",
      state,
      result: {
        terminal: false,
        loserTeamIds: ["team-4"],
        endReason: "ongoing",
        actionCount: 100_000,
        rewards: {},
      },
    });

    expect(PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS).toEqual({
      safetyMaxTurns: 1_000,
      safetyMaxActions: 100_000,
    });
    expect(diagnostic).toMatchObject({
      environmentIndex: 3,
      currentEpisodeSeed: 27,
      generation: 2,
      episodeDecisionCount: 100_000,
      currentTurn: 1_001,
      currentPhase: "attack_input",
      currentActorTeamId: "team-2",
      terminal: false,
      endReason: "ongoing",
      winnerTeamId: null,
      loserTeamIds: ["team-4"],
      naturalVictoryPredicateSatisfied: false,
      victoryBlockers: {
        activeNonNeutralTeamCount: 3,
        requiredMaximumActiveNonNeutralTeamCount: 1,
        excessActiveNonNeutralTeamCount: 2,
        activeNonNeutralTeamIds: [
          "team-1",
          "team-2",
          "team-3",
        ],
      },
      wouldHitStandardTurnLimit: true,
      wouldHitStandardActionLimit: true,
    });
    expect(diagnostic.victoryPredicateInputs.teams).toEqual(
      state.teams.map((team) => ({
        teamId: team.id,
        isNeutral: Boolean(team.isNeutral),
        status: team.status,
      })),
    );
  });

  it("reports the exact winner predicate boundary", () => {
    const state = createHeadlessInitialState(4);
    for (const team of state.teams) {
      if (!team.isNeutral && team.id !== "team-2") {
        team.status = "defeated";
      }
    }

    const diagnostic = createPpoRolloutEnvironmentDiagnosticV8({
      environmentIndex: 0,
      currentEpisodeSeed: 7,
      generation: 0,
      episodeDecisionCount: 99_999,
      state,
      result: {
        terminal: true,
        winnerTeamId: "team-2",
        loserTeamIds: ["team-1", "team-3", "team-4"],
        endReason: "victory",
        actionCount: 99_999,
        rewards: {},
      },
    });

    expect(diagnostic.naturalVictoryPredicateSatisfied).toBe(true);
    expect(diagnostic.victoryBlockers).toMatchObject({
      activeNonNeutralTeamCount: 1,
      excessActiveNonNeutralTeamCount: 0,
      activeNonNeutralTeamIds: ["team-2"],
    });
    expect(diagnostic.wouldHitStandardActionLimit).toBe(false);
  });
});
