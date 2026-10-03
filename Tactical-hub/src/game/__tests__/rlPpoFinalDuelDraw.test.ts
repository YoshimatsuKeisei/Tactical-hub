import { describe, expect, it } from "vitest";
import { createInitialGameState } from "../initialState";
import type { GameState } from "../types";
import { RlEnvironmentV2, type RlResult } from "../cpu/rlEnvironment";
import {
  finalizeTerminalTrajectory,
  type PpoTrajectoryStep,
} from "../cpu/rlPpoSelfPlay";
import {
  classifyPpoGameTerminalOutcome,
  countPpoEpisodeOutcomes,
  createFinalDuelDrawPpoRewards,
  isPpoLearnableOutcomeKind,
} from "../cpu/rlPpoTerminalOutcome";
import { createPpoShapingDiagnostics } from "../cpu/rlPpoShapingDiagnostics";
import {
  createPpoRolloutWorkerV7ReplayRollout,
  type PpoRolloutWorkerV7EpisodeSummary,
} from "../cpu/rlPpoRolloutWorkerV7Messages";
import {
  assertPpoFastBatchV7FinalizedConsistency,
  createPpoFastBatchV7ShapingDiagnostics,
} from "../cpu/rlPpoFastBatchV7Workers";

function finalDuelDrawState(
  participantIds: [string, string],
  defeatedIds: string[],
) {
  const state = createInitialGameState();
  for (const team of state.teams) {
    if (participantIds.includes(team.id)) team.status = "active";
    else if (!team.isNeutral) team.status = "defeated";
  }
  state.config.playerCount = participantIds.length + defeatedIds.length;
  state.finalDuel = {
    active: false,
    teamIds: participantIds,
    entryTurn: 120,
    lastEvaluatedTurn: 169,
    consecutiveAdvantageTurns: {
      [participantIds[0]]: 0,
      [participantIds[1]]: 0,
    },
  };
  state.gameResult = { reason: "final_duel_timeout_draw" };
  return state;
}

function result(overrides: Partial<RlResult> = {}): RlResult {
  return {
    terminal: true,
    endReason: "victory",
    resultReason: "final_duel_timeout_draw",
    winnerTeamId: undefined,
    loserTeamIds: [],
    actionCount: 1,
    rewards: {},
    ...overrides,
  };
}

function step(teamId: string, reward = 0): PpoTrajectoryStep {
  return {
    decisionIndex: 0,
    turnNumber: 169,
    phase: "movement_input",
    teamId,
    selectedActionIndex: 0,
    selectedActionKey: `action:${teamId}`,
    oldLogProbability: -0.5,
    value: 0.1,
    reward,
    done: false,
  };
}

function workerSummary(
  outcomeKind: PpoRolloutWorkerV7EpisodeSummary["outcomeKind"],
): PpoRolloutWorkerV7EpisodeSummary {
  return {
    environmentIndex: 0,
    seed: 7,
    decisionCount: 1,
    environmentResult: result(),
    finalStateHash: "draw-state",
    outcomeKind,
    reason: "victory",
  };
}

describe("PPO Final Duel terminal draw", () => {
  it("classifies only the explicit winnerless Final Duel timeout draw", () => {
    expect(classifyPpoGameTerminalOutcome(result())).toBe("terminal_draw");
    expect(classifyPpoGameTerminalOutcome(result(), "phase_stall")).toBeUndefined();
    expect(classifyPpoGameTerminalOutcome(result({
      resultReason: "natural_victory",
    }))).toBeUndefined();
    expect(classifyPpoGameTerminalOutcome(result({
      resultReason: "natural_victory",
      winnerTeamId: "team-1",
    }))).toBe("victory");
    expect(classifyPpoGameTerminalOutcome(result({
      terminal: false,
      endReason: "ongoing",
      resultReason: undefined,
    }))).toBeUndefined();
  });

  it.each([
    [
      "four teams",
      finalDuelDrawState(["team-3", "team-4"], ["team-1", "team-2"]),
      { "team-1": -1, "team-2": -1, "team-3": 0, "team-4": 0 },
    ],
    [
      "two teams",
      finalDuelDrawState(["team-1", "team-2"], []),
      { "team-1": 0, "team-2": 0 },
    ],
    [
      "three teams",
      finalDuelDrawState(["team-2", "team-3"], ["team-1"]),
      { "team-1": -1, "team-2": 0, "team-3": 0 },
    ],
  ] as const)("creates explicit %s PPO rewards without neutral", (_name, state, expected) => {
    const rewards = createFinalDuelDrawPpoRewards(state as GameState);
    expect(rewards).toEqual(expected);
    expect(rewards).not.toHaveProperty("neutral");
  });

  it("does not change the environment's existing all-minus-one draw rewards", () => {
    const state = finalDuelDrawState(
      ["team-3", "team-4"],
      ["team-1", "team-2"],
    );
    const environment = new RlEnvironmentV2();
    environment.reset(7, 4, state);
    expect(environment.getResult()).toMatchObject({
      terminal: true,
      winnerTeamId: undefined,
      resultReason: "final_duel_timeout_draw",
      rewards: {
        "team-1": -1,
        "team-2": -1,
        "team-3": -1,
        "team-4": -1,
      },
    });
    expect(createFinalDuelDrawPpoRewards(state)).toEqual({
      "team-1": -1,
      "team-2": -1,
      "team-3": 0,
      "team-4": 0,
    });
  });

  it("finalizes zero-reward participants and defeated teams with shaping preserved", () => {
    const state = finalDuelDrawState(
      ["team-3", "team-4"],
      ["team-1", "team-2"],
    );
    const rewards = createFinalDuelDrawPpoRewards(state);
    const steps = [
      step("team-1", -0.004),
      step("team-3", -0.012),
    ];
    finalizeTerminalTrajectory(
      steps,
      rewards,
      { gamma: 0.99, gaeLambda: 0.95 },
      { preserveExistingRewards: true },
    );

    expect(steps[0]).toMatchObject({ reward: -1.004, done: true });
    expect(steps[1]).toMatchObject({ reward: -0.012, done: true });
    expect(steps.every((entry) =>
      Number.isFinite(entry.advantage) && Number.isFinite(entry.return),
    )).toBe(true);

    const diagnostics = createPpoShapingDiagnostics({
      battleAdvantageShapingBeta: 0.02,
      gamma: 0.99,
      rollouts: [{ seed: 7, trajectory: steps, baseRewards: rewards }],
    });
    expect(diagnostics.reward.baseReward.sum).toBe(-1);
    expect(diagnostics.reward.shapingReward.sum).toBeCloseTo(-0.016);
    expect(diagnostics.reward.trainingReward.sum).toBeCloseTo(-1.016);
    expect(diagnostics.reward.trainingReward.finiteCount)
      .toBe(diagnostics.reward.trainingReward.count);

    const summary = workerSummary("terminal_draw");
    const rollout = createPpoRolloutWorkerV7ReplayRollout({
      summary,
      trajectory: steps,
    });
    const workerDiagnostics = createPpoFastBatchV7ShapingDiagnostics({
      battleAdvantageShapingBeta: 0.02,
      gamma: 0.99,
      finalized: [{
        environmentIndex: 0,
        summary,
        rollout,
        baseRewards: rewards,
      }],
    });
    expect(workerDiagnostics.reward.baseReward.sum).toBe(-1);
    expect(workerDiagnostics.reward.shapingReward.sum).toBeCloseTo(-0.016);
    expect(workerDiagnostics.reward.trainingReward.sum).toBeCloseTo(-1.016);
  });

  it("treats terminal_draw as learnable in the V7 worker and parent contract", () => {
    expect(isPpoLearnableOutcomeKind("terminal_draw")).toBe(true);
    expect(isPpoLearnableOutcomeKind("abnormal_truncated")).toBe(false);
    const summary = workerSummary("terminal_draw");
    const trajectory = [step("team-3")];
    const rollout = createPpoRolloutWorkerV7ReplayRollout({
      summary,
      trajectory,
    });
    expect(rollout).toMatchObject({
      outcomeKind: "terminal_draw",
      terminal: true,
      winnerTeamId: undefined,
      trajectory,
    });
    expect(() => assertPpoFastBatchV7FinalizedConsistency({
      environmentIndex: 0,
      summary,
      rollout,
      baseRewards: { "team-3": 0, "team-4": 0 },
    })).not.toThrow();
    expect(() => assertPpoFastBatchV7FinalizedConsistency({
      environmentIndex: 0,
      summary,
      rollout: undefined,
    })).toThrow("missing learnable rollout");
  });

  it("counts terminal draws separately from every existing outcome", () => {
    expect(countPpoEpisodeOutcomes([
      "victory",
      "time_limit_adjudicated",
      "terminal_draw",
      "terminal_draw",
      "abnormal_truncated",
    ])).toEqual({
      victoryEpisodeCount: 1,
      adjudicatedEpisodeCount: 1,
      drawEpisodeCount: 2,
      truncatedEpisodeCount: 1,
    });
  });

  it("keeps abnormal truncations out of the V7 replay contract", () => {
    const summary = workerSummary("abnormal_truncated");
    expect(createPpoRolloutWorkerV7ReplayRollout({
      summary,
      trajectory: [step("team-1")],
    })).toBeUndefined();
    expect(() => assertPpoFastBatchV7FinalizedConsistency({
      environmentIndex: 0,
      summary,
    })).not.toThrow();
  });
});
