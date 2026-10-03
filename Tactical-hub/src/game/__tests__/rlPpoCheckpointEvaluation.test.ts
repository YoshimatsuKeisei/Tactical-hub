import { describe, expect, it, vi } from "vitest";
import type { EncodedLegalActionsV2 } from "../cpu/rlActionEncoder";
import type { EncodedObservation } from "../cpu/rlObservationEncoder";
import {
  createPpoEvaluationSeatAssignments,
  runPpoCheckpointEvaluation,
  runPpoCheckpointEvaluationMatch,
  type PpoEvaluationPolicyClientFactory,
} from "../cpu/rlPpoCheckpointEvaluation";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";

class OneDecisionFinalDuelDrawEnvironment extends RlEnvironmentV2 {
  override stepWithoutObservation(actionKey: string) {
    super.stepWithoutObservation(actionKey);
    const state = this.getStateForValidation();
    state.turnNumber = 170;
    state.turnState.turnNumber = 170;
    for (const team of state.teams) {
      if (team.id === "team-3" || team.id === "team-4") {
        team.status = "defeated";
      }
    }
    state.logs.push(
      {
        id: "log-team-defeated-evaluation-1",
        turnNumber: 70,
        type: "battle",
        message: "fixture defeat",
        relatedIds: ["team-3"],
      },
      {
        id: "log-team-defeated-evaluation-2",
        turnNumber: 80,
        type: "battle",
        message: "fixture defeat",
        relatedIds: ["team-4"],
      },
    );
    state.finalDuel = {
      active: false,
      teamIds: ["team-1", "team-2"],
      entryTurn: 120,
      lastEvaluatedTurn: 169,
      consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
    };
    state.gameResult = { reason: "final_duel_timeout_draw" };
    return this.getResult();
  }
}

function fakePolicyClients() {
  const clients: Array<{
    teamId: string;
    checkpointLabel: "update2" | "update3";
    checkpointPath: string;
    seed: number;
    start: ReturnType<typeof vi.fn>;
    act: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
    beginUpdate: ReturnType<typeof vi.fn>;
    finishUpdate: ReturnType<typeof vi.fn>;
    save: ReturnType<typeof vi.fn>;
  }> = [];
  const factory: PpoEvaluationPolicyClientFactory = (input) => {
    const client = {
      teamId: input.teamId,
      checkpointLabel: input.checkpointLabel,
      checkpointPath: input.checkpointPath,
      seed: input.seed,
      start: vi.fn(async () => ({ selectedDevice: "cpu" })),
      act: vi.fn(async (
        _observation: EncodedObservation,
        legalActions: EncodedLegalActionsV2,
      ) => ({ actionKey: legalActions.actionKeys[0] })),
      close: vi.fn(async () => {}),
      beginUpdate: vi.fn(async () => {}),
      finishUpdate: vi.fn(async () => {}),
      save: vi.fn(async () => {}),
    };
    clients.push(client);
    return client;
  };
  return { clients, factory };
}

describe("PPO checkpoint evaluation", () => {
  it("rotates update3 through all four seats for every seed", () => {
    const assignments = createPpoEvaluationSeatAssignments({
      seedStart: 50,
      seedCount: 2,
    });

    expect(assignments).toHaveLength(8);
    for (const [seedOffset, seed] of [50, 51].entries()) {
      const rotations = assignments.slice(seedOffset * 4, seedOffset * 4 + 4);
      expect(rotations.map((rotation) => rotation.seed)).toEqual([
        seed, seed, seed, seed,
      ]);
      expect(rotations.map((rotation) => rotation.update3TeamId)).toEqual([
        "team-1", "team-2", "team-3", "team-4",
      ]);
      for (const rotation of rotations) {
        expect(Object.values(rotation.checkpointByTeam).filter(
          (checkpoint) => checkpoint === "update3",
        )).toHaveLength(1);
        expect(rotation.checkpointByTeam[rotation.update3TeamId])
          .toBe("update3");
      }
    }
  });

  it("runs same-checkpoint seats, includes turn diagnostics, and never trains", async () => {
    const policies = fakePolicyClients();
    const result = await runPpoCheckpointEvaluation({
      update2Checkpoint: "same-checkpoint.pt",
      update3Checkpoint: "same-checkpoint.pt",
      update2Sha256: "same-sha",
      update3Sha256: "same-sha",
      seedStart: 7,
      seedCount: 1,
      clientFactory: policies.factory,
      environmentFactory: () => new OneDecisionFinalDuelDrawEnvironment(),
    });

    expect(result.matches).toHaveLength(4);
    expect(result.matches.map((match) => match.update3TeamId)).toEqual([
      "team-1", "team-2", "team-3", "team-4",
    ]);
    for (const match of result.matches) {
      expect(match).toMatchObject({
        seed: 7,
        winnerTeamId: null,
        resultReason: "final_duel_timeout_draw",
        decisionCount: 1,
        turnDiagnostics: {
          finalStateTurnNumber: 170,
          defeatedTeamTurns: { "team-3": 70, "team-4": 80 },
          finalDuel: {
            teamIds: ["team-1", "team-2"],
            entryTurn: 120,
            lastEvaluatedTurn: 169,
            evaluatedTurns: 50,
            activeAtEnd: false,
            consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
          },
        },
      });
      expect(match.finalStateHash).toMatch(/^[a-f0-9]{8}$/);
      expect(match.teams).toHaveLength(4);
      expect(match.teams.every((team) =>
        Number.isInteger(team.rank)
        && Number.isFinite(team.battleAdvantageRaw)
        && Number.isFinite(team.battleAdvantageShare),
      )).toBe(true);
    }
    expect(result.aggregate).toMatchObject({
      matchCount: 4,
      update3WinCount: 0,
      averageMatchTurn: 170,
      averageFirstDefeatTurn: 70,
      averageFinalDuelEntryTurn: 120,
      averageFinalDuelDurationTurns: 50,
      resultReasonCounts: { final_duel_timeout_draw: 4 },
    });
    expect(Object.values(result.aggregate.update3RankDistribution)
      .reduce((sum, count) => sum + count, 0)).toBe(4);
    expect(Object.values(result.aggregate.update3AverageFinalMetrics)
      .every((value) => Number.isFinite(value))).toBe(true);

    expect(policies.clients).toHaveLength(16);
    expect(policies.clients.every((client) =>
      client.checkpointPath === "same-checkpoint.pt",
    )).toBe(true);
    expect(policies.clients.every((client) =>
      client.start.mock.calls[0]?.[0]?.evaluationCheckpoint
        === "same-checkpoint.pt",
    )).toBe(true);
    expect(policies.clients.every((client) =>
      client.seed === 7 && client.start.mock.calls[0]?.[0]?.seed === 7,
    )).toBe(true);
    const team1Policies = policies.clients.filter((client) =>
      client.teamId === "team-1",
    );
    expect(team1Policies.map((client) => client.checkpointLabel)).toEqual([
      "update3", "update2", "update2", "update2",
    ]);
    expect(team1Policies.map((client) => client.start.mock.calls[0]?.[0]?.seed))
      .toEqual([7, 7, 7, 7]);
    expect(policies.clients.reduce(
      (sum, client) => sum + client.act.mock.calls.length,
      0,
    )).toBe(4);
    for (const client of policies.clients) {
      expect(client.beginUpdate).not.toHaveBeenCalled();
      expect(client.finishUpdate).not.toHaveBeenCalled();
      expect(client.save).not.toHaveBeenCalled();
      expect(client.close).toHaveBeenCalledOnce();
    }
    expect(() => JSON.parse(JSON.stringify(result))).not.toThrow();
  });

  it("loads update3 only for the explicitly rotated team", async () => {
    const policies = fakePolicyClients();
    const assignment = createPpoEvaluationSeatAssignments({
      seedStart: 19,
      seedCount: 1,
    })[2];

    await runPpoCheckpointEvaluationMatch({
      assignment,
      checkpointPaths: {
        update2: "update2.pt",
        update3: "update3.pt",
      },
      clientFactory: policies.factory,
      environmentFactory: () => new OneDecisionFinalDuelDrawEnvironment(),
    });

    expect(assignment.update3TeamId).toBe("team-3");
    expect(policies.clients.map((client) => [
      client.teamId,
      client.checkpointLabel,
      client.checkpointPath,
    ])).toEqual([
      ["team-1", "update2", "update2.pt"],
      ["team-2", "update2", "update2.pt"],
      ["team-3", "update3", "update3.pt"],
      ["team-4", "update2", "update2.pt"],
    ]);
  });
});
