import { describe, expect, it } from "vitest";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import { calculatePpoBattleAdvantagePotential } from "../cpu/rlPpoBattleAdvantageShaping";
import { PpoRolloutWorkerV7Pool } from "../cpu/rlPpoRolloutWorkerV7Pool";
import { DEFAULT_PPO_HYPERPARAMETERS } from "../cpu/rlPpoSelfPlay";
import { createPpoFastBatchV7ShapingDiagnostics } from "../cpu/rlPpoFastBatchV7Workers";

describe("PPO V7 worker Battle Advantage shaping", () => {
  it("uses the worker-owned full state and composes terminal shaping with adjudication", async () => {
    const seed = 31;
    const beta = 0.02;
    const direct = new RlEnvironmentV2(undefined, true, {
      cpuStep: {
        rlInPlacePhaseTransitions: true,
        rlPrevalidatedMovement: true,
        rlInPlaceProduction: true,
      },
    });
    const firstObservation = direct.reset(seed, 4);
    const actor = direct.getCurrentActorTeamId()!;
    const phi = calculatePpoBattleAdvantagePotential(
      direct.getStateForValidation(),
      actor,
    );
    const pool = await PpoRolloutWorkerV7Pool.create({
      workerCount: 1,
      environmentCount: 1,
      firstGameSeed: seed,
      featureSpec: createRlFeatureSpecV2(firstObservation),
      hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
      battleAdvantageShapingBeta: beta,
      safetyMaxTurns: 1_000,
      safetyMaxActions: 1,
    });

    try {
      const prepared = await pool.prepareWorker(0, 0);
      const group = prepared.group!;
      const actionKey = group.actionKeys[0][0];
      const finalized = await pool.advanceWorker(0, 0, [{
        environmentIndex: 0,
        actionIndex: 0,
        actionKey,
        logProbability: -0.5,
        value: 0.1,
      }]);
      expect(finalized.group).toBeUndefined();
      expect(finalized.finalized).toHaveLength(1);
      const item = finalized.finalized[0];
      expect(item.summary.outcomeKind).toBe("time_limit_adjudicated");
      const step = item.rollout!.trajectory[0];
      const baseReward = item.summary.adjudication!
        .find((team) => team.teamId === actor)!.reward;
      expect(step.teamId).toBe(actor);
      expect(step.done).toBe(true);
      expect(step.reward).toBeCloseTo(baseReward - beta * phi);
      const diagnostics = createPpoFastBatchV7ShapingDiagnostics({
        battleAdvantageShapingBeta: beta,
        gamma: DEFAULT_PPO_HYPERPARAMETERS.gamma,
        finalized: finalized.finalized,
      });
      expect(diagnostics).toMatchObject({
        battleAdvantageShapingBeta: beta,
        gamma: DEFAULT_PPO_HYPERPARAMETERS.gamma,
        reward: {
          baseReward: { count: 1, finiteCount: 1 },
          shapingReward: { count: 1, nonZeroCount: 1, finiteCount: 1 },
          trainingReward: { count: 1, finiteCount: 1 },
        },
        advantage: { count: 1, finiteCount: 1 },
        return: { count: 1, finiteCount: 1 },
      });
      expect(diagnostics.reward.shapingReward.sum).toBeCloseTo(-beta * phi);
    } finally {
      await pool.close();
    }
  }, 20_000);
});
