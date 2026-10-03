import { createHash } from "node:crypto";

export const PPO_SHAPING_DIAGNOSTICS_FLAG = "--shaping-diagnostics";

export function isPpoShapingDiagnosticsEnabled(args: readonly string[]) {
  return args.includes(PPO_SHAPING_DIAGNOSTICS_FLAG);
}

type DiagnosticTrajectoryStep = {
  decisionIndex: number;
  turnNumber: number;
  phase: string;
  teamId: string;
  selectedActionIndex: number;
  selectedActionKey: string;
  reward: number;
  advantage?: number;
  return?: number;
};

export type PpoShapingDiagnosticRollout = {
  seed: number;
  environmentIndex?: number;
  trajectory: readonly DiagnosticTrajectoryStep[];
  baseRewards: Readonly<Record<string, number>>;
};

export type PpoRewardDiagnosticStatistics = {
  count: number;
  sum: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  nonZeroCount: number;
  finiteCount: number;
};

export type PpoDistributionDiagnosticStatistics = {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  finiteCount: number;
};

class NumericStatisticsAccumulator {
  count = 0;
  finiteCount = 0;
  sum = 0;
  meanValue = 0;
  squaredDeviationSum = 0;
  min: number | null = null;
  max: number | null = null;

  add(value: number | undefined) {
    this.count += 1;
    if (!Number.isFinite(value)) return;
    const finite = value as number;
    this.finiteCount += 1;
    this.sum += finite;
    const delta = finite - this.meanValue;
    this.meanValue += delta / this.finiteCount;
    this.squaredDeviationSum += delta * (finite - this.meanValue);
    this.min = this.min === null ? finite : Math.min(this.min, finite);
    this.max = this.max === null ? finite : Math.max(this.max, finite);
  }

  distribution(): PpoDistributionDiagnosticStatistics {
    if (!this.finiteCount) {
      return {
        count: this.count,
        min: null,
        max: null,
        mean: null,
        std: null,
        finiteCount: 0,
      };
    }
    const mean = Number.isFinite(this.meanValue) ? this.meanValue : null;
    const variance = this.squaredDeviationSum / this.finiteCount;
    const std = Number.isFinite(variance)
      ? Math.sqrt(Math.max(0, variance))
      : null;
    return {
      count: this.count,
      min: this.min,
      max: this.max,
      mean,
      std,
      finiteCount: this.finiteCount,
    };
  }
}

class RewardStatisticsAccumulator extends NumericStatisticsAccumulator {
  nonZeroCount = 0;

  override add(value: number | undefined) {
    super.add(value);
    if (Number.isFinite(value) && value !== 0) this.nonZeroCount += 1;
  }

  reward(): PpoRewardDiagnosticStatistics {
    const distribution = this.distribution();
    return {
      count: distribution.count,
      sum: this.sum,
      min: distribution.min,
      max: distribution.max,
      mean: distribution.mean,
      nonZeroCount: this.nonZeroCount,
      finiteCount: distribution.finiteCount,
    };
  }
}

type DiagnosticAccumulators = {
  baseReward: RewardStatisticsAccumulator;
  shapingReward: RewardStatisticsAccumulator;
  trainingReward: RewardStatisticsAccumulator;
  advantage: NumericStatisticsAccumulator;
  return: NumericStatisticsAccumulator;
};

function createAccumulators(): DiagnosticAccumulators {
  return {
    baseReward: new RewardStatisticsAccumulator(),
    shapingReward: new RewardStatisticsAccumulator(),
    trainingReward: new RewardStatisticsAccumulator(),
    advantage: new NumericStatisticsAccumulator(),
    return: new NumericStatisticsAccumulator(),
  };
}

function summarize(accumulators: DiagnosticAccumulators) {
  return {
    reward: {
      baseReward: accumulators.baseReward.reward(),
      shapingReward: accumulators.shapingReward.reward(),
      trainingReward: accumulators.trainingReward.reward(),
    },
    advantage: accumulators.advantage.distribution(),
    return: accumulators.return.distribution(),
  };
}

export function summarizePpoDistribution(values: readonly number[]) {
  const accumulator = new NumericStatisticsAccumulator();
  for (const value of values) accumulator.add(value);
  return accumulator.distribution();
}

export function calculatePpoRolloutActionHash(
  rollouts: readonly PpoShapingDiagnosticRollout[],
) {
  const hash = createHash("sha256");
  hash.update("tactical-hub-ppo-rollout-actions-v1\n");
  for (const [rolloutIndex, rollout] of rollouts.entries()) {
    hash.update(`${JSON.stringify({
      rolloutIndex,
      seed: rollout.seed,
      environmentIndex: rollout.environmentIndex ?? null,
    })}\n`);
    for (const step of rollout.trajectory) {
      hash.update(`${JSON.stringify({
        teamId: step.teamId,
        decisionIndex: step.decisionIndex,
        turnNumber: step.turnNumber,
        phase: step.phase,
        selectedActionIndex: step.selectedActionIndex,
        selectedActionKey: step.selectedActionKey,
      })}\n`);
    }
  }
  return hash.digest("hex");
}

export function createPpoShapingDiagnostics(input: {
  battleAdvantageShapingBeta: number;
  gamma: number;
  rollouts: readonly PpoShapingDiagnosticRollout[];
}) {
  const overall = createAccumulators();
  const byTeam = new Map<string, DiagnosticAccumulators>();

  for (const rollout of input.rollouts) {
    const lastDecisionByTeam = new Map<string, number>();
    for (const step of rollout.trajectory) {
      lastDecisionByTeam.set(step.teamId, step.decisionIndex);
    }
    for (const step of rollout.trajectory) {
      const baseReward = lastDecisionByTeam.get(step.teamId) === step.decisionIndex
        ? rollout.baseRewards[step.teamId] ?? 0
        : 0;
      const trainingReward = step.reward;
      const shapingReward = trainingReward - baseReward;
      const team = byTeam.get(step.teamId) ?? createAccumulators();
      if (!byTeam.has(step.teamId)) byTeam.set(step.teamId, team);

      for (const accumulators of [overall, team]) {
        accumulators.baseReward.add(baseReward);
        accumulators.shapingReward.add(shapingReward);
        accumulators.trainingReward.add(trainingReward);
        accumulators.advantage.add(step.advantage);
        accumulators.return.add(step.return);
      }
    }
  }

  return {
    battleAdvantageShapingBeta: input.battleAdvantageShapingBeta,
    gamma: input.gamma,
    rolloutActionHash: calculatePpoRolloutActionHash(input.rollouts),
    ...summarize(overall),
    byTeam: Object.fromEntries(
      [...byTeam.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([teamId, accumulators]) => [teamId, summarize(accumulators)]),
    ),
  };
}
