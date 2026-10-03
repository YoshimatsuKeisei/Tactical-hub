import { describe, expect, it } from "vitest";
import { formatPpoCliJson } from "../cpu/rlPpoCliJson";
import {
  calculatePpoRolloutActionHash,
  createPpoShapingDiagnostics,
  isPpoShapingDiagnosticsEnabled,
  summarizePpoDistribution,
  type PpoShapingDiagnosticRollout,
} from "../cpu/rlPpoShapingDiagnostics";

function rollout(overrides: Partial<PpoShapingDiagnosticRollout> = {}): PpoShapingDiagnosticRollout {
  return {
    seed: 7,
    environmentIndex: 0,
    baseRewards: { "team-1": 1, "team-2": -1 },
    trajectory: [
      {
        decisionIndex: 0,
        turnNumber: 1,
        phase: "movement_input",
        teamId: "team-1",
        selectedActionIndex: 0,
        selectedActionKey: "move:a",
        reward: 0.1,
        advantage: 1,
        return: 2,
      },
      {
        decisionIndex: 1,
        turnNumber: 1,
        phase: "movement_input",
        teamId: "team-2",
        selectedActionIndex: 1,
        selectedActionKey: "move:b",
        reward: -1.2,
        advantage: 3,
        return: 4,
      },
      {
        decisionIndex: 2,
        turnNumber: 2,
        phase: "production",
        teamId: "team-1",
        selectedActionIndex: 2,
        selectedActionKey: "produce:c",
        reward: 1.3,
        advantage: 5,
        return: 6,
      },
    ],
    ...overrides,
  };
}

describe("PPO shaping diagnostics", () => {
  it("is opt-in and defaults to disabled", () => {
    expect(isPpoShapingDiagnosticsEnabled([])).toBe(false);
    expect(isPpoShapingDiagnosticsEnabled(["--shaping-diagnostics"])).toBe(true);
  });

  it("separates base, shaping and training rewards and reports nonzero shaping", () => {
    const diagnostics = createPpoShapingDiagnostics({
      battleAdvantageShapingBeta: 0.02,
      gamma: 0.99,
      rollouts: [rollout()],
    });
    expect(diagnostics.reward.baseReward).toMatchObject({
      count: 3,
      sum: 0,
      nonZeroCount: 2,
      finiteCount: 3,
    });
    expect(diagnostics.reward.shapingReward.sum).toBeCloseTo(0.2);
    expect(diagnostics.reward.shapingReward.nonZeroCount).toBe(3);
    expect(diagnostics.reward.trainingReward.sum).toBeCloseTo(0.2);
    expect(
      diagnostics.reward.baseReward.sum + diagnostics.reward.shapingReward.sum,
    ).toBeCloseTo(diagnostics.reward.trainingReward.sum);
  });

  it("reports an exact zero shaping summary for a beta-zero trajectory", () => {
    const diagnostics = createPpoShapingDiagnostics({
      battleAdvantageShapingBeta: 0,
      gamma: 0.99,
      rollouts: [rollout({
        trajectory: [
          { ...rollout().trajectory[0], reward: 0 },
          { ...rollout().trajectory[1], reward: -1 },
          { ...rollout().trajectory[2], reward: 1 },
        ],
      })],
    });
    expect(diagnostics.reward.shapingReward).toMatchObject({
      count: 3,
      sum: 0,
      min: 0,
      max: 0,
      mean: 0,
      nonZeroCount: 0,
      finiteCount: 3,
    });
  });

  it("calculates advantage and return population statistics", () => {
    const diagnostics = createPpoShapingDiagnostics({
      battleAdvantageShapingBeta: 0.02,
      gamma: 0.99,
      rollouts: [rollout()],
    });
    expect(diagnostics.advantage).toMatchObject({
      count: 3,
      min: 1,
      max: 5,
      mean: 3,
      finiteCount: 3,
    });
    expect(diagnostics.advantage.std).toBeCloseTo(Math.sqrt(8 / 3));
    expect(diagnostics.return).toMatchObject({
      count: 3,
      min: 2,
      max: 6,
      mean: 4,
      finiteCount: 3,
    });
    expect(diagnostics.return.std).toBeCloseTo(Math.sqrt(8 / 3));
    expect(diagnostics.byTeam["team-1"].advantage.count).toBe(2);
  });

  it("keeps count-zero and count-one standard deviations JSON-safe", () => {
    expect(summarizePpoDistribution([])).toEqual({
      count: 0,
      min: null,
      max: null,
      mean: null,
      std: null,
      finiteCount: 0,
    });
    expect(summarizePpoDistribution([4])).toEqual({
      count: 1,
      min: 4,
      max: 4,
      mean: 4,
      std: 0,
      finiteCount: 1,
    });
  });

  it("counts non-finite values without emitting them into JSON", () => {
    const statistics = summarizePpoDistribution([2, Number.NaN, Number.POSITIVE_INFINITY]);
    expect(statistics).toEqual({
      count: 3,
      min: 2,
      max: 2,
      mean: 2,
      std: 0,
      finiteCount: 1,
    });
    expect(JSON.parse(JSON.stringify(statistics))).toEqual(statistics);
  });

  it("hashes identical action columns identically and ignores reward fields", () => {
    const original = rollout();
    const rewardChanged = rollout({
      trajectory: original.trajectory.map((step) => ({
        ...step,
        reward: step.reward + 100,
        advantage: (step.advantage ?? 0) - 50,
        return: (step.return ?? 0) + 25,
      })),
    });
    expect(calculatePpoRolloutActionHash([rollout()]))
      .toBe(calculatePpoRolloutActionHash([rollout()]));
    expect(calculatePpoRolloutActionHash([original]))
      .toBe(calculatePpoRolloutActionHash([rewardChanged]));
  });

  it("changes the action hash when one action changes", () => {
    const original = rollout();
    const changed = rollout({
      trajectory: original.trajectory.map((step, index) => index === 1
        ? { ...step, selectedActionKey: "move:different" }
        : step),
    });
    expect(calculatePpoRolloutActionHash([original]))
      .not.toBe(calculatePpoRolloutActionHash([changed]));
  });

  it("formats one parseable stdout JSON document containing diagnostics", () => {
    const diagnostics = createPpoShapingDiagnostics({
      battleAdvantageShapingBeta: 0.02,
      gamma: 0.99,
      rollouts: [rollout()],
    });
    const output = formatPpoCliJson({ existing: true, shapingDiagnostics: diagnostics });
    expect(output.endsWith("\n")).toBe(true);
    expect(JSON.parse(output)).toEqual({ existing: true, shapingDiagnostics: diagnostics });
  });
});
