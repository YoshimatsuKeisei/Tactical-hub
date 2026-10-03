import { calculateBattleAdvantage } from "../engine/battleAdvantage";
import type { GameState } from "../types";

export const DEFAULT_BATTLE_ADVANTAGE_SHAPING_BETA = 0;
export const BATTLE_ADVANTAGE_SHAPING_BETA_FLAG = "--battle-advantage-shaping-beta";

type MutableRewardStep = { reward: number };

export type PpoBattleAdvantageShapingRuntime = {
  readonly beta: number;
  readonly gamma: number;
  readonly previousStepByTeam: Map<string, MutableRewardStep>;
};

export function validateBattleAdvantageShapingBeta(beta: number) {
  if (!Number.isFinite(beta) || beta < 0) {
    throw new Error("battleAdvantageShapingBeta must be a non-negative finite number");
  }
  return beta;
}

export function parseBattleAdvantageShapingBeta(
  args: readonly string[],
  fallback = DEFAULT_BATTLE_ADVANTAGE_SHAPING_BETA,
) {
  const index = args.indexOf(BATTLE_ADVANTAGE_SHAPING_BETA_FLAG);
  const raw = index >= 0 ? args[index + 1] : fallback;
  return validateBattleAdvantageShapingBeta(Number(raw));
}

/**
 * Creates no shaping state at all for the exact legacy beta=0 path.
 */
export function createPpoBattleAdvantageShapingRuntime(
  beta: number,
  gamma: number,
): PpoBattleAdvantageShapingRuntime | undefined {
  validateBattleAdvantageShapingBeta(beta);
  if (!Number.isFinite(gamma) || gamma < 0) {
    throw new Error("PPO shaping gamma must be a non-negative finite number");
  }
  if (beta === 0) return undefined;
  return { beta, gamma, previousStepByTeam: new Map() };
}

/** Full-GameState potential used only by PPO training reward shaping. */
export function calculatePpoBattleAdvantagePotential(
  state: GameState,
  teamId: string,
  trainingTerminal = false,
) {
  if (trainingTerminal) return 0;
  const team = state.teams.find((candidate) => candidate.id === teamId);
  if (!team || team.isNeutral || team.status !== "active") return 0;
  return calculateBattleAdvantage(state)
    .find((advantage) => advantage.teamId === teamId && advantage.active)
    ?.battleAdvantageShare ?? 0;
}

/**
 * Advances the same-team decision clock. The returned reward initializes the
 * new sample; the previous same-team sample receives its gamma * Phi(next)
 * term. A missing next decision therefore leaves terminal Phi equal to zero.
 */
export function beginPpoBattleAdvantageDecision(
  runtime: PpoBattleAdvantageShapingRuntime | undefined,
  state: GameState,
  teamId: string,
) {
  if (!runtime) return 0;
  const potential = calculatePpoBattleAdvantagePotential(state, teamId);
  const previous = runtime.previousStepByTeam.get(teamId);
  if (previous) previous.reward += runtime.beta * runtime.gamma * potential;
  return -runtime.beta * potential;
}

export function trackPpoBattleAdvantageDecision(
  runtime: PpoBattleAdvantageShapingRuntime | undefined,
  teamId: string,
  step: MutableRewardStep,
) {
  if (runtime) runtime.previousStepByTeam.set(teamId, step);
}
