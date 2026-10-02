import type { GameState } from "../types";
import type { RlResult } from "./rlEnvironment";

export const PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS = {
  safetyMaxTurns: 1_000,
  safetyMaxActions: 100_000,
} as const;

export type PpoRolloutEnvironmentDiagnosticV8 = {
  environmentIndex: number;
  currentEpisodeSeed: number;
  generation: number;
  episodeDecisionCount: number;
  currentTurn: number;
  currentPhase: GameState["phase"];
  currentActorTeamId: string | null;
  terminal: boolean;
  endReason: RlResult["endReason"];
  winnerTeamId: string | null;
  loserTeamIds: string[];
  naturalVictoryPredicateSatisfied: boolean;
  victoryPredicateInputs: {
    teams: Array<{
      teamId: string;
      isNeutral: boolean;
      status: GameState["teams"][number]["status"];
    }>;
  };
  victoryBlockers: {
    activeNonNeutralTeamCount: number;
    requiredMaximumActiveNonNeutralTeamCount: 1;
    excessActiveNonNeutralTeamCount: number;
    activeNonNeutralTeamIds: string[];
  };
  wouldHitStandardTurnLimit: boolean;
  wouldHitStandardActionLimit: boolean;
};

export function createPpoRolloutEnvironmentDiagnosticV8(input: {
  environmentIndex: number;
  currentEpisodeSeed: number;
  generation: number;
  episodeDecisionCount: number;
  currentActorTeamId?: string;
  state: GameState;
  result: RlResult;
}): PpoRolloutEnvironmentDiagnosticV8 {
  const teams = input.state.teams.map((team) => ({
    teamId: team.id,
    isNeutral: Boolean(team.isNeutral),
    status: team.status,
  }));
  const activeNonNeutralTeamIds = teams
    .filter((team) => !team.isNeutral && team.status === "active")
    .map((team) => team.teamId);
  const activeNonNeutralTeamCount = activeNonNeutralTeamIds.length;

  return {
    environmentIndex: input.environmentIndex,
    currentEpisodeSeed: input.currentEpisodeSeed,
    generation: input.generation,
    episodeDecisionCount: input.episodeDecisionCount,
    currentTurn: input.state.turnNumber,
    currentPhase: input.state.phase,
    currentActorTeamId: input.currentActorTeamId ?? null,
    terminal: input.result.terminal,
    endReason: input.result.endReason,
    winnerTeamId: input.result.winnerTeamId ?? null,
    loserTeamIds: [...input.result.loserTeamIds],
    naturalVictoryPredicateSatisfied:
      activeNonNeutralTeamCount <= 1,
    victoryPredicateInputs: { teams },
    victoryBlockers: {
      activeNonNeutralTeamCount,
      requiredMaximumActiveNonNeutralTeamCount: 1,
      excessActiveNonNeutralTeamCount: Math.max(
        0,
        activeNonNeutralTeamCount - 1,
      ),
      activeNonNeutralTeamIds,
    },
    wouldHitStandardTurnLimit:
      input.state.turnNumber
      > PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS.safetyMaxTurns,
    wouldHitStandardActionLimit:
      input.episodeDecisionCount
      >= PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS.safetyMaxActions,
  };
}
