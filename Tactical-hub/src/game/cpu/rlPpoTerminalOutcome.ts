import type { GameState } from "../types";
import type { RlResult } from "./rlEnvironment";

export type PpoLearnableOutcomeKind =
  | "victory"
  | "time_limit_adjudicated"
  | "terminal_draw";

export type PpoEpisodeOutcomeKind =
  | PpoLearnableOutcomeKind
  | "abnormal_truncated";

export function classifyPpoGameTerminalOutcome(
  result: RlResult,
  reason?: string,
): Extract<PpoLearnableOutcomeKind, "victory" | "terminal_draw"> | undefined {
  if (reason || !result.terminal || result.endReason !== "victory") {
    return undefined;
  }
  if (
    result.resultReason === "final_duel_timeout_draw"
    && result.winnerTeamId === undefined
  ) {
    return "terminal_draw";
  }
  return result.winnerTeamId ? "victory" : undefined;
}

export function isPpoLearnableOutcomeKind(
  outcomeKind: PpoEpisodeOutcomeKind,
): outcomeKind is PpoLearnableOutcomeKind {
  return outcomeKind === "victory"
    || outcomeKind === "time_limit_adjudicated"
    || outcomeKind === "terminal_draw";
}

export function countPpoEpisodeOutcomes(
  outcomes: readonly PpoEpisodeOutcomeKind[],
) {
  return {
    victoryEpisodeCount: outcomes.filter((outcome) => outcome === "victory").length,
    adjudicatedEpisodeCount: outcomes.filter(
      (outcome) => outcome === "time_limit_adjudicated",
    ).length,
    drawEpisodeCount: outcomes.filter(
      (outcome) => outcome === "terminal_draw",
    ).length,
    truncatedEpisodeCount: outcomes.filter(
      (outcome) => outcome === "abnormal_truncated",
    ).length,
  };
}

/** PPO-only rewards for the normal, winnerless Final Duel terminal. */
export function createFinalDuelDrawPpoRewards(
  state: GameState,
): Record<string, number> {
  if (state.gameResult?.reason !== "final_duel_timeout_draw") {
    throw new Error("PPO Final Duel draw rewards require final_duel_timeout_draw");
  }
  const participantIds = new Set(state.finalDuel?.teamIds ?? []);
  if (participantIds.size !== 2) {
    throw new Error("PPO Final Duel draw rewards require exactly two participants");
  }

  const rewards: Record<string, number> = {};
  const participatingTeams = state.teams
    .filter((team) => !team.isNeutral && team.status !== "neutral")
    .slice(0, state.config.playerCount);
  const participatingTeamIds = new Set(
    participatingTeams.map((team) => team.id),
  );
  if ([...participantIds].some((teamId) => !participatingTeamIds.has(teamId))) {
    throw new Error("PPO Final Duel draw participant is outside configured players");
  }
  for (const team of participatingTeams) {
    if (participantIds.has(team.id)) {
      if (team.status !== "active") {
        throw new Error(`PPO Final Duel draw participant is not active: ${team.id}`);
      }
      rewards[team.id] = 0;
    } else if (team.status === "defeated") {
      rewards[team.id] = -1;
    }
  }
  return rewards;
}
