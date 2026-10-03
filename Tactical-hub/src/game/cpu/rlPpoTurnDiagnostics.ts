import type { GameState } from "../types";

export type PpoFinalDuelTurnDiagnostics = {
  teamIds: [string, string];
  entryTurn: number;
  lastEvaluatedTurn?: number;
  evaluatedTurns?: number;
  activeAtEnd: boolean;
  consecutiveAdvantageTurns: Record<string, number>;
};

export type PpoTurnDiagnostics = {
  finalStateTurnNumber: number;
  defeatedTeamTurns: Record<string, number>;
  finalDuel?: PpoFinalDuelTurnDiagnostics;
};

/** Read-only episode-end projection for game-design diagnostics. */
export function createPpoTurnDiagnostics(
  state: Readonly<GameState>,
): PpoTurnDiagnostics {
  const nonNeutralTeamIds = new Set(
    state.teams
      .filter((team) => !team.isNeutral && team.status !== "neutral")
      .map((team) => team.id),
  );
  const defeatedTeamTurns: Record<string, number> = {};
  for (const log of state.logs) {
    if (!log.id.startsWith("log-team-defeated-")) continue;
    for (const relatedId of log.relatedIds ?? []) {
      if (
        nonNeutralTeamIds.has(relatedId)
        && defeatedTeamTurns[relatedId] === undefined
      ) {
        defeatedTeamTurns[relatedId] = log.turnNumber;
      }
    }
  }

  const finalDuel = state.finalDuel;
  return {
    finalStateTurnNumber: state.turnNumber,
    defeatedTeamTurns,
    ...(finalDuel
      ? {
          finalDuel: {
            teamIds: [...finalDuel.teamIds] as [string, string],
            entryTurn: finalDuel.entryTurn,
            ...(finalDuel.lastEvaluatedTurn === undefined
              ? {}
              : {
                  lastEvaluatedTurn: finalDuel.lastEvaluatedTurn,
                  evaluatedTurns:
                    finalDuel.lastEvaluatedTurn - finalDuel.entryTurn + 1,
                }),
            activeAtEnd: finalDuel.active,
            consecutiveAdvantageTurns: {
              ...finalDuel.consecutiveAdvantageTurns,
            },
          },
        }
      : {}),
  };
}
