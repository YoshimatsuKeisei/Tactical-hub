import type { GameResultReason, GameState, StoredGameResult } from "../types";
import { calculateBattleAdvantage } from "./battleAdvantage";

export const FINAL_DUEL_CONSECUTIVE_ADVANTAGE_TURNS = 20;
export const FINAL_DUEL_MAX_TURNS = 50;

function activeTeamIds(state: GameState) {
  return state.teams
    .filter((team) => !team.isNeutral && team.status === "active")
    .map((team) => team.id);
}

/** Records the exact 3/4 -> 2 transition point after a rules resolution. */
export function syncBattleRoyaleFinalDuel(state: GameState) {
  const active = activeTeamIds(state);
  if (active.length <= 1) {
    if (state.finalDuel?.active) state.finalDuel.active = false;
    return;
  }
  if (state.finalDuel || state.gameResult || active.length !== 2) return;
  state.finalDuel = {
    active: true,
    teamIds: [active[0], active[1]],
    entryTurn: state.turnNumber,
    consecutiveAdvantageTurns: { [active[0]]: 0, [active[1]]: 0 },
  };
}

function finishFinalDuel(state: GameState, reason: StoredGameResult["reason"], winnerTeamId?: string) {
  if (!state.finalDuel || state.gameResult) return;
  state.finalDuel.active = false;
  state.gameResult = winnerTeamId ? { reason, winnerTeamId } : { reason };
}

/**
 * Applies exactly one Final Duel sample at a completed formal game turn.
 * completedTurn is the number before movement advances state.turnNumber.
 */
export function resolveBattleRoyaleFinalDuelTurnEnd(state: GameState, completedTurn: number) {
  syncBattleRoyaleFinalDuel(state);
  const duel = state.finalDuel;
  if (!duel?.active || state.gameResult || completedTurn < duel.entryTurn) return;
  if (duel.lastEvaluatedTurn !== undefined && completedTurn <= duel.lastEvaluatedTurn) return;

  const active = new Set(activeTeamIds(state));
  if (active.size <= 1) {
    duel.active = false;
    return;
  }
  if (active.size !== 2 || duel.teamIds.some((teamId) => !active.has(teamId))) return;

  const advantages = new Map(
    calculateBattleAdvantage(state).map((advantage) => [advantage.teamId, advantage]),
  );
  const [firstTeamId, secondTeamId] = duel.teamIds;
  const firstRaw = advantages.get(firstTeamId)?.battleAdvantageRaw ?? 0;
  const secondRaw = advantages.get(secondTeamId)?.battleAdvantageRaw ?? 0;
  duel.lastEvaluatedTurn = completedTurn;

  if (firstRaw > secondRaw) {
    duel.consecutiveAdvantageTurns[firstTeamId] += 1;
    duel.consecutiveAdvantageTurns[secondTeamId] = 0;
  } else if (secondRaw > firstRaw) {
    duel.consecutiveAdvantageTurns[secondTeamId] += 1;
    duel.consecutiveAdvantageTurns[firstTeamId] = 0;
  } else {
    duel.consecutiveAdvantageTurns[firstTeamId] = 0;
    duel.consecutiveAdvantageTurns[secondTeamId] = 0;
  }

  const consecutiveWinner = duel.teamIds.find(
    (teamId) => duel.consecutiveAdvantageTurns[teamId] >= FINAL_DUEL_CONSECUTIVE_ADVANTAGE_TURNS,
  );
  if (consecutiveWinner) {
    finishFinalDuel(state, "final_duel_consecutive_advantage", consecutiveWinner);
    return;
  }

  const elapsedTurns = completedTurn - duel.entryTurn + 1;
  if (elapsedTurns < FINAL_DUEL_MAX_TURNS) return;
  if (firstRaw > secondRaw) finishFinalDuel(state, "final_duel_timeout_advantage", firstTeamId);
  else if (secondRaw > firstRaw) finishFinalDuel(state, "final_duel_timeout_advantage", secondTeamId);
  else finishFinalDuel(state, "final_duel_timeout_draw");
}

export type GameTerminalResult = {
  terminal: boolean;
  winnerTeamId?: string;
  resultReason?: GameResultReason;
};

/** Natural victory is deliberately checked before a stored Final Duel result. */
export function getGameTerminalResult(state: GameState): GameTerminalResult {
  const active = activeTeamIds(state);
  if (active.length <= 1) {
    return {
      terminal: true,
      winnerTeamId: active.length === 1 ? active[0] : undefined,
      resultReason: "natural_victory",
    };
  }
  if (state.gameResult) {
    return {
      terminal: true,
      winnerTeamId: state.gameResult.winnerTeamId,
      resultReason: state.gameResult.reason,
    };
  }
  return { terminal: false };
}
