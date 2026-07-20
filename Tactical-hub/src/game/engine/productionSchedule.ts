import type { GameState } from "../types";

export function isProductionTurn(state: Pick<GameState, "turnNumber" | "config">) {
  return (state.turnNumber - 1) % state.config.productionInterval === 0;
}

export function isTeamProductionPending(state: GameState, teamId: string) {
  return state.phase === "movement_input"
    && state.currentMovementTeamId === teamId
    && isProductionTurn(state)
    && !state.productionCompletedTeamIdsThisTurn.includes(teamId);
}
