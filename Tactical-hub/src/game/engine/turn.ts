import type { GameState } from "../types";
import { resolveBattle } from "./battle";
import { resolveMovement } from "./movement";
import { resolveProduction } from "./production";

export function resolveTurnInputs(state: GameState): GameState {
  return resolveBattle(resolveMovement(resolveProduction(state)));
}
