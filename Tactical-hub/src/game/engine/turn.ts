import type { GameState } from "../types";
import { resolveMovement } from "./movement";
import { resolveProduction } from "./production";

export function resolveTurnInputs(state: GameState): GameState {
  return resolveMovement(resolveProduction(state));
}
