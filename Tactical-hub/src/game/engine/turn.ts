import type { GameState } from "../types";
import { resolveBattle } from "./battle";
import { resolveMovement } from "./movement";
import { resolveProduction } from "./production";

export function resolveTurnInputs(state: GameState): GameState {
  const produced = resolveProduction(state);
  const moved = produced.phase === "movement_input" ? resolveMovement(produced) : produced;
  return moved.phase === "attack_input" ? resolveBattle(moved) : moved;
}
