import type { GameState } from "../types";
import { createHeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { getRandomCpuDecision } from "./randomCpuPolicy";
import type { CpuDecision, CpuPolicy, CpuRuntime, CpuTeamSettings, TeamController } from "./types";

const isCpuController = (controller: TeamController | undefined) => controller === "random_cpu" || controller === "heuristic_cpu" || controller === "bc_cpu";

export function getVisualCpuActorTeamId(state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings) {
  const activeTeamIds = state.teams.filter((team) => team.status === "active").map((team) => team.id);
  if (state.phase === "production") return activeTeamIds.find((id) => isCpuController(settings[id]) && !runtime.completedProductionTeamIds.includes(id));
  if (state.phase === "movement_input") return state.currentMovementTeamId;
  if (state.phase === "attack_input") return activeTeamIds.find((id) => isCpuController(settings[id]) && !runtime.completedAttackTeamIds.includes(id));
  if (state.phase === "reward_placement") return state.rewardPlacementRequests.find((request) => !request.completed && !request.expired && isCpuController(settings[request.teamId]))?.teamId;
  if (state.phase === "strategist_action_input") return activeTeamIds.find((id) => isCpuController(settings[id]) && !state.strategistSubmittedTeamIds.includes(id));
  return undefined;
}

export function createVisualCpuPolicyRouter(overrides: { randomPolicy?: CpuPolicy; heuristicPolicy?: CpuPolicy } = {}): CpuPolicy {
  const randomPolicy = overrides.randomPolicy ?? getRandomCpuDecision;
  const heuristicPolicy = overrides.heuristicPolicy ?? createHeuristicCpuPolicy();

  return (state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings): CpuDecision | undefined => {
    const activeTeamIds = state.teams.filter((team) => team.status === "active").map((team) => team.id);
    let teamId = getVisualCpuActorTeamId(state, runtime, settings);

    if (state.phase === "production") {
      if (!teamId && activeTeamIds.every((id) => isCpuController(settings[id]))) return { kind: "resolve_production", teamId: "all" };
    } else if (state.phase === "attack_input") {
      if (!teamId && activeTeamIds.every((id) => isCpuController(settings[id]))) return { kind: "resolve_battle", teamId: "all" };
    } else if (state.phase === "strategist_action_resolution") {
      return { kind: "resolve_strategists", teamId: "all" };
    }

    if (!teamId) return undefined;
    if (settings[teamId] === "random_cpu") return randomPolicy(state, runtime, settings);
    if (settings[teamId] === "heuristic_cpu") return heuristicPolicy(state, runtime, settings);
    if (settings[teamId] === "bc_cpu") return undefined;
    return undefined;
  };
}

export { isCpuController };
