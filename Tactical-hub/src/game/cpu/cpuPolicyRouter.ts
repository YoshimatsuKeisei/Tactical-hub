import type { GameState } from "../types";
import { createHeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { getRandomCpuDecision } from "./randomCpuPolicy";
import type { CpuDecision, CpuPolicy, CpuRuntime, CpuTeamSettings, TeamController } from "./types";

const isCpuController = (controller: TeamController | undefined) => controller === "random_cpu" || controller === "heuristic_cpu";

export function createVisualCpuPolicyRouter(overrides: { randomPolicy?: CpuPolicy; heuristicPolicy?: CpuPolicy } = {}): CpuPolicy {
  const randomPolicy = overrides.randomPolicy ?? getRandomCpuDecision;
  const heuristicPolicy = overrides.heuristicPolicy ?? createHeuristicCpuPolicy();

  return (state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings): CpuDecision | undefined => {
    const activeTeamIds = state.teams.filter((team) => team.status === "active").map((team) => team.id);
    let teamId: string | undefined;

    if (state.phase === "production") {
      teamId = activeTeamIds.find((id) => isCpuController(settings[id]) && !runtime.completedProductionTeamIds.includes(id));
      if (!teamId && activeTeamIds.every((id) => isCpuController(settings[id]))) return { kind: "resolve_production", teamId: "all" };
    } else if (state.phase === "movement_input") {
      teamId = state.currentMovementTeamId;
    } else if (state.phase === "attack_input") {
      teamId = activeTeamIds.find((id) => isCpuController(settings[id]) && !runtime.completedAttackTeamIds.includes(id));
      if (!teamId && activeTeamIds.every((id) => isCpuController(settings[id]))) return { kind: "resolve_battle", teamId: "all" };
    } else if (state.phase === "reward_placement") {
      teamId = state.rewardPlacementRequests.find((request) => !request.completed && !request.expired && isCpuController(settings[request.teamId]))?.teamId;
    } else if (state.phase === "strategist_action_input") {
      teamId = activeTeamIds.find((id) => isCpuController(settings[id]) && !state.strategistSubmittedTeamIds.includes(id));
    } else if (state.phase === "strategist_action_resolution") {
      return { kind: "resolve_strategists", teamId: "all" };
    }

    if (!teamId) return undefined;
    if (settings[teamId] === "random_cpu") return randomPolicy(state, runtime, settings);
    if (settings[teamId] === "heuristic_cpu") return heuristicPolicy(state, runtime, settings);
    return undefined;
  };
}

export { isCpuController };
