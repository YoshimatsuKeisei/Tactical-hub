import type { GameState } from "../types";
import type { CpuRuntime, CpuTeamSettings, TeamController } from "./types";

export function getPolicyActorTeamId(
  state: GameState,
  runtime: CpuRuntime,
  settings: CpuTeamSettings,
  controller: Exclude<TeamController, "human">,
) {
  const active = state.teams.filter((team) => team.status === "active").map((team) => team.id);
  if (state.phase === "production") return active.find((id) => settings[id] === controller && !runtime.completedProductionTeamIds.includes(id));
  if (state.phase === "movement_input") return settings[state.currentMovementTeamId ?? ""] === controller ? state.currentMovementTeamId : undefined;
  if (state.phase === "attack_input") return active.find((id) => settings[id] === controller && !runtime.completedAttackTeamIds.includes(id));
  if (state.phase === "reward_placement") return state.rewardPlacementRequests.find((request) => !request.completed && !request.expired && settings[request.teamId] === controller)?.teamId;
  if (state.phase === "strategist_action_input") return active.find((id) => settings[id] === controller && !state.strategistSubmittedTeamIds.includes(id));
  return undefined;
}
