import type { GameState, TeamStatus } from "../types";

export type PpoTimeLimitReason = "safety_turn_limit" | "safety_action_limit";

export function isPpoTimeLimitReason(reason: string | undefined): reason is PpoTimeLimitReason {
  return reason === "safety_turn_limit" || reason === "safety_action_limit";
}

export type PpoTeamAdjudication = {
  teamId: string;
  status: TeamStatus;
  active: boolean;
  ownedBaseCount: number;
  livingKingHp: number;
  totalLivingHp: number;
  livingUnitCount: number;
  /** One-based competition rank; tied teams share the first occupied rank. */
  rank: number;
  reward: number;
};

function compareTeams(a: PpoTeamAdjudication, b: PpoTeamAdjudication) {
  return Number(b.active) - Number(a.active)
    || b.ownedBaseCount - a.ownedBaseCount
    || b.livingKingHp - a.livingKingHp
    || b.totalLivingHp - a.totalLivingHp
    || b.livingUnitCount - a.livingUnitCount;
}

/** PPO-only cutoff evaluation. Does not mutate the game or declare a winner. */
export function adjudicatePpoTimeLimit(state: GameState): PpoTeamAdjudication[] {
  const teams = state.teams.filter((team) => !team.isNeutral && team.status !== "neutral").map((team) => {
    // Same living semantics as battle.ts and checkHeadlessInvariants (helpers are private).
    const living = state.units.filter((unit) => unit.ownerTeamId === team.id && unit.hp > 0 && unit.position.kind !== "removed");
    return {
      teamId: team.id, status: team.status, active: team.status === "active",
      // transferBaseOwnership uses the base owner as the current ownership authority.
      ownedBaseCount: state.bases.filter((base) => base.ownerTeamId === team.id).length,
      livingKingHp: living.filter((unit) => unit.type === "king").reduce((sum, unit) => sum + unit.hp, 0),
      totalLivingHp: living.reduce((sum, unit) => sum + unit.hp, 0),
      livingUnitCount: living.length, rank: 0, reward: 0,
    };
  }).sort(compareTeams);

  for (let start = 0; start < teams.length;) {
    let end = start + 1;
    while (end < teams.length && compareTeams(teams[start], teams[end]) === 0) end += 1;
    // The linear slots' mean equals the reward at their mean position.
    // This also makes a complete tie exactly zero, including a single team.
    const meanPosition = (start + end - 1) / 2;
    const reward = teams.length > 1 ? 0.25 * (1 - 2 * meanPosition / (teams.length - 1)) : 0;
    for (let index = start; index < end; index += 1) {
      teams[index].rank = start + 1;
      teams[index].reward = reward;
    }
    start = end;
  }
  return teams;
}
