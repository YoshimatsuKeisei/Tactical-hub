import type { GameState } from "../types";

export type TeamBattleAdvantage = {
  teamId: string;
  active: boolean;
  ownedBaseCount: number;
  livingKingHp: number;
  livingUnitCount: number;
  battleAdvantageRaw: number;
  battleAdvantageShare: number;
};

/**
 * Shared game-state evaluation for battle royale. Neutral teams are not part of
 * the result. Living units use the same hp/position semantics as PPO cutoff
 * adjudication, but this score is otherwise independent from that adjudication.
 */
export function calculateBattleAdvantage(state: GameState): TeamBattleAdvantage[] {
  const advantages = state.teams
    .filter((team) => !team.isNeutral && team.status !== "neutral")
    .map((team): TeamBattleAdvantage => {
      const active = team.status === "active";
      const living = state.units.filter(
        (unit) => unit.ownerTeamId === team.id && unit.hp > 0 && unit.position.kind !== "removed",
      );
      const ownedBaseCount = state.bases.filter((base) => base.ownerTeamId === team.id).length;
      const livingKingHp = living
        .filter((unit) => unit.type === "king")
        .reduce((sum, unit) => sum + unit.hp, 0);
      const livingUnitCount = living.length;
      return {
        teamId: team.id,
        active,
        ownedBaseCount,
        livingKingHp,
        livingUnitCount,
        battleAdvantageRaw: active
          ? ownedBaseCount + livingKingHp * 1.5 + livingUnitCount * 0.25
          : 0,
        battleAdvantageShare: 0,
      };
    });

  const active = advantages.filter((advantage) => advantage.active);
  const totalRaw = active.reduce((sum, advantage) => sum + advantage.battleAdvantageRaw, 0);
  if (totalRaw > 0) {
    for (const advantage of active) {
      advantage.battleAdvantageShare = advantage.battleAdvantageRaw / totalRaw;
    }
  } else if (active.length > 0) {
    // A malformed/custom state can have active teams with no bases or living
    // units. Keep shares finite and normalized without inventing a raw lead.
    for (const advantage of active) advantage.battleAdvantageShare = 1 / active.length;
  }
  return advantages;
}
