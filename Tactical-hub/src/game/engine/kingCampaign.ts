import type { GameState, KingCampaignState } from "../types";

export function getKingCampaign(state: GameState, kingUnitId: string) {
  return state.kingCampaignStates.find((campaign) => campaign.kingUnitId === kingUnitId);
}

function ensureCampaign(state: GameState, kingUnitId: string, kingTeamId: string): KingCampaignState {
  let campaign = getKingCampaign(state, kingUnitId);
  if (!campaign) {
    campaign = { kingUnitId, kingTeamId, contributions: [] };
    state.kingCampaignStates.push(campaign);
  }
  return campaign;
}

function contribution(campaign: KingCampaignState, teamId: string) {
  let entry = campaign.contributions.find((candidate) => candidate.teamId === teamId);
  if (!entry) {
    entry = { teamId, cumulativeDamage: 0, effectiveAttackTurns: 0 };
    campaign.contributions.push(entry);
  }
  return entry;
}

export function recordKingAttackTurns(state: GameState, attacks: { kingUnitId: string; kingTeamId: string; attackingTeamId: string }[]) {
  const unique = new Map(attacks.map((attack) => [`${attack.kingUnitId}:${attack.attackingTeamId}`, attack]));
  for (const attack of unique.values()) {
    contribution(ensureCampaign(state, attack.kingUnitId, attack.kingTeamId), attack.attackingTeamId).effectiveAttackTurns += 1;
    state.logs.push({ id: `log-king-attack-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `王への有効攻撃ターン: ${attack.attackingTeamId} → ${attack.kingUnitId}`, relatedIds: [attack.kingUnitId, attack.attackingTeamId] });
  }
}

export function recordKingDamage(state: GameState, kingUnitId: string, kingTeamId: string, attackingTeamId: string, damage: number) {
  contribution(ensureCampaign(state, kingUnitId, kingTeamId), attackingTeamId).cumulativeDamage += damage;
  state.logs.push({ id: `log-king-damage-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `王への累積ダメージ更新: ${attackingTeamId} → ${kingUnitId} +${damage}`, relatedIds: [kingUnitId, attackingTeamId] });
}

export function selectConquestTeam(campaign: KingCampaignState, candidateTeamIds: string[], rng: () => number) {
  let candidates = [...new Set(candidateTeamIds)].filter((teamId) => teamId !== campaign.kingTeamId);
  if (candidates.length <= 1) return candidates[0];
  const entry = (teamId: string) => campaign.contributions.find((candidate) => candidate.teamId === teamId);
  const maxDamage = Math.max(...candidates.map((teamId) => entry(teamId)?.cumulativeDamage ?? 0));
  candidates = candidates.filter((teamId) => (entry(teamId)?.cumulativeDamage ?? 0) === maxDamage);
  if (candidates.length <= 1) return candidates[0];
  const maxTurns = Math.max(...candidates.map((teamId) => entry(teamId)?.effectiveAttackTurns ?? 0));
  candidates = candidates.filter((teamId) => (entry(teamId)?.effectiveAttackTurns ?? 0) === maxTurns);
  return candidates.length === 1 ? candidates[0] : candidates[Math.min(candidates.length - 1, Math.floor(rng() * candidates.length))];
}

export function clearKingCampaignsForTeam(state: GameState, teamId: string) {
  const removed = state.kingCampaignStates.filter((campaign) => campaign.kingTeamId === teamId);
  state.kingCampaignStates = state.kingCampaignStates.filter((campaign) => campaign.kingTeamId !== teamId);
  if (removed.length) state.logs.push({ id: `log-king-campaign-clear-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `王攻略記録の破棄: ${teamId}`, relatedIds: [teamId, ...removed.map((campaign) => campaign.kingUnitId)] });
}
