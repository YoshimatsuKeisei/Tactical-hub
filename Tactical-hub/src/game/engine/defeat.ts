import type { GameState, KingCampaignState, SiegeState, UnitPosition } from "../types";
import { requestSiegeRewards, selectCaptureTeam, transferBaseOwnership } from "./capture";
import { clearKingCampaignsForTeam, selectConquestTeam } from "./kingCampaign";
import { enqueueRewardRequest } from "./reward";

export type FallenBasePlan = { baseId: string; defendingTeamId: string; siege: SiegeState; candidateTeamIds: string[]; intendedCaptureTeamId?: string };
export type DefeatedKingPlan = { kingUnitId: string; kingTeamId: string; candidateTeamIds: string[]; campaign: KingCampaignState };

function removeTeamUnits(state: GameState, teamId: string) {
  const unitIds = new Set(state.units.filter((unit) => unit.ownerTeamId === teamId && unit.position.kind !== "removed").map((unit) => unit.id));
  for (const base of state.bases) for (const slot of base.slots) if (slot.unitId && unitIds.has(slot.unitId)) slot.unitId = undefined;
  state.units = state.units.map((unit) => unit.ownerTeamId === teamId && unit.position.kind !== "removed" ? { ...unit, hp: 0, position: { kind: "removed", reason: "team_defeat" } as UnitPosition, statuses: [] } : unit);
  state.logs.push({ id: `log-team-units-removed-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `敗北チーム残存駒の消失: ${teamId}`, relatedIds: [teamId, ...unitIds] });
}

function markTeamDefeated(state: GameState, teamId: string, reason: string) {
  state.teams = state.teams.map((team) => team.id === teamId ? { ...team, status: "defeated" } : team);
  state.logs.push({ id: `log-team-defeated-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `${reason}: ${teamId}`, relatedIds: [teamId] });
  removeTeamUnits(state, teamId);
  clearKingCampaignsForTeam(state, teamId);
}

function ownedBases(state: GameState, teamId: string) {
  return state.bases.filter((base) => base.ownerTeamId === teamId || state.teams.find((team) => team.id === teamId)?.controlledBaseIds.includes(base.id));
}

function enqueueKingContributionCompensation(state: GameState, campaign: KingCampaignState, conquestTeamId: string) {
  const conquestDamage = campaign.contributions.find((entry) => entry.teamId === conquestTeamId)?.cumulativeDamage ?? 0;
  for (const entry of campaign.contributions.filter((candidate) => candidate.teamId !== conquestTeamId && candidate.cumulativeDamage > conquestDamage)) {
    enqueueRewardRequest(state, { teamId: entry.teamId, rewardType: "king_contribution_compensation", sourceBaseId: `king:${campaign.kingUnitId}`, sourceKingUnitId: campaign.kingUnitId });
  }
}

export function resolveKingDefeats(state: GameState, defeatedKings: DefeatedKingPlan[], fallenBases: FallenBasePlan[], rng: () => number) {
  if (!defeatedKings.length) return false;
  if (defeatedKings.length > 1) {
    state.logs.push({ id: `log-multiple-kings-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `複数王同時撃破: ${defeatedKings.map((plan) => plan.kingUnitId).join(", ")}`, relatedIds: defeatedKings.map((plan) => plan.kingUnitId) });
    const losingTeams = [...new Set(defeatedKings.map((plan) => plan.kingTeamId))];
    for (const teamId of losingTeams) {
      const bases = ownedBases(state, teamId);
      markTeamDefeated(state, teamId, "複数王同時撃破による敗北");
      for (const base of bases) {
        transferBaseOwnership(state, base.id, "neutral");
        state.logs.push({ id: `log-base-neutralized-${state.logs.length}`, turnNumber: state.turnNumber, type: "capture", message: `拠点中立化: ${base.id}`, relatedIds: [base.id, teamId] });
      }
    }
    return true;
  }

  const plan = defeatedKings[0];
  const conquestTeamId = selectConquestTeam(plan.campaign, plan.candidateTeamIds, rng);
  if (!conquestTeamId) return false;
  const bases = ownedBases(state, plan.kingTeamId);
  state.logs.push({ id: `log-conquest-team-${state.logs.length}`, turnNumber: state.turnNumber, type: "battle", message: `征服チーム: ${conquestTeamId} / ${plan.kingUnitId}`, relatedIds: [conquestTeamId, plan.kingUnitId] });
  markTeamDefeated(state, plan.kingTeamId, "王撃破による敗北");

  for (const fallen of fallenBases.filter((entry) => entry.defendingTeamId === plan.kingTeamId)) {
    const intended = fallen.intendedCaptureTeamId ?? selectCaptureTeam(state, fallen.siege, fallen.candidateTeamIds, rng);
    if (intended) {
      requestSiegeRewards(state, fallen.siege, intended, false);
      if (intended !== conquestTeamId) enqueueRewardRequest(state, { teamId: intended, rewardType: "overridden_capture_compensation", sourceBaseId: fallen.baseId });
      state.logs.push({ id: `log-overridden-capture-${state.logs.length}`, turnNumber: state.turnNumber, type: "capture", message: `王撃破優先による占領褒賞変換: ${fallen.baseId} / 本来 ${intended}`, relatedIds: [fallen.baseId, intended, conquestTeamId] });
    }
  }
  for (const base of bases) {
    transferBaseOwnership(state, base.id, conquestTeamId);
    enqueueRewardRequest(state, { teamId: conquestTeamId, rewardType: "king_conquest_reward", sourceBaseId: base.id, sourceKingUnitId: plan.kingUnitId, fixedBaseId: base.id });
  }
  enqueueKingContributionCompensation(state, plan.campaign, conquestTeamId);
  return true;
}

export function defeatTeamsWithoutBases(state: GameState, excludedTeamIds = new Set<string>()) {
  for (const team of [...state.teams]) {
    if (team.isNeutral || team.status !== "active" || excludedTeamIds.has(team.id)) continue;
    if (state.bases.some((base) => base.ownerTeamId === team.id)) continue;
    markTeamDefeated(state, team.id, "所有拠点0による敗北");
  }
}
