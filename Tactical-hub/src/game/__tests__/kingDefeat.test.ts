import { describe, expect, it } from "vitest";
import { defeatTeamsWithoutBases, resolveKingDefeats, type DefeatedKingPlan, type FallenBasePlan } from "../engine/defeat";
import { getKingCampaign, recordKingAttackTurns, recordKingDamage, selectConquestTeam } from "../engine/kingCampaign";
import { createInitialGameState } from "../initialState";
import type { KingCampaignState, SiegeState } from "../types";

function campaign(kingUnitId = "home-2-king", kingTeamId = "team-2"): KingCampaignState {
  return { kingUnitId, kingTeamId, contributions: [
    { teamId: "team-1", cumulativeDamage: 2, effectiveAttackTurns: 2 },
    { teamId: "team-3", cumulativeDamage: 4, effectiveAttackTurns: 1 },
  ] };
}

function defeatPlan(value = campaign(), candidates = ["team-1"]): DefeatedKingPlan {
  return { kingUnitId: value.kingUnitId, kingTeamId: value.kingTeamId, candidateTeamIds: candidates, campaign: value };
}

describe("王攻略と勢力敗北", () => {
  it("王への攻撃ターンを同一チーム・同一ターンで1回だけ記録する", () => {
    const state = createInitialGameState();
    recordKingAttackTurns(state, [
      { kingUnitId: "home-2-king", kingTeamId: "team-2", attackingTeamId: "team-1" },
      { kingUnitId: "home-2-king", kingTeamId: "team-2", attackingTeamId: "team-1" },
    ]);
    expect(getKingCampaign(state, "home-2-king")?.contributions[0].effectiveAttackTurns).toBe(1);
  });

  it("同時成功分の王累積ダメージをチーム別に上限補正せず記録する", () => {
    const state = createInitialGameState();
    recordKingDamage(state, "home-2-king", "team-2", "team-1", 1);
    recordKingDamage(state, "home-2-king", "team-2", "team-3", 1);
    expect(getKingCampaign(state, "home-2-king")?.contributions.map((entry) => [entry.teamId, entry.cumulativeDamage])).toEqual([["team-1", 1], ["team-3", 1]]);
  });

  it("単独の王撃破候補は過去貢献に関係なく征服する", () => {
    expect(selectConquestTeam(campaign(), ["team-1"], () => 0.9)).toBe("team-1");
  });

  it("複数候補は累積ダメージ、攻撃ターン、注入乱数の順で決める", () => {
    expect(selectConquestTeam(campaign(), ["team-1", "team-3"], () => 0)).toBe("team-3");
    const tied = campaign(); tied.contributions[0] = { teamId: "team-1", cumulativeDamage: 4, effectiveAttackTurns: 3 };
    expect(selectConquestTeam(tied, ["team-1", "team-3"], () => 0)).toBe("team-1");
  });

  it("王撃破で全残存駒を消し全拠点を継承して拠点ごとの褒賞を作る", () => {
    const state = createInitialGameState();
    state.kingCampaignStates.push(campaign());
    resolveKingDefeats(state, [defeatPlan()], [], () => 0);
    expect(state.teams.find((team) => team.id === "team-2")?.status).toBe("defeated");
    expect(state.units.filter((unit) => unit.ownerTeamId === "team-2").every((unit) => unit.position.kind === "removed")).toBe(true);
    expect(state.bases.filter((base) => base.id === "home-2").every((base) => base.ownerTeamId === "team-1")).toBe(true);
    expect(state.rewardPlacementRequests.filter((request) => request.rewardType === "king_conquest_reward" && request.fixedBaseId === "home-2")).toHaveLength(1);
    expect(state.kingCampaignStates.some((entry) => entry.kingTeamId === "team-2")).toBe(false);
  });

  it("征服チームより累積ダメージが厳密に多い全チームへ王攻略補償を作る", () => {
    const state = createInitialGameState();
    const value = campaign(); value.contributions.push({ teamId: "team-4", cumulativeDamage: 3, effectiveAttackTurns: 1 });
    state.kingCampaignStates.push(value);
    resolveKingDefeats(state, [defeatPlan(value)], [], () => 0);
    expect(state.rewardPlacementRequests.filter((request) => request.rewardType === "king_contribution_compensation").map((request) => request.teamId).sort()).toEqual(["team-3", "team-4"]);
  });

  it("複数王同時撃破では敗北チーム拠点を中立化し褒賞を作らない", () => {
    const state = createInitialGameState();
    const second = campaign("home-3-king", "team-3");
    resolveKingDefeats(state, [defeatPlan(), defeatPlan(second, ["team-4"])], [], () => 0);
    expect(state.bases.find((base) => base.id === "home-2")?.ownerTeamId).toBe("neutral");
    expect(state.bases.find((base) => base.id === "home-3")?.ownerTeamId).toBe("neutral");
    expect(state.rewardPlacementRequests).toHaveLength(0);
  });

  it("同時陥落の本来占領チームへ変換補償を与え攻略功労補償も維持する", () => {
    const state = createInitialGameState();
    const siege: SiegeState = { baseId: "home-2", defendingTeamId: "team-2", active: true, defenderLossOccurred: true, fallCandidateTeamIds: [], teamRecords: [
      { teamId: "team-3", defenderKills: 1, effectiveAttackTurns: 2 },
      { teamId: "team-4", defenderKills: 2, effectiveAttackTurns: 1 },
    ] };
    const fallen: FallenBasePlan = { baseId: "home-2", defendingTeamId: "team-2", siege, candidateTeamIds: ["team-3"], intendedCaptureTeamId: "team-3" };
    resolveKingDefeats(state, [defeatPlan()], [fallen], () => 0);
    expect(state.rewardPlacementRequests.some((request) => request.rewardType === "overridden_capture_compensation" && request.teamId === "team-3")).toBe(true);
    expect(state.rewardPlacementRequests.some((request) => request.rewardType === "contribution_compensation" && request.teamId === "team-4")).toBe(true);
  });

  it("征服チームと本来占領チームが同じなら変換補償を重複生成しない", () => {
    const state = createInitialGameState();
    const siege: SiegeState = { baseId: "home-2", defendingTeamId: "team-2", active: true, defenderLossOccurred: true, fallCandidateTeamIds: [], teamRecords: [{ teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 1 }] };
    resolveKingDefeats(state, [defeatPlan()], [{ baseId: "home-2", defendingTeamId: "team-2", siege, candidateTeamIds: ["team-1"], intendedCaptureTeamId: "team-1" }], () => 0);
    expect(state.rewardPlacementRequests.filter((request) => request.rewardType === "overridden_capture_compensation")).toHaveLength(0);
  });

  it("所有拠点0では王を含む残存駒を除去し王攻略記録を破棄する", () => {
    const state = createInitialGameState();
    state.kingCampaignStates.push(campaign());
    state.bases.find((base) => base.id === "home-2")!.ownerTeamId = "team-1";
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds = [];
    defeatTeamsWithoutBases(state);
    expect(state.teams.find((team) => team.id === "team-2")?.status).toBe("defeated");
    expect(state.units.filter((unit) => unit.ownerTeamId === "team-2").every((unit) => unit.position.kind === "removed")).toBe(true);
    expect(state.rewardPlacementRequests).toHaveLength(0);
    expect(state.kingCampaignStates).toHaveLength(0);
  });
});
