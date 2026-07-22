import { describe, expect, it } from "vitest";
import { completeSiegeCapture, selectCaptureTeam, transferBaseOwnership } from "../engine/capture";
import { placeRewardUnit } from "../engine/reward";
import { getSiegeState, recordDefenderKill, recordEffectiveBaseAttacks, resetInactiveSieges } from "../engine/siege";
import { createInitialGameState } from "../initialState";
import type { SiegeState } from "../types";

describe("攻略状態・占領・褒賞", () => {
  it("同一ターン・同一チームの有効攻撃を拠点ごとに1回だけ数える", () => {
    const state = createInitialGameState();
    recordEffectiveBaseAttacks(state, [
      { baseId: "neutral-north", defendingTeamId: "neutral", attackingTeamId: "team-1" },
      { baseId: "neutral-north", defendingTeamId: "neutral", attackingTeamId: "team-1" },
    ]);
    expect(getSiegeState(state, "neutral-north")?.teamRecords[0].effectiveAttackTurns).toBe(1);
  });

  it("守備駒撃破をチーム別に記録する", () => {
    const state = createInitialGameState();
    recordDefenderKill(state, "neutral-north", "neutral", ["team-1", "team-2"]);
    expect(getSiegeState(state, "neutral-north")?.teamRecords).toEqual(expect.arrayContaining([
      expect.objectContaining({ teamId: "team-1", defenderKills: 1 }),
      expect.objectContaining({ teamId: "team-2", defenderKills: 1 }),
    ]));
  });

  it("9ターンでは維持し10ターン無攻撃でリセットする", () => {
    const state = createInitialGameState();
    state.turnNumber = 20;
    recordEffectiveBaseAttacks(state, [{ baseId: "neutral-north", defendingTeamId: "neutral", attackingTeamId: "team-1" }]);
    state.turnNumber = 29; resetInactiveSieges(state);
    expect(getSiegeState(state, "neutral-north")).toBeDefined();
    state.turnNumber = 30; resetInactiveSieges(state);
    expect(getSiegeState(state, "neutral-north")).toBeUndefined();
  });

  it("撃破数、距離、攻撃ターン数の順で候補を絞る", () => {
    const state = createInitialGameState();
    const siege: SiegeState = { baseId: "neutral-north", defendingTeamId: "neutral", active: true, defenderLossOccurred: true, fallCandidateTeamIds: [], teamRecords: [
      { teamId: "team-1", defenderKills: 2, effectiveAttackTurns: 1 },
      { teamId: "team-2", defenderKills: 1, effectiveAttackTurns: 9 },
    ] };
    expect(selectCaptureTeam(state, siege, ["team-1", "team-2"], () => 0.9)).toBe("team-1");
  });

  it("単独の陥落決定候補を総撃破数に関係なく選ぶ", () => {
    const state = createInitialGameState();
    const siege: SiegeState = { baseId: "neutral-north", defendingTeamId: "neutral", active: true, defenderLossOccurred: true, fallCandidateTeamIds: [], teamRecords: [
      { teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 1 },
      { teamId: "team-2", defenderKills: 8, effectiveAttackTurns: 8 },
    ] };
    expect(selectCaptureTeam(state, siege, ["team-1"])).toBe("team-1");
  });

  it("所有権情報を重複なく一括更新し攻略状態を消す", () => {
    const state = createInitialGameState();
    recordDefenderKill(state, "neutral-north", "neutral", ["team-1"]);
    transferBaseOwnership(state, "neutral-north", "team-1");
    expect(state.bases.find((base) => base.id === "neutral-north")?.ownerTeamId).toBe("team-1");
    expect(state.teams.find((team) => team.id === "team-1")?.controlledBaseIds.filter((id) => id === "neutral-north")).toHaveLength(1);
    expect(state.teams.filter((team) => team.controlledBaseIds.includes("neutral-north"))).toHaveLength(1);
    expect(getSiegeState(state, "neutral-north")).toBeUndefined();
  });

  it("全滅占領で占領褒賞と条件を満たす全補償要求を作る", () => {
    const state = createInitialGameState();
    const siege: SiegeState = { baseId: "neutral-north", defendingTeamId: "neutral", active: true, defenderLossOccurred: true, fallCandidateTeamIds: [], teamRecords: [
      { teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 3 },
      { teamId: "team-2", defenderKills: 2, effectiveAttackTurns: 1 },
      { teamId: "team-3", defenderKills: 2, effectiveAttackTurns: 1 },
    ] };
    state.siegeStates.push(siege);
    expect(completeSiegeCapture(state, siege, ["team-1"], "annihilation")).toBe(true);
    expect(state.rewardPlacementRequests.filter((request) => request.rewardType === "capture_reward")).toHaveLength(1);
    expect(state.rewardPlacementRequests.filter((request) => request.rewardType === "contribution_compensation").map((request) => request.teamId).sort()).toEqual(["team-2", "team-3"]);
  });

  it("褒賞駒を通常生産Intentを消費せず指定拠点へ配置する", () => {
    const state = createInitialGameState();
    const base = state.bases.find((entry) => entry.id === "neutral-north")!;
    for (const slot of base.slots) slot.unitId = undefined;
    state.units = state.units.filter((unit) => unit.position.kind !== "base" || unit.position.baseId !== base.id);
    transferBaseOwnership(state, base.id, "team-1");
    state.phase = "reward_placement"; state.turnState.phase = "reward_placement"; state.phaseAfterRewards = "attack_input";
    state.rewardPlacementRequests.push({ id: "reward-test", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: base.id, destinationKind: "fixed", fixedBaseId: base.id, eligibleBaseIds: [base.id], completed: false, expired: false });
    const resolved = placeRewardUnit(state, "reward-test", base.id, "infantry");
    expect(resolved.rewardPlacementRequests[0].completed).toBe(true);
    expect(resolved.units.some((unit) => unit.ownerTeamId === "team-1" && unit.position.kind === "base" && unit.position.baseId === base.id)).toBe(true);
    expect(resolved.turnState.actionIntents).toEqual(state.turnState.actionIntents);
    expect(resolved.phase).toBe("attack_input");
  });

  it("does not let a defeated team capture a base or receive its reward later in the same resolution", () => {
    const state = createInitialGameState();
    state.teams.find((team) => team.id === "team-1")!.status = "defeated";
    const siege: SiegeState = {
      baseId: "neutral-north",
      defendingTeamId: "neutral",
      active: true,
      defenderLossOccurred: true,
      fallCandidateTeamIds: ["team-1"],
      teamRecords: [{ teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 1 }],
    };
    state.siegeStates.push(siege);
    expect(completeSiegeCapture(state, siege, ["team-1"], "annihilation")).toBe(false);
    expect(state.bases.find((base) => base.id === siege.baseId)?.ownerTeamId).toBe("neutral");
    expect(state.rewardPlacementRequests).toHaveLength(0);
  });
});
