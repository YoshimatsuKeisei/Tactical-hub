import { describe, expect, it, vi } from "vitest";
import { resolveBattle } from "../engine/battle";
import { getMovementCandidates, saveMovementIntent, submitMovement } from "../engine/movement";
import { submitTeamProduction } from "../engine/production";
import { enqueueRewardRequest, getPendingRewardRequests, getRewardPlacementCandidates, normalizeRewardPlacementState, placeRewardUnit } from "../engine/reward";
import { createInitialGameState } from "../initialState";
import type { GameState, RewardPlacementRequest, RewardType } from "../types";

function request(id: string, rewardType: RewardType = "capture_reward", destinationKind: RewardPlacementRequest["destinationKind"] = "fixed"): RewardPlacementRequest {
  return {
    id,
    teamId: "team-1",
    rewardType,
    sourceBaseId: "home-1",
    destinationKind,
    fixedBaseId: destinationKind === "fixed" ? "home-1" : undefined,
    eligibleBaseIds: ["home-1"],
    completed: false,
    expired: false,
  };
}

function enterRewardPhase(state: GameState, requests: RewardPlacementRequest[], after: GameState["phaseAfterRewards"] = "strategist_action_input") {
  state.phase = state.turnState.phase = "reward_placement";
  state.phaseAfterRewards = after;
  state.rewardPlacementRequests.push(...requests);
  return state;
}

describe("mandatory reward and compensation placement flow", () => {
  it.each([
    ["capture reward", "capture_reward"],
    ["contribution compensation", "contribution_compensation"],
  ] as const)("does not enter strategist actions while a %s request remains", (_label, rewardType) => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    state.rewardPlacementRequests.push(request(`pending-${rewardType}`, rewardType));
    const resolved = resolveBattle(state, () => 0);
    expect(resolved.phase).toBe("reward_placement");
    expect(resolved.phaseAfterRewards).toBe("strategist_action_input");
    expect(resolved.strategistSubmittedTeamIds).toEqual([]);
  });

  it("interrupts movement immediately when combat abandonment creates a reward and resumes at the next team", () => {
    let state = submitTeamProduction(createInitialGameState(), "team-1");
    const base = state.bases.find((entry) => entry.id === "home-1")!;
    const mover = state.units.find((unit) => unit.id === "home-1-strategist")!;
    for (const slot of base.slots) {
      if (slot.unitId && slot.unitId !== mover.id) {
        const unit = state.units.find((entry) => entry.id === slot.unitId);
        if (unit) { unit.hp = 0; unit.position = { kind: "removed", reason: "defeated" }; }
        slot.unitId = undefined;
      }
    }
    state.siegeStates.push({
      baseId: base.id,
      defendingTeamId: "team-1",
      active: true,
      defenderLossOccurred: true,
      fallCandidateTeamIds: ["team-2", "team-3"],
      teamRecords: [
        { teamId: "team-2", defenderKills: 1, effectiveAttackTurns: 1 },
        { teamId: "team-3", defenderKills: 1, effectiveAttackTurns: 1 },
      ],
    });
    const tiedPosition = { kind: "tile", x: 4, y: 1 } as const;
    for (const unit of state.units.filter((entry) => entry.ownerTeamId === "team-2" || entry.ownerTeamId === "team-3")) unit.position = tiedPosition;
    const destination = getMovementCandidates(state, mover.id).find((candidate) => candidate.kind === "tile");
    expect(destination).toBeDefined();
    state = saveMovementIntent(state, { teamId: "team-1", unitId: mover.id, from: mover.position, to: destination!, stay: false });
    let rngCalls = 0;
    const unseededRandom = vi.spyOn(Math, "random").mockImplementation(() => { throw new Error("unseeded Math.random was used"); });
    try {
      state = submitMovement(state, "team-1", () => { rngCalls += 1; return 0; });
    } finally {
      unseededRandom.mockRestore();
    }
    expect(rngCalls).toBe(1);
    expect(state).toMatchObject({ phase: "reward_placement", phaseAfterRewards: "movement_input", currentMovementTeamId: "team-2" });
    const generated = state.rewardPlacementRequests.find((entry) => entry.rewardType === "capture_reward" && !entry.completed && !entry.expired);
    expect(generated).toMatchObject({ teamId: "team-2", fixedBaseId: "home-1" });
    state = placeRewardUnit(state, generated!.id, "home-1", "infantry");
    expect(state).toMatchObject({ phase: "movement_input", currentMovementTeamId: "team-2" });
  });

  it("preserves the current team and its processing position when a pending request interrupts before submission", () => {
    let state = submitTeamProduction(createInitialGameState(), "team-1");
    state.rewardPlacementRequests.push(request("same-team-interrupt"));
    const completedBefore = [...state.movementCompletedTeamIds];
    state = submitMovement(state, "team-1");
    expect(state).toMatchObject({ phase: "reward_placement", phaseAfterRewards: "movement_input", currentMovementTeamId: "team-1" });
    expect(state.movementCompletedTeamIds).toEqual(completedBefore);
    state = placeRewardUnit(state, "same-team-interrupt", "home-1", "infantry");
    expect(state).toMatchObject({ phase: "movement_input", currentMovementTeamId: "team-1" });
  });

  it("resolves multiple requests one at a time against the latest base occupancy", () => {
    const state = createInitialGameState();
    const home = state.bases.find((base) => base.id === "home-1")!;
    for (const slot of home.slots.slice(0, -1)) slot.unitId ??= `occupied-${slot.id}`;
    const lastSlot = home.slots.at(-1)!;
    if (lastSlot.unitId) {
      const occupant = state.units.find((unit) => unit.id === lastSlot.unitId);
      if (occupant) { occupant.hp = 0; occupant.position = { kind: "removed", reason: "defeated" }; }
    }
    lastSlot.unitId = undefined;
    const fallback = state.bases.find((base) => base.id === "neutral-north")!;
    fallback.ownerTeamId = "team-1";
    state.teams.find((team) => team.id === "team-1")!.controlledBaseIds.push(fallback.id);
    enterRewardPhase(state, [request("selectable-1", "contribution_compensation", "selectable"), request("selectable-2", "king_contribution_compensation", "selectable")]);

    let next = placeRewardUnit(state, "selectable-1", "home-1", "infantry");
    expect(next.phase).toBe("reward_placement");
    expect(getRewardPlacementCandidates(next, "team-1").filter((candidate) => candidate.requestId === "selectable-2").map((candidate) => candidate.baseId)).toContain("neutral-north");
    next = placeRewardUnit(next, "selectable-2", "neutral-north", "infantry");
    expect(next.rewardPlacementRequests.every((entry) => entry.completed)).toBe(true);
    expect(next.phase).toBe("strategist_action_input");
  });

  it("keeps the ordinary transition when no request exists", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    expect(resolveBattle(state, () => 0).phase).toBe("strategist_action_input");
  });

  it.each([
    ["neutral", "neutral", "neutral-north"],
    ["defeated", "team-1", "home-1"],
    ["eliminated", "team-1", "home-1"],
  ] as const)("rejects reward placement for a %s recipient without creating a unit", (status, teamId, baseId) => {
    const state = createInitialGameState();
    const team = state.teams.find((candidate) => candidate.id === teamId)!;
    if (status !== "neutral") team.status = status;
    enterRewardPhase(state, [{ ...request(`invalid-${status}`), teamId, sourceBaseId: baseId, fixedBaseId: baseId, eligibleBaseIds: [baseId] }]);
    const unitCount = state.units.length;
    const slots = structuredClone(state.bases.map((base) => [base.id, base.slots.map((slot) => slot.unitId)]));

    const resolved = placeRewardUnit(state, `invalid-${status}`, baseId, "infantry");

    expect(resolved.units).toHaveLength(unitCount);
    expect(resolved.bases.map((base) => [base.id, base.slots.map((slot) => slot.unitId)])).toEqual(slots);
    expect(resolved.rewardPlacementRequests.find((entry) => entry.id === `invalid-${status}`)).toMatchObject({
      expired: true,
      expirationReason: "recipient_not_active_player",
    });
    expect(resolved.phase).toBe("strategist_action_input");
  });

  it("rejects a fixed reward base not owned by the recipient", () => {
    const state = enterRewardPhase(createInitialGameState(), [{
      ...request("foreign-fixed"),
      fixedBaseId: "home-2",
      sourceBaseId: "home-2",
      eligibleBaseIds: ["home-2"],
    }]);
    const unitCount = state.units.length;
    const slots = structuredClone(state.bases.map((base) => [base.id, base.slots.map((slot) => slot.unitId)]));

    const resolved = placeRewardUnit(state, "foreign-fixed", "home-2", "infantry");

    expect(resolved.units).toHaveLength(unitCount);
    expect(resolved.bases.map((base) => [base.id, base.slots.map((slot) => slot.unitId)])).toEqual(slots);
    expect(resolved.rewardPlacementRequests[0]).toMatchObject({ expired: true, expirationReason: "fixed_base_not_owned_by_recipient" });
    expect(resolved.phase).toBe("strategist_action_input");
  });

  it("does not enqueue rewards for neutral, defeated, eliminated, or foreign fixed-base recipients", () => {
    const state = createInitialGameState();
    const team1 = state.teams.find((team) => team.id === "team-1")!;
    expect(enqueueRewardRequest(state, { teamId: "neutral", rewardType: "capture_reward", sourceBaseId: "neutral-north", fixedBaseId: "neutral-north" })).toBeUndefined();
    expect(enqueueRewardRequest(state, { teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-2", fixedBaseId: "home-2" })).toBeUndefined();
    team1.status = "defeated";
    expect(enqueueRewardRequest(state, { teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", fixedBaseId: "home-1" })).toBeUndefined();
    team1.status = "eliminated";
    expect(enqueueRewardRequest(state, { teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", fixedBaseId: "home-1" })).toBeUndefined();
    expect(state.rewardPlacementRequests).toHaveLength(0);
  });

  it("expires an invalid legacy request and resumes its saved continuation phase", () => {
    const state = createInitialGameState();
    state.teams.find((team) => team.id === "team-1")!.status = "defeated";
    enterRewardPhase(state, [request("legacy-defeated")], "attack_input");

    const normalized = normalizeRewardPlacementState(state);

    expect(getPendingRewardRequests(normalized)).toHaveLength(0);
    expect(normalized.rewardPlacementRequests[0]).toMatchObject({ expired: true, expirationReason: "recipient_not_active_player" });
    expect(normalized.phase).toBe("attack_input");
    expect(normalized.turnState.phase).toBe("attack_input");
    expect(normalized.phaseAfterRewards).toBeUndefined();
  });
});
