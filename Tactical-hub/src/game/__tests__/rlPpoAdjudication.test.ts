import { describe, expect, it } from "vitest";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";
import { adjudicatePpoTimeLimit } from "../cpu/rlPpoAdjudication";
import type { GameState, TeamStatus, Unit } from "../types";

function fixture() {
  const state = createHeadlessInitialState(4);
  state.units = [];
  state.bases = [];
  return state;
}

function unit(state: GameState, teamId: string, hp: number, type: Unit["type"] = "infantry", position: Unit["position"] = { kind: "tile", x: 0, y: 0 }) {
  state.units.push({ id: `unit-${state.units.length}`, ownerTeamId: teamId, type, hp, position, statuses: [] });
}

function base(state: GameState, ownerTeamId: string) {
  state.bases.push({ id: `base-${state.bases.length}`, name: "test", type: "home", ownerTeamId, coords: [], slots: [] });
}

describe("Phase 12B PPO time-limit adjudication", () => {
  it("assigns four linear reward slots and ignores stale controlledBaseIds", () => {
    const state = fixture();
    for (let index = 0; index < 4; index += 1) {
      for (let count = 0; count < 3 - index; count += 1) base(state, `team-${index + 1}`);
      state.teams[index].controlledBaseIds = index === 3 ? ["fake-1", "fake-2", "fake-3", "fake-4"] : [];
    }
    const before = structuredClone(state);
    const result = adjudicatePpoTimeLimit(state);
    expect(result.map((team) => team.rank)).toEqual([1, 2, 3, 4]);
    for (const [index, reward] of [0.25, 1 / 12, -1 / 12, -0.25].entries()) expect(result[index].reward).toBeCloseTo(reward);
    expect(result.map((team) => team.ownedBaseCount)).toEqual([3, 2, 1, 0]);
    expect(state).toEqual(before);
  });

  it("gives exact zero to complete ties and never breaks ties by team ID or input order", () => {
    const state = fixture();
    const result = adjudicatePpoTimeLimit(state);
    expect(result.map((team) => [team.rank, team.reward])).toEqual(Array.from({ length: 4 }, () => [1, 0]));
    state.teams.reverse();
    expect(Object.fromEntries(adjudicatePpoTimeLimit(state).map((team) => [team.teamId, team.reward])))
      .toEqual(Object.fromEntries(result.map((team) => [team.teamId, team.reward])));
  });

  it("averages occupied reward slots for partial ties", () => {
    const state = fixture();
    unit(state, "team-1", 3); unit(state, "team-2", 3); unit(state, "team-3", 2); unit(state, "team-4", 1);
    const result = adjudicatePpoTimeLimit(state);
    expect(result.map((team) => team.rank)).toEqual([1, 1, 3, 4]);
    expect(result[0].reward).toBeCloseTo(1 / 6);
    expect(result[1].reward).toBeCloseTo(1 / 6);
    expect(result[2].reward).toBeCloseTo(-1 / 12);
    expect(result[3].reward).toBe(-0.25);
  });

  it.each(["active", "bases", "king_hp", "total_hp", "unit_count"])("honors %s priority over all lower metrics", (metric) => {
    const state = fixture();
    state.teams = state.teams.slice(0, 2);
    if (metric === "active") state.teams[1].status = "defeated";
    if (metric === "bases") base(state, "team-1");
    if (metric === "active") base(state, "team-2");
    if (["active", "bases"].includes(metric)) {
      unit(state, "team-2", 100, "king"); unit(state, "team-2", 100);
    } else if (metric === "king_hp") {
      unit(state, "team-1", 2, "king"); unit(state, "team-2", 1, "king"); unit(state, "team-2", 100);
    } else if (metric === "total_hp") {
      unit(state, "team-1", 5); unit(state, "team-2", 2); unit(state, "team-2", 2);
    } else {
      unit(state, "team-1", 2); unit(state, "team-1", 2); unit(state, "team-2", 4);
    }
    expect(adjudicatePpoTimeLimit(state).map((team) => [team.teamId, team.reward])).toEqual([["team-1", 0.25], ["team-2", -0.25]]);
  });

  it("counts living owned kings and units using hp and removed-position semantics, excluding neutral teams", () => {
    const state = fixture();
    unit(state, "team-1", 2, "king", { kind: "base", baseId: "home-1", slotId: "slot" });
    unit(state, "team-1", 3, "king");
    unit(state, "team-1", 4, "ninja", { kind: "water", x: 0, y: 1 });
    unit(state, "team-1", 99, "king", { kind: "removed", reason: "team_defeat" });
    unit(state, "team-1", 0, "king");
    unit(state, "team-2", 100, "king");
    state.teams.push({ ...state.teams[0], id: "status-neutral", status: "neutral", isNeutral: false });
    const result = adjudicatePpoTimeLimit(state);
    expect(result).toHaveLength(4);
    expect(result.find((team) => team.teamId === "team-1")).toMatchObject({ livingKingHp: 5, totalLivingHp: 9, livingUnitCount: 3 });
  });

  it("handles zero or one participant and ties defeated/eliminated with equal metrics", () => {
    const state = fixture();
    state.teams = state.teams.slice(0, 2);
    state.teams.forEach((team, index) => { team.status = ["defeated", "eliminated"][index] as TeamStatus; });
    expect(adjudicatePpoTimeLimit(state).map((team) => team.reward)).toEqual([0, 0]);
    state.teams.pop();
    expect(adjudicatePpoTimeLimit(state)[0]).toMatchObject({ rank: 1, reward: 0 });
    state.teams = [];
    expect(adjudicatePpoTimeLimit(state)).toEqual([]);
  });
});
