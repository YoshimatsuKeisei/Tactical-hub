import { describe, expect, it } from "vitest";
import { calculateBattleAdvantage } from "../engine/battleAdvantage";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit } from "../types";

function participant(state: GameState, teamId: string) {
  return calculateBattleAdvantage(state).find((entry) => entry.teamId === teamId)!;
}

describe("shared Battle Advantage", () => {
  it("uses base 1.0, king HP 1.5 and each living unit 0.25 for an active team", () => {
    const state = createInitialGameState();
    state.bases.forEach((base) => { base.ownerTeamId = "neutral"; });
    state.bases[0].ownerTeamId = "team-1";
    state.units = [
      { id: "king", ownerTeamId: "team-1", type: "king", hp: 2, position: { kind: "tile", x: 0, y: 0 }, statuses: [] },
      ...[1, 2, 3].map((index): Unit => ({ id: `unit-${index}`, ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "tile", x: index, y: 0 }, statuses: [] })),
      { id: "dead", ownerTeamId: "team-1", type: "infantry", hp: 0, position: { kind: "removed", reason: "defeated" }, statuses: [] },
    ];

    expect(participant(state, "team-1")).toMatchObject({
      active: true,
      ownedBaseCount: 1,
      livingKingHp: 2,
      livingUnitCount: 4,
      battleAdvantageRaw: 5,
    });
  });

  it("gives defeated teams zero and excludes neutral teams", () => {
    const state = createInitialGameState();
    state.teams.find((team) => team.id === "team-1")!.status = "defeated";
    const advantages = calculateBattleAdvantage(state);
    expect(advantages.some((entry) => entry.teamId === "neutral")).toBe(false);
    expect(participant(state, "team-1")).toMatchObject({ active: false, battleAdvantageRaw: 0, battleAdvantageShare: 0 });
  });

  it.each([2, 3, 4])("normalizes %i equal active teams to a total share of 1", (activeCount) => {
    const state = createInitialGameState();
    state.teams.filter((team) => !team.isNeutral).forEach((team, index) => {
      if (index >= activeCount) team.status = "defeated";
    });
    const active = calculateBattleAdvantage(state).filter((entry) => entry.active);
    expect(active).toHaveLength(activeCount);
    expect(active.reduce((sum, entry) => sum + entry.battleAdvantageShare, 0)).toBeCloseTo(1);
    active.forEach((entry) => expect(entry.battleAdvantageShare).toBeCloseTo(1 / activeCount));
  });

  it("normalizes unequal raw scores by their active-team total", () => {
    const state = createInitialGameState();
    state.teams.find((team) => team.id === "team-3")!.status = "defeated";
    state.teams.find((team) => team.id === "team-4")!.status = "defeated";
    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-1";
    const advantages = calculateBattleAdvantage(state);
    const first = advantages.find((entry) => entry.teamId === "team-1")!;
    const second = advantages.find((entry) => entry.teamId === "team-2")!;
    expect(first.battleAdvantageShare).toBeCloseTo(first.battleAdvantageRaw / (first.battleAdvantageRaw + second.battleAdvantageRaw));
    expect(first.battleAdvantageShare + second.battleAdvantageShare).toBeCloseTo(1);
  });

  it("is safe with no active teams and with zero total raw", () => {
    const none = createInitialGameState();
    none.teams.filter((team) => !team.isNeutral).forEach((team) => { team.status = "defeated"; });
    expect(calculateBattleAdvantage(none).every((entry) => entry.battleAdvantageShare === 0)).toBe(true);

    const zero = createInitialGameState();
    zero.teams.find((team) => team.id === "team-3")!.status = "defeated";
    zero.teams.find((team) => team.id === "team-4")!.status = "defeated";
    zero.bases.forEach((base) => { base.ownerTeamId = "neutral"; });
    zero.units = [];
    const active = calculateBattleAdvantage(zero).filter((entry) => entry.active);
    expect(active.map((entry) => entry.battleAdvantageRaw)).toEqual([0, 0]);
    expect(active.map((entry) => entry.battleAdvantageShare)).toEqual([0.5, 0.5]);
  });
});
