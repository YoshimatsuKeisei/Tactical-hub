import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { resolveBattleRoyaleFinalDuelTurnEnd } from "../game/engine/finalDuel";
import { createInitialGameState } from "../game/initialState";
import type { GameState, Unit } from "../game/types";
import { createTeamVisibleState } from "../game/visibility";
import { BattleAdvantageGauge, roundBattleAdvantagePercentages } from "./BattleAdvantageGauge";

function renderGauge(state: GameState) {
  return renderToStaticMarkup(<BattleAdvantageGauge state={state} />);
}

function defeatTeamsAfter(state: GameState, activeCount: number) {
  state.teams.filter((team) => !team.isNeutral).forEach((team, index) => {
    if (index >= activeCount) team.status = "defeated";
  });
}

function waterNinja(id: string, teamId: string): Unit {
  return {
    id,
    ownerTeamId: teamId,
    type: "ninja",
    hp: 1,
    position: { kind: "water", x: 4, y: 2 },
    statuses: [],
  };
}

function displayedPercentages(markup: string) {
  return [...markup.matchAll(/<strong>(\d+)%<\/strong>/g)].map((match) => Number(match[1]));
}

describe("BattleAdvantageGauge", () => {
  it("uses largest-remainder display rounding so percentages total exactly 100", () => {
    const percentages = roundBattleAdvantagePercentages([1 / 3, 1 / 3, 1 / 3]);

    expect(percentages).toEqual([34, 33, 33]);
    expect(percentages.reduce((sum, percentage) => sum + percentage, 0)).toBe(100);
    expect(roundBattleAdvantagePercentages([])).toEqual([]);
  });

  it.each([4, 3, 2, 1])("renders only the %i active non-neutral teams", (activeCount) => {
    const state = createInitialGameState();
    defeatTeamsAfter(state, activeCount);

    const markup = renderGauge(state);

    expect(markup.match(/role="meter"/g) ?? []).toHaveLength(activeCount);
    for (let teamNumber = 1; teamNumber <= activeCount; teamNumber += 1) {
      expect(markup).toContain(`Team ${teamNumber}`);
    }
    for (let teamNumber = activeCount + 1; teamNumber <= 4; teamNumber += 1) {
      expect(markup).not.toContain(`Team ${teamNumber}`);
    }
    expect(markup).not.toContain("Neutral Guard");
  });

  it("shows one remaining team at 100% using its existing team color", () => {
    const state = createInitialGameState();
    defeatTeamsAfter(state, 1);

    const markup = renderGauge(state);

    expect(markup).toContain("Team 1");
    expect(markup).toContain("100%");
    expect(markup).toContain("aria-valuenow=\"100\"");
    expect(markup).toContain("background-color:#d94a4a");
  });

  it("updates shares when bases, king HP and living units change", () => {
    const state = createInitialGameState();
    const before = renderGauge(state);
    const teamOneKing = state.units.find((unit) => unit.ownerTeamId === "team-1" && unit.type === "king")!;
    const teamTwoStrategist = state.units.find((unit) => unit.ownerTeamId === "team-2" && unit.type === "strategist")!;

    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-1";
    teamOneKing.hp -= 1;
    teamTwoStrategist.hp = 0;
    teamTwoStrategist.position = { kind: "removed", reason: "defeated" };
    const after = renderGauge(state);

    expect(before).not.toBe(after);
    expect(before).toContain("Team 1</span><strong>25%");
    expect(after).not.toContain("Team 1</span><strong>25%");
  });

  it("does not reflect an unrevealed enemy water ninja in the viewer's gauge", () => {
    const state = createInitialGameState();
    const baseline = renderGauge(createTeamVisibleState(state, "team-1"));
    state.units.push(waterNinja("hidden-enemy-ninja", "team-2"));

    expect(renderGauge(state)).not.toBe(baseline);
    expect(renderGauge(createTeamVisibleState(state, "team-1"))).toBe(baseline);
  });

  it("reflects an enemy water ninja after it is revealed to the viewer", () => {
    const state = createInitialGameState();
    const ninja = waterNinja("revealed-enemy-ninja", "team-2");
    const baseline = renderGauge(createTeamVisibleState(state, "team-1"));
    state.units.push(ninja);
    state.ninjaRevealStates = [{ ninjaUnitId: ninja.id, visibleToTeamIds: ["team-1"] }];

    const markup = renderGauge(createTeamVisibleState(state, "team-1"));

    expect(markup).not.toBe(baseline);
    expect(displayedPercentages(markup)[1]).toBeGreaterThan(displayedPercentages(markup)[0]);
  });

  it("continues to reflect an ordinary enemy unit in the viewer's gauge", () => {
    const state = createInitialGameState();
    const baseline = renderGauge(createTeamVisibleState(state, "team-1"));
    state.units.push({
      id: "visible-enemy-unit",
      ownerTeamId: "team-2",
      type: "infantry",
      hp: 1,
      position: { kind: "tile", x: 4, y: 1 },
      statuses: [],
    });

    const markup = renderGauge(createTeamVisibleState(state, "team-1"));

    expect(markup).not.toBe(baseline);
    expect(displayedPercentages(markup)[1]).toBeGreaterThan(displayedPercentages(markup)[0]);
  });

  it("keeps visible-state display percentages normalized to 100%", () => {
    const state = createInitialGameState();
    const ninja = waterNinja("viewer-hidden-ninja", "team-2");
    state.units.push(ninja);

    const hiddenPercentages = displayedPercentages(renderGauge(createTeamVisibleState(state, "team-1")));
    state.ninjaRevealStates = [{ ninjaUnitId: ninja.id, visibleToTeamIds: ["team-1"] }];
    const revealedPercentages = displayedPercentages(renderGauge(createTeamVisibleState(state, "team-1")));

    expect(hiddenPercentages.reduce((sum, percentage) => sum + percentage, 0)).toBe(100);
    expect(revealedPercentages.reduce((sum, percentage) => sum + percentage, 0)).toBe(100);
  });

  it("re-renders without defeated teams after a status change", () => {
    const state = createInitialGameState();
    expect(renderGauge(state)).toContain("Team 4");

    state.teams.find((team) => team.id === "team-4")!.status = "defeated";

    expect(renderGauge(state)).not.toContain("Team 4");
  });

  it("uses the calculator's equal shares when active-team raw total is zero", () => {
    const state = createInitialGameState();
    defeatTeamsAfter(state, 2);
    state.bases.forEach((base) => { base.ownerTeamId = "neutral"; });
    state.units = [];

    const markup = renderGauge(state);

    expect(markup.match(/<strong>50%<\/strong>/g) ?? []).toHaveLength(2);
    expect(markup.match(/aria-valuenow="50"/g) ?? []).toHaveLength(2);
  });

  it("uses the same two-team gauge during Final Duel", () => {
    const state = createInitialGameState();
    defeatTeamsAfter(state, 2);
    state.finalDuel = {
      active: true,
      teamIds: ["team-1", "team-2"],
      entryTurn: state.turnNumber,
      consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
    };

    const markup = renderGauge(state);

    expect(markup.match(/role="meter"/g) ?? []).toHaveLength(2);
    expect(markup).toContain("Team 1");
    expect(markup).toContain("Team 2");
    expect(markup).not.toContain("Final Duel");
  });

  it("does not change Final Duel's full-state Battle Advantage evaluation", () => {
    const state = createInitialGameState();
    defeatTeamsAfter(state, 2);
    state.units.push(waterNinja("official-hidden-ninja", "team-2"));
    state.finalDuel = {
      active: true,
      teamIds: ["team-1", "team-2"],
      entryTurn: state.turnNumber,
      consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
    };

    const visiblePercentages = displayedPercentages(
      renderGauge(createTeamVisibleState(state, "team-1")),
    );
    resolveBattleRoyaleFinalDuelTurnEnd(state, state.turnNumber);

    expect(visiblePercentages).toEqual([50, 50]);
    expect(state.finalDuel.consecutiveAdvantageTurns).toEqual({ "team-1": 0, "team-2": 1 });
  });

  it("shows a safe empty state when no active teams remain", () => {
    const state = createInitialGameState();
    defeatTeamsAfter(state, 0);

    const markup = renderGauge(state);

    expect(markup).toContain("表示できるactive teamがありません。");
    expect(markup).not.toContain("role=\"meter\"");
  });
});
