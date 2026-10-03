import { describe, expect, it } from "vitest";
import { createInitialGameState } from "../initialState";
import { createPpoTurnDiagnostics } from "../cpu/rlPpoTurnDiagnostics";

describe("PPO turn diagnostics", () => {
  it("reports the exact final state turn without inventing defeats or a duel", () => {
    const state = createInitialGameState();
    state.turnNumber = 37;

    expect(createPpoTurnDiagnostics(state)).toEqual({
      finalStateTurnNumber: 37,
      defeatedTeamTurns: {},
    });
  });

  it("reads the first actual defeat turn from defeat-log relatedIds only", () => {
    const state = createInitialGameState();
    state.turnNumber = 170;
    state.logs.push(
      {
        id: "not-a-team-defeat",
        turnNumber: 41,
        type: "battle",
        message: "not authoritative",
        relatedIds: ["team-1"],
      },
      {
        id: "log-team-defeated-1",
        turnNumber: 73,
        type: "battle",
        message: "defeated",
        relatedIds: ["team-1"],
      },
      {
        id: "log-team-defeated-2",
        turnNumber: 91,
        type: "battle",
        message: "defeated",
        relatedIds: ["team-2", "neutral", "missing-team"],
      },
      {
        id: "log-team-defeated-3",
        turnNumber: 99,
        type: "battle",
        message: "duplicate",
        relatedIds: ["team-1"],
      },
      {
        id: "log-team-defeated-4",
        turnNumber: 101,
        type: "battle",
        message: "missing relatedIds",
      },
    );

    expect(createPpoTurnDiagnostics(state)).toMatchObject({
      finalStateTurnNumber: 170,
      defeatedTeamTurns: {
        "team-1": 73,
        "team-2": 91,
      },
    });
  });

  it("copies all Final Duel fields and derives evaluated turns inclusively", () => {
    const state = createInitialGameState();
    state.turnNumber = 170;
    state.finalDuel = {
      active: false,
      teamIds: ["team-3", "team-4"],
      entryTurn: 120,
      lastEvaluatedTurn: 169,
      consecutiveAdvantageTurns: { "team-3": 0, "team-4": 7 },
    };
    const before = structuredClone(state);

    const diagnostics = createPpoTurnDiagnostics(state);

    expect(diagnostics.finalDuel).toEqual({
      teamIds: ["team-3", "team-4"],
      entryTurn: 120,
      lastEvaluatedTurn: 169,
      evaluatedTurns: 50,
      activeAtEnd: false,
      consecutiveAdvantageTurns: { "team-3": 0, "team-4": 7 },
    });
    expect(state).toEqual(before);
    expect(diagnostics.finalDuel?.teamIds).not.toBe(state.finalDuel.teamIds);
    expect(diagnostics.finalDuel?.consecutiveAdvantageTurns)
      .not.toBe(state.finalDuel.consecutiveAdvantageTurns);
  });

  it("omits evaluation fields until the Final Duel has been evaluated", () => {
    const state = createInitialGameState();
    state.finalDuel = {
      active: true,
      teamIds: ["team-1", "team-2"],
      entryTurn: 50,
      consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
    };

    expect(createPpoTurnDiagnostics(state).finalDuel).toEqual({
      teamIds: ["team-1", "team-2"],
      entryTurn: 50,
      activeAtEnd: true,
      consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
    });
  });
});
