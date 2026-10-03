import { describe, expect, it } from "vitest";
import {
  getGameTerminalResult,
  resolveBattleRoyaleFinalDuelTurnEnd,
  syncBattleRoyaleFinalDuel,
} from "../engine/finalDuel";
import { submitMovement } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import { runHeadlessMatch } from "../cpu/headlessSimulation";
import { RlEnvironment } from "../cpu/rlEnvironment";
import type { GameState } from "../types";

function setActiveTeams(state: GameState, teamIds: string[]) {
  for (const team of state.teams) {
    if (!team.isNeutral) team.status = teamIds.includes(team.id) ? "active" : "defeated";
  }
}

function duelState(entryTurn = 120, advantage: "first" | "second" | "tie" = "first") {
  const state = createInitialGameState();
  state.turnNumber = state.turnState.turnNumber = entryTurn;
  setActiveTeams(state, ["team-1", "team-2"]);
  const bonus = state.bases.find((base) => base.id === "neutral-north")!;
  if (advantage === "first") bonus.ownerTeamId = "team-1";
  else if (advantage === "second") bonus.ownerTeamId = "team-2";
  syncBattleRoyaleFinalDuel(state);
  return state;
}

describe("battle royale Final Duel", () => {
  it("starts on 3 -> 2 active teams", () => {
    const state = createInitialGameState();
    setActiveTeams(state, ["team-1", "team-2", "team-3"]);
    syncBattleRoyaleFinalDuel(state);
    expect(state.finalDuel).toBeUndefined();
    state.teams.find((team) => team.id === "team-3")!.status = "defeated";
    syncBattleRoyaleFinalDuel(state);
    expect(state.finalDuel).toMatchObject({ active: true, teamIds: ["team-1", "team-2"], entryTurn: 1 });
  });

  it("starts on 4 -> 2 but not while 3 or more remain", () => {
    const state = createInitialGameState();
    syncBattleRoyaleFinalDuel(state);
    expect(state.finalDuel).toBeUndefined();
    setActiveTeams(state, ["team-1", "team-2"]);
    syncBattleRoyaleFinalDuel(state);
    expect(state.finalDuel?.active).toBe(true);
  });

  it("records a simultaneous 4 -> 2 rules transition before the formal turn ends", () => {
    const state = createInitialGameState();
    for (const teamId of ["team-3", "team-4"]) {
      state.units = state.units.filter((unit) => unit.ownerTeamId !== teamId);
      for (const base of state.bases.filter((candidate) => candidate.ownerTeamId === teamId)) {
        base.ownerTeamId = "neutral";
        base.slots.forEach((slot) => { slot.unitId = undefined; });
      }
    }
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    const resolved = submitMovement(state, "team-1", () => 0);
    expect(resolved.finalDuel).toMatchObject({
      active: true,
      teamIds: ["team-1", "team-2"],
      entryTurn: 1,
      consecutiveAdvantageTurns: { "team-1": 0, "team-2": 0 },
    });
    expect(resolved.turnNumber).toBe(1);
  });

  it("does not start at one or zero active teams and leaves natural victory authoritative", () => {
    for (const ids of [["team-2"], []]) {
      const state = createInitialGameState();
      setActiveTeams(state, ids);
      syncBattleRoyaleFinalDuel(state);
      expect(state.finalDuel).toBeUndefined();
      expect(getGameTerminalResult(state)).toMatchObject({ terminal: true, resultReason: "natural_victory" });
    }
  });

  it("does not finish at 19 consecutive leads and finishes at 20", () => {
    const state = duelState();
    for (let turn = 120; turn <= 138; turn += 1) resolveBattleRoyaleFinalDuelTurnEnd(state, turn);
    expect(state.finalDuel?.consecutiveAdvantageTurns["team-1"]).toBe(19);
    expect(state.gameResult).toBeUndefined();
    resolveBattleRoyaleFinalDuelTurnEnd(state, 139);
    expect(state.finalDuel?.consecutiveAdvantageTurns["team-1"]).toBe(20);
    expect(state.gameResult).toEqual({ reason: "final_duel_consecutive_advantage", winnerTeamId: "team-1" });
  });

  it("resets the old leader on reversal and resets both on an exact tie", () => {
    const state = duelState();
    resolveBattleRoyaleFinalDuelTurnEnd(state, 120);
    expect(state.finalDuel?.consecutiveAdvantageTurns).toMatchObject({ "team-1": 1, "team-2": 0 });

    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-2";
    resolveBattleRoyaleFinalDuelTurnEnd(state, 121);
    expect(state.finalDuel?.consecutiveAdvantageTurns).toMatchObject({ "team-1": 0, "team-2": 1 });

    state.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "neutral";
    resolveBattleRoyaleFinalDuelTurnEnd(state, 122);
    expect(state.finalDuel?.consecutiveAdvantageTurns).toMatchObject({ "team-1": 0, "team-2": 0 });
  });

  it("does not timeout at 49 turns and chooses the raw leader at 50", () => {
    const state = duelState();
    resolveBattleRoyaleFinalDuelTurnEnd(state, 168);
    expect(state.gameResult).toBeUndefined();
    resolveBattleRoyaleFinalDuelTurnEnd(state, 169);
    expect(state.gameResult).toEqual({ reason: "final_duel_timeout_advantage", winnerTeamId: "team-1" });
  });

  it("returns a winnerless draw on an exact raw tie at the 50-turn timeout", () => {
    const state = duelState(120, "tie");
    resolveBattleRoyaleFinalDuelTurnEnd(state, 169);
    expect(state.gameResult).toEqual({ reason: "final_duel_timeout_draw" });
    expect(getGameTerminalResult(state)).toEqual({ terminal: true, winnerTeamId: undefined, resultReason: "final_duel_timeout_draw" });
  });

  it("counts entry turn 120 as 1, turn 139 as 20, and turn 169 as timeout 50", () => {
    const consecutive = duelState();
    resolveBattleRoyaleFinalDuelTurnEnd(consecutive, 120);
    expect(consecutive.finalDuel?.consecutiveAdvantageTurns["team-1"]).toBe(1);
    for (let turn = 121; turn <= 139; turn += 1) resolveBattleRoyaleFinalDuelTurnEnd(consecutive, turn);
    expect(consecutive.gameResult?.reason).toBe("final_duel_consecutive_advantage");

    const timeout = duelState(120, "tie");
    resolveBattleRoyaleFinalDuelTurnEnd(timeout, 168);
    expect(timeout.gameResult).toBeUndefined();
    resolveBattleRoyaleFinalDuelTurnEnd(timeout, 169);
    expect(timeout.gameResult?.reason).toBe("final_duel_timeout_draw");
  });

  it("samples once when the formal movement turn closes and turnNumber increments", () => {
    const state = duelState();
    state.units.find((unit) => unit.ownerTeamId === "team-2" && unit.type === "king")!.hp = 2;
    state.movementSeatOrderTeamIds = ["team-1", "team-2"];
    state.movementOrderTeamIds = ["team-1", "team-2"];
    state.movementCompletedTeamIds = ["team-1"];
    state.currentMovementTeamId = "team-2";
    state.productionCompletedTeamIdsThisTurn = ["team-2"];
    const resolved = submitMovement(state, "team-2", () => 0);
    expect(resolved.turnNumber).toBe(121);
    expect(resolved.finalDuel?.lastEvaluatedTurn).toBe(120);
    expect(resolved.finalDuel?.consecutiveAdvantageTurns["team-1"]).toBe(1);
  });

  it("prioritizes later natural victory and never replaces it with a second Final Duel winner", () => {
    const state = duelState();
    state.finalDuel!.consecutiveAdvantageTurns["team-1"] = 19;
    state.teams.find((team) => team.id === "team-2")!.status = "defeated";
    resolveBattleRoyaleFinalDuelTurnEnd(state, 120);
    expect(state.gameResult).toBeUndefined();
    expect(getGameTerminalResult(state)).toEqual({ terminal: true, winnerTeamId: "team-1", resultReason: "natural_victory" });
    resolveBattleRoyaleFinalDuelTurnEnd(state, 121);
    expect(state.gameResult).toBeUndefined();
  });

  it("surfaces the distinct result reason through RL and headless results", () => {
    const state = duelState();
    resolveBattleRoyaleFinalDuelTurnEnd(state, 169);

    const environment = new RlEnvironment();
    environment.reset(1, 4, state);
    expect(environment.getResult()).toMatchObject({ terminal: true, winnerTeamId: "team-1", loserTeamIds: ["team-2", "team-3", "team-4"], endReason: "victory", resultReason: "final_duel_timeout_advantage" });

    expect(runHeadlessMatch({ participantCount: 4, seed: 1, maxTurns: 500, initialState: state })).toMatchObject({
      endReason: "victory",
      winnerTeamId: "team-1",
      resultReason: "final_duel_timeout_advantage",
    });
  });
});
