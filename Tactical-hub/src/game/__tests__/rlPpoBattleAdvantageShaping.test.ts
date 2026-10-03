import { describe, expect, it, vi } from "vitest";
import * as battleAdvantage from "../engine/battleAdvantage";
import { createInitialGameState } from "../initialState";
import { createTeamVisibleState } from "../visibility";
import {
  beginPpoBattleAdvantageDecision,
  calculatePpoBattleAdvantagePotential,
  createPpoBattleAdvantageShapingRuntime,
  DEFAULT_BATTLE_ADVANTAGE_SHAPING_BETA,
  parseBattleAdvantageShapingBeta,
  trackPpoBattleAdvantageDecision,
} from "../cpu/rlPpoBattleAdvantageShaping";
import type { GameState, Unit } from "../types";

type RewardStep = { reward: number };

function potential(state: GameState, teamId: string) {
  return calculatePpoBattleAdvantagePotential(state, teamId);
}

function addDecision(
  runtime: ReturnType<typeof createPpoBattleAdvantageShapingRuntime>,
  state: GameState,
  teamId: string,
) {
  const step: RewardStep = {
    reward: beginPpoBattleAdvantageDecision(runtime, state, teamId),
  };
  trackPpoBattleAdvantageDecision(runtime, teamId, step);
  return step;
}

function stateSequence() {
  const first = createInitialGameState();
  const second = structuredClone(first) as GameState;
  second.bases.find((base) => base.id === "neutral-north")!.ownerTeamId = "team-1";
  const third = structuredClone(second) as GameState;
  third.units.find((unit) => unit.ownerTeamId === "team-1" && unit.type === "king")!.hp -= 1;
  return [first, second, third];
}

describe("PPO Battle Advantage potential shaping", () => {
  it("uses an exact allocation-free legacy branch at beta zero", () => {
    const runtime = createPpoBattleAdvantageShapingRuntime(0, 0.99);
    expect(DEFAULT_BATTLE_ADVANTAGE_SHAPING_BETA).toBe(0);
    expect(runtime).toBeUndefined();
    expect(beginPpoBattleAdvantageDecision(runtime, undefined as unknown as GameState, "team-1")).toBe(0);
    expect(() => trackPpoBattleAdvantageDecision(runtime, "team-1", { reward: 0 })).not.toThrow();
  });

  it("uses the shared Battle Advantage calculator on the supplied full GameState", () => {
    const state = createInitialGameState();
    const expected = battleAdvantage.calculateBattleAdvantage(state)
      .find((entry) => entry.teamId === "team-1")!.battleAdvantageShare;
    const calculator = vi.spyOn(battleAdvantage, "calculateBattleAdvantage");
    expect(calculatePpoBattleAdvantagePotential(state, "team-1")).toBe(expected);
    expect(calculator).toHaveBeenCalledTimes(1);
    expect(calculator).toHaveBeenCalledWith(state);
  });

  it("returns zero for inactive or explicit training-terminal teams", () => {
    const state = createInitialGameState();
    state.teams.find((team) => team.id === "team-1")!.status = "defeated";
    expect(calculatePpoBattleAdvantagePotential(state, "team-1")).toBe(0);
    state.teams.find((team) => team.id === "team-1")!.status = "active";
    expect(calculatePpoBattleAdvantagePotential(state, "team-1", true)).toBe(0);
  });

  it.each([0.99, 0.7])("uses the run gamma %s in the same-team potential formula", (gamma) => {
    const beta = 0.02;
    const [first, second] = stateSequence();
    const runtime = createPpoBattleAdvantageShapingRuntime(beta, gamma);
    const firstStep = addDecision(runtime, first, "team-1");
    const phi0 = potential(first, "team-1");
    const phi1 = potential(second, "team-1");
    addDecision(runtime, second, "team-1");
    expect(firstStep.reward).toBeCloseTo(beta * (gamma * phi1 - phi0));
  });

  it("uses the next same-team decision across interleaved opponent decisions", () => {
    const beta = 0.02;
    const gamma = 0.99;
    const [first, second] = stateSequence();
    const runtime = createPpoBattleAdvantageShapingRuntime(beta, gamma);
    const teamOneFirst = addDecision(runtime, first, "team-1");
    const beforeOpponent = teamOneFirst.reward;
    addDecision(runtime, first, "team-2");
    expect(teamOneFirst.reward).toBe(beforeOpponent);
    addDecision(runtime, second, "team-1");
    expect(teamOneFirst.reward).toBeCloseTo(
      beta * (gamma * potential(second, "team-1") - potential(first, "team-1")),
    );
  });

  it("leaves the final defeated/terminal sample at -beta * Phi and never positive", () => {
    const beta = 0.02;
    const state = createInitialGameState();
    const runtime = createPpoBattleAdvantageShapingRuntime(beta, 0.99);
    const last = addDecision(runtime, state, "team-1");
    expect(last.reward).toBeCloseTo(-beta * potential(state, "team-1"));
    expect(last.reward).toBeLessThanOrEqual(0);
  });

  it("uses hidden water ninjas from the formal full GameState", () => {
    const state = createInitialGameState();
    const hidden: Unit = {
      id: "hidden-water-ninja",
      ownerTeamId: "team-2",
      type: "ninja",
      hp: 1,
      position: { kind: "water", x: 0, y: 0 },
      statuses: [],
    };
    state.units.push(hidden);
    const visible = createTeamVisibleState(state, "team-1");
    expect(visible.units.some((unit) => unit.id === hidden.id)).toBe(false);
    expect(potential(state, "team-1")).toBeLessThan(potential(visible, "team-1"));
  });

  it("gives synchronous and worker-style runtimes identical rewards", () => {
    const states = stateSequence();
    const sync = createPpoBattleAdvantageShapingRuntime(0.02, 0.99);
    const worker = createPpoBattleAdvantageShapingRuntime(0.02, 0.99);
    const syncSteps: RewardStep[] = [];
    const workerSteps: RewardStep[] = [];
    for (const [index, state] of states.entries()) {
      const teamId = index === 1 ? "team-2" : "team-1";
      syncSteps.push(addDecision(sync, state, teamId));
      workerSteps.push(addDecision(worker, state, teamId));
    }
    expect(workerSteps.map((step) => step.reward)).toEqual(syncSteps.map((step) => step.reward));
  });

  it("telescopes to -beta * Phi0 with terminal Phi zero", () => {
    const beta = 0.02;
    const gamma = 0.99;
    const states = stateSequence();
    const runtime = createPpoBattleAdvantageShapingRuntime(beta, gamma);
    const steps = states.map((state) => addDecision(runtime, state, "team-1"));
    const discounted = steps.reduce(
      (sum, step, index) => sum + gamma ** index * step.reward,
      0,
    );
    expect(discounted).toBeCloseTo(-beta * potential(states[0], "team-1"));
  });

  it("parses default, zero and 0.02 beta and rejects negative beta", () => {
    expect(parseBattleAdvantageShapingBeta([])).toBe(0);
    expect(parseBattleAdvantageShapingBeta(["--battle-advantage-shaping-beta", "0"])).toBe(0);
    expect(parseBattleAdvantageShapingBeta(["--battle-advantage-shaping-beta", "0.02"])).toBe(0.02);
    expect(() => parseBattleAdvantageShapingBeta(["--battle-advantage-shaping-beta", "-0.01"]))
      .toThrow("non-negative");
  });
});
