import { describe, expect, it } from "vitest";
import { filterPpoLegalActionsForEvaluation } from "../cpu/rlPpoEvaluationActionFilter";

describe("PPO evaluation legacy-only production filter", () => {
  const actions = [
    { actionType: "production", unitType: "infantry", actionKey: "p-i" },
    { actionType: "production", unitType: "cavalry", actionKey: "p-c" },
    { actionType: "production", unitType: "archer", actionKey: "p-a" },
    { actionType: "production", unitType: "engineer", actionKey: "p-e" },
    { actionType: "production", unitType: "ninja", actionKey: "p-n" },
    { actionType: "production", unitType: "strategist", actionKey: "p-s" },
    { actionType: "production", actionKey: "p-pass" },
    { actionType: "reward", unitType: "infantry", actionKey: "r-i" },
    { actionType: "reward", unitType: "engineer", actionKey: "r-e" },
    { actionType: "reward", unitType: "ninja", actionKey: "r-n" },
    { actionType: "reward", unitType: "strategist", actionKey: "r-s" },
    { actionType: "movement", unitType: "engineer", actionKey: "m-e" },
  ];

  it("removes special-unit production and reward choices but preserves legacy units and non-production actions", () => {
    expect(filterPpoLegalActionsForEvaluation(actions, true).map((action) => action.actionKey)).toEqual([
      "p-i",
      "p-c",
      "p-a",
      "p-pass",
      "r-i",
      "m-e",
    ]);
  });

  it("returns all actions unchanged when legacy-only mode is disabled", () => {
    expect(filterPpoLegalActionsForEvaluation(actions, false)).toBe(actions);
  });
});
