import { describe, expect, it } from "vitest";
import { getTeamMovementCandidates } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { advanceVisualCpuOneStep, advanceVisualCpuTick, resolveBattleWithHiddenCpuIntents } from "../cpu/visualCpuRunner";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { positionKey } from "../utils/position";
import { submitTeamProduction } from "../engine/production";
import { getHeadlessProgressSignature } from "../cpu/headlessSimulation";

const allCpu = (): CpuTeamSettings => ({ "team-1": "random_cpu", "team-2": "random_cpu", "team-3": "random_cpu", "team-4": "random_cpu" });
const allHuman = (): CpuTeamSettings => ({ "team-1": "human", "team-2": "human", "team-3": "human", "team-4": "human" });

function runSteps(count: number, seed: number, settings = allCpu()) {
  let state = createInitialGameState();
  let runtime = createCpuRuntime(seed);
  for (let index = 0; index < count; index += 1) {
    const result = advanceVisualCpuOneStep(state, runtime, settings);
    state = result.state; runtime = result.runtime;
    if (!result.applied) break;
  }
  return { state, runtime };
}

describe("Phase 5-B Random CPU and Visual Runner", () => {
  it("randomly selects an available production choice, confirms it, then continues into movement", () => {
    let state = createInitialGameState();
    let runtime = createCpuRuntime(37);
    const initialUnitCount = state.units.length;
    for (let index = 0; index < 20 && !state.productionCompletedTeamIdsThisTurn.includes("team-1"); index += 1) {
      const result = advanceVisualCpuOneStep(state, runtime, { ...allHuman(), "team-1": "random_cpu" });
      state = result.state;
      runtime = result.runtime;
    }
    expect(state.productionCompletedTeamIdsThisTurn).toContain("team-1");
    expect(state.units.length).toBeGreaterThan(initialUnitCount);
    expect(state.currentMovementTeamId).toBe("team-1");
    expect(getRandomCpuDecision(state, runtime, { ...allHuman(), "team-1": "random_cpu" })?.kind).toMatch(/movement|teleport/);
  });

  it("treats production skip as progress and continues with the same team's movement", () => {
    let state = createInitialGameState();
    for (const slot of state.bases.find((base) => base.id === "home-1")!.slots) slot.unitId ??= `occupied-${slot.id}`;
    let runtime = createCpuRuntime(1020);
    const before = getHeadlessProgressSignature(state, runtime);
    const confirmed = advanceVisualCpuOneStep(state, runtime, { ...allHuman(), "team-1": "random_cpu" });
    state = confirmed.state; runtime = confirmed.runtime;
    expect(confirmed.runtime.logs.at(-1)?.action).toBe("confirm production / skip");
    expect(state.productionCompletedTeamIdsThisTurn).toContain("team-1");
    expect(getHeadlessProgressSignature(state, runtime)).not.toBe(before);
    const next = advanceVisualCpuOneStep(state, runtime, { ...allHuman(), "team-1": "random_cpu" });
    expect(next.runtime.logs.at(-1)?.action).toMatch(/move|movement pass/);
  });

  it("processes every production base before confirmation and then enters movement", () => {
    let state = createInitialGameState();
    const extra = state.bases.find((base) => base.id === "neutral-north")!;
    extra.ownerTeamId = "team-1";
    state.teams.find((team) => team.id === "team-1")!.controlledBaseIds.push(extra.id);
    let runtime = createCpuRuntime(2040);
    const initialUnits = state.units.length;
    for (let index = 0; index < 20 && !state.productionCompletedTeamIdsThisTurn.includes("team-1"); index += 1) {
      const step = advanceVisualCpuOneStep(state, runtime, { ...allHuman(), "team-1": "random_cpu" });
      state = step.state; runtime = step.runtime;
    }
    expect(runtime.processedKeys).toEqual(expect.arrayContaining(["movement-production:team-1:home-1", "movement-production:team-1:neutral-north"]));
    expect(state.productionCompletedTeamIdsThisTurn).toContain("team-1");
    expect(state.units.length).toBeGreaterThan(initialUnits);
    expect(getRandomCpuDecision(state, runtime, { ...allHuman(), "team-1": "random_cpu" })?.kind).toMatch(/movement|teleport/);
  });

  it("selects movement only from the Phase 5-A legal candidates", () => {
    const state = submitTeamProduction(createInitialGameState(), "team-1");
    const runtime = createCpuRuntime(7);
    const legal = getTeamMovementCandidates(state, "team-1");
    const decision = getRandomCpuDecision(state, runtime, allCpu());
    expect(decision?.kind).toBe("movement");
    if (decision?.kind === "movement" && decision.to) {
      expect(legal.find((entry) => entry.unitId === decision.unitId)?.destinations.map(positionKey)).toContain(positionKey(decision.to));
    }
  });

  it("does not operate a Human team or a CPU team that is not the current mover", () => {
    const state = createInitialGameState();
    expect(advanceVisualCpuOneStep(state, createCpuRuntime(1), allHuman()).applied).toBe(false);
    const onlyTeam2Cpu = { ...allHuman(), "team-2": "random_cpu" as const };
    expect(advanceVisualCpuOneStep(state, createCpuRuntime(1), onlyTeam2Cpu).applied).toBe(false);
  });

  it("confirms sequential CPU movement, advances teams, and reaches attack input after all teams", () => {
    let state = createInitialGameState();
    let runtime = createCpuRuntime(11);
    let sawTeam2 = false;
    for (let index = 0; index < 200 && state.phase === "movement_input"; index += 1) {
      const result = advanceVisualCpuOneStep(state, runtime, allCpu());
      state = result.state; runtime = result.runtime;
      if (state.currentMovementTeamId === "team-2") sawTeam2 = true;
    }
    expect(sawTeam2).toBe(true);
    expect(state.phase).toBe("attack_input");
  });

  it("keeps CPU attack intents hidden until Human confirmation, then resolves through simultaneous battle", () => {
    let state = createInitialGameState();
    state.phase = state.turnState.phase = "attack_input";
    const settings = { ...allHuman(), "team-2": "random_cpu" as const };
    let runtime = createCpuRuntime(19);
    for (let index = 0; index < 100 && !runtime.completedAttackTeamIds.includes("team-2"); index += 1) {
      const result = advanceVisualCpuOneStep(state, runtime, settings);
      state = result.state; runtime = result.runtime;
    }
    expect(runtime.completedAttackTeamIds).toContain("team-2");
    expect(runtime.hiddenAttackIntents.some((intent) => intent.teamId === "team-2")).toBe(true);
    expect(state.turnState.actionIntents.flatMap((entry) => entry.attackIntents).some((intent) => intent.teamId === "team-2")).toBe(false);
    for (const intent of runtime.hiddenAttackIntents.filter((entry) => entry.target))
      expect(runtime.logs.some((entry) => entry.detail?.includes(intent.target!.unitId))).toBe(false);
    const resolved = resolveBattleWithHiddenCpuIntents(state, runtime);
    expect(resolved.state.phase).not.toBe("attack_input");
    for (const intent of runtime.hiddenAttackIntents.filter((entry) => entry.target))
      expect(resolved.runtime.logs.some((entry) => entry.detail?.includes(intent.target!.unitId))).toBe(true);
  });

  it("uses pass and confirmation to avoid stopping when a team has no action candidates", () => {
    let state = createInitialGameState();
    state.turnNumber = state.turnState.turnNumber = 2;
    state.units = state.units.filter((unit) => unit.ownerTeamId !== "team-1");
    let runtime = createCpuRuntime(3);
    const result = advanceVisualCpuOneStep(state, runtime, { ...allHuman(), "team-1": "random_cpu" });
    state = result.state; runtime = result.runtime;
    expect(result.applied).toBe(true);
    expect(state.currentMovementTeamId).toBe("team-2");
    expect(runtime.logs.at(-1)?.action).toBe("confirm movement");
  });

  it("connects Random CPU to production, reward placement, and strategist actions", () => {
    const team1Cpu = { ...allHuman(), "team-1": "random_cpu" as const };

    const production = createInitialGameState();
    production.phase = production.turnState.phase = "production";
    const producedInput = advanceVisualCpuOneStep(production, createCpuRuntime(21), team1Cpu);
    expect(producedInput.state.turnState.actionIntents.find((entry) => entry.teamId === "team-1")?.productionChoices).toHaveLength(1);

    const reward = createInitialGameState();
    reward.phase = reward.turnState.phase = "reward_placement";
    reward.rewardPlacementRequests.push({ id: "cpu-runner-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    const placed = advanceVisualCpuOneStep(reward, createCpuRuntime(22), team1Cpu);
    expect(placed.state.rewardPlacementRequests.find((request) => request.id === "cpu-runner-reward")?.completed).toBe(true);

    const construction = createInitialGameState();
    construction.phase = construction.turnState.phase = "strategist_action_input";
    construction.units.find((unit) => unit.id === "home-1-strategist")!.role = "builder";
    const planned = advanceVisualCpuOneStep(construction, createCpuRuntime(23), team1Cpu);
    expect(planned.state.strategistActionIntents.some((intent) => intent.teamId === "team-1")).toBe(true);
  });

  it("confirms production and advances to sequential movement when every team is CPU", () => {
    let state = createInitialGameState();
    state.phase = state.turnState.phase = "production";
    let runtime = createCpuRuntime(31);
    for (let index = 0; index < 100 && state.phase === "production"; index += 1) {
      const result = advanceVisualCpuOneStep(state, runtime, allCpu());
      state = result.state; runtime = result.runtime;
    }
    expect(state.phase).toBe("movement_input");
    expect(state.currentMovementTeamId).toBe("team-1");
  });

  it("replays the same action sequence for the same seed and permits different choices for another seed", () => {
    const first = runSteps(25, 1234).runtime.logs.map((entry) => `${entry.action}:${entry.detail ?? ""}`);
    const replay = runSteps(25, 1234).runtime.logs.map((entry) => `${entry.action}:${entry.detail ?? ""}`);
    const other = runSteps(25, 9876).runtime.logs.map((entry) => `${entry.action}:${entry.detail ?? ""}`);
    expect(replay).toEqual(first);
    expect(other).not.toEqual(first);
  });

  it("does not advance while paused and applies exactly one action for a one-step request", () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(5);
    const paused = advanceVisualCpuTick(state, runtime, allCpu(), { running: true, paused: true });
    expect(paused.applied).toBe(false);
    expect(paused.state).toBe(state);
    expect(paused.runtime.logs).toHaveLength(0);
    const single = advanceVisualCpuOneStep(state, runtime, allCpu());
    expect(single.applied).toBe(true);
    expect(single.runtime.logs).toHaveLength(1);
    expect(single.runtime.appliedStepCount).toBe(1);
  });

  it("runs CPU versus CPU across multiple phases and turns", () => {
    const result = runSteps(500, 42);
    expect(result.state.turnNumber).toBeGreaterThan(1);
    expect(new Set(result.runtime.logs.map((entry) => entry.phase)).size).toBeGreaterThan(2);
    expect(result.runtime.stoppedReason).toBeUndefined();
  });

  it("stops and logs a reason when the safety limit is reached", () => {
    const state = createInitialGameState();
    const result = advanceVisualCpuOneStep(state, createCpuRuntime(1, 0), allCpu());
    expect(result.applied).toBe(false);
    expect(result.runtime.stoppedReason).toContain("safety limit");
    expect(result.runtime.logs.at(-1)?.error).toContain("safety limit");
  });
});
