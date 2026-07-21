import { describe, expect, it } from "vitest";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { checkHeadlessInvariants, createHeadlessInitialState, runHeadlessBatch, runHeadlessMatch } from "../cpu/headlessSimulation";

describe("Phase 5-C headless simulation", () => {
  it("uses the Phase 5-B Random CPU policy through the replaceable policy entry", () => {
    let calls = 0;
    const wrapped = (...args: Parameters<typeof getRandomCpuDecision>) => { calls += 1; return getRandomCpuDecision(...args); };
    const explicit = runHeadlessMatch({ participantCount: 4, seed: 9, maxTurns: 2, policy: wrapped });
    const defaulted = runHeadlessMatch({ participantCount: 4, seed: 9, maxTurns: 2 });
    expect(calls).toBeGreaterThan(0);
    expect(explicit.endReason).toBe(defaulted.endReason);
    expect(explicit.recentActions?.map((entry) => [entry.action, entry.detail])).toEqual(defaulted.recentActions?.map((entry) => [entry.action, entry.detail]));
  });

  it("runs multiple turns without rendering or timers and is reproducible by seed", () => {
    const first = runHeadlessMatch({ participantCount: 4, seed: 77, maxTurns: 3, historyLimit: 500 });
    const second = runHeadlessMatch({ participantCount: 4, seed: 77, maxTurns: 3, historyLimit: 500 });
    expect(first.endTurn).toBeGreaterThan(1);
    expect(first.endReason).toBe(second.endReason);
    expect(first.winnerTeamId).toBe(second.winnerTeamId);
    expect(first.recentActions).toEqual(second.recentActions);
  });

  it("returns victory when only one participant remains active", () => {
    const state = createHeadlessInitialState(4);
    state.teams.filter((team) => !team.isNeutral && team.id !== "team-2").forEach((team) => { team.status = "defeated"; });
    const result = runHeadlessMatch({ participantCount: 4, seed: 1, maxTurns: 10, initialState: state });
    expect(result).toMatchObject({ endReason: "victory", winnerTeamId: "team-2" });
  });

  it("reports the turn limit independently", () => {
    expect(runHeadlessMatch({ participantCount: 4, seed: 2, maxTurns: 1 }).endReason).toBe("turn_limit");
  });

  it("detects invariant violations with seed and recent history", () => {
    const state = createHeadlessInitialState(4);
    state.units.push(structuredClone(state.units.find((unit) => unit.id === "home-1-king")!));
    const result = runHeadlessMatch({ participantCount: 4, seed: 123, maxTurns: 3, initialState: state });
    expect(result.endReason).toBe("invariant_violation");
    expect(result.seed).toBe(123);
    expect(result.violations.join(" ")).toContain("duplicate living unit id");
    expect(checkHeadlessInvariants(state).length).toBeGreaterThan(0);
  });

  it("stops when a policy cannot advance and when the action safety limit is reached", () => {
    const stalled = runHeadlessMatch({ participantCount: 4, seed: 4, maxTurns: 3, policy: () => undefined });
    expect(stalled.endReason).toBe("phase_stall");
    expect(stalled.error).toContain("no action");
    const limited = runHeadlessMatch({ participantCount: 4, seed: 4, maxTurns: 3, maxActions: 1 });
    expect(limited.endReason).toBe("action_limit");
  });

  it("captures policy exceptions as reproducible match results", () => {
    const result = runHeadlessMatch({ participantCount: 4, seed: 404, maxTurns: 3, policy: () => { throw new Error("policy exploded"); } });
    expect(result).toMatchObject({ endReason: "exception", seed: 404, endTurn: 1, phase: "movement_input" });
    expect(result.error).toContain("policy exploded");
  });

  it.each([3, 4] as const)("runs multiple %i-player matches and aggregates outcomes", (participantCount) => {
    const batch = runHeadlessBatch({ participantCount, matchCount: 2, seedStart: 30, maxTurns: 2 });
    expect(batch.matches.map((match) => match.seed)).toEqual([30, 31]);
    expect(batch.matches.every((match) => match.participantCount === participantCount)).toBe(true);
    expect(batch.settledCount + batch.turnLimitCount + batch.exceptionCount + batch.invariantViolationCount + batch.phaseStallCount + batch.actionLimitCount).toBe(2);
  });

  it("keeps the CPU action sequence and match outcome identical across execution modes", () => {
    const run = (mode: "debug" | "sweep" | "training") => runHeadlessMatch({ participantCount: 4, seed: 919, maxTurns: 1, mode });
    const results = [run("debug"), run("sweep"), run("training")];
    expect(results.map(({ endReason, winnerTeamId, endTurn, actionCount, actionSequenceHash }) => ({ endReason, winnerTeamId, endTurn, actionCount, actionSequenceHash })))
      .toEqual(Array(3).fill(expect.objectContaining({
        endReason: results[0].endReason,
        winnerTeamId: results[0].winnerTeamId,
        endTurn: results[0].endTurn,
        actionCount: results[0].actionCount,
        actionSequenceHash: results[0].actionSequenceHash,
      })));
  });

  it("checks every action in debug, boundaries in sweep, and only the end in training", () => {
    const debug = runHeadlessMatch({ participantCount: 4, seed: 12, maxTurns: 1, mode: "debug" });
    const sweep = runHeadlessMatch({ participantCount: 4, seed: 12, maxTurns: 1, mode: "sweep" });
    const training = runHeadlessMatch({ participantCount: 4, seed: 12, maxTurns: 1, mode: "training" });
    expect(debug.invariantCheckCount).toBe(debug.actionCount + 1);
    expect(sweep.invariantCheckCount).toBeGreaterThan(0);
    expect(sweep.invariantCheckCount).toBeLessThan(debug.invariantCheckCount);
    expect(training.invariantCheckCount).toBe(0);
    expect(training.recentActions).toBeUndefined();
    expect(sweep.recentActions).toBeUndefined();
  });

  it("retains only the latest 50 actions for a sweep failure", () => {
    const result = runHeadlessMatch({ participantCount: 4, seed: 88, maxTurns: 10, maxActions: 60, mode: "sweep" });
    expect(result.endReason).toBe("action_limit");
    expect(result.recentActions?.length).toBe(50);
  });

  it("returns an end-only profile with counts and timing summaries", () => {
    const result = runHeadlessMatch({ participantCount: 4, seed: 45, maxTurns: 1, mode: "training", profile: true });
    expect(result.profile?.total.calls).toBe(1);
    expect(result.profile?.total.totalMs).toBeGreaterThan(0);
    expect(result.profile?.total.percentage).toBe(100);
    expect(result.profile?.legalEnumeration.calls).toBeGreaterThan(0);
    expect(result.profile?.policySelection.calls).toBeGreaterThan(0);
    expect(result.profile?.actionApplication.calls).toBe(result.actionCount);
    expect(result.profile?.stallDetection.calls).toBe(result.actionCount);
    expect(result.profile?.invariantChecks.calls).toBe(result.invariantCheckCount);
    expect(result.profile?.actionLogging.calls).toBeGreaterThan(0);
    expect(result.profile?.otherRunner.calls).toBe(1);
    for (const entry of Object.values(result.profile!)) {
      expect(entry).toEqual(expect.objectContaining({ calls: expect.any(Number), totalMs: expect.any(Number), averageMs: expect.any(Number), maxMs: expect.any(Number), percentage: expect.any(Number) }));
    }
  });
});
