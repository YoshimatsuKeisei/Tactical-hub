import { describe, expect, it } from "vitest";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
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
    const result = runHeadlessMatch({ participantCount: 4, seed: 45, maxTurns: 2, mode: "training", profile: true });
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
    expect(result.legalEnumerationBreakdown?.byCategory.movement).toEqual(expect.objectContaining({ calls: expect.any(Number), totalMs: expect.any(Number), averageMs: expect.any(Number), maxMs: expect.any(Number) }));
    expect(result.legalEnumerationBreakdown?.byCategory.movementRangePathSearch.calls).toBeGreaterThan(0);
    expect(result.legalEnumerationBreakdown?.byCategory.attackTargetSearch.calls).toBeGreaterThan(0);
    for (const category of ["attackLivingEnemyList", "attackUnitCoordinateBaseSearch", "attackStaticRangePrefilter", "attackRangeDistance", "attackRoadSectionConnection", "attackAcrossBaseBlocking", "attackBaseBlocking", "attackBridgeConnection", "attackLakeNinjaRule", "attackBasicFilter", "attackCandidateIdGeneration", "attackPostProcessing", "attackFinalLegalCheck"]) {
      expect(result.legalEnumerationBreakdown?.byCategory[category]).toEqual(expect.objectContaining({ calls: expect.any(Number), totalMs: expect.any(Number), averageMs: expect.any(Number), maxMs: expect.any(Number) }));
    }
    expect(Object.keys(result.legalEnumerationBreakdown?.byPhase ?? {})).toContain("movement_input");
    for (const entry of Object.values(result.profile!)) {
      expect(entry).toEqual(expect.objectContaining({ calls: expect.any(Number), totalMs: expect.any(Number), averageMs: expect.any(Number), maxMs: expect.any(Number), percentage: expect.any(Number) }));
    }
    expect(result.profiling).toEqual(expect.objectContaining({ enabled: true, matchElapsedMs: expect.any(Number), overlapWarning: expect.stringContaining("overlap") }));
    expect(result.profiling?.turnBuckets.reduce((sum, bucket) => sum + bucket.actionCount, 0)).toBe(result.actionCount);
    expect(result.profiling?.sections.map((entry) => entry.totalMs)).toEqual([...result.profiling!.sections.map((entry) => entry.totalMs)].sort((left, right) => right - left));
    for (const section of result.profiling?.sections ?? []) {
      expect(Number.isFinite(section.totalMs)).toBe(true);
      expect(Number.isFinite(section.averageMs)).toBe(true);
      expect(section.totalMs).toBeGreaterThanOrEqual(0);
      expect(section.callCount).toBeGreaterThanOrEqual(0);
    }
  });

  it("keeps Random and Heuristic outcomes unchanged when profiling is enabled", () => {
    const randomPlain = runHeadlessMatch({ participantCount: 4, seed: 451, maxTurns: 1, mode: "training" });
    const randomProfiled = runHeadlessMatch({ participantCount: 4, seed: 451, maxTurns: 1, mode: "training", profile: true });
    expect({ hash: randomProfiled.actionSequenceHash, count: randomProfiled.actionCount, reason: randomProfiled.endReason }).toEqual({ hash: randomPlain.actionSequenceHash, count: randomPlain.actionCount, reason: randomPlain.endReason });
    const heuristicPlain = runHeadlessMatch({ participantCount: 4, seed: 452, maxTurns: 1, mode: "training", policy: createHeuristicCpuPolicy() });
    const heuristicProfiled = runHeadlessMatch({ participantCount: 4, seed: 452, maxTurns: 1, mode: "training", policy: createHeuristicCpuPolicy(), profile: true });
    expect({ hash: heuristicProfiled.actionSequenceHash, count: heuristicProfiled.actionCount, reason: heuristicProfiled.endReason }).toEqual({ hash: heuristicPlain.actionSequenceHash, count: heuristicPlain.actionCount, reason: heuristicPlain.endReason });
    expect(heuristicProfiled.profiling?.sections.some((entry) => entry.name === "legal.heuristicCandidateEvaluation")).toBe(true);
    expect(heuristicProfiled.profiling?.heuristicDistanceCache).toEqual(expect.objectContaining({ searchCount: expect.any(Number), hitCount: expect.any(Number), missCount: expect.any(Number), hitRate: expect.any(Number) }));
    expect(heuristicProfiled.profiling!.heuristicDistanceCache!.searchCount).toBeGreaterThan(0);
    expect(heuristicProfiled.profiling!.heuristicDistanceCache!.hitCount).toBeGreaterThan(0);
    expect(randomPlain.profiling).toBeUndefined();
  });

  it("does not falsely stall after team-4 confirms production for seed 1020", () => {
    const debug = runHeadlessMatch({ participantCount: 4, seed: 1020, maxTurns: 31, maxActions: 30_000, mode: "debug", historyLimit: 500 });
    const training = runHeadlessMatch({ participantCount: 4, seed: 1020, maxTurns: 31, maxActions: 30_000, mode: "training" });
    const replay = runHeadlessMatch({ participantCount: 4, seed: 1020, maxTurns: 31, maxActions: 30_000, mode: "training" });
    expect(debug.endReason).not.toBe("phase_stall");
    expect(training.endReason).not.toBe("phase_stall");
    expect(replay.endReason).toBe(training.endReason);
    expect(replay.actionSequenceHash).toBe(training.actionSequenceHash);
    const actions = debug.recentActions ?? [];
    const confirmation = actions.findIndex((entry) => entry.turnNumber === 31 && entry.teamId === "team-4" && entry.action === "confirm production / skip");
    expect(confirmation).toBeGreaterThanOrEqual(0);
    expect(actions.slice(confirmation + 1).some((entry) => entry.teamId === "team-4" && /move|movement pass/.test(entry.action))).toBe(true);
  });
});
