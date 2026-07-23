import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { runHeadlessBatch } from "../cpu/headlessSimulation";
import { runParallelHeadlessBatch } from "../cpu/headlessParallel";

const deterministicResult = (match: Awaited<ReturnType<typeof runParallelHeadlessBatch>>["matches"][number]) => ({
  seed: match.seed,
  endReason: match.endReason,
  winnerTeamId: match.winnerTeamId,
  endTurn: match.endTurn,
  phase: match.phase,
  actionCount: match.actionCount,
  actionSequenceHash: match.actionSequenceHash,
  violations: match.violations,
  invariantCheckCount: match.invariantCheckCount,
});

describe("parallel headless batch", () => {
  it("keeps the existing serial Random Policy results when workers=1", async () => {
    const options = { participantCount: 4 as const, matchCount: 2, seedStart: 2001, maxTurns: 1, mode: "training" as const };
    const existing = runHeadlessBatch(options);
    const parallelEntry = await runParallelHeadlessBatch({ ...options, policyName: "random", workerCount: 1 });
    expect(parallelEntry.matches.map(deterministicResult)).toEqual(existing.matches.map(deterministicResult));
    expect(parallelEntry).toMatchObject({ requestedWorkerCount: 1, effectiveWorkerCount: 1, totalActionCount: 50 });
  });

  it.each(["random", "heuristic"] as const)("is deterministic between one and four requested workers for %s policy", async (policyName) => {
    const options = { participantCount: 4 as const, matchCount: 4, seedStart: 2101, maxTurns: 1, mode: "training" as const, policyName };
    const serial = await runParallelHeadlessBatch({ ...options, workerCount: 1 });
    const concurrent = await runParallelHeadlessBatch({ ...options, workerCount: 4 });
    expect(concurrent.requestedWorkerCount).toBe(4);
    expect(concurrent.effectiveWorkerCount).toBe(4);
    expect(concurrent.matches.map(deterministicResult)).toEqual(serial.matches.map(deterministicResult));
    expect(concurrent.matches.map((match) => match.seed)).toEqual([2101, 2102, 2103, 2104]);
  }, 30_000);

  it("handles fewer matches than workers and a non-divisible dynamic queue", async () => {
    const single = await runParallelHeadlessBatch({ participantCount: 4, matchCount: 1, seedStart: 2201, maxTurns: 1, mode: "training", policyName: "random", workerCount: 4 });
    expect(single).toMatchObject({ requestedWorkerCount: 4, effectiveWorkerCount: 1 });
    const uneven = await runParallelHeadlessBatch({ participantCount: 4, matchCount: 3, seedStart: 2202, maxTurns: 1, mode: "training", policyName: "random", workerCount: 2 });
    expect(uneven.matches.map((match) => match.seed)).toEqual([2202, 2203, 2204]);
    expect(uneven.workerMemory.reduce((sum, worker) => sum + worker.completedMatchCount, 0)).toBe(3);
    expect(uneven.matchesPerSecond).toBeGreaterThan(0);
    expect(uneven.actionsPerSecond).toBeGreaterThan(0);
  }, 30_000);

  it("collects an isolated profile from every worker and keeps seed ordering", async () => {
    const result = await runParallelHeadlessBatch({ participantCount: 4, matchCount: 2, seedStart: 2251, maxTurns: 1, mode: "training", profile: true, policyName: "heuristic", workerCount: 2 });
    expect(result.matches.map((match) => match.seed)).toEqual([2251, 2252]);
    expect(result.matches.every((match) => match.profiling?.enabled && match.profiling.turnBuckets.reduce((sum, bucket) => sum + bucket.actionCount, 0) === match.actionCount)).toBe(true);
  }, 30_000);

  it("keeps trace state isolated between workers and matches", async () => {
    const result = await runParallelHeadlessBatch({ participantCount: 4, matchCount: 2, seedStart: 2261, maxTurns: 1, mode: "training", trace: true, policyName: "heuristic", workerCount: 2 });
    expect(result.matches.map((match) => match.trace?.[0].seed)).toEqual([2261, 2262]);
    expect(result.matches[0].trace).not.toBe(result.matches[1].trace);
    expect(result.matches.every((match) => match.trace?.length === match.actionCount)).toBe(true);
  }, 30_000);

  it("rejects invalid worker counts", async () => {
    await expect(runParallelHeadlessBatch({ participantCount: 4, matchCount: 1, seedStart: 1, maxTurns: 1, policyName: "random", workerCount: 0 })).rejects.toThrow("positive integer");
    await expect(runParallelHeadlessBatch({ participantCount: 4, matchCount: 1, seedStart: 1, maxTurns: 1, policyName: "random", workerCount: 1.5 })).rejects.toThrow("positive integer");
  });

  it("reports the assigned task and seed when a worker exits abnormally", async () => {
    const missingWorker = fileURLToPath(new URL("./fixtures/missing-headless-worker.ts", import.meta.url));
    await expect(runParallelHeadlessBatch({ participantCount: 4, matchCount: 2, seedStart: 2301, maxTurns: 1, mode: "training", policyName: "random", workerCount: 2, workerEntryPath: missingWorker }))
      .rejects.toThrow(/task match-[01] seed 230[12]/);
  }, 30_000);
});
