import { existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { runBehavioralCloning } from "../cpu/rlBehavioralCloning";
import type { BehavioralCloningProgress } from "../cpu/rlBehavioralCloning";
import { runParallelRlImitationCollection } from "../cpu/rlImitationParallel";

const directory = mkdtempSync(join(tmpdir(), "tactical-hub-bc-smoke-"));
afterAll(() => rmSync(directory, { recursive: true, force: true }));

describe("RL-4B Behavioral Cloning smoke", () => {
  it("keeps sample coverage and training valid with workers=1 and workers=4", async () => {
    const dataPath = join(directory, "smoke-replay.jsonl");
    const serialCheckpoint = join(directory, "serial-best.pt");
    const parallelCheckpoint = join(directory, "parallel-best.pt");
    const resumeBestCheckpoint = join(directory, "resume-best.pt");
    const latestCheckpoint = join(directory, "resume-latest.pt");
    const continuousCheckpoint = join(directory, "continuous-best.pt");
    const collection = await runParallelRlImitationCollection({
      seedStart: 701,
      episodeCount: 6,
      maxTurns: 1,
      outputPath: dataPath,
      workerCount: 4,
    });
    expect(collection.failedEpisodeCount).toBe(0);

    const common = {
      dataPath,
      epochs: 1,
      batchSize: 4,
      learningRate: 1e-3,
      trainRange: { from: 1, to: 4 },
      validationRange: { from: 5, to: 5 },
      testRange: { from: 6, to: 6 },
      seed: 19,
      runTest: true,
    };
    await expect(runBehavioralCloning({
      ...common,
      checkpointPath: join(directory, "invalid.pt"),
      torchThreads: 0,
    })).rejects.toThrow(/torchThreads must be a positive integer/);
    await expect(runBehavioralCloning({
      ...common,
      checkpointPath: join(directory, "invalid-interop.pt"),
      torchInteropThreads: 1.5,
    })).rejects.toThrow(/torchInteropThreads must be a positive integer/);
    const progress: BehavioralCloningProgress[] = [];
    const serial = await runBehavioralCloning({
      ...common,
      checkpointPath: serialCheckpoint,
      workerCount: 1,
      device: "auto" as const,
      onProgress: (event) => progress.push(event),
    });
    const parallel = await runBehavioralCloning({
      ...common,
      checkpointPath: parallelCheckpoint,
      workerCount: 4,
      torchThreads: 2,
      torchInteropThreads: 1,
      device: "cpu",
    });

    expect(parallel.epochs[0].train.sampleCount).toBe(serial.epochs[0].train.sampleCount);
    expect(parallel.epochs[0].validation.sampleCount).toBe(serial.epochs[0].validation.sampleCount);
    expect(parallel.test!.sampleCount).toBe(serial.test!.sampleCount);
    expect(parallel.torchThreads).toBe(2);
    expect(parallel.torchInteropThreads).toBe(1);
    expect(parallel.selectedDevice).toBe("cpu");
    expect(["cpu", "cuda"]).toContain(serial.selectedDevice);
    expect(serial.replayCache.generatedSidecarCount).toBe(6);
    expect(serial.replayCache.reusedSidecarCount).toBe(0);
    expect(parallel.replayCache.generatedSidecarCount).toBe(0);
    expect(parallel.replayCache.reusedSidecarCount).toBe(6);
    expect(serial.replayCache.sidecarPreparationMs).toBeGreaterThanOrEqual(0);
    expect(serial.replayCache.directReplayMs).toBeGreaterThanOrEqual(0);
    expect(progress.filter((event) => event.kind === "episode" && event.phase === "train")).toHaveLength(4);
    expect(progress.filter((event) => event.kind === "episode" && event.phase === "validation")).toHaveLength(1);
    expect(progress.filter((event) => event.kind === "episode" && event.phase === "test")).toHaveLength(1);
    expect(progress.every((event) => event.processedSamples >= 0 && event.processedBatches >= 0)).toBe(true);
    for (const result of [serial, parallel]) {
      expect(result.epochs).toHaveLength(1);
      expect(result.epochs[0].train.sampleCount).toBeGreaterThan(0);
      expect(Number.isFinite(result.epochs[0].train.loss)).toBe(true);
      expect(Number.isFinite(result.epochs[0].validation.loss)).toBe(true);
      expect(Number.isFinite(result.test!.loss)).toBe(true);
      expect(result.initialParameterHash).not.toBe(result.trainedParameterHash);
      expect(result.reloadedParameterHash).toBe(result.trainedParameterHash);
      expect(existsSync(result.checkpointPath)).toBe(true);
      expect(statSync(result.checkpointPath).size).toBeGreaterThan(0);
    }

    const firstStageStatus: string[] = [];
    const firstStageEvents: string[] = [];
    const firstStage = await runBehavioralCloning({
      ...common,
      checkpointPath: resumeBestCheckpoint,
      latestCheckpointPath: latestCheckpoint,
      workerCount: 1,
      device: "cpu",
      onProgress: (progress) => {
        if (progress.kind === "episode" && progress.phase === "validation") firstStageEvents.push("validation-completed");
      },
      onStatus: (message) => {
        firstStageStatus.push(message);
        firstStageEvents.push(message);
      },
    });
    expect(firstStage.completedEpoch).toBe(1);
    expect(existsSync(latestCheckpoint)).toBe(true);
    expect(firstStageStatus.some((message) => message.includes("latest checkpoint saved epoch=2 phase=train nextEpisode=1"))).toBe(true);
    expect(firstStageEvents.indexOf("validation-completed")).toBeLessThan(
      firstStageEvents.findIndex((event) => event.includes("latest checkpoint saved epoch=2 phase=train nextEpisode=1")),
    );

    const resumeStatus: string[] = [];
    const resumed = await runBehavioralCloning({
      ...common,
      epochs: 2,
      checkpointPath: resumeBestCheckpoint,
      latestCheckpointPath: latestCheckpoint,
      resumePath: latestCheckpoint,
      workerCount: 1,
      device: "cpu",
      onStatus: (message) => resumeStatus.push(message),
    });
    expect(resumed.epochs.map((epoch) => epoch.epoch)).toEqual([2]);
    expect(resumed.completedEpoch).toBe(2);
    expect(resumed.resumedFrom).toBe(latestCheckpoint);
    expect(resumeStatus).toEqual(expect.arrayContaining([
      `[BC] resumed checkpoint=${latestCheckpoint}`,
      "[BC] completedEpoch=1",
      "[BC] nextEpoch=2",
    ]));
    expect(resumeStatus.some((message) => message.includes("latest checkpoint saved epoch=3 phase=train nextEpisode=1"))).toBe(true);

    const continuous = await runBehavioralCloning({
      ...common,
      epochs: 2,
      checkpointPath: continuousCheckpoint,
      workerCount: 1,
      device: "cpu",
    });
    expect(continuous.epochs.map((epoch) => epoch.epoch)).toEqual([1, 2]);
    expect(continuous.trainedParameterHash).toBe(resumed.trainedParameterHash);

    await expect(runBehavioralCloning({
      ...common,
      checkpointPath: join(directory, "missing-best.pt"),
      resumePath: join(directory, "does-not-exist.pt"),
    })).rejects.toThrow(/Resume checkpoint does not exist/);
  }, 180_000);
});
