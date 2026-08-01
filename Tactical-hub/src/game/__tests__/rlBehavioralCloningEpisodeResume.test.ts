import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { runBehavioralCloning } from "../cpu/rlBehavioralCloning";
import { runParallelRlImitationCollection } from "../cpu/rlImitationParallel";

const directory = mkdtempSync(join(tmpdir(), "tactical-hub-bc-episode-resume-"));
const dataPath = join(directory, "replay.jsonl");
const common = {
  dataPath, epochs: 1, batchSize: 32, learningRate: 1e-3, seed: 27, workerCount: 1, device: "cpu" as const,
  trainRange: { from: 1, to: 3 }, validationRange: { from: 4, to: 5 }, testRange: { from: 6, to: 6 },
};
beforeAll(async () => {
  const collection = await runParallelRlImitationCollection({ seedStart: 910, episodeCount: 6, maxTurns: 1, outputPath: dataPath, workerCount: 1 });
  expect(collection.failedEpisodeCount).toBe(0);
  expect(collection.decisionCount).toBeLessThanOrEqual(256);
});
afterAll(() => rmSync(directory, { recursive: true, force: true }));

describe("BC episode-boundary resume", () => {
  it("train_resume_episode_2", async () => {
    let continuousFinal: { model: string; optimizer: string } | undefined;
    let continuousAccumulator: unknown;
    await expect(runBehavioralCloning({
      ...common, checkpointPath: join(directory, "train-continuous-best.pt"), latestCheckpointPath: join(directory, "train-continuous-latest.pt"),
      onEpisodeCheckpointSaved: (state, _path, episodeNumber, hashes) => {
        if (episodeNumber === 3) {
          continuousFinal = hashes; continuousAccumulator = state.trainAccumulator;
          throw new Error("stop-continuous-after-train-3");
        }
      },
    })).rejects.toThrow("stop-continuous-after-train-3");
    const latest = join(directory, "train-latest.pt");
    const before: number[] = [];
    const startedBefore: Array<[number, string]> = [];
    let episodeTwoState: unknown;
    await expect(runBehavioralCloning({
      ...common, checkpointPath: join(directory, "train-resume-best.pt"), latestCheckpointPath: latest,
      onReplayEpisodeStarted: (episode, execution) => startedBefore.push([episode, execution]),
      onEpisodeCheckpointSaved: (state, _path, episodeNumber) => {
        if (episodeNumber <= 3) before.push(episodeNumber);
        if (state.phase === "train" && state.nextEpisodeNumber === 3) {
          episodeTwoState = state;
          throw new Error("stop-after-train-2");
        }
      },
    })).rejects.toThrow("stop-after-train-2");
    const after: number[] = [];
    const startedAfter: Array<[number, string]> = [];
    let resumedFinal: { model: string; optimizer: string } | undefined;
    let resumedAccumulator: unknown;
    await expect(runBehavioralCloning({
      ...common, checkpointPath: join(directory, "train-resume-best.pt"), latestCheckpointPath: latest, resumePath: latest,
      onReplayEpisodeStarted: (episode, execution) => startedAfter.push([episode, execution]),
      onEpisodeCheckpointSaved: (state, _path, episodeNumber, hashes) => {
        if (episodeNumber <= 3) after.push(episodeNumber);
        if (episodeNumber === 3) {
          resumedFinal = hashes; resumedAccumulator = state.trainAccumulator;
          throw new Error("stop-resumed-after-train-3");
        }
      },
    })).rejects.toThrow("stop-resumed-after-train-3");
    expect(episodeTwoState).toMatchObject({ currentEpoch: 1, phase: "train", nextEpisodeNumber: 3, completedTrainEpisodes: [1, 2] });
    expect([...before, ...after]).toEqual([1, 2, 3]);
    expect(startedBefore).toEqual([[1, "direct"], [2, "direct"]]);
    expect(startedAfter).toEqual([[3, "direct"]]);
    expect(resumedFinal).toEqual(continuousFinal);
    expect(resumedAccumulator).toEqual(continuousAccumulator);
  }, 30_000);

  it("validation_resume_mid_split", async () => {
    const validationCommon = {
      ...common,
      trainRange: { from: 1, to: 1 },
      validationRange: { from: 2, to: 4 },
      testRange: { from: 5, to: 5 },
    };
    const continuous = await runBehavioralCloning({ ...validationCommon, checkpointPath: join(directory, "validation-continuous-best.pt") });
    const latest = join(directory, "validation-latest.pt");
    const before: number[] = [];
    const startedBefore: Array<[number, string]> = [];
    let interruptedState: unknown;
    await expect(runBehavioralCloning({
      ...validationCommon, checkpointPath: join(directory, "validation-resume-best.pt"), latestCheckpointPath: latest,
      onReplayEpisodeStarted: (episode, execution) => startedBefore.push([episode, execution]),
      onEpisodeCheckpointSaved: (state, _path, episodeNumber) => {
        if (episodeNumber >= 2) before.push(episodeNumber);
        if (state.phase === "validation" && state.nextEpisodeNumber === 4) {
          interruptedState = state;
          throw new Error("stop-after-validation-4");
        }
      },
    })).rejects.toThrow("stop-after-validation-4");
    const after: number[] = [];
    const startedAfter: Array<[number, string]> = [];
    const resumed = await runBehavioralCloning({
      ...validationCommon, checkpointPath: join(directory, "validation-resume-best.pt"), latestCheckpointPath: latest, resumePath: latest,
      onReplayEpisodeStarted: (episode, execution) => startedAfter.push([episode, execution]),
      onEpisodeCheckpointSaved: (_state, _path, episodeNumber) => { if (episodeNumber >= 2) after.push(episodeNumber); },
    });
    expect(interruptedState).toMatchObject({ currentEpoch: 1, phase: "validation", nextEpisodeNumber: 4, completedTrainEpisodes: [1], completedValidationEpisodes: [2, 3] });
    expect([...before, ...after]).toEqual([2, 3, 4]);
    expect(startedBefore).toEqual([[1, "direct"], [2, "direct"], [3, "direct"]]);
    expect(startedAfter).toEqual([[4, "direct"]]);
    expect(resumed.trainedParameterHash).toBe(continuous.trainedParameterHash);
    expect(resumed.optimizerStateHash).toBe(continuous.optimizerStateHash);
    expect(resumed.epochs[0].train).toMatchObject({ loss: continuous.epochs[0].train.loss, accuracy: continuous.epochs[0].train.accuracy, sampleCount: continuous.epochs[0].train.sampleCount });
    expect(resumed.epochs[0].validation).toMatchObject({ loss: continuous.epochs[0].validation.loss, accuracy: continuous.epochs[0].validation.accuracy, sampleCount: continuous.epochs[0].validation.sampleCount });
    expect(resumed.bestEpoch).toBe(continuous.bestEpoch);
    expect(resumed.bestValidationAccuracy).toBe(continuous.bestValidationAccuracy);
  }, 30_000);

  it("does not modify latest checkpoint while optional test runs", async () => {
    const withoutTestLatest = join(directory, "without-test-latest.pt");
    const withoutTestEpisodes: Array<[number, string]> = [];
    const withoutTest = await runBehavioralCloning({
      ...common,
      checkpointPath: join(directory, "without-test-best.pt"),
      latestCheckpointPath: withoutTestLatest,
      onReplayEpisodeStarted: (episode, execution) => withoutTestEpisodes.push([episode, execution]),
    });
    expect(withoutTest.test).toBeUndefined();
    expect(withoutTestEpisodes).toEqual([[1, "direct"], [2, "direct"], [3, "direct"], [4, "direct"], [5, "direct"]]);
    expect(statSync(withoutTestLatest).size).toBeGreaterThan(0);

    const latest = join(directory, "test-latest.pt");
    let beforeTest: { hash: string; size: number; mtimeMs: number } | undefined;
    const withTestEpisodes: Array<[number, string]> = [];
    const result = await runBehavioralCloning({
      ...common, checkpointPath: join(directory, "test-best.pt"), latestCheckpointPath: latest, runTest: true,
      onReplayEpisodeStarted: (episode, execution) => withTestEpisodes.push([episode, execution]),
      onStatus: (message) => {
        if (message.includes("epoch=2 phase=train nextEpisode=1")) {
          const info = statSync(latest);
          beforeTest = { hash: createHash("sha256").update(readFileSync(latest)).digest("hex"), size: info.size, mtimeMs: info.mtimeMs };
        }
      },
    });
    const after = statSync(latest);
    expect(result.test?.sampleCount).toBeGreaterThan(0);
    const afterTest = { hash: createHash("sha256").update(readFileSync(latest)).digest("hex"), size: after.size, mtimeMs: after.mtimeMs };
    expect(beforeTest).toEqual(afterTest);
    expect(withTestEpisodes).toEqual([[1, "direct"], [2, "direct"], [3, "direct"], [4, "direct"], [5, "direct"], [6, "direct"]]);
    console.info(JSON.stringify({ withoutTestEpisodes, withTestEpisodes, latestBefore: beforeTest, latestAfter: afterTest, optimizerStateHash: result.optimizerStateHash }));
  }, 30_000);
});
