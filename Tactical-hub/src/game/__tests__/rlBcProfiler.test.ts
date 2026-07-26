import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { runRlBcShortProfile } from "../cpu/rlBcProfiler";
import { runParallelRlImitationCollection } from "../cpu/rlImitationParallel";

const directory = mkdtempSync(join(tmpdir(), "tactical-hub-bc-profile-"));
afterAll(() => rmSync(directory, { recursive: true, force: true }));

describe("RL BC short profiler", () => {
  it("uses the same episode workload with workers=1, workers=2 and workers=4", async () => {
    const dataPath = join(directory, "replay.jsonl");
    const collection = await runParallelRlImitationCollection({
      seedStart: 7701,
      episodeCount: 4,
      maxTurns: 1,
      outputPath: dataPath,
      workerCount: 1,
    });
    expect(collection.failedEpisodeCount).toBe(0);
    let expectedWorkload: Pick<Awaited<ReturnType<typeof runRlBcShortProfile>>, "episodeNumbers" | "samplesPerEpisode"> | undefined;
    for (const workerCount of [1, 2, 4]) {
      const result = await runRlBcShortProfile({
        dataPath,
        samples: 16,
        warmupSamples: 8,
        batchSize: 4,
        workerCount,
        profileEpisodeCount: 4,
        device: "cpu",
        torchThreads: 2,
        torchInteropThreads: 1,
      });
      expect(result.measuredSamples).toBe(16);
      expect(result.requestedWorkers).toBe(workerCount);
      expect(result.effectiveWorkers).toBe(workerCount);
      expect(result.profileEpisodeCount).toBe(4);
      if (!expectedWorkload) {
        expectedWorkload = {
          episodeNumbers: result.episodeNumbers,
          samplesPerEpisode: result.samplesPerEpisode,
        };
      } else {
        expect({
          episodeNumbers: result.episodeNumbers,
          samplesPerEpisode: result.samplesPerEpisode,
        }).toEqual(expectedWorkload);
      }
      expect(result.batchCount).toBeGreaterThanOrEqual(4);
      expect(result.selectedDevice).toBe("cpu");
      expect(result.samplesPerSecond).toBeGreaterThan(0);
      expect(result.sections.map((section) => section.name)).toEqual(expect.arrayContaining([
        "getObservationMs", "getLegalActionsMs", "encodeObservationMs", "encodeLegalActionsMs",
        "stepReplayActionMs", "sidecarLoadMs", "workerBatchQueueAndProcessingMs", "nodePythonRoundTripMs",
        "pythonDeserializeMs", "pythonBinaryDecodeMs", "pythonTensorPreparationMs", "pythonForwardMs", "pythonLossMs",
        "pythonBackwardMs", "pythonOptimizerStepMs",
      ]));
      expect(result.sections.every((section) => [section.totalMs, section.msPerSample, section.percentOfElapsed].every(Number.isFinite))).toBe(true);
    }
  }, 120_000);
});
