import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { runRlBcShortProfile } from "../cpu/rlBcProfiler";
import { runParallelRlImitationCollection } from "../cpu/rlImitationParallel";

const directory = mkdtempSync(join(tmpdir(), "tactical-hub-bc-profile-"));
afterAll(() => rmSync(directory, { recursive: true, force: true }));

describe("RL BC short profiler", () => {
  it("stops at the requested measured samples with workers=1 and workers=4", async () => {
    const dataPath = join(directory, "replay.jsonl");
    const collection = await runParallelRlImitationCollection({
      seedStart: 7701,
      episodeCount: 3,
      maxTurns: 1,
      outputPath: dataPath,
      workerCount: 1,
    });
    expect(collection.failedEpisodeCount).toBe(0);
    for (const workerCount of [1, 4]) {
      const result = await runRlBcShortProfile({
        dataPath,
        samples: 16,
        warmupSamples: 8,
        batchSize: 4,
        workerCount,
        device: "cpu",
        torchThreads: 2,
        torchInteropThreads: 1,
      });
      expect(result.measuredSamples).toBe(16);
      expect(result.batchCount).toBe(4);
      expect(result.selectedDevice).toBe("cpu");
      expect(result.samplesPerSecond).toBeGreaterThan(0);
      expect(result.sections.map((section) => section.name)).toEqual(expect.arrayContaining([
        "getObservationMs", "getLegalActionsMs", "encodeObservationMs", "encodeLegalActionsMs",
        "stepReplayActionMs", "sidecarLoadMs", "workerIpcBatchWaitMs", "nodePythonRoundTripMs",
        "pythonDeserializeMs", "pythonBinaryDecodeMs", "pythonTensorPreparationMs", "pythonForwardMs", "pythonLossMs",
        "pythonBackwardMs", "pythonOptimizerStepMs",
      ]));
      expect(result.sections.every((section) => [section.totalMs, section.msPerSample, section.percentOfElapsed].every(Number.isFinite))).toBe(true);
    }
  }, 120_000);
});
