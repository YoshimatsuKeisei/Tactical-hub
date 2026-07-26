import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { runParallelRlImitationCollection } from "../cpu/rlImitationParallel";
import { prepareRlReplaySidecars } from "../cpu/rlPrepareSidecars";

const directory = mkdtempSync(join(tmpdir(), "tactical-hub-sidecars-"));
afterAll(() => rmSync(directory, { recursive: true, force: true }));

describe("RL Replay RNG sidecar preparation", () => {
  it("generates each episode once with workers=4 and reuses all with workers=1", async () => {
    const dataPath = join(directory, "replay.jsonl");
    const collection = await runParallelRlImitationCollection({
      seedStart: 9901,
      episodeCount: 4,
      maxTurns: 1,
      outputPath: dataPath,
      workerCount: 1,
    });
    expect(collection.failedEpisodeCount).toBe(0);

    const parallel = await prepareRlReplaySidecars({ dataPath, workerCount: 4 });
    expect(parallel).toMatchObject({
      requestedWorkerCount: 4,
      effectiveWorkerCount: 4,
      episodeCount: 4,
      generatedCount: 4,
      reusedCount: 0,
      failedCount: 0,
      failures: [],
    });

    const serial = await prepareRlReplaySidecars({ dataPath, workerCount: 1 });
    expect(serial).toMatchObject({
      requestedWorkerCount: 1,
      effectiveWorkerCount: 1,
      episodeCount: 4,
      generatedCount: 0,
      reusedCount: 4,
      failedCount: 0,
      failures: [],
    });
    expect(serial.sidecarDirectory).toBe(parallel.sidecarDirectory);
    expect(Number.isFinite(parallel.elapsedMs)).toBe(true);
    expect(Number.isFinite(serial.elapsedMs)).toBe(true);
  }, 120_000);
});
