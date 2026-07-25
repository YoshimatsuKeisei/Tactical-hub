import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import {
  parseRlImitationReplayRecords,
  replayHeuristicImitationEpisode,
  type RlImitationReplayRecord,
} from "../cpu/rlImitationCollector";
import { runParallelRlImitationCollection } from "../cpu/rlImitationParallel";

const directory = mkdtempSync(join(tmpdir(), "tactical-hub-rl4a-parallel-test-"));
afterAll(() => rmSync(directory, { recursive: true, force: true }));

function readEpisodes(path: string) {
  const records = readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RlImitationReplayRecord);
  return parseRlImitationReplayRecords(records);
}

describe("RL-4A parallel replay collection", () => {
  it("preserves episodes for workers=1 and a dynamic multi-worker queue", async () => {
    const serialPath = join(directory, "serial.jsonl");
    const parallelPath = join(directory, "parallel.jsonl");
    const serial = await runParallelRlImitationCollection({
      seedStart: 501,
      episodeCount: 1,
      maxTurns: 1,
      outputPath: serialPath,
      workerCount: 1,
    });
    const parallel = await runParallelRlImitationCollection({
      seedStart: 501,
      episodeCount: 2,
      maxTurns: 1,
      outputPath: parallelPath,
      workerCount: 2,
    });

    expect(serial.successEpisodeCount).toBe(1);
    expect(serial.failedEpisodeCount).toBe(0);
    expect(serial.effectiveWorkerCount).toBe(1);
    expect(parallel.successEpisodeCount).toBe(2);
    expect(parallel.failedEpisodeCount).toBe(0);
    expect(parallel.effectiveWorkerCount).toBe(2);

    const serialEpisodes = readEpisodes(serialPath);
    const parallelEpisodes = readEpisodes(parallelPath);
    expect(parallelEpisodes.map((episode) => episode.header.seed)).toEqual([501, 502]);
    expect(new Set(parallelEpisodes.map((episode) => episode.header.episodeId)).size).toBe(2);
    expect(parallelEpisodes).toHaveLength(2);
    expect(parallel.episodes[0].finalStateHash).toBe(serial.episodes[0].finalStateHash);

    for (const episode of parallelEpisodes) {
      const replayed = await replayHeuristicImitationEpisode({ episode });
      expect(replayed.finalStateHash).toBe(episode.end.finalStateHash);
    }
  }, 60_000);
});
