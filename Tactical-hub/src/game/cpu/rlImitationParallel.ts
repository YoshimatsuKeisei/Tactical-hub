import { createReadStream, createWriteStream, mkdtempSync, rmSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { fork, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { runHeuristicImitationEpisode, type RlImitationEpisodeResult } from "./rlImitationCollector";
import type { RlImitationWorkerRequest, RlImitationWorkerResponse } from "./rlImitationWorkerMessages";

export type ParallelRlImitationOptions = {
  seedStart: number;
  episodeCount: number;
  maxTurns: number;
  outputPath: string;
  workerCount?: number;
  workerEntryPath?: string;
};

export type RlImitationFailure = {
  taskId: string;
  episodeId: string;
  seed: number;
  error: string;
};

export type ParallelRlImitationResult = {
  requestedWorkerCount: number;
  effectiveWorkerCount: number;
  episodeCount: number;
  decisionCount: number;
  successEpisodeCount: number;
  failedEpisodeCount: number;
  elapsedMs: number;
  episodesPerSecond: number;
  decisionsPerSecond: number;
  outputPath: string;
  episodes: RlImitationEpisodeResult[];
  failures: RlImitationFailure[];
};

type Task = {
  taskId: string;
  episodeId: string;
  seed: number;
  shardPath: string;
};

async function writeEpisodeShard(task: Task, maxTurns: number) {
  const output = createWriteStream(task.shardPath, { encoding: "utf8" });
  try {
    const result = await runHeuristicImitationEpisode({
      episodeId: task.episodeId,
      seed: task.seed,
      maxTurns,
      onRecord: async (record) => {
        if (!output.write(`${JSON.stringify(record)}\n`)) await once(output, "drain");
      },
    });
    output.end();
    await once(output, "finish");
    return result;
  } catch (error) {
    output.destroy();
    throw error;
  }
}

async function runWorkers(
  input: ParallelRlImitationOptions,
  tasks: Task[],
  workerCount: number,
): Promise<{ results: Map<string, RlImitationEpisodeResult>; failures: RlImitationFailure[] }> {
  const workerEntry = input.workerEntryPath ?? fileURLToPath(new URL("./rlImitationWorker.ts", import.meta.url));
  const viteNodeEntry = resolve(process.cwd(), "node_modules/vite-node/vite-node.mjs");
  const results = new Map<string, RlImitationEpisodeResult>();
  const failures: RlImitationFailure[] = [];
  const pending = new Map<string, { workerId: number; task: Task }>();
  const workers: ChildProcess[] = [];
  let nextTaskIndex = 0;

  return await new Promise((resolveBatch, rejectBatch) => {
    let settled = false;
    const cleanup = () => {
      for (const worker of workers) if (worker.connected) worker.send({ type: "shutdown" } satisfies RlImitationWorkerRequest);
      for (const worker of workers) worker.kill();
    };
    const finishIfDone = () => {
      if (!settled && results.size + failures.length === tasks.length) {
        settled = true;
        cleanup();
        resolveBatch({ results, failures });
        return true;
      }
      return false;
    };
    const failBatch = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      rejectBatch(error);
    };
    const assign = (worker: ChildProcess, workerId: number) => {
      if (finishIfDone() || nextTaskIndex >= tasks.length) return;
      const task = tasks[nextTaskIndex++];
      pending.set(task.taskId, { workerId, task });
      worker.send({
        type: "runEpisode",
        taskId: task.taskId,
        episodeId: task.episodeId,
        seed: task.seed,
        maxTurns: input.maxTurns,
        shardPath: task.shardPath,
      } satisfies RlImitationWorkerRequest);
    };

    for (let workerId = 0; workerId < workerCount; workerId += 1) {
      const worker = fork(viteNodeEntry, [workerEntry], { stdio: ["ignore", "ignore", "ignore", "ipc"] });
      workers.push(worker);
      worker.on("message", (raw: RlImitationWorkerResponse) => {
        if (settled || !raw || !["episodeCompleted", "workerError"].includes(raw.type)) {
          if (!settled) failBatch(new Error(`Worker ${workerId} sent an invalid message`));
          return;
        }
        if (!raw.taskId) {
          failBatch(new Error(`Worker ${workerId} failed outside a known episode: ${raw.type === "workerError" ? raw.error : "invalid response"}`));
          return;
        }
        const assignment = pending.get(raw.taskId);
        if (!assignment || assignment.workerId !== workerId) {
          failBatch(new Error(`Worker ${workerId} completed unknown or duplicate task ${raw.taskId}`));
          return;
        }
        pending.delete(raw.taskId);
        const task = assignment.task;
        if (raw.type === "workerError") {
          failures.push({ taskId: task.taskId, episodeId: task.episodeId, seed: task.seed, error: raw.error });
        } else {
          if (raw.seed !== task.seed || raw.episodeId !== task.episodeId || raw.result.seed !== task.seed) {
            failBatch(new Error(`Worker ${workerId} returned mismatched episode ${raw.taskId}`));
            return;
          }
          results.set(task.taskId, raw.result);
        }
        assign(worker, workerId);
      });
      worker.on("error", (error) => failBatch(new Error(`Worker ${workerId} process error: ${error.message}`)));
      worker.on("exit", (code, signal) => {
        if (!settled) {
          const active = [...pending.values()].find((entry) => entry.workerId === workerId);
          const suffix = active ? ` episode ${active.task.episodeId} seed ${active.task.seed}` : "";
          failBatch(new Error(`Worker ${workerId} exited unexpectedly${suffix} (code=${code}, signal=${signal})`));
        }
      });
      assign(worker, workerId);
    }
  });
}

export async function runParallelRlImitationCollection(input: ParallelRlImitationOptions): Promise<ParallelRlImitationResult> {
  const requestedWorkerCount = input.workerCount ?? 1;
  if (!Number.isInteger(requestedWorkerCount) || requestedWorkerCount <= 0) throw new Error("workerCount must be a positive integer");
  if (!Number.isInteger(input.episodeCount) || input.episodeCount <= 0) throw new Error("episodeCount must be a positive integer");
  const effectiveWorkerCount = Math.min(requestedWorkerCount, input.episodeCount);
  const outputPath = resolve(input.outputPath);
  await mkdir(dirname(outputPath), { recursive: true });
  const shardDirectory = mkdtempSync(join(dirname(outputPath), ".rl-imitation-shards-"));
  const tasks = Array.from({ length: input.episodeCount }, (_, index): Task => {
    const seed = input.seedStart + index;
    return {
      taskId: `episode-${index}`,
      episodeId: `episode-${index + 1}-seed-${seed}`,
      seed,
      shardPath: join(shardDirectory, `episode-${index.toString().padStart(6, "0")}.jsonl`),
    };
  });
  const started = performance.now();
  let resultMap = new Map<string, RlImitationEpisodeResult>();
  let failures: RlImitationFailure[] = [];
  try {
    if (effectiveWorkerCount === 1) {
      for (const task of tasks) {
        try {
          resultMap.set(task.taskId, await writeEpisodeShard(task, input.maxTurns));
        } catch (error) {
          failures.push({ taskId: task.taskId, episodeId: task.episodeId, seed: task.seed, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
        }
      }
    } else {
      ({ results: resultMap, failures } = await runWorkers(input, tasks, effectiveWorkerCount));
    }

    const output = createWriteStream(outputPath, { encoding: "utf8" });
    try {
      for (const task of tasks) {
        if (!resultMap.has(task.taskId)) continue;
        const shard = createReadStream(task.shardPath);
        shard.pipe(output, { end: false });
        await once(shard, "end");
      }
    } finally {
      output.end();
      await once(output, "finish");
    }
  } finally {
    rmSync(shardDirectory, { recursive: true, force: true });
  }
  const elapsedMs = performance.now() - started;
  const episodes = tasks.flatMap((task) => {
    const result = resultMap.get(task.taskId);
    return result ? [result] : [];
  });
  const decisionCount = episodes.reduce((sum, episode) => sum + episode.decisionCount, 0);
  return {
    requestedWorkerCount,
    effectiveWorkerCount,
    episodeCount: input.episodeCount,
    decisionCount,
    successEpisodeCount: episodes.length,
    failedEpisodeCount: failures.length,
    elapsedMs,
    episodesPerSecond: episodes.length / (elapsedMs / 1000),
    decisionsPerSecond: decisionCount / (elapsedMs / 1000),
    outputPath,
    episodes,
    failures,
  };
}
