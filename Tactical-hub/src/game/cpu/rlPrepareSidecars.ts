import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { generateRlReplayRngSidecar, type RlImitationEpisode } from "./rlImitationCollector";
import type { RlPrepareSidecarsWorkerRequest, RlPrepareSidecarsWorkerResponse } from "./rlPrepareSidecarsWorkerMessages";
import { RL_PROJECT_ROOT, RL_VITE_NODE_ENTRY } from "./rlProjectPaths";
import { getDefaultRlReplaySidecarDirectory, loadOrCreateRlReplayRngSidecar } from "./rlReplayRngSidecar";
import { readRlImitationEpisodes } from "./rlReplayReader";

type EpisodeTask = { taskId: string; episodeNumber: number; episode: RlImitationEpisode };
export type RlPrepareSidecarsFailure = { episodeNumber: number; episodeId: string; error: string };
export type RlPrepareSidecarsResult = {
  requestedWorkerCount: number;
  effectiveWorkerCount: number;
  episodeCount: number;
  generatedCount: number;
  reusedCount: number;
  failedCount: number;
  elapsedMs: number;
  sidecarDirectory: string;
  failures: RlPrepareSidecarsFailure[];
};

export async function prepareRlReplaySidecars(input: {
  dataPath: string;
  workerCount?: number;
  workerEntryPath?: string;
}): Promise<RlPrepareSidecarsResult> {
  const requestedWorkerCount = input.workerCount ?? 1;
  if (!Number.isInteger(requestedWorkerCount) || requestedWorkerCount <= 0) throw new Error("workerCount must be a positive integer");
  const tasks: EpisodeTask[] = [];
  for await (const item of readRlImitationEpisodes(input.dataPath, { from: 1, to: Number.MAX_SAFE_INTEGER })) {
    tasks.push({ taskId: `prepare-sidecar-${item.episodeNumber}`, ...item });
  }
  if (!tasks.length) throw new Error("No Replay episodes found");
  const effectiveWorkerCount = Math.min(requestedWorkerCount, tasks.length);
  const sidecarDirectory = getDefaultRlReplaySidecarDirectory(input.dataPath);
  const started = performance.now();
  const completed = new Map<number, boolean>();
  const failures: RlPrepareSidecarsFailure[] = [];

  if (effectiveWorkerCount === 1) {
    for (const task of tasks) {
      try {
        const result = await loadOrCreateRlReplayRngSidecar(
          task.episode,
          sidecarDirectory,
          () => generateRlReplayRngSidecar(task.episode),
        );
        completed.set(task.episodeNumber, result.generated);
      } catch (error) {
        failures.push({
          episodeNumber: task.episodeNumber,
          episodeId: task.episode.header.episodeId,
          error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        });
      }
    }
  } else {
    const workerEntry = input.workerEntryPath ?? fileURLToPath(new URL("./rlPrepareSidecarsWorker.ts", import.meta.url));
    const workers: ChildProcess[] = [];
    const pending = new Map<string, { workerId: number; task: EpisodeTask }>();
    let nextIndex = 0;
    await new Promise<void>((resolveRun, rejectRun) => {
      let settled = false;
      const cleanup = () => {
        for (const worker of workers) if (worker.connected) worker.send({ type: "shutdown" } satisfies RlPrepareSidecarsWorkerRequest);
        for (const worker of workers) worker.kill();
      };
      const failRun = (error: Error) => {
        if (settled) return;
        settled = true;
        cleanup();
        rejectRun(error);
      };
      const finishIfDone = () => {
        if (!settled && completed.size + failures.length === tasks.length) {
          settled = true;
          cleanup();
          resolveRun();
          return true;
        }
        return false;
      };
      const assign = (worker: ChildProcess, workerId: number) => {
        if (finishIfDone() || nextIndex >= tasks.length) return;
        const task = tasks[nextIndex++];
        pending.set(task.taskId, { workerId, task });
        worker.send({
          type: "prepareEpisode",
          taskId: task.taskId,
          episodeNumber: task.episodeNumber,
          episode: task.episode,
          sidecarDirectory,
        } satisfies RlPrepareSidecarsWorkerRequest);
      };
      for (let workerId = 0; workerId < effectiveWorkerCount; workerId += 1) {
        const worker = fork(RL_VITE_NODE_ENTRY, [workerEntry], {
          cwd: RL_PROJECT_ROOT,
          stdio: ["ignore", "ignore", "ignore", "ipc"],
        });
        workers.push(worker);
        worker.on("message", (raw: RlPrepareSidecarsWorkerResponse) => {
          if (settled || !raw || !["episodePrepared", "workerError"].includes(raw.type)) {
            if (!settled) failRun(new Error(`RNG sidecar worker ${workerId} sent an invalid message`));
            return;
          }
          if (!raw.taskId) {
            failRun(new Error(`RNG sidecar worker ${workerId} failed outside a known episode`));
            return;
          }
          const assignment = pending.get(raw.taskId);
          if (!assignment || assignment.workerId !== workerId) {
            failRun(new Error(`RNG sidecar worker ${workerId} referenced unknown task ${raw.taskId}`));
            return;
          }
          pending.delete(raw.taskId);
          if (raw.type === "workerError") {
            failures.push({
              episodeNumber: assignment.task.episodeNumber,
              episodeId: assignment.task.episode.header.episodeId,
              error: raw.error,
            });
          } else {
            if (raw.episodeNumber !== assignment.task.episodeNumber || completed.has(raw.episodeNumber)) {
              failRun(new Error(`RNG sidecar worker ${workerId} returned a mismatched or duplicate episode`));
              return;
            }
            completed.set(raw.episodeNumber, raw.generated);
          }
          assign(worker, workerId);
        });
        worker.on("error", (error) => failRun(new Error(`RNG sidecar worker ${workerId} process error: ${error.message}`)));
        worker.on("exit", (code, signal) => {
          if (!settled) failRun(new Error(`RNG sidecar worker ${workerId} exited unexpectedly (code=${code}, signal=${signal})`));
        });
        assign(worker, workerId);
      }
    });
  }

  return {
    requestedWorkerCount,
    effectiveWorkerCount,
    episodeCount: tasks.length,
    generatedCount: [...completed.values()].filter(Boolean).length,
    reusedCount: [...completed.values()].filter((generated) => !generated).length,
    failedCount: failures.length,
    elapsedMs: performance.now() - started,
    sidecarDirectory,
    failures: failures.sort((left, right) => left.episodeNumber - right.episodeNumber),
  };
}
