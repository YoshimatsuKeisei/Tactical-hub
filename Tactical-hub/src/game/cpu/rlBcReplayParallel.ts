import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import type { RlImitationEpisode } from "./rlImitationCollector";
import type { PackedBcBatch } from "./rlBcPackedBatch";
import type { RlBcReplayWorkerRequest, RlBcReplayWorkerResponse } from "./rlBcReplayWorkerMessages";
import { RL_PROJECT_ROOT, RL_VITE_NODE_ENTRY } from "./rlProjectPaths";

export type NumberedReplayEpisode = { episodeNumber: number; episode: RlImitationEpisode };

export async function runParallelBcReplay(input: {
  episodes: NumberedReplayEpisode[];
  workerCount: number;
  batchSize: number;
  sidecarDirectory: string;
  onBatch: (packedBatch: PackedBcBatch) => void | Promise<void>;
  onEpisodeCompleted?: (completed: number, total: number) => void;
  workerEntryPath?: string;
}) {
  if (!input.episodes.length) throw new Error("Parallel BC replay requires at least one episode");
  if (!Number.isInteger(input.workerCount) || input.workerCount <= 0) throw new Error("workerCount must be positive");
  const effectiveWorkerCount = Math.min(input.workerCount, input.episodes.length);
  const workerEntry = input.workerEntryPath ?? fileURLToPath(new URL("./rlBcReplayWorker.ts", import.meta.url));
  const viteNodeEntry = RL_VITE_NODE_ENTRY;
  const workers: ChildProcess[] = [];
  const pending = new Map<string, { workerId: number; item: NumberedReplayEpisode; nextBatchSequence: number; sampleCount: number }>();
  const completed = new Map<number, { sampleCount: number; sidecarGenerated: boolean; sidecarPreparationMs: number; directReplayMs: number }>();
  let nextIndex = 0;

  return await new Promise<{
    sampleCount: number;
    episodeCount: number;
    effectiveWorkerCount: number;
    generatedSidecarCount: number;
    reusedSidecarCount: number;
    sidecarPreparationMs: number;
    directReplayMs: number;
  }>((resolveRun, rejectRun) => {
    let settled = false;
    const cleanup = () => {
      for (const worker of workers) if (worker.connected) worker.send({ type: "shutdown" } satisfies RlBcReplayWorkerRequest);
      for (const worker of workers) worker.kill();
    };
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      rejectRun(error);
    };
    const finishIfDone = () => {
      if (!settled && completed.size === input.episodes.length) {
        settled = true;
        cleanup();
        resolveRun({
          sampleCount: [...completed.values()].reduce((sum, value) => sum + value.sampleCount, 0),
          episodeCount: completed.size,
          effectiveWorkerCount,
          generatedSidecarCount: [...completed.values()].filter((value) => value.sidecarGenerated).length,
          reusedSidecarCount: [...completed.values()].filter((value) => !value.sidecarGenerated).length,
          sidecarPreparationMs: [...completed.values()].reduce((sum, value) => sum + value.sidecarPreparationMs, 0),
          directReplayMs: [...completed.values()].reduce((sum, value) => sum + value.directReplayMs, 0),
        });
        return true;
      }
      return false;
    };
    const assign = (worker: ChildProcess, workerId: number) => {
      if (finishIfDone() || nextIndex >= input.episodes.length) return;
      const item = input.episodes[nextIndex++];
      const taskId = `bc-replay-${item.episodeNumber}`;
      if (pending.has(taskId) || completed.has(item.episodeNumber)) {
        fail(new Error(`Duplicate BC replay episode ${item.episodeNumber}`));
        return;
      }
      pending.set(taskId, { workerId, item, nextBatchSequence: 0, sampleCount: 0 });
      worker.send({
        type: "runEpisode",
        taskId,
        episodeNumber: item.episodeNumber,
        episode: item.episode,
        batchSize: input.batchSize,
        sidecarDirectory: input.sidecarDirectory,
      } satisfies RlBcReplayWorkerRequest);
    };

    for (let workerId = 0; workerId < effectiveWorkerCount; workerId += 1) {
      const worker = fork(viteNodeEntry, [workerEntry], {
        cwd: RL_PROJECT_ROOT,
        stdio: ["ignore", "ignore", "ignore", "ipc"],
        serialization: "advanced",
      });
      workers.push(worker);
      worker.on("message", (raw: RlBcReplayWorkerResponse) => {
        if (settled || !raw || !["encodedBatch", "episodeCompleted", "workerError"].includes(raw.type)) {
          if (!settled) fail(new Error(`BC replay worker ${workerId} sent an invalid message`));
          return;
        }
        if (!raw.taskId) {
          fail(new Error(`BC replay worker ${workerId} failed outside a known episode`));
          return;
        }
        const assignment = pending.get(raw.taskId);
        if (!assignment || assignment.workerId !== workerId) {
          fail(new Error(`BC replay worker ${workerId} referenced unknown task ${raw.taskId}`));
          return;
        }
        if (raw.type === "workerError") {
          fail(new Error(`BC replay worker ${workerId} failed episode ${assignment.item.episodeNumber}: ${raw.error}`));
          return;
        }
        if (raw.episodeNumber !== assignment.item.episodeNumber) {
          fail(new Error(`BC replay worker ${workerId} returned mismatched episode`));
          return;
        }
        if (raw.type === "encodedBatch") {
          if (raw.batchSequence !== assignment.nextBatchSequence || raw.packedBatch.batchSize <= 0) {
            fail(new Error(`BC replay worker ${workerId} returned duplicate/out-of-order batch for episode ${raw.episodeNumber}`));
            return;
          }
          assignment.nextBatchSequence += 1;
          assignment.sampleCount += raw.packedBatch.batchSize;
          void Promise.resolve(input.onBatch(raw.packedBatch)).then(() => {
            if (!settled) worker.send({
              type: "batchConsumed",
              taskId: raw.taskId,
              batchSequence: raw.batchSequence,
            } satisfies RlBcReplayWorkerRequest);
          }).catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
          return;
        }
        if (raw.sampleCount !== assignment.sampleCount || raw.sampleCount !== assignment.item.episode.end.decisionCount) {
          fail(new Error(`BC replay episode ${raw.episodeNumber} sample count mismatch: ${raw.sampleCount}/${assignment.sampleCount}/${assignment.item.episode.end.decisionCount}`));
          return;
        }
        pending.delete(raw.taskId);
        if (completed.has(raw.episodeNumber)) {
          fail(new Error(`BC replay episode completed twice: ${raw.episodeNumber}`));
          return;
        }
        completed.set(raw.episodeNumber, {
          sampleCount: raw.sampleCount,
          sidecarGenerated: raw.sidecarGenerated,
          sidecarPreparationMs: raw.sidecarPreparationMs,
          directReplayMs: raw.directReplayMs,
        });
        input.onEpisodeCompleted?.(completed.size, input.episodes.length);
        assign(worker, workerId);
      });
      worker.on("error", (error) => fail(new Error(`BC replay worker ${workerId} process error: ${error.message}`)));
      worker.on("exit", (code, signal) => {
        if (!settled) {
          const active = [...pending.values()].find((entry) => entry.workerId === workerId);
          fail(new Error(`BC replay worker ${workerId} exited${active ? ` during episode ${active.item.episodeNumber}` : ""} (code=${code}, signal=${signal})`));
        }
      });
      assign(worker, workerId);
    }
  });
}
