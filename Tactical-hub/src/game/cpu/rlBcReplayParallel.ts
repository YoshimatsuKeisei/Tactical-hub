import { fork, type ChildProcess } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { RlImitationEpisode } from "./rlImitationCollector";
import type { BcEncodedSample } from "./pythonBcTrainerClient";
import type { RlBcReplayWorkerRequest, RlBcReplayWorkerResponse } from "./rlBcReplayWorkerMessages";

export type NumberedReplayEpisode = { episodeNumber: number; episode: RlImitationEpisode };

export async function runParallelBcReplay(input: {
  episodes: NumberedReplayEpisode[];
  workerCount: number;
  batchSize: number;
  onBatch: (samples: BcEncodedSample[]) => void | Promise<void>;
  workerEntryPath?: string;
}) {
  if (!input.episodes.length) throw new Error("Parallel BC replay requires at least one episode");
  if (!Number.isInteger(input.workerCount) || input.workerCount <= 0) throw new Error("workerCount must be positive");
  const effectiveWorkerCount = Math.min(input.workerCount, input.episodes.length);
  const workerEntry = input.workerEntryPath ?? fileURLToPath(new URL("./rlBcReplayWorker.ts", import.meta.url));
  const viteNodeEntry = resolve(process.cwd(), "node_modules/vite-node/vite-node.mjs");
  const workers: ChildProcess[] = [];
  const pending = new Map<string, { workerId: number; item: NumberedReplayEpisode; nextBatchSequence: number; sampleCount: number }>();
  const completed = new Map<number, number>();
  let nextIndex = 0;

  return await new Promise<{ sampleCount: number; episodeCount: number; effectiveWorkerCount: number }>((resolveRun, rejectRun) => {
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
          sampleCount: [...completed.values()].reduce((sum, count) => sum + count, 0),
          episodeCount: completed.size,
          effectiveWorkerCount,
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
      } satisfies RlBcReplayWorkerRequest);
    };

    for (let workerId = 0; workerId < effectiveWorkerCount; workerId += 1) {
      const worker = fork(viteNodeEntry, [workerEntry], { stdio: ["ignore", "ignore", "ignore", "ipc"] });
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
          if (raw.batchSequence !== assignment.nextBatchSequence || !raw.samples.length) {
            fail(new Error(`BC replay worker ${workerId} returned duplicate/out-of-order batch for episode ${raw.episodeNumber}`));
            return;
          }
          assignment.nextBatchSequence += 1;
          assignment.sampleCount += raw.samples.length;
          void Promise.resolve(input.onBatch(raw.samples)).then(() => {
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
        completed.set(raw.episodeNumber, raw.sampleCount);
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
