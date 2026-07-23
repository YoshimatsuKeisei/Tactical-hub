import { fork, type ChildProcess } from "node:child_process";
import { availableParallelism } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { runHeadlessMatch, summarizeHeadlessMatches, type HeadlessBatchResult, type HeadlessMatchResult, type HeadlessMode } from "./headlessSimulation";
import type { HeadlessPolicyName, HeadlessWorkerRequest, HeadlessWorkerResponse, SerializableHeadlessMatchOptions, WorkerMemoryPeak } from "./headlessWorkerMessages";

export type ParallelHeadlessBatchOptions = {
  participantCount: 3 | 4;
  matchCount: number;
  seedStart: number;
  maxTurns: number;
  maxActions?: number;
  maxActionsPerPhase?: number;
  historyLimit?: number;
  mode?: HeadlessMode;
  profile?: boolean;
  trace?: boolean;
  policyName: HeadlessPolicyName;
  workerCount?: number;
  workerEntryPath?: string;
};

export type WorkerMemoryResult = WorkerMemoryPeak & { workerId: number; completedMatchCount: number };
export type ParallelHeadlessBatchResult = HeadlessBatchResult & {
  requestedWorkerCount: number;
  effectiveWorkerCount: number;
  elapsedMs: number;
  matchesPerSecond: number;
  totalActionCount: number;
  actionsPerSecond: number;
  startProcessRssBytes: number;
  peakProcessRssBytes: number;
  endProcessRssBytes: number;
  peakProcessRssDeltaBytes: number;
  workerMemory: WorkerMemoryResult[];
};

type Task = { taskId: string; seed: number; options: SerializableHeadlessMatchOptions };

function updatePeak(target: WorkerMemoryPeak, value: WorkerMemoryPeak) {
  target.peakHeapUsedBytes = Math.max(target.peakHeapUsedBytes, value.peakHeapUsedBytes);
  target.peakHeapTotalBytes = Math.max(target.peakHeapTotalBytes, value.peakHeapTotalBytes);
  target.peakExternalBytes = Math.max(target.peakExternalBytes, value.peakExternalBytes);
  target.peakArrayBuffersBytes = Math.max(target.peakArrayBuffersBytes, value.peakArrayBuffersBytes);
}

function validate(input: ParallelHeadlessBatchOptions) {
  const requested = input.workerCount ?? 1;
  if (!Number.isInteger(requested) || requested <= 0) throw new Error("workerCount must be a positive integer");
  if (!Number.isInteger(input.matchCount) || input.matchCount <= 0) throw new Error("matchCount must be a positive integer");
  return { requested, effective: Math.min(requested, input.matchCount) };
}

function taskOptions(input: ParallelHeadlessBatchOptions, seed: number): SerializableHeadlessMatchOptions {
  return { participantCount: input.participantCount, seed, maxTurns: input.maxTurns, maxActions: input.maxActions, maxActionsPerPhase: input.maxActionsPerPhase, historyLimit: input.historyLimit, mode: input.mode, profile: input.profile, trace: input.trace };
}

async function runParallel(input: ParallelHeadlessBatchOptions, tasks: Task[], workerCount: number) {
  const workerEntry = input.workerEntryPath ?? fileURLToPath(new URL("./headlessWorker.ts", import.meta.url));
  const viteNodeEntry = resolve(process.cwd(), "node_modules/vite-node/vite-node.mjs");
  const results = new Map<string, HeadlessMatchResult>();
  const pending = new Map<string, { workerId: number; seed: number }>();
  const workers: ChildProcess[] = [];
  const workerMemory = Array.from({ length: workerCount }, (_, workerId): WorkerMemoryResult => ({ workerId, completedMatchCount: 0, peakHeapUsedBytes: 0, peakHeapTotalBytes: 0, peakExternalBytes: 0, peakArrayBuffersBytes: 0 }));
  let nextTaskIndex = 0;

  return await new Promise<{ matches: HeadlessMatchResult[]; workerMemory: WorkerMemoryResult[] }>((resolveBatch, rejectBatch) => {
    let settled = false;
    const cleanup = () => { for (const worker of workers) if (worker.connected) worker.send({ type: "shutdown" } satisfies HeadlessWorkerRequest); for (const worker of workers) worker.kill(); };
    const fail = (error: Error) => { if (settled) return; settled = true; cleanup(); rejectBatch(error); };
    const assign = (worker: ChildProcess, workerId: number) => {
      if (nextTaskIndex >= tasks.length) {
        if (results.size === tasks.length && !settled) { settled = true; cleanup(); resolveBatch({ matches: tasks.map((task) => results.get(task.taskId)!), workerMemory }); }
        return;
      }
      const task = tasks[nextTaskIndex++];
      pending.set(task.taskId, { workerId, seed: task.seed });
      worker.send({ type: "runMatch", taskId: task.taskId, options: task.options, policyName: input.policyName } satisfies HeadlessWorkerRequest);
    };

    for (let workerId = 0; workerId < workerCount; workerId += 1) {
      const worker = fork(viteNodeEntry, [workerEntry], { stdio: ["ignore", "ignore", "ignore", "ipc"] });
      workers.push(worker);
      worker.on("message", (raw: HeadlessWorkerResponse) => {
        if (settled) return;
        if (!raw || !["matchCompleted", "workerError"].includes(raw.type)) { fail(new Error(`Worker ${workerId} sent an invalid message`)); return; }
        if (raw.type === "workerError") { fail(new Error(`Worker ${workerId} failed task ${raw.taskId ?? "unknown"} seed ${raw.seed ?? "unknown"}: ${raw.error}`)); return; }
        const task = pending.get(raw.taskId);
        if (!task) { fail(new Error(`Worker ${workerId} completed unknown or duplicate task ${raw.taskId}`)); return; }
        if (task.workerId !== workerId || task.seed !== raw.seed || raw.result.seed !== raw.seed) { fail(new Error(`Worker ${workerId} returned mismatched task ${raw.taskId}/seed ${raw.seed}`)); return; }
        pending.delete(raw.taskId);
        if (results.has(raw.taskId)) { fail(new Error(`Worker ${workerId} completed task twice: ${raw.taskId}`)); return; }
        results.set(raw.taskId, raw.result);
        workerMemory[workerId].completedMatchCount += 1;
        updatePeak(workerMemory[workerId], raw.memory);
        assign(worker, workerId);
      });
      worker.on("error", (error) => fail(new Error(`Worker ${workerId} process error: ${error.message}`)));
      worker.on("exit", (code, signal) => {
        if (!settled && results.size < tasks.length) {
          const activeTask = [...pending.entries()].find(([, assignment]) => assignment.workerId === workerId);
          const detail = activeTask ? ` task ${activeTask[0]} seed ${activeTask[1].seed}` : " without an assigned task";
          fail(new Error(`Worker ${workerId} exited before completing${detail} (code=${code}, signal=${signal})`));
        }
      });
      assign(worker, workerId);
    }
  });
}

export async function runParallelHeadlessBatch(input: ParallelHeadlessBatchOptions): Promise<ParallelHeadlessBatchResult> {
  const { requested, effective } = validate(input);
  const startProcessRssBytes = process.memoryUsage().rss;
  let peakProcessRssBytes = startProcessRssBytes;
  const sampleParent = () => { peakProcessRssBytes = Math.max(peakProcessRssBytes, process.memoryUsage().rss); };
  const interval = setInterval(sampleParent, 25);
  const started = performance.now();
  const tasks = Array.from({ length: input.matchCount }, (_, index): Task => {
    const seed = input.seedStart + index;
    return { taskId: `match-${index}`, seed, options: taskOptions(input, seed) };
  });
  let matches: HeadlessMatchResult[];
  let workerMemory: WorkerMemoryResult[];
  try {
    if (effective === 1) {
      const memory: WorkerMemoryResult = { workerId: 0, completedMatchCount: 0, peakHeapUsedBytes: 0, peakHeapTotalBytes: 0, peakExternalBytes: 0, peakArrayBuffersBytes: 0 };
      matches = tasks.map((task) => {
        const policy = input.policyName === "heuristic" ? createHeuristicCpuPolicy() : undefined;
        const result = runHeadlessMatch({ ...task.options, policy, onProgress: () => { sampleParent(); const current = process.memoryUsage(); updatePeak(memory, { peakHeapUsedBytes: current.heapUsed, peakHeapTotalBytes: current.heapTotal, peakExternalBytes: current.external, peakArrayBuffersBytes: current.arrayBuffers }); } });
        memory.completedMatchCount += 1; sampleParent(); return result;
      });
      workerMemory = [memory];
    } else ({ matches, workerMemory } = await runParallel(input, tasks, effective));
  } finally { clearInterval(interval); sampleParent(); }
  matches.sort((left, right) => left.seed - right.seed);
  const elapsedMs = performance.now() - started;
  const endProcessRssBytes = process.memoryUsage().rss;
  const summary = summarizeHeadlessMatches(matches);
  const totalActionCount = matches.reduce((sum, match) => sum + match.actionCount, 0);
  return { ...summary, requestedWorkerCount: requested, effectiveWorkerCount: effective, elapsedMs, matchesPerSecond: input.matchCount / (elapsedMs / 1000), totalActionCount, actionsPerSecond: totalActionCount / (elapsedMs / 1000), startProcessRssBytes, peakProcessRssBytes, endProcessRssBytes, peakProcessRssDeltaBytes: peakProcessRssBytes - startProcessRssBytes, workerMemory };
}

export function getAvailableHeadlessParallelism() { return availableParallelism(); }
