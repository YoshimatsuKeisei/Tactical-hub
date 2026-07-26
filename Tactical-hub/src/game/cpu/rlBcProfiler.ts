import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { getRlImitationEpisodeFeatureSpec, type RlImitationEpisode, type RlReplayPrefixProfile } from "./rlImitationCollector";
import type { RlBcProfileWorkerRequest, RlBcProfileWorkerResponse } from "./rlBcProfileWorkerMessages";
import { PythonBcTrainerClient } from "./pythonBcTrainerClient";
import { RL_PROJECT_ROOT, RL_VITE_NODE_ENTRY } from "./rlProjectPaths";
import { getDefaultRlReplaySidecarDirectory } from "./rlReplayRngSidecar";
import { readRlImitationEpisodes } from "./rlReplayReader";
import type { RlTorchDevice } from "./rlTorchDevice";

type SectionName = keyof RlReplayPrefixProfile
  | "sidecarLoadMs"
  | "workerBatchQueueAndProcessingMs"
  | "workerPackMs"
  | "workerSendPackedMs"
  | "nodePythonRoundTripMs"
  | "nodePackMs"
  | "pythonRoundTripAfterPackMs"
  | "parentPythonRoundTripMs"
  | "pythonDeserializeMs"
  | "pythonBinaryDecodeMs"
  | "pythonTensorPreparationMs"
  | "pythonForwardMs"
  | "pythonLossMs"
  | "pythonBackwardMs"
  | "pythonOptimizerStepMs";

export type RlBcProfileResult = {
  profileEpisodeCount: number;
  episodeNumbers: number[];
  samplesPerEpisode: Array<{ episodeNumber: number; warmupSamples: number; measuredSamples: number }>;
  requestedWorkers: number;
  effectiveWorkers: number;
  requestedSamples: number;
  warmupSamples: number;
  measuredSamples: number;
  batchSize: number;
  batchCount: number;
  elapsedMs: number;
  samplesPerSecond: number;
  batchesPerSecond: number;
  selectedDevice: "cpu" | "cuda";
  workerParentPackedPayloadBytes: number;
  workerParentPayloadDescription: string;
  overlapWarning: string;
  sections: Array<{ name: SectionName; totalMs: number; msPerSample: number; percentOfElapsed: number }>;
};

const replayKeys: Array<keyof RlReplayPrefixProfile> = [
  "getObservationMs", "getLegalActionsMs", "encodeObservationMs", "encodeLegalActionsMs", "stepReplayActionMs",
];

export async function runRlBcShortProfile(input: {
  dataPath: string;
  samples: number;
  warmupSamples: number;
  batchSize: number;
  workerCount: number;
  profileEpisodeCount?: number;
  learningRate?: number;
  seed?: number;
  pythonCommand?: string;
  device?: RlTorchDevice;
  torchThreads?: number;
  torchInteropThreads?: number;
}): Promise<RlBcProfileResult> {
  for (const [name, value, allowZero] of [
    ["samples", input.samples, false],
    ["warmupSamples", input.warmupSamples, true],
    ["batchSize", input.batchSize, false],
    ["workerCount", input.workerCount, false],
    ["profileEpisodeCount", input.profileEpisodeCount ?? 4, false],
  ] as const) {
    if (!Number.isInteger(value) || value < (allowZero ? 0 : 1)) throw new Error(`${name} must be ${allowZero ? "non-negative" : "positive"}`);
  }
  const profileEpisodeCount = input.profileEpisodeCount ?? 4;
  const episodes: Array<{ episodeNumber: number; episode: RlImitationEpisode }> = [];
  for await (const item of readRlImitationEpisodes(input.dataPath, { from: 1, to: Number.MAX_SAFE_INTEGER })) {
    episodes.push(item);
    if (episodes.length >= profileEpisodeCount) break;
  }
  if (episodes.length < profileEpisodeCount) {
    throw new Error(`Replay contains only ${episodes.length} episodes; ${profileEpisodeCount} profile episodes required`);
  }

  const client = new PythonBcTrainerClient();
  await client.start({
    featureSpec: getRlImitationEpisodeFeatureSpec(episodes[0].episode),
    learningRate: input.learningRate ?? 1e-4,
    seed: input.seed ?? 1,
    command: input.pythonCommand,
    device: input.device ?? "auto",
    torchThreads: input.torchThreads,
    torchInteropThreads: input.torchInteropThreads,
  });
  const selectedDevice = client.getAppliedThreads().selectedDevice;
  const totals = Object.fromEntries([
    ...replayKeys,
    "sidecarLoadMs", "workerBatchQueueAndProcessingMs", "workerPackMs", "workerSendPackedMs", "nodePythonRoundTripMs",
    "nodePackMs", "pythonRoundTripAfterPackMs", "parentPythonRoundTripMs", "pythonDeserializeMs", "pythonBinaryDecodeMs",
    "pythonTensorPreparationMs", "pythonForwardMs", "pythonLossMs", "pythonBackwardMs", "pythonOptimizerStepMs",
  ].map((name) => [name, 0])) as Record<SectionName, number>;
  let measured = 0;
  let batchCount = 0;
  let workerParentPackedPayloadBytes = 0;
  let measurementStartedAt: number | undefined;
  const workers: ChildProcess[] = [];
  type ProfileBatchMessage = Extract<RlBcProfileWorkerResponse, { type: "profileBatch" }>;
  type ProfileTask = {
    taskId: string;
    workerId: number;
    episodeIndex: number;
    warmupRemaining: number;
    measuredRemaining: number;
    nextSequence: number;
    batches: Map<number, { raw: ProfileBatchMessage; receivedAt: number; worker: ChildProcess }>;
    measuredRatios: Map<number, number>;
    sendTimings: Map<number, number>;
    assigned: boolean;
  };
  const effectiveWorkers = Math.min(input.workerCount, profileEpisodeCount);
  const workerEntry = fileURLToPath(new URL("./rlBcProfileWorker.ts", import.meta.url));
  let settled = false;
  let processing = Promise.resolve();

  const addReplayTimings = (profile: RlReplayPrefixProfile, ratio: number) => {
    for (const key of replayKeys) totals[key] += profile[key] * ratio;
  };
  const distribute = (total: number) => Array.from(
    { length: profileEpisodeCount },
    (_, index) => Math.floor(total / profileEpisodeCount) + (index < total % profileEpisodeCount ? 1 : 0),
  );
  const warmupQuotas = distribute(input.warmupSamples);
  const measuredQuotas = distribute(input.samples);
  const tasks: ProfileTask[] = episodes.map((item, index) => ({
    taskId: `bc-profile-${item.episodeNumber}`,
    workerId: -1,
    episodeIndex: index,
    warmupRemaining: warmupQuotas[index],
    measuredRemaining: measuredQuotas[index],
    nextSequence: 0,
    batches: new Map(),
    measuredRatios: new Map(),
    sendTimings: new Map(),
    assigned: false,
  }));
  const taskById = new Map(tasks.map((task) => [task.taskId, task]));
  let nextTaskToAssign = 0;
  let currentTaskToProcess = 0;
  const processPackedBatch = async (packedBatch: ProfileBatchMessage["packedBatch"], measuredPart: boolean) => {
    if (!packedBatch.batchSize) return;
    if (measuredPart && measurementStartedAt === undefined) measurementStartedAt = performance.now();
    const result = await client.profilePackedBatch(packedBatch);
    if (!measuredPart) return;
    batchCount += 1;
    measured += packedBatch.batchSize;
    totals.nodePythonRoundTripMs += result.roundTripMs;
    totals.nodePackMs += result.nodePackMs;
    totals.pythonRoundTripAfterPackMs += result.pythonRoundTripAfterPackMs;
    totals.parentPythonRoundTripMs += result.pythonRoundTripAfterPackMs;
    totals.pythonDeserializeMs += result.deserializeMs;
    totals.pythonBinaryDecodeMs += result.binaryDecodeMs;
    totals.pythonTensorPreparationMs += result.timings.tensorPreparationMs;
    totals.pythonForwardMs += result.timings.forwardMs;
    totals.pythonLossMs += result.timings.lossMs;
    totals.pythonBackwardMs += result.timings.backwardMs;
    totals.pythonOptimizerStepMs += result.timings.optimizerStepMs;
  };

  try {
    await new Promise<void>((resolveRun, rejectRun) => {
      const cleanup = () => {
        for (const worker of workers) if (worker.connected) worker.send({ type: "shutdown" } satisfies RlBcProfileWorkerRequest);
        for (const worker of workers) worker.kill();
      };
      const finish = () => {
        if (settled) return;
        settled = true;
        cleanup();
        resolveRun();
      };
      const fail = (error: Error) => {
        if (settled) return;
        settled = true;
        cleanup();
        rejectRun(error);
      };
      const assign = (worker: ChildProcess, workerId: number) => {
        if (settled || nextTaskToAssign >= tasks.length) return;
        const task = tasks[nextTaskToAssign++];
        const item = episodes[task.episodeIndex];
        task.workerId = workerId;
        task.assigned = true;
        const maxDecisions = task.warmupRemaining + task.measuredRemaining;
        if (item.episode.end.decisionCount < maxDecisions) {
          fail(new Error(
            `Episode ${item.episodeNumber} has ${item.episode.end.decisionCount} decisions; worker ${workerId} needs ${maxDecisions}`,
          ));
          return;
        }
        worker.send({
          type: "runEpisode",
          taskId: task.taskId,
          episode: item.episode,
          batchSize: input.batchSize,
          warmupDecisions: task.warmupRemaining,
          maxDecisions,
          sidecarDirectory: getDefaultRlReplaySidecarDirectory(input.dataPath),
        } satisfies RlBcProfileWorkerRequest);
      };
      const drainInDeterministicOrder = async () => {
        while (!settled && currentTaskToProcess < tasks.length) {
          const task = tasks[currentTaskToProcess];
          const entry = task.batches.get(task.nextSequence);
          if (!entry) return;
          task.batches.delete(task.nextSequence);
          task.nextSequence += 1;
          const { raw, receivedAt, worker } = entry;
          const batchSize = raw.packedBatch.batchSize;
          const isWarmup = task.warmupRemaining > 0;
          if (isWarmup && batchSize > task.warmupRemaining) throw new Error(`Packed profile batch crosses warmup boundary for ${raw.taskId}`);
          if (!isWarmup && batchSize > task.measuredRemaining) throw new Error(`Packed profile batch exceeds measured quota for ${raw.taskId}`);
          if (isWarmup) task.warmupRemaining -= batchSize;
          else task.measuredRemaining -= batchSize;
          const measuredRatio = isWarmup ? 0 : 1;
          task.measuredRatios.set(raw.sequence, measuredRatio);
          const workerSendPackedMs = task.sendTimings.get(raw.sequence);
          if (workerSendPackedMs !== undefined) {
            totals.workerSendPackedMs += workerSendPackedMs * measuredRatio;
            task.sendTimings.delete(raw.sequence);
            task.measuredRatios.delete(raw.sequence);
          }
          await processPackedBatch(raw.packedBatch, !isWarmup);
          if (!isWarmup) {
            addReplayTimings(raw.replayTimings, measuredRatio);
            totals.sidecarLoadMs += raw.sidecarLoadMs * measuredRatio;
            totals.workerPackMs += raw.workerPackMs;
            workerParentPackedPayloadBytes += raw.workerParentPackedPayloadBytes;
          }
          totals.workerBatchQueueAndProcessingMs += performance.now() - receivedAt;
          worker.send({ type: "batchConsumed", taskId: raw.taskId, sequence: raw.sequence } satisfies RlBcProfileWorkerRequest);
        }
      };
      for (let workerId = 0; workerId < effectiveWorkers; workerId += 1) {
        const worker = fork(RL_VITE_NODE_ENTRY, [workerEntry], {
          cwd: RL_PROJECT_ROOT,
          stdio: ["ignore", "ignore", "ignore", "ipc"],
          serialization: "advanced",
        });
        workers.push(worker);
        worker.on("message", (raw: RlBcProfileWorkerResponse) => {
          if (settled) return;
          if (!raw?.taskId || !taskById.has(raw.taskId)) { fail(new Error(`BC profile worker ${workerId} sent an invalid task`)); return; }
          if (raw.type === "workerError") { fail(new Error(raw.error)); return; }
          if (raw.type === "profileBatchSendTiming") {
            const task = taskById.get(raw.taskId)!;
            processing = processing.then(() => {
              const ratio = task.measuredRatios.get(raw.sequence);
              if (ratio === undefined) {
                if (task.sendTimings.has(raw.sequence)) throw new Error(`Duplicate worker send timing ${raw.taskId}/${raw.sequence}`);
                task.sendTimings.set(raw.sequence, raw.workerSendPackedMs);
              } else {
                totals.workerSendPackedMs += raw.workerSendPackedMs * ratio;
                task.measuredRatios.delete(raw.sequence);
              }
            }).catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
            return;
          }
          if (raw.type === "episodeCompleted") {
            const task = taskById.get(raw.taskId);
            if (!task || task.warmupRemaining !== 0 || task.measuredRemaining !== 0) {
              fail(new Error(`BC profile worker ${workerId} completed before satisfying its sample quota`));
              return;
            }
            processing = processing.then(async () => {
              if (settled) return;
              if (tasks[currentTaskToProcess] !== task) {
                throw new Error(`Episode ${task.episodeIndex + 1} completed outside deterministic processing order`);
              }
              currentTaskToProcess += 1;
              assign(workers[workerId], workerId);
              await drainInDeterministicOrder();
              if (currentTaskToProcess === tasks.length && measured === input.samples) finish();
            }).catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
            return;
          }
          const task = taskById.get(raw.taskId)!;
          if (task.workerId !== workerId || task.batches.has(raw.sequence)) {
            fail(new Error(`BC profile worker ${workerId} sent an invalid or duplicate batch`));
            return;
          }
          task.batches.set(raw.sequence, { raw, receivedAt: performance.now(), worker });
          processing = processing.then(drainInDeterministicOrder)
            .catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
        });
        worker.on("error", (error) => fail(error));
        worker.on("exit", (code, signal) => { if (!settled) fail(new Error(`BC profile worker exited (code=${code}, signal=${signal})`)); });
        assign(worker, workerId);
      }
    });
  } finally {
    await client.close();
  }
  const elapsedMs = performance.now() - (measurementStartedAt ?? performance.now());
  return {
    profileEpisodeCount,
    episodeNumbers: episodes.map((item) => item.episodeNumber),
    samplesPerEpisode: episodes.map((item, index) => ({
      episodeNumber: item.episodeNumber,
      warmupSamples: warmupQuotas[index],
      measuredSamples: measuredQuotas[index],
    })),
    requestedWorkers: input.workerCount,
    effectiveWorkers,
    requestedSamples: input.samples,
    warmupSamples: input.warmupSamples,
    measuredSamples: measured,
    batchSize: input.batchSize,
    batchCount,
    elapsedMs,
    samplesPerSecond: measured / Math.max(elapsedMs / 1000, 0.001),
    batchesPerSecond: batchCount / Math.max(elapsedMs / 1000, 0.001),
    selectedDevice,
    workerParentPackedPayloadBytes,
    workerParentPayloadDescription: "Exact packed-v1 binary payload bytes; small IPC descriptors and advanced-serialization framing are excluded.",
    overlapWarning: "Nested timings overlap and do not sum to 100%. workerSendPackedMs ends at the child_process.send callback and includes binary serialization/copy/queue acceptance, so it is not pure transport latency. parentPythonRoundTripMs equals the packed-payload Python round trip; nodePackMs remains for compatibility and is zero on the worker-packed path.",
    sections: (Object.entries(totals) as Array<[SectionName, number]>)
      .map(([name, totalMs]) => ({ name, totalMs, msPerSample: totalMs / measured, percentOfElapsed: elapsedMs > 0 ? totalMs / elapsedMs * 100 : 0 }))
      .sort((left, right) => right.totalMs - left.totalMs),
  };
}
