import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { getRlImitationEpisodeFeatureSpec, type RlImitationEpisode, type RlReplayPrefixProfile } from "./rlImitationCollector";
import type { RlBcProfileWorkerRequest, RlBcProfileWorkerResponse } from "./rlBcProfileWorkerMessages";
import { PythonBcTrainerClient, type BcEncodedSample } from "./pythonBcTrainerClient";
import { RL_PROJECT_ROOT, RL_VITE_NODE_ENTRY } from "./rlProjectPaths";
import { getDefaultRlReplaySidecarDirectory } from "./rlReplayRngSidecar";
import { readRlImitationEpisodes } from "./rlReplayReader";
import type { RlTorchDevice } from "./rlTorchDevice";

type SectionName = keyof RlReplayPrefixProfile
  | "sidecarLoadMs"
  | "workerBatchQueueAndProcessingMs"
  | "nodePythonRoundTripMs"
  | "pythonDeserializeMs"
  | "pythonBinaryDecodeMs"
  | "pythonTensorPreparationMs"
  | "pythonForwardMs"
  | "pythonLossMs"
  | "pythonBackwardMs"
  | "pythonOptimizerStepMs";

export type RlBcProfileResult = {
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
  ] as const) {
    if (!Number.isInteger(value) || value < (allowZero ? 0 : 1)) throw new Error(`${name} must be ${allowZero ? "non-negative" : "positive"}`);
  }
  const totalNeeded = input.samples + input.warmupSamples;
  const episodes: Array<{ episodeNumber: number; episode: RlImitationEpisode }> = [];
  let available = 0;
  for await (const item of readRlImitationEpisodes(input.dataPath, { from: 1, to: Number.MAX_SAFE_INTEGER })) {
    episodes.push(item);
    available += item.episode.end.decisionCount;
    if (available >= totalNeeded && episodes.length >= input.workerCount) break;
  }
  if (available < totalNeeded) throw new Error(`Replay contains only ${available} decisions; ${totalNeeded} required`);

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
    "sidecarLoadMs", "workerBatchQueueAndProcessingMs", "nodePythonRoundTripMs", "pythonDeserializeMs", "pythonBinaryDecodeMs",
    "pythonTensorPreparationMs", "pythonForwardMs", "pythonLossMs", "pythonBackwardMs", "pythonOptimizerStepMs",
  ].map((name) => [name, 0])) as Record<SectionName, number>;
  let measured = 0;
  let batchCount = 0;
  let measurementStartedAt: number | undefined;
  const workers: ChildProcess[] = [];
  const pending = new Map<string, {
    workerId: number;
    episodeIndex: number;
    warmupRemaining: number;
    measuredRemaining: number;
  }>();
  const effectiveWorkers = Math.min(input.workerCount, episodes.length);
  const workerEntry = fileURLToPath(new URL("./rlBcProfileWorker.ts", import.meta.url));
  let settled = false;
  let processing = Promise.resolve();
  let warmupRemainingGlobally = input.warmupSamples;
  const deferredMeasured: Array<() => Promise<void>> = [];

  const addReplayTimings = (profile: RlReplayPrefixProfile, ratio: number) => {
    for (const key of replayKeys) totals[key] += profile[key] * ratio;
  };
  const distribute = (total: number) => Array.from(
    { length: effectiveWorkers },
    (_, index) => Math.floor(total / effectiveWorkers) + (index < total % effectiveWorkers ? 1 : 0),
  );
  const warmupQuotas = distribute(input.warmupSamples);
  const measuredQuotas = distribute(input.samples);
  const processSamples = async (samples: BcEncodedSample[], measuredPart: boolean) => {
    if (!samples.length) return;
    if (measuredPart && measurementStartedAt === undefined) measurementStartedAt = performance.now();
    const result = await client.profileBatch(samples);
    if (!measuredPart) return;
    batchCount += 1;
    measured += samples.length;
    totals.nodePythonRoundTripMs += result.roundTripMs;
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
        if (settled || workerId >= effectiveWorkers) return;
        const item = episodes[workerId];
        const taskId = `bc-profile-${item.episodeNumber}`;
        const warmupRemaining = warmupQuotas[workerId];
        const measuredRemaining = measuredQuotas[workerId];
        const maxDecisions = warmupRemaining + measuredRemaining;
        if (item.episode.end.decisionCount < maxDecisions) {
          fail(new Error(
            `Episode ${item.episodeNumber} has ${item.episode.end.decisionCount} decisions; worker ${workerId} needs ${maxDecisions}`,
          ));
          return;
        }
        pending.set(taskId, { workerId, episodeIndex: item.episodeNumber, warmupRemaining, measuredRemaining });
        worker.send({
          type: "runEpisode",
          taskId,
          episode: item.episode,
          batchSize: input.batchSize,
          maxDecisions,
          sidecarDirectory: getDefaultRlReplaySidecarDirectory(input.dataPath),
        } satisfies RlBcProfileWorkerRequest);
      };
      for (let workerId = 0; workerId < effectiveWorkers; workerId += 1) {
        const worker = fork(RL_VITE_NODE_ENTRY, [workerEntry], { cwd: RL_PROJECT_ROOT, stdio: ["ignore", "ignore", "ignore", "ipc"] });
        workers.push(worker);
        worker.on("message", (raw: RlBcProfileWorkerResponse) => {
          if (settled) return;
          if (!raw?.taskId || !pending.has(raw.taskId)) { fail(new Error(`BC profile worker ${workerId} sent an invalid task`)); return; }
          if (raw.type === "workerError") { fail(new Error(raw.error)); return; }
          if (raw.type === "episodeCompleted") {
            const task = pending.get(raw.taskId);
            if (!task || task.warmupRemaining !== 0 || task.measuredRemaining !== 0) {
              fail(new Error(`BC profile worker ${workerId} completed before satisfying its sample quota`));
              return;
            }
            pending.delete(raw.taskId);
            if (pending.size === 0 && measured === input.samples) finish();
            return;
          }
          const receivedAt = performance.now();
          processing = processing.then(async () => {
            if (settled) return;
            const task = pending.get(raw.taskId);
            if (!task) throw new Error(`Missing BC profile task ${raw.taskId}`);
            const warmup = raw.samples.slice(0, task.warmupRemaining);
            task.warmupRemaining -= warmup.length;
            const measuredSamples = raw.samples.slice(
              warmup.length,
              warmup.length + task.measuredRemaining,
            );
            task.measuredRemaining -= measuredSamples.length;
            if (warmup.length) {
              await processSamples(warmup, false);
              warmupRemainingGlobally -= warmup.length;
            }
            const processMeasured = async () => {
              if (!measuredSamples.length) return;
              const ratio = measuredSamples.length / raw.samples.length;
              addReplayTimings(raw.replayTimings, ratio);
              totals.sidecarLoadMs += raw.sidecarLoadMs * ratio;
              await processSamples(measuredSamples, true);
            };
            if (measuredSamples.length) {
              if (warmupRemainingGlobally === 0) {
                while (deferredMeasured.length) await deferredMeasured.shift()!();
                await processMeasured();
              } else {
                deferredMeasured.push(processMeasured);
              }
            }
            totals.workerBatchQueueAndProcessingMs += performance.now() - receivedAt;
            worker.send({ type: "batchConsumed", taskId: raw.taskId, sequence: raw.sequence } satisfies RlBcProfileWorkerRequest);
          }).catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
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
    overlapWarning: "Nested timings overlap and do not sum to 100%. workerBatchQueueAndProcessingMs measures parent queue plus downstream batch processing after IPC receipt; it is not pure IPC transport time.",
    sections: (Object.entries(totals) as Array<[SectionName, number]>)
      .map(([name, totalMs]) => ({ name, totalMs, msPerSample: totalMs / measured, percentOfElapsed: elapsedMs > 0 ? totalMs / elapsedMs * 100 : 0 }))
      .sort((left, right) => right.totalMs - left.totalMs),
  };
}
