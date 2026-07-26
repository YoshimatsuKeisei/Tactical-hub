import { generateRlReplayRngSidecar, getRlImitationEpisodeFeatureSpec, replayHeuristicImitationEpisode } from "./rlImitationCollector";
import type { BcEncodedSample } from "./pythonBcTrainerClient";
import { packBcEncodedSamples } from "./rlBcPackedBatch";
import type { RlBcReplayWorkerRequest, RlBcReplayWorkerResponse } from "./rlBcReplayWorkerMessages";
import { loadOrCreateRlReplayRngSidecar } from "./rlReplayRngSidecar";

declare const process: NodeJS.Process & { send?: (message: RlBcReplayWorkerResponse) => boolean };

function send(message: RlBcReplayWorkerResponse) {
  if (!process.send) throw new Error("RL BC replay worker IPC channel is unavailable");
  process.send(message);
}

let activeTaskId: string | undefined;
let waitingAck: { taskId: string; sequence: number; resolve: () => void } | undefined;

async function runEpisode(message: Extract<RlBcReplayWorkerRequest, { type: "runEpisode" }>) {
  const featureSpec = getRlImitationEpisodeFeatureSpec(message.episode);
  let batch: BcEncodedSample[] = [];
  let batchSequence = 0;
  let sampleCount = 0;
  const flush = async () => {
    if (!batch.length) return;
    const samples = batch;
    batch = [];
    const sequence = batchSequence++;
    const packedBatch = packBcEncodedSamples(samples, featureSpec);
    const acknowledged = new Promise<void>((resolve) => {
      waitingAck = { taskId: message.taskId, sequence, resolve };
    });
    send({
      type: "encodedBatch",
      taskId: message.taskId,
      episodeNumber: message.episodeNumber,
      batchSequence: sequence,
      packedBatch,
    });
    await acknowledged;
    waitingAck = undefined;
  };
  const sidecarStarted = performance.now();
  const cached = await loadOrCreateRlReplayRngSidecar(
    message.episode,
    message.sidecarDirectory,
    () => generateRlReplayRngSidecar(message.episode),
  );
  const sidecarPreparationMs = performance.now() - sidecarStarted;
  const replayTiming = { directReplayMs: 0 };
  const replayResult = await replayHeuristicImitationEpisode({
    episode: message.episode,
    rngSidecar: cached.sidecar,
    timing: replayTiming,
    onEncodedDecision: async (decision) => {
      if (decision.encodedLegalActions.actionKeys[decision.selectedActionIndex] !== decision.record.selectedActionKey) {
        throw new Error(`Replay target index mismatch for ${decision.record.selectedActionKey}`);
      }
      batch.push({
        observation: decision.encodedObservation,
        actions: decision.encodedLegalActions.actions,
        targetIndex: decision.selectedActionIndex,
      });
      sampleCount += 1;
      if (batch.length >= message.batchSize) await flush();
    },
  });
  const directReplayMs = replayTiming.directReplayMs;
  await flush();
  send({
    type: "episodeCompleted",
    taskId: message.taskId,
    episodeNumber: message.episodeNumber,
    sampleCount,
    replayResult,
    sidecarGenerated: cached.generated,
    sidecarPreparationMs,
    directReplayMs,
  });
}

process.on("message", (message: RlBcReplayWorkerRequest) => {
  if (message.type === "shutdown") {
    process.disconnect?.();
    return;
  }
  if (message.type === "batchConsumed") {
    if (!waitingAck || waitingAck.taskId !== message.taskId || waitingAck.sequence !== message.batchSequence) {
      send({ type: "workerError", taskId: message.taskId, error: `Unexpected batchConsumed ${message.batchSequence}` });
      return;
    }
    waitingAck.resolve();
    return;
  }
  if (activeTaskId) {
    send({ type: "workerError", taskId: message.taskId, episodeNumber: message.episodeNumber, error: `Worker is already processing ${activeTaskId}` });
    return;
  }
  activeTaskId = message.taskId;
  void runEpisode(message)
    .catch((error) => {
      send({
        type: "workerError",
        taskId: message.taskId,
        episodeNumber: message.episodeNumber,
        error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
      });
    })
    .finally(() => {
      activeTaskId = undefined;
    });
});

process.on("uncaughtException", (error) => send({ type: "workerError", taskId: activeTaskId, error: `${error.name}: ${error.message}` }));
process.on("unhandledRejection", (error) => send({ type: "workerError", taskId: activeTaskId, error: `Unhandled rejection: ${String(error)}` }));
