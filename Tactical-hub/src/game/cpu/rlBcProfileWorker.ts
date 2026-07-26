import { generateRlReplayRngSidecar, replayRlImitationEpisodePrefix, type RlReplayPrefixProfile } from "./rlImitationCollector";
import type { BcEncodedSample } from "./pythonBcTrainerClient";
import type { RlBcProfileWorkerRequest, RlBcProfileWorkerResponse } from "./rlBcProfileWorkerMessages";
import { loadOrCreateRlReplayRngSidecar } from "./rlReplayRngSidecar";

declare const process: NodeJS.Process & { send?: (message: RlBcProfileWorkerResponse) => boolean };
const zeroProfile = (): RlReplayPrefixProfile => ({ getObservationMs: 0, getLegalActionsMs: 0, encodeObservationMs: 0, encodeLegalActionsMs: 0, stepReplayActionMs: 0 });
const subtract = (current: RlReplayPrefixProfile, previous: RlReplayPrefixProfile): RlReplayPrefixProfile => Object.fromEntries(
  Object.keys(current).map((key) => [key, current[key as keyof RlReplayPrefixProfile] - previous[key as keyof RlReplayPrefixProfile]]),
) as RlReplayPrefixProfile;
function send(message: RlBcProfileWorkerResponse) {
  if (!process.send) throw new Error("BC profile worker IPC channel is unavailable");
  process.send(message);
}

let activeTaskId: string | undefined;
let waitingAck: { taskId: string; sequence: number; started: number; resolve: () => void } | undefined;

async function runEpisode(message: Extract<RlBcProfileWorkerRequest, { type: "runEpisode" }>) {
  const sidecarStarted = performance.now();
  const cached = await loadOrCreateRlReplayRngSidecar(
    message.episode,
    message.sidecarDirectory,
    () => generateRlReplayRngSidecar(message.episode),
  );
  const sidecarLoadMs = performance.now() - sidecarStarted;
  let batch: BcEncodedSample[] = [];
  let sequence = 0;
  let previousProfile = zeroProfile();
  let currentProfile = zeroProfile();
  const flush = async () => {
    if (!batch.length) return;
    const samples = batch;
    batch = [];
    const currentSequence = sequence++;
    const replayTimings = subtract(currentProfile, previousProfile);
    previousProfile = { ...currentProfile };
    const acknowledged = new Promise<void>((resolve) => {
      waitingAck = { taskId: message.taskId, sequence: currentSequence, started: performance.now(), resolve };
    });
    send({ type: "profileBatch", taskId: message.taskId, sequence: currentSequence, samples, replayTimings, sidecarLoadMs: currentSequence === 0 ? sidecarLoadMs : 0 });
    await acknowledged;
    waitingAck = undefined;
  };
  await replayRlImitationEpisodePrefix({
    episode: message.episode,
    rngSidecar: cached.sidecar,
    onEncodedDecision: (decision) => {
      batch.push({ observation: decision.encodedObservation, actions: decision.encodedLegalActions.actions, targetIndex: decision.selectedActionIndex });
    },
    onDecisionCompleted: async (profile) => {
      currentProfile = profile;
      if (batch.length >= message.batchSize) await flush();
    },
  });
  await flush();
  send({ type: "episodeCompleted", taskId: message.taskId });
}

process.on("message", (message: RlBcProfileWorkerRequest) => {
  if (message.type === "shutdown") { process.disconnect?.(); return; }
  if (message.type === "batchConsumed") {
    if (!waitingAck || waitingAck.taskId !== message.taskId || waitingAck.sequence !== message.sequence) {
      send({ type: "workerError", taskId: message.taskId, error: `Unexpected batchConsumed ${message.sequence}` });
      return;
    }
    waitingAck.resolve();
    return;
  }
  if (activeTaskId) { send({ type: "workerError", taskId: message.taskId, error: `Worker is already processing ${activeTaskId}` }); return; }
  activeTaskId = message.taskId;
  void runEpisode(message).catch((error) => {
    send({ type: "workerError", taskId: message.taskId, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  }).finally(() => { activeTaskId = undefined; });
});
