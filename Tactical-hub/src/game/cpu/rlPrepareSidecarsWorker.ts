import { generateRlReplayRngSidecar } from "./rlImitationCollector";
import type { RlPrepareSidecarsWorkerRequest, RlPrepareSidecarsWorkerResponse } from "./rlPrepareSidecarsWorkerMessages";
import { loadOrCreateRlReplayRngSidecar } from "./rlReplayRngSidecar";

declare const process: NodeJS.Process & { send?: (message: RlPrepareSidecarsWorkerResponse) => boolean };

function send(message: RlPrepareSidecarsWorkerResponse) {
  if (!process.send) throw new Error("RNG sidecar worker IPC channel is unavailable");
  process.send(message);
}

let activeTaskId: string | undefined;

process.on("message", (message: RlPrepareSidecarsWorkerRequest) => {
  if (message.type === "shutdown") {
    process.disconnect?.();
    return;
  }
  if (activeTaskId) {
    send({ type: "workerError", taskId: message.taskId, episodeNumber: message.episodeNumber, error: `Worker is already processing ${activeTaskId}` });
    return;
  }
  activeTaskId = message.taskId;
  const started = performance.now();
  void loadOrCreateRlReplayRngSidecar(
    message.episode,
    message.sidecarDirectory,
    () => generateRlReplayRngSidecar(message.episode),
  ).then((result) => {
    send({
      type: "episodePrepared",
      taskId: message.taskId,
      episodeNumber: message.episodeNumber,
      generated: result.generated,
      elapsedMs: performance.now() - started,
      sidecarPath: result.path,
    });
  }).catch((error) => {
    send({
      type: "workerError",
      taskId: message.taskId,
      episodeNumber: message.episodeNumber,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  }).finally(() => {
    activeTaskId = undefined;
  });
});

process.on("uncaughtException", (error) => send({ type: "workerError", taskId: activeTaskId, error: `${error.name}: ${error.message}` }));
process.on("unhandledRejection", (error) => send({ type: "workerError", taskId: activeTaskId, error: `Unhandled rejection: ${String(error)}` }));
