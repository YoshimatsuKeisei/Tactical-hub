import { performance } from "node:perf_hooks";
import { validatePpoTrajectoryReplay } from "./rlPpoSelfPlay";
import type {
  PpoValidationWorkerRequest,
  PpoValidationWorkerResponse,
} from "./rlPpoValidationWorkerMessages";

declare const process: NodeJS.Process & {
  send?: (message: PpoValidationWorkerResponse) => boolean;
};

function send(message: PpoValidationWorkerResponse) {
  if (!process.send) {
    throw new Error("PPO validation worker IPC channel is unavailable");
  }
  process.send(message);
}

let activeTaskId: string | undefined;

process.on("message", (message: PpoValidationWorkerRequest) => {
  if (message.type === "shutdown") {
    process.disconnect?.();
    return;
  }

  if (activeTaskId) {
    send({
      type: "workerError",
      taskId: message.taskId,
      seed: message.rollout.seed,
      error: `PPO validation worker is already processing ${activeTaskId}`,
    });
    return;
  }

  activeTaskId = message.taskId;
  const started = performance.now();

  void validatePpoTrajectoryReplay({
    rollout: message.rollout,
    memoryLogInterval: message.memoryLogInterval,
    fastRlMovement: message.fastRlMovement,
    fastRlPhaseTransitions: message.fastRlPhaseTransitions,
  }).then((sampleCount) => {
    send({
      type: "validated",
      taskId: message.taskId,
      seed: message.rollout.seed,
      sampleCount,
      elapsedMs: performance.now() - started,
    });
  }).catch((error) => {
    send({
      type: "workerError",
      taskId: message.taskId,
      seed: message.rollout.seed,
      error: error instanceof Error
        ? `${error.name}: ${error.message}`
        : String(error),
    });
  }).finally(() => {
    activeTaskId = undefined;
  });
});

process.on("uncaughtException", (error) => {
  send({
    type: "workerError",
    taskId: activeTaskId,
    error: `${error.name}: ${error.message}`,
  });
});

process.on("unhandledRejection", (error) => {
  send({
    type: "workerError",
    taskId: activeTaskId,
    error: `Unhandled rejection: ${String(error)}`,
  });
});
