import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { runHeuristicImitationEpisode } from "./rlImitationCollector";
import type { RlImitationWorkerRequest, RlImitationWorkerResponse } from "./rlImitationWorkerMessages";

declare const process: NodeJS.Process & { send?: (message: RlImitationWorkerResponse) => boolean };

function send(message: RlImitationWorkerResponse) {
  if (!process.send) throw new Error("RL imitation worker IPC channel is unavailable");
  process.send(message);
}

async function writeEpisode(message: Extract<RlImitationWorkerRequest, { type: "runEpisode" }>) {
  const output = createWriteStream(message.shardPath, { encoding: "utf8" });
  try {
    const result = await runHeuristicImitationEpisode({
      episodeId: message.episodeId,
      seed: message.seed,
      maxTurns: message.maxTurns,
      onRecord: async (record) => {
        if (!output.write(`${JSON.stringify(record)}\n`)) await once(output, "drain");
      },
    });
    output.end();
    await once(output, "finish");
    send({
      type: "episodeCompleted",
      taskId: message.taskId,
      episodeId: message.episodeId,
      seed: message.seed,
      result,
    });
  } catch (error) {
    output.destroy();
    send({
      type: "workerError",
      taskId: message.taskId,
      episodeId: message.episodeId,
      seed: message.seed,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  }
}

let active = false;
process.on("message", (message: RlImitationWorkerRequest) => {
  if (message.type === "shutdown") {
    process.disconnect?.();
    return;
  }
  if (active) {
    send({ type: "workerError", taskId: message.taskId, episodeId: message.episodeId, seed: message.seed, error: "Worker received a second task while busy" });
    return;
  }
  active = true;
  void writeEpisode(message).finally(() => {
    active = false;
  });
});

process.on("uncaughtException", (error) => send({ type: "workerError", error: `${error.name}: ${error.message}` }));
process.on("unhandledRejection", (error) => send({ type: "workerError", error: `Unhandled rejection: ${String(error)}` }));
