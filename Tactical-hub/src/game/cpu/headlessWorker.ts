import { createHeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { runHeadlessMatch } from "./headlessSimulation";
import type { HeadlessWorkerRequest, HeadlessWorkerResponse, WorkerMemoryPeak } from "./headlessWorkerMessages";

declare const process: NodeJS.Process & { send?: (message: HeadlessWorkerResponse) => boolean };

function emptyPeak(): WorkerMemoryPeak {
  return { peakHeapUsedBytes: 0, peakHeapTotalBytes: 0, peakExternalBytes: 0, peakArrayBuffersBytes: 0 };
}

function sample(peak: WorkerMemoryPeak) {
  const memory = process.memoryUsage();
  peak.peakHeapUsedBytes = Math.max(peak.peakHeapUsedBytes, memory.heapUsed);
  peak.peakHeapTotalBytes = Math.max(peak.peakHeapTotalBytes, memory.heapTotal);
  peak.peakExternalBytes = Math.max(peak.peakExternalBytes, memory.external);
  peak.peakArrayBuffersBytes = Math.max(peak.peakArrayBuffersBytes, memory.arrayBuffers);
}

function send(message: HeadlessWorkerResponse) {
  const sender = process.send;
  if (!sender) throw new Error("Headless worker IPC channel is unavailable");
  sender.call(process, message);
}

process.on("message", (message: HeadlessWorkerRequest) => {
  if (message.type === "shutdown") { process.disconnect?.(); return; }
  const peak = emptyPeak();
  try {
    sample(peak);
    const policy = message.policyName === "heuristic" ? createHeuristicCpuPolicy() : undefined;
    const result = runHeadlessMatch({ ...message.options, policy, onProgress: () => sample(peak) });
    sample(peak);
    send({ type: "matchCompleted", taskId: message.taskId, seed: message.options.seed, result, memory: peak });
  } catch (caught) {
    send({ type: "workerError", taskId: message.taskId, seed: message.options.seed, error: caught instanceof Error ? `${caught.name}: ${caught.message}` : String(caught) });
  }
});

process.on("uncaughtException", (error) => send({ type: "workerError", error: `${error.name}: ${error.message}` }));
process.on("unhandledRejection", (error) => send({ type: "workerError", error: `Unhandled rejection: ${String(error)}` }));
