import type { HeadlessMatchOptions, HeadlessMatchResult } from "./headlessSimulation";

export type HeadlessPolicyName = "random" | "heuristic";
export type SerializableHeadlessMatchOptions = Omit<HeadlessMatchOptions, "policy" | "initialState" | "onProgress">;

export type HeadlessWorkerRequest =
  | { type: "runMatch"; taskId: string; options: SerializableHeadlessMatchOptions; policyName: HeadlessPolicyName }
  | { type: "shutdown" };

export type WorkerMemoryPeak = {
  peakHeapUsedBytes: number;
  peakHeapTotalBytes: number;
  peakExternalBytes: number;
  peakArrayBuffersBytes: number;
};

export type HeadlessWorkerResponse =
  | { type: "matchCompleted"; taskId: string; seed: number; result: HeadlessMatchResult; memory: WorkerMemoryPeak }
  | { type: "workerError"; taskId?: string; seed?: number; error: string };
