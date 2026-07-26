import type { RlImitationEpisode, RlReplayPrefixProfile } from "./rlImitationCollector";
import type { PackedBcBatch } from "./rlBcPackedBatch";

export type RlBcProfileWorkerRequest =
  | {
    type: "runEpisode";
    taskId: string;
    episode: RlImitationEpisode;
    batchSize: number;
    warmupDecisions: number;
    maxDecisions: number;
    sidecarDirectory: string;
  }
  | { type: "batchConsumed"; taskId: string; sequence: number }
  | { type: "shutdown" };

export type RlBcProfileWorkerResponse =
  | {
    type: "profileBatch";
    taskId: string;
    sequence: number;
    packedBatch: PackedBcBatch;
    replayTimings: RlReplayPrefixProfile;
    sidecarLoadMs: number;
    workerPackMs: number;
    workerParentPackedPayloadBytes: number;
  }
  | { type: "profileBatchSendTiming"; taskId: string; sequence: number; workerSendPackedMs: number }
  | { type: "episodeCompleted"; taskId: string }
  | { type: "workerError"; taskId?: string; error: string };
