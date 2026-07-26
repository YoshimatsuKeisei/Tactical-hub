import type { RlImitationEpisode, RlReplayPrefixProfile } from "./rlImitationCollector";
import type { BcEncodedSample } from "./pythonBcTrainerClient";

export type RlBcProfileWorkerRequest =
  | {
    type: "runEpisode";
    taskId: string;
    episode: RlImitationEpisode;
    batchSize: number;
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
    samples: BcEncodedSample[];
    replayTimings: RlReplayPrefixProfile;
    sidecarLoadMs: number;
  }
  | { type: "episodeCompleted"; taskId: string }
  | { type: "workerError"; taskId?: string; error: string };
