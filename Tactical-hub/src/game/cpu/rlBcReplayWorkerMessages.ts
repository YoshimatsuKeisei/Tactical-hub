import type { RlImitationEpisode, RlImitationEpisodeEnd } from "./rlImitationCollector";
import type { BcEncodedSample } from "./pythonBcTrainerClient";

export type RlBcReplayWorkerRequest =
  | {
    type: "runEpisode";
    taskId: string;
    episodeNumber: number;
    episode: RlImitationEpisode;
    batchSize: number;
    sidecarDirectory: string;
  }
  | {
    type: "batchConsumed";
    taskId: string;
    batchSequence: number;
  }
  | { type: "shutdown" };

export type RlBcReplayWorkerResponse =
  | {
    type: "encodedBatch";
    taskId: string;
    episodeNumber: number;
    batchSequence: number;
    samples: BcEncodedSample[];
  }
  | {
    type: "episodeCompleted";
    taskId: string;
    episodeNumber: number;
    sampleCount: number;
    replayResult: RlImitationEpisodeEnd;
    sidecarGenerated: boolean;
    sidecarPreparationMs: number;
    directReplayMs: number;
  }
  | {
    type: "workerError";
    taskId?: string;
    episodeNumber?: number;
    error: string;
  };
