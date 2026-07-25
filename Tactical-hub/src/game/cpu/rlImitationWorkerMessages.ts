import type { RlImitationEpisodeResult } from "./rlImitationCollector";

export type RlImitationWorkerRequest =
  | {
    type: "runEpisode";
    taskId: string;
    episodeId: string;
    seed: number;
    maxTurns: number;
    shardPath: string;
  }
  | { type: "shutdown" };

export type RlImitationWorkerResponse =
  | {
    type: "episodeCompleted";
    taskId: string;
    episodeId: string;
    seed: number;
    result: RlImitationEpisodeResult;
  }
  | {
    type: "workerError";
    taskId?: string;
    episodeId?: string;
    seed?: number;
    error: string;
  };
