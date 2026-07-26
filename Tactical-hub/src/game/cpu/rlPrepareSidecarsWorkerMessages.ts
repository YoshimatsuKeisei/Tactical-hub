import type { RlImitationEpisode } from "./rlImitationCollector";

export type RlPrepareSidecarsWorkerRequest =
  | {
    type: "prepareEpisode";
    taskId: string;
    episodeNumber: number;
    episode: RlImitationEpisode;
    sidecarDirectory: string;
  }
  | { type: "shutdown" };

export type RlPrepareSidecarsWorkerResponse =
  | {
    type: "episodePrepared";
    taskId: string;
    episodeNumber: number;
    generated: boolean;
    elapsedMs: number;
    sidecarPath: string;
  }
  | {
    type: "workerError";
    taskId?: string;
    episodeNumber?: number;
    error: string;
  };
