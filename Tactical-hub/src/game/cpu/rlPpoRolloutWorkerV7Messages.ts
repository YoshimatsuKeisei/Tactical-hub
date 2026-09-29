import type { RlResult } from "./rlEnvironment";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { PpoHyperparameters } from "./pythonPpoClient";
import type {
  PpoReplayRollout,
} from "./rlPpoSelfPlay";
import type {
  TransferablePackedBcBatch,
} from "./rlPpoWorkerPackedV7";
import type {
  PpoTeamAdjudication,
  PpoTimeLimitReason,
} from "./rlPpoAdjudication";

export type PpoRolloutWorkerV7EpisodeSummary = {
  environmentIndex: number;
  seed: number;
  decisionCount: number;
  environmentResult: RlResult;
  finalStateHash: string;
  outcomeKind: PpoReplayRollout["outcomeKind"] | "abnormal_truncated";
  reason: string;
  limitReason?: PpoTimeLimitReason;
  adjudication?: PpoTeamAdjudication[];
};

export type PpoRolloutWorkerV7Finalized = {
  environmentIndex: number;
  summary: PpoRolloutWorkerV7EpisodeSummary;
  rollout?: PpoReplayRollout;
};

export type PpoRolloutWorkerV7PreparedSample = {
  environmentIndex: number;
  decisionIndex: number;
  progressHash: string;
  actionKeys: string[];
  packed: TransferablePackedBcBatch;
};

export type PpoRolloutWorkerV7Timing = {
  observationMs: number;
  legalActionsMs: number;
  encodeObservationMs: number;
  encodeActionsMs: number;
  packMs: number;
  gameStepMs: number;
};

export type PpoRolloutWorkerV7Request =
  | {
      type: "init";
      requestId: number;
      workerId: number;
      environments: Array<{
        environmentIndex: number;
        seed: number;
      }>;
      featureSpec: RlFeatureSpecV2;
      hyperparameters: PpoHyperparameters;
      safetyMaxTurns: number;
      safetyMaxActions: number;
    }
  | {
      type: "prepare";
      requestId: number;
      round: number;
    }
  | {
      type: "apply";
      requestId: number;
      round: number;
      actions: Array<{
        environmentIndex: number;
        actionIndex: number;
        actionKey: string;
        logProbability: number;
        value: number;
      }>;
    }
  | {
      type: "shutdown";
      requestId: number;
    };

export type PpoRolloutWorkerV7Response =
  | {
      type: "ready";
      requestId: number;
      workerId: number;
      environmentIndices: number[];
    }
  | {
      type: "prepared";
      requestId: number;
      workerId: number;
      round: number;
      samples: PpoRolloutWorkerV7PreparedSample[];
      finalized: PpoRolloutWorkerV7Finalized[];
      mergeLegalActionCount: number;
      timing: PpoRolloutWorkerV7Timing;
    }
  | {
      type: "applied";
      requestId: number;
      workerId: number;
      round: number;
      finalized: PpoRolloutWorkerV7Finalized[];
      timing: PpoRolloutWorkerV7Timing;
    }
  | {
      type: "closed";
      requestId: number;
      workerId: number;
    }
  | {
      type: "workerError";
      requestId?: number;
      workerId?: number;
      error: string;
    };
