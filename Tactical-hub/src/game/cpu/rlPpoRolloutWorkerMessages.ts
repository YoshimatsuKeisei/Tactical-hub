import type { GameState } from "../types";
import type { RlResult } from "./rlEnvironment";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { PpoHyperparameters } from "./pythonPpoClient";
import type {
  PpoReplayRollout,
} from "./rlPpoSelfPlay";
import type {
  PpoTeamAdjudication,
  PpoTimeLimitReason,
} from "./rlPpoAdjudication";
import type { PackedTensorDescriptor } from "./rlBcPackedBatch";

export type PpoRolloutWorkerEpisodeSummary = {
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

export type PpoRolloutWorkerFinalized = {
  environmentIndex: number;
  summary: PpoRolloutWorkerEpisodeSummary;
  rollout?: PpoReplayRollout;
};

export type PpoRolloutWorkerPackedSample = {
  environmentIndex: number;
  decisionIndex: number;
  turnNumber: number;
  phase: GameState["phase"];
  teamId: string;
  progressHash: string;
  actionKeys: string[];
  packed: {
    batchSize: 1;
    tensors: PackedTensorDescriptor[];
    payload: ArrayBuffer;
  };
};

export type PpoRolloutWorkerTiming = {
  observationMs: number;
  legalActionsMs: number;
  encodeObservationMs: number;
  encodeActionsMs: number;
  packMs: number;
  gameStepMs: number;
};

export type PpoRolloutWorkerRequest =
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

export type PpoRolloutWorkerResponse =
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
      samples: PpoRolloutWorkerPackedSample[];
      finalized: PpoRolloutWorkerFinalized[];
      mergeLegalActionCount: number;
      timing: PpoRolloutWorkerTiming;
    }
  | {
      type: "applied";
      requestId: number;
      workerId: number;
      round: number;
      finalized: PpoRolloutWorkerFinalized[];
      timing: PpoRolloutWorkerTiming;
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
