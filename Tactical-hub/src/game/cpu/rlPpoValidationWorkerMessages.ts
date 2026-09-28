import type { PpoReplayValidationRollout } from "./rlPpoSelfPlay";

export type PpoValidationWorkerRequest =
  | {
      type: "validate";
      taskId: string;
      rollout: PpoReplayValidationRollout;
      memoryLogInterval: number;
      fastRlMovement: boolean;
      fastRlPhaseTransitions: boolean;
    }
  | {
      type: "shutdown";
    };

export type PpoValidationWorkerResponse =
  | {
      type: "validated";
      taskId: string;
      seed: number;
      sampleCount: number;
      elapsedMs: number;
    }
  | {
      type: "workerError";
      taskId?: string;
      seed?: number;
      error: string;
    };
