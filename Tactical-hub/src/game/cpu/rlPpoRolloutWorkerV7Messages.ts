import type { RlResult } from "./rlEnvironment";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { PpoHyperparameters } from "./pythonPpoClient";
import type { PpoReplayRollout } from "./rlPpoSelfPlay";
import type { TransferablePackedBcBatch } from "./rlPpoWorkerPackedV7";
import type { PpoTeamAdjudication, PpoTimeLimitReason } from "./rlPpoAdjudication";
import type { PpoRolloutEnvironmentDiagnosticV8 } from "./rlPpoRolloutDiagnosticsV8";
import type {
  PpoDefeatDiagnosticEventV8,
  PpoDefeatEnvironmentSnapshotV8,
} from "./rlPpoDefeatDiagnosticsV8";
import {
  isPpoLearnableOutcomeKind,
  type PpoEpisodeOutcomeKind,
} from "./rlPpoTerminalOutcome";
import type { PpoTurnDiagnostics } from "./rlPpoTurnDiagnostics";
import type { PpoCombatDiagnostics } from "./rlPpoCombatDiagnostics";

export type PpoRolloutWorkerV7EpisodeSummary = {
  environmentIndex: number;
  seed: number;
  decisionCount: number;
  environmentResult: RlResult;
  finalStateHash: string;
  turnDiagnostics: PpoTurnDiagnostics;
  outcomeKind: PpoEpisodeOutcomeKind;
  reason: string;
  limitReason?: PpoTimeLimitReason;
  adjudication?: PpoTeamAdjudication[];
  combatDiagnostics?: PpoCombatDiagnostics;
};
export type PpoRolloutWorkerV7Finalized = {
  environmentIndex: number;
  summary: PpoRolloutWorkerV7EpisodeSummary;
  rollout?: PpoReplayRollout;
  /** Runtime-only terminal reward separation for optional diagnostics. */
  baseRewards?: Record<string, number>;
};

export function createPpoRolloutWorkerV7ReplayRollout(input: {
  summary: PpoRolloutWorkerV7EpisodeSummary;
  trajectory: PpoReplayRollout["trajectory"];
}): PpoReplayRollout | undefined {
  if (!isPpoLearnableOutcomeKind(input.summary.outcomeKind)) return undefined;
  const result = input.summary.environmentResult;
  return {
    seed: input.summary.seed,
    outcomeKind: input.summary.outcomeKind,
    limitReason: input.summary.limitReason,
    adjudication: input.summary.adjudication,
    terminal: result.terminal,
    endReason: result.endReason,
    winnerTeamId: result.winnerTeamId,
    loserTeamIds: result.loserTeamIds,
    finalStateHash: input.summary.finalStateHash,
    trajectory: input.trajectory,
  };
}
export type PpoRolloutWorkerV7PreparedGroup = {
  environmentIndices: number[];
  decisionIndices: number[];
  actionKeys: string[][];
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
  | { type: "init"; requestId: number; workerId: number; environments: Array<{environmentIndex:number; seed:number}>; featureSpec: RlFeatureSpecV2; hyperparameters: PpoHyperparameters; battleAdvantageShapingBeta: number; safetyMaxTurns: number; safetyMaxActions: number; autoRecycle?: boolean; recycleSeedStride?: number; defeatDiagnostics?: boolean }
  | { type: "setAutoRecycle"; requestId: number; enabled: boolean }
  | { type: "getDiagnostics"; requestId: number }
  | { type: "getDefeatDiagnostics"; requestId: number }
  | { type: "prepare"; requestId: number; round: number }
  | { type: "apply"; requestId: number; round: number; actions: Array<{environmentIndex:number; actionIndex:number; actionKey:string; logProbability:number; value:number}> }
  | { type: "advance"; requestId: number; round: number; actions: Array<{environmentIndex:number; actionIndex:number; actionKey:string; logProbability:number; value:number}> }
  | { type: "shutdown"; requestId: number };
export type PpoRolloutWorkerV7Response =
  | { type: "ready"; requestId: number; workerId: number; environmentIndices: number[] }
  | { type: "autoRecycleSet"; requestId: number; workerId: number; enabled: boolean }
  | { type: "diagnostics"; requestId: number; workerId: number; environments: PpoRolloutEnvironmentDiagnosticV8[] }
  | { type: "defeatDiagnostics"; requestId: number; workerId: number; environments: PpoDefeatEnvironmentSnapshotV8[] }
  | { type: "prepared"; requestId: number; workerId: number; round: number; group?: PpoRolloutWorkerV7PreparedGroup; finalized: PpoRolloutWorkerV7Finalized[]; mergeLegalActionCount: number; timing: PpoRolloutWorkerV7Timing }
  | { type: "applied"; requestId: number; workerId: number; round: number; finalized: PpoRolloutWorkerV7Finalized[]; timing: PpoRolloutWorkerV7Timing; defeatDiagnosticEvents?: PpoDefeatDiagnosticEventV8[] }
  | { type: "advanced"; requestId: number; workerId: number; round: number; group?: PpoRolloutWorkerV7PreparedGroup; finalized: PpoRolloutWorkerV7Finalized[]; mergeLegalActionCount: number; timing: PpoRolloutWorkerV7Timing; defeatDiagnosticEvents?: PpoDefeatDiagnosticEventV8[] }
  | { type: "closed"; requestId: number; workerId: number }
  | { type: "workerError"; requestId?: number; workerId?: number; error: string };
