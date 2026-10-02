import { performance } from "node:perf_hooks";
import { parentPort } from "node:worker_threads";
import { encodeRlLegalActionsSparseV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationCompactPackedMapV2,
} from "./rlObservationEncoder";
import {
  adjudicatePpoTimeLimit,
  isPpoTimeLimitReason,
} from "./rlPpoAdjudication";
import {
  finalizeTerminalTrajectory,
  finalizeVictoryTrajectory,
  type PpoTrajectoryStep,
} from "./rlPpoSelfPlay";
import type {
  PpoRolloutWorkerV7Finalized,
  PpoRolloutWorkerV7Request,
  PpoRolloutWorkerV7Response,
  PpoRolloutWorkerV7Timing,
} from "./rlPpoRolloutWorkerV7Messages";
import type { PpoHyperparameters } from "./pythonPpoClient";
import { packPpoActBatchInput } from "./rlPpoPackedBatch";
import { toTransferablePackedBcBatch } from "./rlPpoWorkerPackedV7";
import { createPpoRolloutEnvironmentDiagnosticV8 } from "./rlPpoRolloutDiagnosticsV8";

if (!parentPort) {
  throw new Error("PPO V7 rollout worker requires worker_threads parentPort");
}

type PendingDecision = {
  decisionIndex: number;
  turnNumber: number;
  phase: PpoTrajectoryStep["phase"];
  teamId: string;
  progressHash: string;
  actionKeys: string[];
};

type WorkerSlot = {
  environmentIndex: number;
  seed: number;
  environment: RlEnvironmentV2;
  encoderCache: ReturnType<typeof createRlObservationEncoderCache>;
  trajectory: PpoTrajectoryStep[];
  generation: number;
  finished: boolean;
  reason?: string;
  pending?: PendingDecision;
};

let workerId = -1;
let featureSpec: RlFeatureSpecV2 | undefined;
let hyperparameters: PpoHyperparameters | undefined;
let safetyMaxTurns = 0;
let safetyMaxActions = 0;
let autoRecycle = false;
let recycleSeedStride = 1;
let slots: WorkerSlot[] = [];

function timing(): PpoRolloutWorkerV7Timing {
  return {
    observationMs: 0,
    legalActionsMs: 0,
    encodeObservationMs: 0,
    encodeActionsMs: 0,
    packMs: 0,
    gameStepMs: 0,
  };
}

function measure<T>(
  target: PpoRolloutWorkerV7Timing,
  key: keyof PpoRolloutWorkerV7Timing,
  operation: () => T,
): T {
  const started = performance.now();
  try {
    return operation();
  } finally {
    target[key] += performance.now() - started;
  }
}

function createEnvironment() {
  return new RlEnvironmentV2(
    undefined,
    true,
    {
      cpuStep: {
        rlInPlacePhaseTransitions: true,
        rlPrevalidatedMovement: true,
        rlInPlaceProduction: true,
      },
    },
  );
}

async function finalizeSlot(
  slot: WorkerSlot,
): Promise<PpoRolloutWorkerV7Finalized | undefined> {
  if (slot.finished) return undefined;
  if (!hyperparameters) {
    throw new Error("PPO V7 rollout worker is not initialized");
  }

  let reason = slot.reason;
  const result = slot.environment.getResult();
  if (
    result.endReason === "stopped"
    && (!reason || isPpoTimeLimitReason(reason))
  ) {
    reason = "stopped";
  }

  if (
    (
      !reason
      && result.terminal
      && result.endReason === "victory"
    )
    || (
      isPpoTimeLimitReason(reason)
      && !result.terminal
      && result.endReason === "ongoing"
    )
  ) {
    const { checkHeadlessInvariants } = await import("./headlessSimulation");
    const violations = checkHeadlessInvariants(
      slot.environment.getStateForValidation(),
    );
    if (violations.length) {
      reason = `invariant_violation:${violations.join(" | ")}`;
    }
  }

  const finalStateHash = slot.environment.getStateHash();
  const summary: PpoRolloutWorkerV7Finalized["summary"] = {
    environmentIndex: slot.environmentIndex,
    seed: slot.seed,
    decisionCount: slot.trajectory.length,
    environmentResult: result,
    finalStateHash,
    outcomeKind: "abnormal_truncated",
    reason: reason ?? result.endReason,
  };

  if (
    !reason
    && result.terminal
    && result.endReason === "victory"
    && result.winnerTeamId
  ) {
    finalizeVictoryTrajectory(
      slot.trajectory,
      result.rewards,
      hyperparameters,
    );
    summary.outcomeKind = "victory";
  } else if (
    isPpoTimeLimitReason(reason)
    && !result.terminal
    && result.endReason === "ongoing"
  ) {
    summary.outcomeKind = "time_limit_adjudicated";
    summary.limitReason = reason;
    summary.adjudication = adjudicatePpoTimeLimit(
      slot.environment.getStateForValidation(),
    );
    finalizeTerminalTrajectory(
      slot.trajectory,
      Object.fromEntries(
        summary.adjudication.map((team) => [
          team.teamId,
          team.reward,
        ]),
      ),
      hyperparameters,
    );
  }

  const finalized: PpoRolloutWorkerV7Finalized = {
    environmentIndex: slot.environmentIndex,
    summary,
    rollout: summary.outcomeKind === "abnormal_truncated"
      ? undefined
      : {
          seed: slot.seed,
          outcomeKind: summary.outcomeKind,
          limitReason: summary.limitReason,
          adjudication: summary.adjudication,
          terminal: result.terminal,
          endReason: result.endReason,
          winnerTeamId: result.winnerTeamId,
          loserTeamIds: result.loserTeamIds,
          finalStateHash,
          trajectory: slot.trajectory,
        },
  };

  if (autoRecycle && slot.pending) {
    throw new Error(
      "PPO V7 rollout worker cannot recycle env="
      + slot.environmentIndex
      + " with an unapplied action",
    );
  }
  slot.finished = true;
  slot.pending = undefined;

  if (autoRecycle) {
    slot.seed += recycleSeedStride;
    slot.environment.reset(slot.seed, 4);
    slot.encoderCache = createRlObservationEncoderCache();
    slot.trajectory = [];
    slot.generation += 1;
    slot.finished = false;
    slot.reason = undefined;
    slot.pending = undefined;
  }

  return finalized;
}

async function handleInit(
  message: Extract<PpoRolloutWorkerV7Request, { type: "init" }>,
) {
  if (slots.length) {
    throw new Error("PPO V7 rollout worker was initialized twice");
  }
  workerId = message.workerId;
  featureSpec = message.featureSpec;
  hyperparameters = message.hyperparameters;
  safetyMaxTurns = message.safetyMaxTurns;
  safetyMaxActions = message.safetyMaxActions;
  autoRecycle = message.autoRecycle ?? false;
  recycleSeedStride = message.recycleSeedStride ?? 1;
  if (!Number.isInteger(recycleSeedStride) || recycleSeedStride <= 0) {
    throw new Error(
      "PPO V7 rollout recycleSeedStride must be a positive integer",
    );
  }
  slots = message.environments.map((entry) => {
    const environment = createEnvironment();
    environment.reset(entry.seed, 4);
    return {
      environmentIndex: entry.environmentIndex,
      seed: entry.seed,
      environment,
      encoderCache: createRlObservationEncoderCache(),
      trajectory: [],
      generation: 0,
      finished: false,
    };
  });

  parentPort!.postMessage({
    type: "ready",
    requestId: message.requestId,
    workerId,
    environmentIndices: slots.map((slot) => slot.environmentIndex),
  } satisfies PpoRolloutWorkerV7Response);
}

async function prepareRound(
  round: number,
  stageTiming: PpoRolloutWorkerV7Timing,
) {
  if (!featureSpec || !hyperparameters || workerId < 0) {
    throw new Error("PPO rollout worker is not initialized");
  }
  const activeFeatureSpec = featureSpec;
  const finalized: PpoRolloutWorkerV7Finalized[] = [];
  const packedSamples: Parameters<typeof packPpoActBatchInput>[0] = [];
  const environmentIndices: number[] = [];
  const decisionIndices: number[] = [];
  const actionKeys: string[][] = [];
  let mergeLegalActionCount = 0;

  slotLoop: for (const slot of slots) {
    if (slot.finished) continue;
    if (slot.pending) throw new Error("Worker has unapplied env=" + slot.environmentIndex);
    while (true) {
      if (slot.environment.isTerminal()) {
        const item = await finalizeSlot(slot);
        if (item) finalized.push(item);
        if (slot.finished) continue slotLoop;
        continue;
      }
      const actor = slot.environment.getCurrentActorTeamId();
      if (!actor) {
        slot.reason = "no_actor";
        const item = await finalizeSlot(slot);
        if (item) finalized.push(item);
        if (slot.finished) continue slotLoop;
        continue;
      }
      const observation = measure(stageTiming, "observationMs",
        () => slot.environment.getObservationForEncoding(actor));
      const legal = measure(stageTiming, "legalActionsMs",
        () => slot.environment.getLegalActionsForEncoding(actor));
      if (!legal.length) {
        slot.reason = "no_legal_actions";
        const item = await finalizeSlot(slot);
        if (item) finalized.push(item);
        if (slot.finished) continue slotLoop;
        continue;
      }
      if (observation.turnNumber > safetyMaxTurns) {
        slot.reason = "safety_turn_limit";
        const item = await finalizeSlot(slot);
        if (item) finalized.push(item);
        if (slot.finished) continue slotLoop;
        continue;
      }
      if (slot.trajectory.length >= safetyMaxActions) {
        slot.reason = "safety_action_limit";
        const item = await finalizeSlot(slot);
        if (item) finalized.push(item);
        if (slot.finished) continue slotLoop;
        continue;
      }

      const encodedObservation = measure(stageTiming, "encodeObservationMs",
        () => encodeRlObservationCompactPackedMapV2(observation, slot.encoderCache));
      const encodedActions = measure(stageTiming, "encodeActionsMs",
        () => encodeRlLegalActionsSparseV2(observation, legal));
      mergeLegalActionCount += legal.filter(
        (action) => action.actionType === "merge_infantry",
      ).length;

      const pending: PendingDecision = {
        decisionIndex: slot.trajectory.length,
        turnNumber: observation.turnNumber,
        phase: observation.phase,
        teamId: actor,
        progressHash: slot.environment.getProgressHash(),
        actionKeys: encodedActions.actionKeys,
      };
      slot.pending = pending;
      packedSamples.push({
        observation: encodedObservation,
        sparseActions: encodedActions.sparseActions,
      });
      environmentIndices.push(slot.environmentIndex);
      decisionIndices.push(pending.decisionIndex);
      actionKeys.push(encodedActions.actionKeys);
      continue slotLoop;
    }
  }

  const group = packedSamples.length ? {
    environmentIndices,
    decisionIndices,
    actionKeys,
    packed: measure(stageTiming, "packMs", () => toTransferablePackedBcBatch(
      packPpoActBatchInput(packedSamples, activeFeatureSpec, {
        compactMaskedPrefixes: true,
        sparseActions: true,
      }),
    )),
  } : undefined;

  return {
    round,
    group,
    finalized,
    mergeLegalActionCount,
  };
}

async function applyRound(
  actions: Extract<PpoRolloutWorkerV7Request, { type: "apply" }>["actions"],
  stageTiming: PpoRolloutWorkerV7Timing,
) {
  if (workerId < 0) {
    throw new Error("PPO rollout worker is not initialized");
  }
  const finalized: PpoRolloutWorkerV7Finalized[] = [];
  const actionByEnvironment = new Map(
    actions.map((action) => [action.environmentIndex, action]),
  );

  for (const slot of slots) {
    const pending = slot.pending;
    if (!pending) continue;

    const action = actionByEnvironment.get(slot.environmentIndex);
    if (!action) {
      throw new Error(
        "PPO rollout worker missing action for env=" + slot.environmentIndex,
      );
    }
    if (
      !Number.isInteger(action.actionIndex)
      || action.actionIndex < 0
      || action.actionIndex >= pending.actionKeys.length
      || pending.actionKeys[action.actionIndex] !== action.actionKey
    ) {
      throw new Error(
        "PPO rollout worker action mismatch env=" + slot.environmentIndex,
      );
    }
    if (![action.logProbability, action.value].every(Number.isFinite)) {
      throw new Error(
        "PPO rollout worker non-finite action output env=" + slot.environmentIndex,
      );
    }

    measure(
      stageTiming,
      "gameStepMs",
      () => slot.environment.stepWithoutObservation(action.actionKey),
    );
    slot.trajectory.push({
      decisionIndex: pending.decisionIndex,
      turnNumber: pending.turnNumber,
      phase: pending.phase,
      teamId: pending.teamId,
      selectedActionIndex: action.actionIndex,
      selectedActionKey: action.actionKey,
      oldLogProbability: action.logProbability,
      value: action.value,
      reward: 0,
      done: false,
    });
    slot.pending = undefined;

    if (slot.environment.getProgressHash() === pending.progressHash) {
      slot.reason = "phase_stall";
      const item = await finalizeSlot(slot);
      if (item) finalized.push(item);
    } else if (slot.environment.isTerminal()) {
      const item = await finalizeSlot(slot);
      if (item) finalized.push(item);
    }
  }

  if (actionByEnvironment.size !== actions.length) {
    throw new Error("PPO rollout worker duplicate environment action");
  }
  return finalized;
}

async function handlePrepare(
  message: Extract<PpoRolloutWorkerV7Request, { type: "prepare" }>,
) {
  const stageTiming = timing();
  const prepared = await prepareRound(message.round, stageTiming);
  const response = {
    type: "prepared",
    requestId: message.requestId,
    workerId,
    round: message.round,
    ...(prepared.group ? { group: prepared.group } : {}),
    finalized: prepared.finalized,
    mergeLegalActionCount: prepared.mergeLegalActionCount,
    timing: stageTiming,
  } satisfies PpoRolloutWorkerV7Response;
  parentPort!.postMessage(
    response,
    prepared.group
      ? [prepared.group.packed.payload.buffer as ArrayBuffer]
      : [],
  );
}

async function handleApply(
  message: Extract<PpoRolloutWorkerV7Request, { type: "apply" }>,
) {
  const stageTiming = timing();
  const finalized = await applyRound(message.actions, stageTiming);
  parentPort!.postMessage({
    type: "applied",
    requestId: message.requestId,
    workerId,
    round: message.round,
    finalized,
    timing: stageTiming,
  } satisfies PpoRolloutWorkerV7Response);
}

async function handleAdvance(
  message: Extract<PpoRolloutWorkerV7Request, { type: "advance" }>,
) {
  const stageTiming = timing();
  const appliedFinalized = await applyRound(message.actions, stageTiming);
  const prepared = await prepareRound(message.round + 1, stageTiming);
  const finalized = [...appliedFinalized, ...prepared.finalized];
  const response = {
    type: "advanced",
    requestId: message.requestId,
    workerId,
    round: message.round,
    ...(prepared.group ? { group: prepared.group } : {}),
    finalized,
    mergeLegalActionCount: prepared.mergeLegalActionCount,
    timing: stageTiming,
  } satisfies PpoRolloutWorkerV7Response;
  parentPort!.postMessage(
    response,
    prepared.group
      ? [prepared.group.packed.payload.buffer as ArrayBuffer]
      : [],
  );
}

async function handle(message: PpoRolloutWorkerV7Request) {
  if (message.type === "init") {
    await handleInit(message);
    return;
  }
  if (message.type === "prepare") {
    await handlePrepare(message);
    return;
  }
  if (message.type === "setAutoRecycle") {
    if (workerId < 0) {
      throw new Error("PPO rollout worker is not initialized");
    }
    autoRecycle = message.enabled;
    parentPort!.postMessage({
      type: "autoRecycleSet",
      requestId: message.requestId,
      workerId,
      enabled: autoRecycle,
    } satisfies PpoRolloutWorkerV7Response);
    return;
  }
  if (message.type === "getDiagnostics") {
    if (workerId < 0) {
      throw new Error("PPO rollout worker is not initialized");
    }
    parentPort!.postMessage({
      type: "diagnostics",
      requestId: message.requestId,
      workerId,
      environments: slots.map((slot) =>
        createPpoRolloutEnvironmentDiagnosticV8({
          environmentIndex: slot.environmentIndex,
          currentEpisodeSeed: slot.seed,
          generation: slot.generation,
          episodeDecisionCount: slot.trajectory.length,
          currentActorTeamId:
            slot.environment.getCurrentActorTeamId(),
          state: slot.environment.getStateForValidation(),
          result: slot.environment.getResult(),
        })),
    } satisfies PpoRolloutWorkerV7Response);
    return;
  }
  if (message.type === "apply") {
    await handleApply(message);
    return;
  }
  if (message.type === "advance") {
    await handleAdvance(message);
    return;
  }
  if (message.type === "shutdown") {
    parentPort!.postMessage({
      type: "closed",
      requestId: message.requestId,
      workerId,
    } satisfies PpoRolloutWorkerV7Response);
    parentPort!.close();
  }
}

parentPort.on(
  "message",
  (message: PpoRolloutWorkerV7Request) => {
    void handle(message).catch((error) => {
      parentPort!.postMessage({
        type: "workerError",
        requestId: "requestId" in message
          ? message.requestId
          : undefined,
        workerId,
        error: error instanceof Error
          ? `${error.name}: ${error.message}`
          : String(error),
      } satisfies PpoRolloutWorkerV7Response);
    });
  },
);
