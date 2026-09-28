import { performance } from "node:perf_hooks";
import { parentPort } from "node:worker_threads";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
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
import { packPpoActInput } from "./rlPpoPackedBatch";
import type {
  PpoRolloutWorkerFinalized,
  PpoRolloutWorkerRequest,
  PpoRolloutWorkerResponse,
  PpoRolloutWorkerTiming,
} from "./rlPpoRolloutWorkerMessages";
import type { PpoHyperparameters } from "./pythonPpoClient";

if (!parentPort) {
  throw new Error("PPO rollout worker requires worker_threads parentPort");
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
  finished: boolean;
  reason?: string;
  pending?: PendingDecision;
};

let workerId = -1;
let featureSpec: RlFeatureSpecV2 | undefined;
let hyperparameters: PpoHyperparameters | undefined;
let safetyMaxTurns = 0;
let safetyMaxActions = 0;
let slots: WorkerSlot[] = [];

function timing(): PpoRolloutWorkerTiming {
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
  target: PpoRolloutWorkerTiming,
  key: keyof PpoRolloutWorkerTiming,
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
    { cpuStep: { rlInPlacePhaseTransitions: true } },
  );
}

async function finalizeSlot(
  slot: WorkerSlot,
): Promise<PpoRolloutWorkerFinalized | undefined> {
  if (slot.finished) return undefined;
  if (!hyperparameters) {
    throw new Error("PPO rollout worker is not initialized");
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
  const summary: PpoRolloutWorkerFinalized["summary"] = {
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

  slot.finished = true;
  slot.pending = undefined;

  return {
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
}

async function handleInit(
  message: Extract<PpoRolloutWorkerRequest, { type: "init" }>,
) {
  if (slots.length) {
    throw new Error("PPO rollout worker was initialized twice");
  }
  workerId = message.workerId;
  featureSpec = message.featureSpec;
  hyperparameters = message.hyperparameters;
  safetyMaxTurns = message.safetyMaxTurns;
  safetyMaxActions = message.safetyMaxActions;
  slots = message.environments.map((entry) => {
    const environment = createEnvironment();
    environment.reset(entry.seed, 4);
    return {
      environmentIndex: entry.environmentIndex,
      seed: entry.seed,
      environment,
      encoderCache: createRlObservationEncoderCache(),
      trajectory: [],
      finished: false,
    };
  });

  parentPort!.postMessage({
    type: "ready",
    requestId: message.requestId,
    workerId,
    environmentIndices: slots.map((slot) => slot.environmentIndex),
  } satisfies PpoRolloutWorkerResponse);
}

async function handlePrepare(
  message: Extract<PpoRolloutWorkerRequest, { type: "prepare" }>,
) {
  if (!featureSpec || !hyperparameters || workerId < 0) {
    throw new Error("PPO rollout worker is not initialized");
  }
  const stageTiming = timing();
  const finalized: PpoRolloutWorkerFinalized[] = [];
  const samples: Array<
    Extract<PpoRolloutWorkerResponse, { type: "prepared" }>["samples"][number]
  > = [];
  const transferList: ArrayBuffer[] = [];
  let mergeLegalActionCount = 0;

  for (const slot of slots) {
    if (slot.finished) continue;
    if (slot.pending) {
      throw new Error(
        `PPO rollout worker env=${slot.environmentIndex} has an unapplied decision`,
      );
    }

    if (slot.environment.isTerminal()) {
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
      continue;
    }

    const actor = slot.environment.getCurrentActorTeamId();
    if (!actor) {
      slot.reason = "no_actor";
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
      continue;
    }

    const observation = measure(
      stageTiming,
      "observationMs",
      () => slot.environment.getObservationForEncoding(actor),
    );
    const legal = measure(
      stageTiming,
      "legalActionsMs",
      () => slot.environment.getLegalActionsForEncoding(actor),
    );

    if (!legal.length) {
      slot.reason = "no_legal_actions";
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
      continue;
    }
    if (observation.turnNumber > safetyMaxTurns) {
      slot.reason = "safety_turn_limit";
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
      continue;
    }
    if (slot.trajectory.length >= safetyMaxActions) {
      slot.reason = "safety_action_limit";
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
      continue;
    }

    const encodedObservation = measure(
      stageTiming,
      "encodeObservationMs",
      () => encodeRlObservationV2(
        observation,
        slot.encoderCache,
      ),
    );
    const encodedActions = measure(
      stageTiming,
      "encodeActionsMs",
      () => encodeRlLegalActionsV2(
        observation,
        legal,
      ),
    );
    const packed = measure(
      stageTiming,
      "packMs",
      () => packPpoActInput(
        encodedObservation,
        encodedActions.actions,
        featureSpec!,
      ),
    );

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

    // A standalone transferable buffer avoids cloning the packed payload
    // through the worker_threads structured-clone path.
    const transferable = Uint8Array.from(
      packed.payload,
    ).buffer as ArrayBuffer;
    transferList.push(transferable);
    samples.push({
      environmentIndex: slot.environmentIndex,
      decisionIndex: pending.decisionIndex,
      turnNumber: pending.turnNumber,
      phase: pending.phase,
      teamId: pending.teamId,
      progressHash: pending.progressHash,
      actionKeys: pending.actionKeys,
      packed: {
        batchSize: 1,
        tensors: packed.tensors,
        payload: transferable,
      },
    });
  }

  parentPort!.postMessage({
    type: "prepared",
    requestId: message.requestId,
    workerId,
    round: message.round,
    samples,
    finalized,
    mergeLegalActionCount,
    timing: stageTiming,
  } satisfies PpoRolloutWorkerResponse, transferList);
}

async function handleApply(
  message: Extract<PpoRolloutWorkerRequest, { type: "apply" }>,
) {
  if (workerId < 0) {
    throw new Error("PPO rollout worker is not initialized");
  }
  const stageTiming = timing();
  const finalized: PpoRolloutWorkerFinalized[] = [];
  const actionByEnvironment = new Map(
    message.actions.map((action) => [
      action.environmentIndex,
      action,
    ]),
  );

  for (const slot of slots) {
    const pending = slot.pending;
    if (!pending) continue;

    const action = actionByEnvironment.get(
      slot.environmentIndex,
    );
    if (!action) {
      throw new Error(
        `PPO rollout worker missing action for env=${slot.environmentIndex}`,
      );
    }
    if (
      !Number.isInteger(action.actionIndex)
      || action.actionIndex < 0
      || action.actionIndex >= pending.actionKeys.length
      || pending.actionKeys[action.actionIndex] !== action.actionKey
    ) {
      throw new Error(
        `PPO rollout worker action mismatch env=${slot.environmentIndex}`,
      );
    }
    if (![action.logProbability, action.value].every(Number.isFinite)) {
      throw new Error(
        `PPO rollout worker non-finite action output env=${slot.environmentIndex}`,
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

    if (
      slot.environment.getProgressHash()
      === pending.progressHash
    ) {
      slot.reason = "phase_stall";
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
    } else if (slot.environment.isTerminal()) {
      const result = await finalizeSlot(slot);
      if (result) finalized.push(result);
    }
  }

  if (actionByEnvironment.size !== message.actions.length) {
    throw new Error("PPO rollout worker duplicate environment action");
  }

  parentPort!.postMessage({
    type: "applied",
    requestId: message.requestId,
    workerId,
    round: message.round,
    finalized,
    timing: stageTiming,
  } satisfies PpoRolloutWorkerResponse);
}

async function handle(
  message: PpoRolloutWorkerRequest,
) {
  if (message.type === "init") {
    await handleInit(message);
    return;
  }
  if (message.type === "prepare") {
    await handlePrepare(message);
    return;
  }
  if (message.type === "apply") {
    await handleApply(message);
    return;
  }
  if (message.type === "shutdown") {
    parentPort!.postMessage({
      type: "closed",
      requestId: message.requestId,
      workerId,
    } satisfies PpoRolloutWorkerResponse);
    parentPort!.close();
  }
}

parentPort.on(
  "message",
  (message: PpoRolloutWorkerRequest) => {
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
      } satisfies PpoRolloutWorkerResponse);
    });
  },
);
