import { performance } from "node:perf_hooks";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  PythonPpoClient,
  type PpoHyperparameters,
  type PpoRetentionStats,
} from "./pythonPpoClient";
import { DEFAULT_PPO_HYPERPARAMETERS } from "./rlPpoSelfPlay";
import {
  PpoRolloutWorkerV7Pool,
  type PpoRolloutWorkerV7WorkerPrepareResult,
} from "./rlPpoRolloutWorkerV7Pool";
import type {
  PpoRolloutWorkerV7EpisodeSummary,
  PpoRolloutWorkerV7Finalized,
  PpoRolloutWorkerV7Timing,
} from "./rlPpoRolloutWorkerV7Messages";
import { PpoDynamicInferenceBrokerV8 } from "./rlPpoDynamicInferenceBrokerV8";

export type PpoAsyncRolloutProbeV8Input = {
  seed: number;
  environmentCount?: number;
  initialCheckpoint: string;
  resume?: string;
  hyperparameters?: Partial<PpoHyperparameters>;
  safetyMaxTurns?: number;
  safetyMaxActions?: number;
  memoryLogInterval?: number;
  rolloutWorkerCount?: number;
  client?: PythonPpoClient;
};

function createFeatureSpecEnvironment() {
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

function retentionSummary(stats: PpoRetentionStats | undefined) {
  if (!stats) return undefined;
  return {
    ...stats,
    averageRawBytesPerSample: stats.storedSamples
      ? stats.rawBytes / stats.storedSamples
      : 0,
    averageCompressedBytesPerSample: stats.storedSamples
      ? stats.compressedBytes / stats.storedSamples
      : 0,
    compressionRatio: stats.compressedBytes
      ? stats.rawBytes / stats.compressedBytes
      : 0,
  };
}

function emptyWorkerTiming(): PpoRolloutWorkerV7Timing {
  return {
    observationMs: 0,
    legalActionsMs: 0,
    encodeObservationMs: 0,
    encodeActionsMs: 0,
    packMs: 0,
    gameStepMs: 0,
  };
}

function addWorkerTiming(
  target: PpoRolloutWorkerV7Timing,
  source: PpoRolloutWorkerV7Timing,
) {
  target.observationMs += source.observationMs;
  target.legalActionsMs += source.legalActionsMs;
  target.encodeObservationMs += source.encodeObservationMs;
  target.encodeActionsMs += source.encodeActionsMs;
  target.packMs += source.packMs;
  target.gameStepMs += source.gameStepMs;
}

export async function runPpoAsyncRolloutProbeV8(
  input: PpoAsyncRolloutProbeV8Input,
) {
  const environmentCount = input.environmentCount ?? 8;
  if (![8, 16, 32, 64].includes(environmentCount)) {
    throw new Error(
      "PPO async rollout V8 structural probe supports exactly 8, 16, 32, or 64 environments",
    );
  }

  const rolloutWorkerCount = input.rolloutWorkerCount ?? 4;
  const safetyMaxTurns = input.safetyMaxTurns ?? 1_000;
  const safetyMaxActions = input.safetyMaxActions ?? 100_000;
  const memoryLogInterval = input.memoryLogInterval ?? 5_000;
  for (const [name, value] of [
    ["rolloutWorkerCount", rolloutWorkerCount],
    ["safetyMaxTurns", safetyMaxTurns],
    ["safetyMaxActions", safetyMaxActions],
    ["memoryLogInterval", memoryLogInterval],
  ] as const) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${name} must be a positive integer`);
    }
  }

  const hyperparameters = {
    ...DEFAULT_PPO_HYPERPARAMETERS,
    ...input.hyperparameters,
  };
  const probe = createFeatureSpecEnvironment();
  const first = probe.reset(input.seed, 4);
  const featureSpec = createRlFeatureSpecV2(first);

  const client = input.client ?? new PythonPpoClient({
    device: "cuda",
    compactPaddedRows: true,
    env: {
      PPO_PACKED_PREPARE_MODE: "fast_batch_v2",
      PPO_SPARSE_ACTION_TRANSPORT: "1",
      PPO_RETENTION_STORAGE_MODE: "raw",
      PPO_PERSISTENT_ACT_H2D: "1",
      PPO_ACT_CUDA_GRAPH_HOT: "1",
      PPO_ACT_CUDA_GRAPH_MIN_HITS: "2",
      PPO_ACT_CUDA_GRAPH_MAX_ENTRIES: "32",
      PPO_PERSISTENT_REPLAY_H2D: "1",
      PPO_IMMUTABLE_RAW_RETENTION: "1",
    },
  });

  let rolloutPool: PpoRolloutWorkerV7Pool | undefined;
  const outstandingRetentionIds = new Set<string>();
  const summaryByEnvironment = new Map<
    number,
    PpoRolloutWorkerV7EpisodeSummary
  >();
  const workerCpuTiming = emptyWorkerTiming();
  const workerAdvanceOperationMsByWorker = Array.from(
    { length: rolloutWorkerCount },
    () => 0,
  );
  const workerRounds = Array.from(
    { length: rolloutWorkerCount },
    () => 0,
  );

  let totalDecisions = 0;
  let mergeLegalActionCount = 0;
  let initialPrepareMs = 0;
  let initialPrepareOperationMsSum = 0;

  const recordFinalized = (
    finalized: PpoRolloutWorkerV7Finalized[],
  ) => {
    for (const item of finalized) {
      if (summaryByEnvironment.has(item.environmentIndex)) {
        throw new Error(
          `PPO async rollout duplicate finalized env=${item.environmentIndex}`,
        );
      }
      summaryByEnvironment.set(
        item.environmentIndex,
        item.summary,
      );
    }
  };

  let broker: PpoDynamicInferenceBrokerV8 | undefined;

  try {
    const initialized = await client.start({
      seed: input.seed,
      featureSpec,
      hyperparameters,
      initialCheckpoint: input.initialCheckpoint,
      resume: input.resume,
    });
    if (initialized.selectedDevice !== "cuda") {
      throw new Error("PPO async rollout V8 probe requires CUDA");
    }

    const firstGameSeed = input.seed + initialized.episodeCount;
    const seeds = Array.from(
      { length: environmentCount },
      (_, environmentIndex) =>
        firstGameSeed + environmentIndex,
    );

    rolloutPool = await PpoRolloutWorkerV7Pool.create({
      workerCount: rolloutWorkerCount,
      environmentCount,
      firstGameSeed,
      featureSpec,
      hyperparameters,
      safetyMaxTurns,
      safetyMaxActions,
    });

    broker = new PpoDynamicInferenceBrokerV8(
      async (packed, actionKeys, flush) => {
        const retentionId =
          `async-v8-flush-${flush.flushIndex}`;
        outstandingRetentionIds.add(retentionId);
        const selected = await client.actPackedBatch(
          packed,
          actionKeys,
          { retentionBatchId: retentionId },
        );
        return selected.map((action) => ({
          actionIndex: action.actionIndex,
          actionKey: action.actionKey,
          logProbability: action.logProbability,
          value: action.value,
        }));
      },
    );

    const workerIds = Array.from(
      { length: rolloutPool.workerCount },
      (_, workerId) => workerId,
    );

    const rolloutStarted = performance.now();
    const prepareStarted = performance.now();
    const initialPrepared = await Promise.all(
      workerIds.map((workerId) =>
        rolloutPool!.prepareWorker(workerId, 0)),
    );
    initialPrepareMs = performance.now() - prepareStarted;
    for (const prepared of initialPrepared) {
      initialPrepareOperationMsSum += prepared.operationMs;
      addWorkerTiming(
        workerCpuTiming,
        prepared.workerCpuTiming,
      );
      mergeLegalActionCount += prepared.mergeLegalActionCount;
      recordFinalized(prepared.finalized);
    }

    const runWorker = async (
      workerId: number,
      initial: PpoRolloutWorkerV7WorkerPrepareResult,
    ) => {
      const assigned = rolloutPool!.getWorkerEnvironmentIndices(
        workerId,
      );
      const assignedSet = new Set(assigned);
      const finalizedAssigned = () =>
        assigned.filter((environmentIndex) =>
          summaryByEnvironment.has(environmentIndex)).length;

      let prepared = initial;
      let round = 0;

      while (finalizedAssigned() < assigned.length) {
        const group = prepared.group;
        if (!group) {
          throw new Error(
            `PPO async rollout worker ${workerId} produced no group before all assigned environments finalized`,
          );
        }
        if (
          group.environmentIndices.some(
            (environmentIndex) =>
              !assignedSet.has(environmentIndex),
          )
        ) {
          throw new Error(
            `PPO async rollout worker ${workerId} returned a foreign environment`,
          );
        }

        const selected = await broker!.submit(group);
        totalDecisions += selected.length;

        if (
          totalDecisions % memoryLogInterval < selected.length
        ) {
          process.stderr.write(
            "[PPO async rollout V8] "
            + JSON.stringify({
              totalDecisions,
              finalizedEnvironments:
                summaryByEnvironment.size,
              broker: broker!.diagnostics(),
            })
            + "\n",
          );
        }

        const advanced = await rolloutPool!.advanceWorker(
          workerId,
          round,
          group.environmentIndices.map(
            (environmentIndex, index) => ({
              environmentIndex,
              actionIndex: selected[index].actionIndex,
              actionKey: selected[index].actionKey,
              logProbability:
                selected[index].logProbability,
              value: selected[index].value,
            }),
          ),
        );
        workerAdvanceOperationMsByWorker[workerId] +=
          advanced.operationMs;
        addWorkerTiming(
          workerCpuTiming,
          advanced.workerCpuTiming,
        );
        mergeLegalActionCount +=
          advanced.mergeLegalActionCount;
        recordFinalized(advanced.finalized);
        prepared = advanced;
        round += 1;
        workerRounds[workerId] = round;
      }
    };

    const workerPromises = initialPrepared.map(
      (prepared, workerId) =>
        runWorker(workerId, prepared).catch((error) => {
          broker!.close(error);
          throw error;
        }),
    );
    const outcomes = await Promise.allSettled(workerPromises);
    const failed = outcomes.find(
      (outcome): outcome is PromiseRejectedResult =>
        outcome.status === "rejected",
    );
    if (failed) throw failed.reason;

    const rolloutMs = performance.now() - rolloutStarted;
    if (summaryByEnvironment.size !== environmentCount) {
      throw new Error(
        `PPO async rollout finalized ${summaryByEnvironment.size}/${environmentCount} environments`,
      );
    }

    const summaries = Array.from(
      { length: environmentCount },
      (_, environmentIndex) => {
        const summary = summaryByEnvironment.get(
          environmentIndex,
        );
        if (!summary) {
          throw new Error(
            `PPO async rollout missing summary env=${environmentIndex}`,
          );
        }
        return summary;
      },
    );

    const brokerDiagnostics = broker.diagnostics();
    if (
      brokerDiagnostics.submittedSampleCount
      !== totalDecisions
    ) {
      throw new Error(
        "PPO async rollout broker/decision count mismatch",
      );
    }

    const retentionBeforeDiscard =
      await client.retentionStats();
    if (outstandingRetentionIds.size) {
      await client.discardRetained(
        [...outstandingRetentionIds],
      );
      outstandingRetentionIds.clear();
    }
    const retentionFinal = await client.retentionStats();
    if (
      retentionFinal.currentChunks !== 0
      || (retentionFinal.pendingChunks ?? 0) !== 0
      || retentionFinal.currentRetainedBytes !== 0
      || (retentionFinal.pendingRawBytes ?? 0) !== 0
    ) {
      throw new Error(
        "PPO async rollout retention leak after discard: "
        + `chunks=${retentionFinal.currentChunks} `
        + `pendingChunks=${retentionFinal.pendingChunks ?? 0} `
        + `bytes=${retentionFinal.currentRetainedBytes} `
        + `pendingRawBytes=${retentionFinal.pendingRawBytes ?? 0}`,
      );
    }

    const diagnostics = await client.diagnostics();

    await rolloutPool.close();
    rolloutPool = undefined;

    return {
      mode: "async_rollout_dynamic_broker_v8",
      purpose: "rollout_only_global_round_barrier_removal",
      structuralProbeOnly: true,
      fixedPolicyForFullProbe: true,
      trainingPerformed: false,
      optimizerStepPerformed: false,
      checkpointSaved: false,
      v7qTrajectoryExactnessExpected: false,
      samplingOrderContract:
        "broker_microbatch_composition_depends_on_worker_readiness",
      samplingOrderNote:
        "Sampling order may differ from V7-Q because broker micro-batch composition depends on worker readiness.",
      environmentCount,
      rolloutWorkerCount: workerIds.length,
      seeds,
      checkpointStart: {
        updateCount: initialized.updateCount,
        episodeCount: initialized.episodeCount,
      },
      totalDecisions,
      finalizedEnvironments: summaries.length,
      summaries,
      mergeLegalActionCount,
      workerRounds,
      timings: {
        rolloutMs,
        rolloutMsPerDecision:
          totalDecisions ? rolloutMs / totalDecisions : 0,
        initialPrepareMs,
        initialPrepareOperationMsSum,
        workerAdvanceOperationMsByWorker,
        workerAdvanceOperationMsSum:
          workerAdvanceOperationMsByWorker.reduce(
            (sum, value) => sum + value,
            0,
          ),
        workerCpuTotals: workerCpuTiming,
      },
      broker: brokerDiagnostics,
      retentionBeforeDiscard:
        retentionSummary(retentionBeforeDiscard),
      retentionFinal: retentionSummary(retentionFinal),
      diagnostics,
    };
  } finally {
    broker?.close(
      new Error("PPO async rollout V8 probe exiting"),
    );
    if (outstandingRetentionIds.size) {
      try {
        await client.discardRetained(
          [...outstandingRetentionIds],
        );
        outstandingRetentionIds.clear();
      } catch {
        // Preserve the original probe failure.
      }
    }
    if (rolloutPool) {
      try {
        await rolloutPool.close();
      } catch {
        // Preserve the original probe failure.
      }
    }
    await client.close();
  }
}