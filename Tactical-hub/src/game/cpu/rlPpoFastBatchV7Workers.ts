import { performance } from "node:perf_hooks";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  PythonPpoClient,
  type PpoHyperparameters,
  type PpoRetentionStats,
} from "./pythonPpoClient";
import { PpoTimingProfiler } from "./rlPpoProfiler";
import { validatePpoRolloutsParallel } from "./rlPpoValidationParallel";
import {
  DEFAULT_PPO_HYPERPARAMETERS,
  validatePpoTrajectoryReplay,
  type PpoReplayRollout,
  type PpoReplayValidationRollout,
} from "./rlPpoSelfPlay";
import type { PpoUpdateScalarSample } from "./rlPpoPackedBatch";
import { PpoRolloutWorkerV7Pool } from "./rlPpoRolloutWorkerV7Pool";
import {
  combinePackedWorkerBatchesV7,
  fromTransferablePackedBcBatch,
} from "./rlPpoWorkerPackedV7";
import type {
  PpoRolloutWorkerV7EpisodeSummary,
  PpoRolloutWorkerV7Finalized,
  PpoRolloutWorkerV7Timing,
} from "./rlPpoRolloutWorkerV7Messages";

type FastBatchRetentionRecord = {
  retentionId: string;
  samples: Array<{
    environmentIndex: number;
    decisionIndex: number;
  }>;
};

export type PpoFastBatchV7WorkersInput = {
  seed: number;
  environmentCount?: number;
  initialCheckpoint: string;
  outputCheckpoint: string;
  bestCheckpoint?: string;
  resume?: string;
  hyperparameters?: Partial<PpoHyperparameters>;
  safetyMaxTurns?: number;
  safetyMaxActions?: number;
  replayChunkSize?: number;
  memoryLogInterval?: number;
  validationWorkerCount?: number;
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

export async function runPpoFastBatchV7WorkersSmoke(
  input: PpoFastBatchV7WorkersInput,
) {
  const profiler = new PpoTimingProfiler();
  const environmentCount = input.environmentCount ?? 8;
  if (environmentCount !== 8) {
    throw new Error("PPO fast_batch_v7_workers requires exactly 8 environments");
  }

  const safetyMaxTurns = input.safetyMaxTurns ?? 1_000;
  const safetyMaxActions = input.safetyMaxActions ?? 100_000;
  const replayChunkSize = input.replayChunkSize ?? 32;
  const memoryLogInterval = input.memoryLogInterval ?? 5_000;
  const validationWorkerCount = input.validationWorkerCount ?? 0;
  const rolloutWorkerCount = input.rolloutWorkerCount ?? 4;

  for (const [name, value] of [
    ["safetyMaxTurns", safetyMaxTurns],
    ["safetyMaxActions", safetyMaxActions],
    ["replayChunkSize", replayChunkSize],
    ["memoryLogInterval", memoryLogInterval],
  ] as const) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${name} must be a positive integer`);
    }
  }
  if (
    !Number.isInteger(validationWorkerCount)
    || validationWorkerCount < 0
  ) {
    throw new Error(
      "validationWorkerCount must be a non-negative integer",
    );
  }
  if (
    !Number.isInteger(rolloutWorkerCount)
    || rolloutWorkerCount <= 0
  ) {
    throw new Error(
      "rolloutWorkerCount must be a positive integer",
    );
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
      PPO_ACT_CUDA_GRAPH_MIN_HITS: "3",
      PPO_ACT_CUDA_GRAPH_MAX_ENTRIES: "32",
      PPO_PERSISTENT_REPLAY_H2D: "1",
      PPO_IMMUTABLE_RAW_RETENTION: "1",
    },
  });

  const outstandingRetentionIds = new Set<string>();
  const retainedBatches: FastBatchRetentionRecord[] = [];
  const started = performance.now();
  const initialized = await client.start({
    seed: input.seed,
    featureSpec,
    hyperparameters,
    initialCheckpoint: input.initialCheckpoint,
    resume: input.resume,
  });
  if (initialized.selectedDevice !== "cuda") {
    throw new Error("PPO fast_batch_v7_workers requires CUDA");
  }

  const firstGameSeed = input.seed + initialized.episodeCount;
  const seeds = Array.from(
    { length: environmentCount },
    (_, environmentIndex) => firstGameSeed + environmentIndex,
  );

  const summaryByEnvironment = new Map<
    number,
    PpoRolloutWorkerV7EpisodeSummary
  >();
  const rolloutByEnvironment = new Map<
    number,
    PpoReplayRollout
  >();
  let rolloutPool: PpoRolloutWorkerV7Pool | undefined;

  let totalDecisions = 0;
  let batchRounds = 0;
  let mergeLegalActionCount = 0;
  let rolloutMs = 0;
  let replayMs = 0;
  let prepareBarrierMs = 0;
  let applyBarrierMs = 0;
  const rolloutWorkerCpuTiming: PpoRolloutWorkerV7Timing = {
    observationMs: 0,
    legalActionsMs: 0,
    encodeObservationMs: 0,
    encodeActionsMs: 0,
    packMs: 0,
    gameStepMs: 0,
  };

  const addWorkerTiming = (
    value: PpoRolloutWorkerV7Timing,
  ) => {
    rolloutWorkerCpuTiming.observationMs += value.observationMs;
    rolloutWorkerCpuTiming.legalActionsMs += value.legalActionsMs;
    rolloutWorkerCpuTiming.encodeObservationMs += value.encodeObservationMs;
    rolloutWorkerCpuTiming.encodeActionsMs += value.encodeActionsMs;
    rolloutWorkerCpuTiming.packMs += value.packMs;
    rolloutWorkerCpuTiming.gameStepMs += value.gameStepMs;
  };

  const recordFinalized = (
    finalized: PpoRolloutWorkerV7Finalized[],
  ) => {
    for (const item of finalized) {
      if (summaryByEnvironment.has(item.environmentIndex)) {
        throw new Error(
          `PPO fast_batch_v7_workers duplicate finalized env=${item.environmentIndex}`,
        );
      }
      summaryByEnvironment.set(
        item.environmentIndex,
        item.summary,
      );
      if (item.rollout) {
        rolloutByEnvironment.set(
          item.environmentIndex,
          item.rollout,
        );
      }
    }
  };

  try {
    rolloutPool = await PpoRolloutWorkerV7Pool.create({
      workerCount: rolloutWorkerCount,
      environmentCount,
      firstGameSeed,
      featureSpec,
      hyperparameters,
      safetyMaxTurns,
      safetyMaxActions,
    });

    const rolloutStarted = performance.now();

    while (summaryByEnvironment.size < environmentCount) {
      const prepared = await profiler.measureAsync(
        "fast_worker_prepare_barrier",
        () => rolloutPool!.prepare(batchRounds),
      );
      prepareBarrierMs += prepared.barrierMs;
      addWorkerTiming(prepared.workerCpuTiming);
      mergeLegalActionCount += prepared.mergeLegalActionCount;
      recordFinalized(prepared.finalized);

      const preparedSamples = prepared.groups.flatMap((group) => {
        if (group.environmentIndices.length !== group.decisionIndices.length
          || group.environmentIndices.length !== group.actionKeys.length
          || group.environmentIndices.length !== group.packed.batchSize) {
          throw new Error("Grouped worker prepare metadata mismatch");
        }
        return group.environmentIndices.map((environmentIndex, index) => ({
          environmentIndex,
          decisionIndex: group.decisionIndices[index],
          actionKeys: group.actionKeys[index],
        }));
      });
      if (!preparedSamples.length) {
        if (summaryByEnvironment.size >= environmentCount) break;
        throw new Error("Grouped workers produced no active samples before all environments finalized");
      }
      for (let index = 1; index < preparedSamples.length; index += 1) {
        if (preparedSamples[index - 1].environmentIndex >= preparedSamples[index].environmentIndex) {
          throw new Error("Grouped worker samples are not in environment-index order");
        }
      }

      const retentionBatchId = "fast-worker-grouped-round-" + batchRounds;
      const packed = profiler.measure(
        "fast_worker_combine_grouped_packed",
        () => combinePackedWorkerBatchesV7(
          prepared.groups.map((group) => fromTransferablePackedBcBatch(group.packed)),
        ),
      );
      const selected = await profiler.measureAsync(
        "fast_worker_batch_act",
        () => client.actPackedBatch(
          packed,
          preparedSamples.map((sample) => sample.actionKeys),
          { retentionBatchId },
        ),
      );

      retainedBatches.push({
        retentionId: retentionBatchId,
        samples: preparedSamples.map((sample) => ({
          environmentIndex: sample.environmentIndex,
          decisionIndex: sample.decisionIndex,
        })),
      });
      outstandingRetentionIds.add(retentionBatchId);
      totalDecisions += preparedSamples.length;

      const applied = await profiler.measureAsync(
        "fast_worker_apply_barrier",
        () => rolloutPool!.apply(
          batchRounds,
          preparedSamples.map((sample, index) => ({
            environmentIndex: sample.environmentIndex,
            actionIndex: selected[index].actionIndex,
            actionKey: selected[index].actionKey,
            logProbability: selected[index].logProbability,
            value: selected[index].value,
          })),
        ),
      );
      applyBarrierMs += applied.barrierMs;
      addWorkerTiming(applied.workerCpuTiming);
      recordFinalized(applied.finalized);

      batchRounds += 1;

      if (
        totalDecisions % memoryLogInterval < preparedSamples.length
        || summaryByEnvironment.size >= environmentCount
      ) {
        process.stderr.write(
          `[PPO fast batch v7 workers] ${JSON.stringify({
            totalDecisions,
            batchRounds,
            finalizedEnvironments: summaryByEnvironment.size,
            rolloutWorkerCount,
          })}\n`,
        );
      }
    }

    rolloutMs = performance.now() - rolloutStarted;

    await rolloutPool.close();
    rolloutPool = undefined;

    const summaries = Array.from(
      { length: environmentCount },
      (_, environmentIndex) => {
        const summary = summaryByEnvironment.get(environmentIndex);
        if (!summary) {
          throw new Error(
            `PPO fast_batch_v7_workers missing summary env=${environmentIndex}`,
          );
        }
        return summary;
      },
    );

    const learnableRollouts = Array.from(
      { length: environmentCount },
      (_, environmentIndex) =>
        rolloutByEnvironment.get(environmentIndex),
    ).filter(
      (rollout): rollout is PpoReplayRollout => Boolean(rollout),
    );

    const totalSamples = learnableRollouts.reduce(
      (sum, rollout) => sum + rollout.trajectory.length,
      0,
    );
    if (
      learnableRollouts.length !== environmentCount
      || totalSamples <= 0
    ) {
      throw new Error(
        "PPO fast_batch_v7_workers requires all 8 environments "
        + "to produce learnable completed/adjudicated trajectories",
      );
    }

    await profiler.measureAsync(
      "fast_begin_update",
      () => client.beginUpdate(totalSamples),
    );
    const retentionBeforeUpdate = await client.retentionStats();

    let validatedSamples = 0;
    let replayedSamples = 0;
    const replayStarted = performance.now();

    const validationRollouts: PpoReplayValidationRollout[] = (
      learnableRollouts.map((rollout) => ({
        seed: rollout.seed,
        terminal: rollout.terminal,
        endReason: rollout.endReason,
        winnerTeamId: rollout.winnerTeamId,
        finalStateHash: rollout.finalStateHash,
        loserTeamIds: rollout.loserTeamIds,
        trajectory: rollout.trajectory.map((step) => ({
          decisionIndex: step.decisionIndex,
          turnNumber: step.turnNumber,
          phase: step.phase,
          teamId: step.teamId,
          selectedActionIndex: step.selectedActionIndex,
          selectedActionKey: step.selectedActionKey,
        })),
      }))
    );

    const parallelValidationPromise = validationWorkerCount > 0
      ? validatePpoRolloutsParallel({
          rollouts: validationRollouts,
          workerCount: validationWorkerCount,
          memoryLogInterval,
          fastRlMovement: true,
          fastRlPhaseTransitions: true,
        }).then(
          (result) => ({ result }),
          (error: unknown) => ({ error }),
        )
      : undefined;

    if (!parallelValidationPromise) {
      for (const rollout of validationRollouts) {
        validatedSamples += await validatePpoTrajectoryReplay({
          rollout,
          memoryLogInterval,
          fastRlMovement: true,
          fastRlPhaseTransitions: true,
          profiler,
        });
      }
    }

    const scalarFor = (
      environmentIndex: number,
      decisionIndex: number,
    ): PpoUpdateScalarSample => {
      const step = rolloutByEnvironment
        .get(environmentIndex)
        ?.trajectory[decisionIndex];
      if (
        !step
        || ![
          step.oldLogProbability,
          step.advantage,
          step.return,
        ].every(Number.isFinite)
      ) {
        throw new Error(
          `PPO fast_batch_v7_workers missing finite scalar at env=${environmentIndex} decision=${decisionIndex}`,
        );
      }
      return {
        oldLogProbability: step.oldLogProbability,
        advantage: step.advantage!,
        return: step.return!,
      };
    };

    let pendingRecords: FastBatchRetentionRecord[] = [];
    let pendingScalars: PpoUpdateScalarSample[] = [];
    let pendingSamples = 0;

    const flushRetainedBatches = async () => {
      if (!pendingRecords.length) return;
      const retentionIds = pendingRecords.map(
        (record) => record.retentionId,
      );
      const accepted = pendingScalars.length;
      await profiler.measureAsync(
        "replay_retention_batch_restore_accumulate",
        () => client.accumulateRetainedBatches(
          retentionIds,
          pendingScalars,
        ),
      );
      for (const retentionId of retentionIds) {
        outstandingRetentionIds.delete(retentionId);
      }
      replayedSamples += accepted;
      pendingRecords = [];
      pendingScalars = [];
      pendingSamples = 0;
    };

    for (const record of retainedBatches) {
      const recordSamples = record.samples.length;
      if (
        pendingSamples > 0
        && pendingSamples + recordSamples > replayChunkSize
      ) {
        await flushRetainedBatches();
      }
      pendingRecords.push(record);
      for (const reference of record.samples) {
        pendingScalars.push(
          scalarFor(
            reference.environmentIndex,
            reference.decisionIndex,
          ),
        );
      }
      pendingSamples += recordSamples;
    }
    await flushRetainedBatches();

    let parallelValidation:
      | Awaited<ReturnType<typeof validatePpoRolloutsParallel>>
      | undefined;
    if (parallelValidationPromise) {
      const outcome = await parallelValidationPromise;
      if ("error" in outcome) {
        throw outcome.error;
      }
      parallelValidation = outcome.result;
      validatedSamples = parallelValidation.sampleCount;
    }

    replayMs = performance.now() - replayStarted;

    if (
      validatedSamples !== totalSamples
      || replayedSamples !== totalSamples
    ) {
      throw new Error(
        `PPO fast batch replay mismatch validated=${validatedSamples} `
        + `replayed=${replayedSamples} total=${totalSamples}`,
      );
    }

    const retentionAfterReplay = await client.retentionStats();
    if (
      retentionAfterReplay.currentChunks !== 0
      || retentionAfterReplay.currentRetainedBytes !== 0
    ) {
      throw new Error(
        "PPO fast batch retention leak after replay: "
        + `chunks=${retentionAfterReplay.currentChunks} `
        + `bytes=${retentionAfterReplay.currentRetainedBytes}`,
      );
    }

    const update = await profiler.measureAsync(
      "fast_finish_update",
      () => client.finishUpdate(learnableRollouts.length),
    );

    const victoryEpisodeCount = summaries.filter(
      (summary) => summary.outcomeKind === "victory",
    ).length;
    const adjudicatedEpisodeCount = summaries.filter(
      (summary) => summary.outcomeKind === "time_limit_adjudicated",
    ).length;
    const truncatedEpisodeCount = summaries.filter(
      (summary) => summary.outcomeKind === "abnormal_truncated",
    ).length;

    const metadata = {
      purpose: "phase_12b_worker_grouped_pack",
      ppoMode: "fast_batch_worker_grouped_pack",
      environmentCount,
      seeds,
      mergeOrder: Array.from(
        { length: environmentCount },
        (_, index) => index,
      ),
      sampleMergeOrder: "round_major_env_index",
      retentionRecordMode: "inference_batch",
      victoryEpisodeCount,
      adjudicatedEpisodeCount,
      truncatedEpisodeCount,
      replayedSamples,
      validationWorkerCount,
      rolloutWorkerCount,
    };

    const saved = await profiler.measureAsync(
      "fast_checkpoint_save",
      () => client.save(
        input.outputCheckpoint,
        metadata,
      ),
    );
    const bestSaved = input.bestCheckpoint
      ? await client.save(
        input.bestCheckpoint,
        { ...metadata, checkpointRole: "best" },
      )
      : undefined;

    const finalRetentionStats = await client.retentionStats();
    const diagnostics = await client.diagnostics();
    const totalMs = performance.now() - started;

    return {
      mode: "fast_batch_worker_grouped_pack",
      environmentCount,
      seeds,
      mergeOrder: Array.from(
        { length: environmentCount },
        (_, index) => index,
      ),
      sampleMergeOrder: "round_major_env_index",
      retentionRecordMode: "inference_batch",
      retainedBatchCount: retainedBatches.length,
      batchRounds,
      totalDecisions,
      totalSamples,
      summaries,
      replayedSamples,
      validatedSamples,
      validationWorkerCount,
      rolloutWorkerCount,
      parallelValidation,
      rolloutWorkerTiming: {
        workerCount: rolloutWorkerCount,
        prepareBarrierMs,
        applyBarrierMs,
        cpuTotals: rolloutWorkerCpuTiming,
      },
      mergeLegalActionCount,
      update,
      saved,
      bestSaved,
      diagnostics,
      selectedDevice: initialized.selectedDevice,
      checkpointStart: {
        updateCount: initialized.updateCount,
        episodeCount: initialized.episodeCount,
      },
      timings: {
        totalMs,
        rolloutMs,
        replayMs,
        rolloutMsPerDecision: rolloutMs / totalDecisions,
        replayMsPerSample: replayMs / replayedSamples,
      },
      retentionBeforeUpdate: retentionSummary(retentionBeforeUpdate),
      retentionAfterReplay: retentionSummary(retentionAfterReplay),
      retentionFinal: retentionSummary(finalRetentionStats),
    };
  } finally {
    if (outstandingRetentionIds.size) {
      try {
        await client.discardRetained([...outstandingRetentionIds]);
      } catch {
        // Preserve the original fast-mode error.
      }
    }
    if (rolloutPool) {
      try {
        await rolloutPool.close();
      } catch {
        // Preserve the original fast-mode error.
      }
    }
    profiler.report("fast_batch_worker_grouped_pack_final");
    await client.close();
  }
}
