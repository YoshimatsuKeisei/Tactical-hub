import { performance } from "node:perf_hooks";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import { RlEnvironmentV2, type RlResult } from "./rlEnvironment";
import {
  adjudicatePpoTimeLimit,
  isPpoTimeLimitReason,
  type PpoTeamAdjudication,
  type PpoTimeLimitReason,
} from "./rlPpoAdjudication";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "./rlObservationEncoder";
import {
  PythonPpoClient,
  type PpoHyperparameters,
  type PpoRetentionStats,
} from "./pythonPpoClient";
import { PpoTimingProfiler } from "./rlPpoProfiler";
import { validatePpoRolloutsParallel } from "./rlPpoValidationParallel";
import {
  DEFAULT_PPO_HYPERPARAMETERS,
  finalizeTerminalTrajectory,
  finalizeVictoryTrajectory,
  validatePpoTrajectoryReplay,
  type PpoReplayRollout,
  type PpoReplayValidationRollout,
  type PpoTrajectoryStep,
} from "./rlPpoSelfPlay";
import type { PpoUpdateScalarSample } from "./rlPpoPackedBatch";

type FastBatchEpisodeSummary = {
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

type FastBatchRetentionRecord = {
  retentionId: string;
  samples: Array<{
    environmentIndex: number;
    decisionIndex: number;
  }>;
};

type FastBatchSlot = {
  environmentIndex: number;
  seed: number;
  environment: RlEnvironmentV2;
  encoderCache: ReturnType<typeof createRlObservationEncoderCache>;
  trajectory: PpoTrajectoryStep[];
  finished: boolean;
  reason?: string;
  summary?: FastBatchEpisodeSummary;
  rollout?: PpoReplayRollout;
};

export type PpoFastBatchInput = {
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
  client?: PythonPpoClient;
};

function createFastEnvironment() {
  return new RlEnvironmentV2(
    undefined,
    true,
    { cpuStep: { rlInPlacePhaseTransitions: true } },
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

export async function runPpoFastBatchV2Smoke(input: PpoFastBatchInput) {
  const profiler = new PpoTimingProfiler();
  const environmentCount = input.environmentCount ?? 8;
  if (environmentCount !== 8) {
    throw new Error("PPO fast_batch_v2 requires exactly 8 environments");
  }

  const safetyMaxTurns = input.safetyMaxTurns ?? 1_000;
  const safetyMaxActions = input.safetyMaxActions ?? 100_000;
  const replayChunkSize = input.replayChunkSize ?? 32;
  const memoryLogInterval = input.memoryLogInterval ?? 5_000;
  const validationWorkerCount = input.validationWorkerCount ?? 0;
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

  const hyperparameters = {
    ...DEFAULT_PPO_HYPERPARAMETERS,
    ...input.hyperparameters,
  };

  const probe = createFastEnvironment();
  const first = probe.reset(input.seed, 4);
  const featureSpec = createRlFeatureSpecV2(first);

  const client = input.client ?? new PythonPpoClient({
    device: "cuda",
    env: {
      PPO_PACKED_PREPARE_MODE: "fast_batch_v2",
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
    throw new Error("PPO fast_batch_v2 requires CUDA");
  }

  const firstGameSeed = input.seed + initialized.episodeCount;
  const slots: FastBatchSlot[] = Array.from(
    { length: environmentCount },
    (_, environmentIndex) => {
      const environment = createFastEnvironment();
      const seed = firstGameSeed + environmentIndex;
      environment.reset(seed, 4);
      return {
        environmentIndex,
        seed,
        environment,
        encoderCache: createRlObservationEncoderCache(),
        trajectory: [],
        finished: false,
      };
    },
  );

  let totalDecisions = 0;
  let batchRounds = 0;
  let mergeLegalActionCount = 0;
  let rolloutMs = 0;
  let replayMs = 0;

  const finalizeSlot = async (slot: FastBatchSlot) => {
    if (slot.finished) return;

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
    const summary: FastBatchEpisodeSummary = {
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
          summary.adjudication.map((team) => [team.teamId, team.reward]),
        ),
        hyperparameters,
      );
    }

    slot.summary = summary;
    if (summary.outcomeKind !== "abnormal_truncated") {
      slot.rollout = {
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
      };
    }
    slot.finished = true;
  };

  try {
    const rolloutStarted = performance.now();

    while (slots.some((slot) => !slot.finished)) {
      const batch: Array<{
        slot: FastBatchSlot;
        actor: string;
        observation: ReturnType<FastBatchSlot["environment"]["getObservationForEncoding"]>;
        encodedObservation: ReturnType<typeof encodeRlObservationV2>;
        encodedActions: ReturnType<typeof encodeRlLegalActionsV2>;
        progressHash: string;
      }> = [];

      for (const slot of slots) {
        if (slot.finished) continue;

        if (slot.environment.isTerminal()) {
          await finalizeSlot(slot);
          continue;
        }

        const actor = slot.environment.getCurrentActorTeamId();
        if (!actor) {
          slot.reason = "no_actor";
          await finalizeSlot(slot);
          continue;
        }

        const observation = profiler.measure(
          "fast_rollout_observation",
          () => slot.environment.getObservationForEncoding(actor),
        );
        const legal = profiler.measure(
          "fast_rollout_legal_actions",
          () => slot.environment.getLegalActionsForEncoding(actor),
        );
        if (!legal.length) {
          slot.reason = "no_legal_actions";
          await finalizeSlot(slot);
          continue;
        }
        if (observation.turnNumber > safetyMaxTurns) {
          slot.reason = "safety_turn_limit";
          await finalizeSlot(slot);
          continue;
        }
        if (slot.trajectory.length >= safetyMaxActions) {
          slot.reason = "safety_action_limit";
          await finalizeSlot(slot);
          continue;
        }

        const encodedObservation = profiler.measure(
          "fast_rollout_encode_observation",
          () => encodeRlObservationV2(
            observation,
            slot.encoderCache,
          ),
        );
        const encodedActions = profiler.measure(
          "fast_rollout_encode_actions",
          () => encodeRlLegalActionsV2(observation, legal),
        );
        mergeLegalActionCount += legal.filter(
          (action) => action.actionType === "merge_infantry",
        ).length;

        batch.push({
          slot,
          actor,
          observation,
          encodedObservation,
          encodedActions,
          progressHash: profiler.measure(
            "fast_rollout_progress_hash",
            () => slot.environment.getProgressHash(),
          ),
        });
      }

      if (!batch.length) continue;

      const retentionBatchId = `fast-v2-round-${batchRounds}`;
      const sampleReferences = batch.map((entry) => ({
        environmentIndex: entry.slot.environmentIndex,
        decisionIndex: entry.slot.trajectory.length,
      }));

      const selected = await profiler.measureAsync(
        "fast_rollout_batch_act",
        () => client.actBatch(
          batch.map((entry) => ({
            observation: entry.encodedObservation,
            legalActions: entry.encodedActions,
          })),
          { retentionBatchId },
        ),
      );

      retainedBatches.push({
        retentionId: retentionBatchId,
        samples: sampleReferences,
      });
      outstandingRetentionIds.add(retentionBatchId);

      batchRounds += 1;
      totalDecisions += batch.length;

      for (let index = 0; index < batch.length; index += 1) {
        const entry = batch[index];
        const action = selected[index];

        profiler.measure(
          "fast_rollout_game_step",
          () => entry.slot.environment.stepWithoutObservation(action.actionKey),
        );
        entry.slot.trajectory.push({
          decisionIndex: entry.slot.trajectory.length,
          turnNumber: entry.observation.turnNumber,
          phase: entry.observation.phase,
          teamId: entry.actor,
          selectedActionIndex: action.actionIndex,
          selectedActionKey: action.actionKey,
          oldLogProbability: action.logProbability,
          value: action.value,
          reward: 0,
          done: false,
        });

        if (
          profiler.measure(
            "fast_rollout_progress_hash_after",
            () => entry.slot.environment.getProgressHash(),
          )
          === entry.progressHash
        ) {
          entry.slot.reason = "phase_stall";
          await finalizeSlot(entry.slot);
        } else if (entry.slot.environment.isTerminal()) {
          await finalizeSlot(entry.slot);
        }
      }

      if (
        totalDecisions % memoryLogInterval < batch.length
        || slots.every((slot) => slot.finished)
      ) {
        process.stderr.write(
          `[PPO fast batch] ${JSON.stringify({
            totalDecisions,
            batchRounds,
            activeEnvironments: slots.filter((slot) => !slot.finished).length,
            perEnvironmentDecisions: slots.map((slot) => slot.trajectory.length),
          })}\n`,
        );
      }
    }

    rolloutMs = performance.now() - rolloutStarted;

    const summaries = slots.map((slot) => {
      if (!slot.summary) {
        throw new Error(
          `PPO fast batch slot ${slot.environmentIndex} has no summary`,
        );
      }
      return slot.summary;
    });

    const learnableRollouts = slots
      .map((slot) => slot.rollout)
      .filter((rollout): rollout is PpoReplayRollout => Boolean(rollout));

    const totalSamples = learnableRollouts.reduce(
      (sum, rollout) => sum + rollout.trajectory.length,
      0,
    );
    if (
      learnableRollouts.length !== environmentCount
      || totalSamples <= 0
    ) {
      throw new Error(
        "PPO fast_batch_v2 requires all 8 environments "
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
      const step = slots[environmentIndex]?.trajectory[decisionIndex];
      if (
        !step
        || ![
          step.oldLogProbability,
          step.advantage,
          step.return,
        ].every(Number.isFinite)
      ) {
        throw new Error(
          `PPO fast_batch_v2 missing finite scalar at env=${environmentIndex} decision=${decisionIndex}`,
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

    // v2 sample merge contract:
    // rollout round-major, then environment-index order within each round.
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
      purpose: "phase_12b_fast_batch_v2",
      ppoMode: "fast_batch_v2",
      environmentCount,
      seeds: slots.map((slot) => slot.seed),
      mergeOrder: slots.map((slot) => slot.environmentIndex),
      sampleMergeOrder: "round_major_env_index",
      retentionRecordMode: "inference_batch",
      victoryEpisodeCount,
      adjudicatedEpisodeCount,
      truncatedEpisodeCount,
      replayedSamples,
      validationWorkerCount,
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
      mode: "fast_batch_v2",
      environmentCount,
      seeds: slots.map((slot) => slot.seed),
      mergeOrder: slots.map((slot) => slot.environmentIndex),
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
      parallelValidation,
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
    profiler.report("fast_batch_v2_final");
    await client.close();
  }
}
