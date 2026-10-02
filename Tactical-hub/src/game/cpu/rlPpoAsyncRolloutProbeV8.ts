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
import {
  PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS,
  type PpoRolloutEnvironmentDiagnosticV8,
} from "./rlPpoRolloutDiagnosticsV8";
import {
  PPO_DEFEAT_DIAGNOSTIC_OBSERVATION_TURNS,
  PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL,
  updatePpoTwoTeamObservationV8,
  type PpoDefeatEnvironmentSnapshotV8,
  type PpoTwoTeamObservationV8,
} from "./rlPpoDefeatDiagnosticsV8";

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
  continuousRecycle?: boolean;
  targetDecisions?: number;
  naturalRecycleProbe?: boolean;
  naturalRecycleDiagnostic?: boolean;
  naturalRecycleDefeatDiagnostic?: boolean;
  client?: PythonPpoClient;
};

export type PpoAsyncRolloutDiagnosticSnapshotV8 = {
  decisionThreshold: number;
  capturedAtTotalDecisions: number;
  environments: PpoRolloutEnvironmentDiagnosticV8[];
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
  const continuousRecycle = input.continuousRecycle ?? false;
  const targetDecisions = input.targetDecisions;
  const naturalRecycleProbe = input.naturalRecycleProbe ?? false;
  const naturalRecycleDiagnostic =
    input.naturalRecycleDiagnostic ?? false;
  const naturalRecycleDefeatDiagnostic =
    input.naturalRecycleDefeatDiagnostic ?? false;
  const naturalTerminationProbe =
    naturalRecycleProbe
    || naturalRecycleDiagnostic
    || naturalRecycleDefeatDiagnostic;
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
  if (
    targetDecisions !== undefined
    && (
      !Number.isInteger(targetDecisions)
      || targetDecisions <= 0
    )
  ) {
    throw new Error("targetDecisions must be a positive integer");
  }
  if (continuousRecycle && targetDecisions === undefined) {
    throw new Error(
      "targetDecisions is required in continuous recycle mode",
    );
  }
  if (naturalRecycleProbe && !continuousRecycle) {
    throw new Error(
      "naturalRecycleProbe requires continuousRecycle",
    );
  }
  if (naturalRecycleDiagnostic && !continuousRecycle) {
    throw new Error(
      "naturalRecycleDiagnostic requires continuousRecycle",
    );
  }
  if (naturalRecycleDefeatDiagnostic && !continuousRecycle) {
    throw new Error(
      "naturalRecycleDefeatDiagnostic requires continuousRecycle",
    );
  }
  if (
    [
      naturalRecycleProbe,
      naturalRecycleDiagnostic,
      naturalRecycleDefeatDiagnostic,
    ].filter(Boolean).length > 1
  ) {
    throw new Error(
      "natural recycle probe modes are mutually exclusive",
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
  const completedSummaries: PpoRolloutWorkerV7EpisodeSummary[] = [];
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
  let draining = false;
  let hardDecisionCapReached = false;
  let stopRequested = false;
  let recycledBeforeDrainEpisodeCount = 0;
  let naturalVictoryRecycledBeforeDrainCount = 0;
  const naturalVictoryPendingProof = new Map<
    number,
    { sourceSeed: number; nextSeed: number }
  >();
  let naturalVictoryRecycleConfirmed = false;
  let naturalVictoryRecycleEnvironmentIndex: number | undefined;
  let naturalVictoryRecycleSourceSeed: number | undefined;
  let naturalVictoryRecycleNextSeed: number | undefined;
  let mergeLegalActionCount = 0;
  let initialPrepareMs = 0;
  let initialPrepareOperationMsSum = 0;
  const diagnosticSnapshots:
    PpoAsyncRolloutDiagnosticSnapshotV8[] = [];
  const pendingDiagnosticSnapshots = new Map<
    number,
    Map<number, PpoRolloutEnvironmentDiagnosticV8[]>
  >();
  let nextDiagnosticDecisionThreshold = 50_000;
  let finalEnvironmentDiagnostics:
    PpoRolloutEnvironmentDiagnosticV8[] = [];
  let twoTeamObservation: PpoTwoTeamObservationV8 | undefined;
  let defeatDiagnosticEventCount = 0;
  let defeatDiagnosticStopReason:
    | "two_team_observation_window_completed"
    | "natural_victory_recycle_confirmed"
    | "hard_decision_cap_reached"
    | undefined;
  let finalDefeatEnvironmentDiagnostics:
    PpoDefeatEnvironmentSnapshotV8[] = [];

  const recordFinalized = (
    finalized: PpoRolloutWorkerV7Finalized[],
  ) => {
    for (const item of finalized) {
      if (
        !continuousRecycle
        && summaryByEnvironment.has(item.environmentIndex)
      ) {
        throw new Error(
          `PPO async rollout duplicate finalized env=${item.environmentIndex}`,
        );
      }
      completedSummaries.push(item.summary);
      if (!continuousRecycle) {
        summaryByEnvironment.set(
          item.environmentIndex,
          item.summary,
        );
      }
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
      ...(continuousRecycle
        ? {
            autoRecycle: true,
            recycleSeedStride: environmentCount,
          }
        : {}),
      ...(naturalRecycleDefeatDiagnostic
        ? { defeatDiagnostics: true }
        : {}),
    });

    broker = new PpoDynamicInferenceBrokerV8(
      async (packed, actionKeys, flush) => {
        const selected = naturalTerminationProbe
          ? await client.actPackedBatch(
              packed,
              actionKeys,
            )
          : await (async () => {
              const retentionId =
                `async-v8-flush-${flush.flushIndex}`;
              outstandingRetentionIds.add(retentionId);
              return client.actPackedBatch(
                packed,
                actionKeys,
                { retentionBatchId: retentionId },
              );
            })();
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

    const publishDiagnosticSnapshot = (
      decisionThreshold: number,
    ) => {
      const byWorker = pendingDiagnosticSnapshots.get(
        decisionThreshold,
      );
      if (!byWorker || byWorker.size !== workerIds.length) return;
      const snapshot: PpoAsyncRolloutDiagnosticSnapshotV8 = {
        decisionThreshold,
        capturedAtTotalDecisions: totalDecisions,
        environments: [...byWorker.values()]
          .flat()
          .sort((left, right) =>
            left.environmentIndex - right.environmentIndex),
      };
      diagnosticSnapshots.push(snapshot);
      pendingDiagnosticSnapshots.delete(decisionThreshold);
      process.stderr.write(
        "[PPO async rollout V8 diagnostic] "
        + JSON.stringify(snapshot)
        + "\n",
      );
    };

    const scheduleDiagnosticSnapshots = () => {
      if (!naturalRecycleDiagnostic) return;
      while (
        totalDecisions >= nextDiagnosticDecisionThreshold
      ) {
        pendingDiagnosticSnapshots.set(
          nextDiagnosticDecisionThreshold,
          new Map(),
        );
        nextDiagnosticDecisionThreshold += 50_000;
      }
    };

    const capturePendingDiagnosticsForWorker = async (
      workerId: number,
    ) => {
      const missingThresholds = [
        ...pendingDiagnosticSnapshots.entries(),
      ].filter(([, byWorker]) => !byWorker.has(workerId));
      if (!missingThresholds.length) return;
      const environments =
        await rolloutPool!.getWorkerDiagnostics(workerId);
      for (const [decisionThreshold, byWorker] of
        missingThresholds) {
        byWorker.set(workerId, environments);
        publishDiagnosticSnapshot(decisionThreshold);
      }
    };

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
      const finalizedAfterRecycleDisabled = new Set<number>();
      let recycleDisabled = false;

      let prepared = initial;
      let round = 0;

      while (
        naturalTerminationProbe
          ? !stopRequested
          : continuousRecycle
            ? (
                !recycleDisabled
                || finalizedAfterRecycleDisabled.size < assigned.length
              )
            : finalizedAssigned() < assigned.length
      ) {
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
          naturalTerminationProbe
          && totalDecisions >= (targetDecisions ?? Number.POSITIVE_INFINITY)
        ) {
          hardDecisionCapReached = true;
          stopRequested = true;
          if (naturalRecycleDefeatDiagnostic) {
            defeatDiagnosticStopReason ??=
              "hard_decision_cap_reached";
          }
        } else if (
          continuousRecycle
          && !draining
          && totalDecisions >= (targetDecisions ?? Number.POSITIVE_INFINITY)
        ) {
          draining = true;
        }
        scheduleDiagnosticSnapshots();

        if (
          !naturalRecycleDiagnostic
          && totalDecisions % memoryLogInterval < selected.length
        ) {
          process.stderr.write(
            "[PPO async rollout V8] "
            + JSON.stringify({
              totalDecisions,
              ...(continuousRecycle
                ? {
                    completedEpisodes:
                      completedSummaries.length,
                    ...(naturalRecycleProbe
                      ? {
                          hardDecisionCapReached,
                          naturalVictoryRecycleConfirmed,
                          naturalVictoryPendingProofCount:
                            naturalVictoryPendingProof.size,
                        }
                      : {}),
                  }
                : {
                    finalizedEnvironments:
                      summaryByEnvironment.size,
                  }),
              broker: broker!.diagnostics(),
            })
            + "\n",
          );
        }

        if (
          continuousRecycle
          && !naturalTerminationProbe
          && draining
          && !recycleDisabled
        ) {
          await rolloutPool!.setWorkerAutoRecycle(
            workerId,
            false,
          );
          recycleDisabled = true;
        }

        const naturalProofCandidates =
          naturalTerminationProbe
            ? group.environmentIndices.filter(
                (environmentIndex) =>
                  naturalVictoryPendingProof.has(
                    environmentIndex,
                  ),
              )
            : [];

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
        if (continuousRecycle && !recycleDisabled) {
          recycledBeforeDrainEpisodeCount +=
            advanced.finalized.length;
          for (const item of advanced.finalized) {
            if (item.summary.outcomeKind === "victory") {
              naturalVictoryRecycledBeforeDrainCount += 1;
              if (naturalTerminationProbe) {
                naturalVictoryPendingProof.set(
                  item.environmentIndex,
                  {
                    sourceSeed: item.summary.seed,
                    nextSeed:
                      item.summary.seed + environmentCount,
                  },
                );
              }
            }
          }
        }
        recordFinalized(advanced.finalized);
        if (naturalRecycleDefeatDiagnostic) {
          for (const event of advanced.defeatDiagnosticEvents) {
            defeatDiagnosticEventCount += 1;
            process.stderr.write(
              "[PPO async rollout V8 defeat diagnostic] "
              + JSON.stringify(event)
              + "\n",
            );
            twoTeamObservation = updatePpoTwoTeamObservationV8(
              twoTeamObservation,
              event,
            );
            if (
              twoTeamObservation?.observationWindowCompleted
            ) {
              defeatDiagnosticStopReason =
                "two_team_observation_window_completed";
              stopRequested = true;
            }
          }
        }
        if (
          naturalTerminationProbe
          && naturalProofCandidates.length
        ) {
          for (const environmentIndex of naturalProofCandidates) {
            const proof =
              naturalVictoryPendingProof.get(
                environmentIndex,
              );
            if (!proof) continue;
            naturalVictoryRecycleConfirmed = true;
            naturalVictoryRecycleEnvironmentIndex ??=
              environmentIndex;
            naturalVictoryRecycleSourceSeed ??=
              proof.sourceSeed;
            naturalVictoryRecycleNextSeed ??=
              proof.nextSeed;
            naturalVictoryPendingProof.delete(
              environmentIndex,
            );
          }
          if (naturalVictoryRecycleConfirmed) {
            if (naturalRecycleDefeatDiagnostic) {
              defeatDiagnosticStopReason =
                "natural_victory_recycle_confirmed";
            }
            stopRequested = true;
          }
        }
        if (naturalRecycleDiagnostic) {
          await capturePendingDiagnosticsForWorker(workerId);
        }
        if (recycleDisabled) {
          for (const item of advanced.finalized) {
            if (assignedSet.has(item.environmentIndex)) {
              finalizedAfterRecycleDisabled.add(
                item.environmentIndex,
              );
            }
          }
        }
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

    if (naturalRecycleDiagnostic) {
      const finalByWorker = await Promise.all(
        workerIds.map((workerId) =>
          rolloutPool!.getWorkerDiagnostics(workerId)),
      );
      finalEnvironmentDiagnostics = finalByWorker
        .flat()
        .sort((left, right) =>
          left.environmentIndex - right.environmentIndex);
      for (const [decisionThreshold, byWorker] of
        pendingDiagnosticSnapshots) {
        for (
          let workerId = 0;
          workerId < finalByWorker.length;
          workerId += 1
        ) {
          if (!byWorker.has(workerId)) {
            byWorker.set(workerId, finalByWorker[workerId]);
          }
        }
        publishDiagnosticSnapshot(decisionThreshold);
      }
    }

    if (naturalRecycleDefeatDiagnostic) {
      const finalByWorker = await Promise.all(
        workerIds.map((workerId) =>
          rolloutPool!.getWorkerDefeatDiagnostics(workerId)),
      );
      finalDefeatEnvironmentDiagnostics = finalByWorker
        .flat()
        .sort((left, right) =>
          left.environmentIndex - right.environmentIndex);
      if (twoTeamObservation) {
        const sameEpisode = finalDefeatEnvironmentDiagnostics.find(
          (snapshot) =>
            snapshot.environmentIndex
              === twoTeamObservation!.selectedEnvironmentIndex
            && snapshot.currentEpisodeSeed
              === twoTeamObservation!.selectedEpisodeSeed
            && snapshot.generation
              === twoTeamObservation!.selectedGeneration,
        );
        if (sameEpisode) {
          twoTeamObservation = {
            ...twoTeamObservation,
            latestSnapshot: sameEpisode,
          };
        }
      }
    }

    const rolloutMs = performance.now() - rolloutStarted;
    if (
      !continuousRecycle
      && summaryByEnvironment.size !== environmentCount
    ) {
      throw new Error(
        `PPO async rollout finalized ${summaryByEnvironment.size}/${environmentCount} environments`,
      );
    }

    const summaries = continuousRecycle
      ? completedSummaries
      : Array.from(
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
    if (
      continuousRecycle
      && new Set(
        summaries.map((summary) => summary.seed),
      ).size !== summaries.length
    ) {
      throw new Error(
        "PPO async continuous recycle produced duplicate episode seeds",
      );
    }

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

    const finalSelectedDefeatSnapshot =
      twoTeamObservation?.latestSnapshot;

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
      continuousRecycle,
      naturalRecycleProbe,
      ...(naturalRecycleDiagnostic
        ? { naturalRecycleDiagnostic: true }
        : {}),
      ...(naturalRecycleDefeatDiagnostic
        ? { naturalRecycleDefeatDiagnostic: true }
        : {}),
      targetDecisions: continuousRecycle
        ? targetDecisions
        : undefined,
      collectedDecisions: totalDecisions,
      decisionOvershoot: continuousRecycle
        ? Math.max(
            0,
            totalDecisions - (targetDecisions ?? 0),
          )
        : 0,
      completedEpisodes: summaries.length,
      drainingTriggered: draining,
      hardDecisionCapReached,
      recycledBeforeDrainEpisodeCount,
      naturalVictoryRecycledBeforeDrainCount,
      naturalVictoryRecycleConfirmed,
      naturalVictoryRecycleEnvironmentIndex,
      naturalVictoryRecycleSourceSeed,
      naturalVictoryRecycleNextSeed,
      episodeSeeds: summaries.map((summary) => summary.seed),
      checkpointStart: {
        updateCount: initialized.updateCount,
        episodeCount: initialized.episodeCount,
      },
      totalDecisions,
      finalizedEnvironments:
        naturalTerminationProbe
          ? new Set(
              summaries.map(
                (summary) => summary.environmentIndex,
              ),
            ).size
          : continuousRecycle
            ? environmentCount
            : summaries.length,
      summaries,
      ...(naturalRecycleDiagnostic
        ? {
            naturalVictorySeen: summaries.some(
              (summary) => summary.outcomeKind === "victory",
            ),
            diagnosticSnapshots,
            finalEnvironmentDiagnostics,
            standardSafetyLimits:
              PPO_PHASE_12B1_STANDARD_SAFETY_LIMITS,
          }
        : {}),
      ...(naturalRecycleDefeatDiagnostic
        ? {
            selectedEnvironmentIndex:
              twoTeamObservation?.selectedEnvironmentIndex
              ?? null,
            selectedEpisodeSeed:
              twoTeamObservation?.selectedEpisodeSeed ?? null,
            selectedGeneration:
              twoTeamObservation?.selectedGeneration ?? null,
            twoTeamEntryTurn:
              twoTeamObservation?.twoTeamEntryTurn ?? null,
            finalObservedTurn:
              finalSelectedDefeatSnapshot?.currentTurn ?? null,
            observedTurnsAfterTwoTeamEntry:
              twoTeamObservation && finalSelectedDefeatSnapshot
                ? Math.max(
                    0,
                    finalSelectedDefeatSnapshot.currentTurn
                    - twoTeamObservation.twoTeamEntryTurn,
                  )
                : 0,
            activeTeamIdsAtTwoTeamEntry:
              twoTeamObservation?.entrySnapshot
                .activeNonNeutralTeamIds ?? [],
            finalActiveTeamIds:
              finalSelectedDefeatSnapshot
                ?.activeNonNeutralTeamIds ?? [],
            defeatDiagnosticsAtEntry:
              twoTeamObservation?.entrySnapshot
                .defeatDiagnostics ?? [],
            finalDefeatDiagnostics:
              finalSelectedDefeatSnapshot
                ?.defeatDiagnostics ?? [],
            finalDefeatEnvironmentDiagnostics,
            naturalVictorySeen: summaries.some(
              (summary) => summary.outcomeKind === "victory",
            ),
            observationWindowCompleted:
              twoTeamObservation?.observationWindowCompleted
              ?? false,
            stopReason:
              defeatDiagnosticStopReason
              ?? (hardDecisionCapReached
                ? "hard_decision_cap_reached"
                : "stopped_without_two_team_observation"),
            observationWindowTurns:
              PPO_DEFEAT_DIAGNOSTIC_OBSERVATION_TURNS,
            periodicTurnInterval:
              PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL,
            defeatDiagnosticEventCount,
          }
        : {}),
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
