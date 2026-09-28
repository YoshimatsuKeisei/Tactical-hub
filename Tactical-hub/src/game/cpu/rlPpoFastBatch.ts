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
import {
  DEFAULT_PPO_HYPERPARAMETERS,
  finalizeTerminalTrajectory,
  finalizeVictoryTrajectory,
  replayPpoTrajectoryFromRetention,
  validatePpoTrajectoryReplay,
  type PpoReplayRollout,
  type PpoTrajectoryStep,
} from "./rlPpoSelfPlay";

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

type FastBatchSlot = {
  environmentIndex: number;
  seed: number;
  environment: RlEnvironmentV2;
  encoderCache: ReturnType<typeof createRlObservationEncoderCache>;
  trajectory: PpoTrajectoryStep[];
  retentionIds: string[];
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

export async function runPpoFastBatchSmoke(input: PpoFastBatchInput) {
  const environmentCount = input.environmentCount ?? 8;
  if (environmentCount !== 8) {
    throw new Error("PPO fast_batch_v1 requires exactly 8 environments");
  }

  const safetyMaxTurns = input.safetyMaxTurns ?? 1_000;
  const safetyMaxActions = input.safetyMaxActions ?? 100_000;
  const replayChunkSize = input.replayChunkSize ?? 32;
  const memoryLogInterval = input.memoryLogInterval ?? 5_000;
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
      PPO_PACKED_PREPARE_MODE: "fast_batch_v1",
    },
  });

  const outstandingRetentionIds = new Set<string>();
  const started = performance.now();
  const initialized = await client.start({
    seed: input.seed,
    featureSpec,
    hyperparameters,
    initialCheckpoint: input.initialCheckpoint,
    resume: input.resume,
  });
  if (initialized.selectedDevice !== "cuda") {
    throw new Error("PPO fast_batch_v1 requires CUDA");
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
        retentionIds: [],
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
    if (slot.retentionIds.length !== slot.trajectory.length) {
      reason = (
        `exception:PPO fast retention sample count mismatch `
        + `${slot.retentionIds.length} != ${slot.trajectory.length}`
      );
    }

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
        retentionIds: slot.retentionIds,
      };
    } else if (slot.retentionIds.length) {
      await client.discardRetained(slot.retentionIds);
      for (const retentionId of slot.retentionIds) {
        outstandingRetentionIds.delete(retentionId);
      }
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
        retentionId: string;
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

        const observation = slot.environment.getObservationForEncoding(actor);
        const legal = slot.environment.getLegalActionsForEncoding(actor);
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

        const encodedObservation = encodeRlObservationV2(
          observation,
          slot.encoderCache,
        );
        const encodedActions = encodeRlLegalActionsV2(observation, legal);
        mergeLegalActionCount += legal.filter(
          (action) => action.actionType === "merge_infantry",
        ).length;

        const retentionId = (
          `fast-v1-env-${slot.environmentIndex}`
          + `-seed-${slot.seed}`
          + `-decision-${slot.trajectory.length}`
        );

        batch.push({
          slot,
          actor,
          observation,
          encodedObservation,
          encodedActions,
          retentionId,
          progressHash: slot.environment.getProgressHash(),
        });
      }

      if (!batch.length) continue;

      const selected = await client.actBatch(
        batch.map((entry) => ({
          observation: entry.encodedObservation,
          legalActions: entry.encodedActions,
        })),
        {
          retentionIds: batch.map((entry) => entry.retentionId),
        },
      );

      batchRounds += 1;
      totalDecisions += batch.length;

      for (let index = 0; index < batch.length; index += 1) {
        const entry = batch[index];
        const action = selected[index];
        entry.slot.retentionIds.push(entry.retentionId);
        outstandingRetentionIds.add(entry.retentionId);

        entry.slot.environment.stepWithoutObservation(action.actionKey);
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
          entry.slot.environment.getProgressHash()
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
        "PPO fast_batch_v1 requires all 8 environments "
        + "to produce learnable completed/adjudicated trajectories",
      );
    }

    await client.beginUpdate(totalSamples);
    const retentionBeforeUpdate = await client.retentionStats();

    let validatedSamples = 0;
    let replayedSamples = 0;
    const replayStarted = performance.now();

    // Stable env-index order is the fast-mode sample merge contract.
    for (const rollout of learnableRollouts) {
      validatedSamples += await validatePpoTrajectoryReplay({
        rollout,
        memoryLogInterval,
        fastRlMovement: true,
        fastRlPhaseTransitions: true,
      });
      replayedSamples += await replayPpoTrajectoryFromRetention({
        rollout,
        client,
        chunkSize: replayChunkSize,
        memoryLogInterval,
        onConsumed: (retentionId) => {
          outstandingRetentionIds.delete(retentionId);
        },
      });
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

    const update = await client.finishUpdate(learnableRollouts.length);

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
      purpose: "phase_12b_fast_batch_v1",
      ppoMode: "fast_batch_v1",
      environmentCount,
      seeds: slots.map((slot) => slot.seed),
      mergeOrder: slots.map((slot) => slot.environmentIndex),
      victoryEpisodeCount,
      adjudicatedEpisodeCount,
      truncatedEpisodeCount,
      replayedSamples,
    };

    const saved = await client.save(
      input.outputCheckpoint,
      metadata,
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
      mode: "fast_batch_v1",
      environmentCount,
      seeds: slots.map((slot) => slot.seed),
      mergeOrder: slots.map((slot) => slot.environmentIndex),
      batchRounds,
      totalDecisions,
      totalSamples,
      summaries,
      replayedSamples,
      validatedSamples,
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
    await client.close();
  }
}
