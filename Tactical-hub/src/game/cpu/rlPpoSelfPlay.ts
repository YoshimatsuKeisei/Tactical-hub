import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import { createRlObservationEncoderCache, encodeRlObservationV2 } from "./rlObservationEncoder";
import { PythonPpoClient, type PpoHyperparameters } from "./pythonPpoClient";
import type { PpoEncodedSample } from "./rlPpoPackedBatch";
import type { GameState } from "../types";

export type PpoClientLike = Pick<PythonPpoClient, "start" | "act" | "beginUpdate" | "accumulatePacked" | "finishUpdate" | "save" | "close">;

export const DEFAULT_PPO_HYPERPARAMETERS: PpoHyperparameters = {
  learningRate: 3e-4, gamma: 0.99, gaeLambda: 0.95, clipEpsilon: 0.2,
  valueCoefficient: 0.5, entropyCoefficient: 0.01, maxGradientNorm: 0.5,
};

/** Scalar-only rollout record. Encoded features are reconstructed after victory. */
export type PpoTrajectoryStep = {
  decisionIndex: number;
  turnNumber: number;
  phase: GameState["phase"];
  teamId: string;
  selectedActionIndex: number;
  selectedActionKey: string;
  oldLogProbability: number;
  value: number;
  reward: number;
  done: boolean;
  advantage?: number;
  return?: number;
};

export type PpoReplayRollout = {
  seed: number;
  terminal: boolean;
  endReason: "ongoing" | "victory" | "stopped";
  winnerTeamId?: string;
  finalStateHash: string;
  loserTeamIds: string[];
  trajectory: PpoTrajectoryStep[];
};

export function calculateTeamGae(steps: PpoTrajectoryStep[], gamma: number, gaeLambda: number) {
  let nextValue = 0;
  let advantage = 0;
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    const continuing = step.done ? 0 : 1;
    const delta = step.reward + gamma * nextValue * continuing - step.value;
    advantage = delta + gamma * gaeLambda * continuing * advantage;
    step.advantage = advantage;
    step.return = advantage + step.value;
    nextValue = step.value;
  }
  return steps;
}

export function finalizeVictoryTrajectory(steps: PpoTrajectoryStep[], rewards: Record<string, number>, hyperparameters: Pick<PpoHyperparameters, "gamma" | "gaeLambda">) {
  const byTeam = new Map<string, PpoTrajectoryStep[]>();
  for (const step of steps) {
    step.reward = 0; step.done = false;
    const teamSteps = byTeam.get(step.teamId) ?? [];
    teamSteps.push(step); byTeam.set(step.teamId, teamSteps);
  }
  for (const [teamId, teamSteps] of byTeam) {
    if (!teamSteps.length) continue;
    const last = teamSteps[teamSteps.length - 1];
    last.reward = rewards[teamId] ?? 0;
    last.done = true;
    calculateTeamGae(teamSteps, hyperparameters.gamma, hyperparameters.gaeLambda);
  }
  return steps;
}

function memorySnapshot(stage: string, decisionCount: number) {
  const usage = process.memoryUsage();
  const mb = (bytes: number) => Math.round(bytes / 1024 / 1024 * 10) / 10;
  const snapshot = { stage, decisionCount, heapUsedMb: mb(usage.heapUsed), heapTotalMb: mb(usage.heapTotal), rssMb: mb(usage.rss) };
  process.stderr.write(`[PPO memory] ${JSON.stringify(snapshot)}\n`);
  return snapshot;
}

function assertFiniteTrajectory(step: PpoTrajectoryStep) {
  if (![step.oldLogProbability, step.value, step.advantage, step.return].every(Number.isFinite)) {
    throw new Error(`PPO trajectory contains NaN or Inf at decision ${step.decisionIndex}`);
  }
}

export async function replayPpoTrajectory(input: {
  rollout: PpoReplayRollout;
  client: PpoClientLike;
  chunkSize: number;
  memoryLogInterval: number;
}) {
  const { rollout, client } = input;
  const environment = new RlEnvironmentV2();
  environment.reset(rollout.seed, 4);
  const encoderCache = createRlObservationEncoderCache();
  let chunk: PpoEncodedSample[] = [];
  let sentSamples = 0;
  memorySnapshot("replay_start", 0);
  const flush = async () => {
    if (!chunk.length) return;
    const sending = chunk;
    chunk = [];
    await client.accumulatePacked(sending);
    sentSamples += sending.length;
  };

  for (const expected of rollout.trajectory) {
    const actorTeamId = environment.getCurrentActorTeamId();
    const observation = actorTeamId ? environment.getObservationForEncoding(actorTeamId) : undefined;
    const prefix = `PPO replay mismatch seed=${rollout.seed} decision=${expected.decisionIndex}`;
    if (!actorTeamId || !observation) throw new Error(`${prefix}: no actor, expected team=${expected.teamId}`);
    if (actorTeamId !== expected.teamId || observation.turnNumber !== expected.turnNumber || observation.phase !== expected.phase) {
      throw new Error(`${prefix}: expected team/turn/phase=${expected.teamId}/${expected.turnNumber}/${expected.phase}, actual=${actorTeamId}/${observation.turnNumber}/${observation.phase}`);
    }
    const legalActions = environment.getLegalActionsForEncoding(actorTeamId);
    const actualIndex = legalActions.findIndex((action) => action.actionKey === expected.selectedActionKey);
    const actualKeyAtExpectedIndex = legalActions[expected.selectedActionIndex]?.actionKey;
    if (actualIndex < 0 || actualIndex !== expected.selectedActionIndex || actualKeyAtExpectedIndex !== expected.selectedActionKey) {
      throw new Error(`${prefix}: expected action index/key=${expected.selectedActionIndex}/${expected.selectedActionKey}, actual index/key=${actualIndex}/${actualKeyAtExpectedIndex ?? "missing"}`);
    }
    assertFiniteTrajectory(expected);
    chunk.push({
      observation: encodeRlObservationV2(observation, encoderCache),
      actions: encodeRlLegalActionsV2(observation, legalActions).actions,
      targetIndex: expected.selectedActionIndex,
      oldLogProbability: expected.oldLogProbability,
      advantage: expected.advantage!,
      return: expected.return!,
    });
    environment.stepWithoutObservation(expected.selectedActionKey);
    if (chunk.length >= input.chunkSize) await flush();
    if ((expected.decisionIndex + 1) % input.memoryLogInterval === 0) memorySnapshot("replay_progress", expected.decisionIndex + 1);
  }
  await flush();
  const result = environment.getResult();
  const finalStateHash = environment.getStateHash();
  if (result.terminal !== rollout.terminal || result.endReason !== rollout.endReason || result.winnerTeamId !== rollout.winnerTeamId
    || finalStateHash !== rollout.finalStateHash || JSON.stringify(result.loserTeamIds) !== JSON.stringify(rollout.loserTeamIds)) {
    throw new Error(`PPO replay final mismatch seed=${rollout.seed}: expected winner/hash=${rollout.winnerTeamId}/${rollout.finalStateHash}, actual=${result.winnerTeamId ?? "none"}/${finalStateHash}`);
  }
  if (sentSamples !== rollout.trajectory.length) throw new Error(`PPO replay sample count mismatch: ${sentSamples} != ${rollout.trajectory.length}`);
  memorySnapshot("replay_end", sentSamples);
  return sentSamples;
}

export async function runPpoSelfPlaySmoke(input: {
  seed: number;
  episodes?: number;
  initialCheckpoint: string;
  outputCheckpoint: string;
  bestCheckpoint?: string;
  resume?: string;
  hyperparameters?: Partial<PpoHyperparameters>;
  safetyMaxTurns?: number;
  safetyMaxActions?: number;
  replayChunkSize?: number;
  memoryLogInterval?: number;
  client?: PpoClientLike;
}) {
  const hyperparameters = { ...DEFAULT_PPO_HYPERPARAMETERS, ...input.hyperparameters };
  const replayChunkSize = input.replayChunkSize ?? 8;
  const memoryLogInterval = input.memoryLogInterval ?? 500;
  if (!Number.isInteger(replayChunkSize) || replayChunkSize <= 0) throw new Error("replayChunkSize must be a positive integer");
  if (!Number.isInteger(memoryLogInterval) || memoryLogInterval <= 0) throw new Error("memoryLogInterval must be a positive integer");
  const probe = new RlEnvironmentV2();
  const first = probe.reset(input.seed, 4);
  const featureSpec = createRlFeatureSpecV2(first);
  const client = input.client ?? new PythonPpoClient();
  const initialized = await client.start({ seed: input.seed, featureSpec, hyperparameters, initialCheckpoint: input.initialCheckpoint, resume: input.resume });
  const completedRollouts: PpoReplayRollout[] = [];
  const truncated: Array<{ seed: number; reason: string; decisionCount: number }> = [];
  let mergeLegalActionCount = 0;
  try {
    for (let episodeIndex = 0; episodeIndex < (input.episodes ?? 1); episodeIndex += 1) {
      const seed = input.seed + initialized.episodeCount + episodeIndex;
      const environment = new RlEnvironmentV2();
      environment.reset(seed, 4);
      const encoderCache = createRlObservationEncoderCache();
      const trajectory: PpoTrajectoryStep[] = [];
      let reason: string | undefined;
      memorySnapshot("rollout_start", 0);
      try {
        while (!environment.isTerminal()) {
          const actor = environment.getCurrentActorTeamId();
          if (!actor) { reason = "no_actor"; break; }
          const observation = environment.getObservationForEncoding(actor);
          if (observation.turnNumber > (input.safetyMaxTurns ?? 1_000)) { reason = "safety_turn_limit"; break; }
          if (trajectory.length >= (input.safetyMaxActions ?? 100_000)) { reason = "safety_action_limit"; break; }
          const legal = environment.getLegalActionsForEncoding(actor);
          if (!legal.length) { reason = "no_legal_actions"; break; }
          const encodedObservation = encodeRlObservationV2(observation, encoderCache);
          const encodedActions = encodeRlLegalActionsV2(observation, legal);
          mergeLegalActionCount += legal.filter((action) => action.actionType === "merge_infantry").length;
          const selected = await client.act(encodedObservation, encodedActions);
          const before = environment.getProgressHash();
          environment.stepWithoutObservation(selected.actionKey);
          trajectory.push({
            decisionIndex: trajectory.length, turnNumber: observation.turnNumber, phase: observation.phase,
            teamId: actor, selectedActionIndex: selected.actionIndex, selectedActionKey: selected.actionKey,
            oldLogProbability: selected.logProbability, value: selected.value, reward: 0, done: false,
          });
          if (environment.getProgressHash() === before) { reason = "phase_stall"; break; }
          if (trajectory.length % memoryLogInterval === 0) memorySnapshot("rollout_progress", trajectory.length);
        }
      } catch (error) {
        reason = `exception:${error instanceof Error ? error.message : String(error)}`;
      }
      const result = environment.getResult();
      if (!reason && result.terminal && result.endReason === "victory") {
        const { checkHeadlessInvariants } = await import("./headlessSimulation");
        const violations = checkHeadlessInvariants(environment.getStateForValidation());
        if (violations.length) reason = `invariant_violation:${violations.join(" | ")}`;
      }
      memorySnapshot("rollout_end", trajectory.length);
      if (!reason && result.terminal && result.endReason === "victory" && result.winnerTeamId) {
        finalizeVictoryTrajectory(trajectory, result.rewards, hyperparameters);
        completedRollouts.push({ seed, terminal: true, endReason: "victory", winnerTeamId: result.winnerTeamId, loserTeamIds: result.loserTeamIds, finalStateHash: environment.getStateHash(), trajectory });
      } else {
        truncated.push({ seed, reason: reason ?? result.endReason, decisionCount: trajectory.length });
      }
    }
    const totalSamples = completedRollouts.reduce((sum, rollout) => sum + rollout.trajectory.length, 0);
    if (!completedRollouts.length || !totalSamples) throw new Error("PPO Smoke produced no normal victory trajectory; truncated episodes are excluded from replay and updates");
    await client.beginUpdate(totalSamples);
    let replayedSamples = 0;
    for (const rollout of completedRollouts) replayedSamples += await replayPpoTrajectory({ rollout, client, chunkSize: replayChunkSize, memoryLogInterval });
    if (replayedSamples !== totalSamples) throw new Error(`PPO replay total mismatch: ${replayedSamples} != ${totalSamples}`);
    const update = await client.finishUpdate(completedRollouts.length);
    memorySnapshot("ppo_update_end", replayedSamples);
    const metadata = { purpose: "phase_12b_smoke", truncatedEpisodeCount: truncated.length, replayedSamples };
    const saved = await client.save(input.outputCheckpoint, metadata);
    const bestSaved = input.bestCheckpoint ? await client.save(input.bestCheckpoint, { ...metadata, checkpointRole: "best" }) : undefined;
    return {
      featureSpec,
      completed: completedRollouts.map((rollout) => ({ seed: rollout.seed, winnerTeamId: rollout.winnerTeamId!, decisionCount: rollout.trajectory.length })),
      truncated, mergeLegalActionCount, replayedSamples, update, saved, bestSaved, selectedDevice: initialized.selectedDevice,
    };
  } finally {
    await client.close();
  }
}
