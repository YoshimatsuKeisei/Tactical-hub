import type { GameConfig } from "../types";
import { encodeRlLegalActions, type EncodedLegalActions } from "./rlActionEncoder";
import { RlEnvironment, type RlResult } from "./rlEnvironment";
import {
  createRlObservationEncoderCache,
  encodeRlObservation,
  type EncodedObservation,
} from "./rlObservationEncoder";
import { createHeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { createRlFeatureSpec, type RlFeatureSpec } from "./rlFeatureSpec";
import {
  getRlReplayIdentity,
  RL_REPLAY_RNG_SIDECAR_SCHEMA_VERSION,
  validateRlReplayRngSidecar,
  type RlReplayRngSidecar,
} from "./rlReplayRngSidecar";

export const RL_IMITATION_REPLAY_SCHEMA_VERSION = 1;

export type RlImitationEpisodeHeader = {
  type: "episode_start";
  schemaVersion: typeof RL_IMITATION_REPLAY_SCHEMA_VERSION;
  episodeId: string;
  seed: number;
  participantCount: 4;
  mapId: string;
  gameConfig: GameConfig;
  maxTurns: number;
};

export type RlImitationDecisionRecord = {
  type: "decision";
  episodeId: string;
  teamId: string;
  selectedActionKey: string;
  turnNumber: number;
};

export type RlImitationEpisodeEnd = {
  type: "episode_end";
  episodeId: string;
  terminal: boolean;
  endReason: RlResult["endReason"] | "max_turns";
  winnerTeamId?: string;
  loserTeamIds: string[];
  endTurn: number;
  decisionCount: number;
  finalStateHash: string;
};

export type RlImitationReplayRecord =
  | RlImitationEpisodeHeader
  | RlImitationDecisionRecord
  | RlImitationEpisodeEnd;

export type RlImitationEpisode = {
  header: RlImitationEpisodeHeader;
  decisions: RlImitationDecisionRecord[];
  end: RlImitationEpisodeEnd;
};

export type RlReplayEncodedDecision = {
  record: RlImitationDecisionRecord;
  encodedObservation: EncodedObservation;
  encodedLegalActions: EncodedLegalActions;
  selectedActionIndex: number;
};
export type RlReplayPrefixProfile = {
  getObservationMs: number;
  getLegalActionsMs: number;
  encodeObservationMs: number;
  encodeLegalActionsMs: number;
  stepReplayActionMs: number;
};

export type RlImitationEpisodeResult = RlImitationEpisodeEnd & {
  seed: number;
  sampleCountByTeam: Record<string, number>;
};

export function getRlImitationHeaderFeatureSpec(header: RlImitationEpisodeHeader) {
  const environment = new RlEnvironment();
  const observation = environment.reset(header.seed, header.participantCount);
  if (
    observation.config.mapId !== header.mapId
    || JSON.stringify(observation.config) !== JSON.stringify(header.gameConfig)
  ) {
    throw new Error(`Replay initial game settings do not match episode ${header.episodeId}`);
  }
  return createRlFeatureSpec(observation);
}
export function getRlImitationEpisodeFeatureSpec(episode: RlImitationEpisode) {
  return getRlImitationHeaderFeatureSpec(episode.header);
}

function currentDecision(
  environment: RlEnvironment,
  expected: Pick<RlImitationDecisionRecord, "teamId" | "turnNumber">,
) {
  const actorTeamId = environment.getCurrentActorTeamId();
  if (actorTeamId !== expected.teamId) {
    throw new Error(`Replay actor mismatch: expected ${expected.teamId}, received ${actorTeamId ?? "none"}`);
  }
  const observation = environment.getObservationForEncoding(actorTeamId);
  if (observation.turnNumber !== expected.turnNumber) {
    throw new Error(`Replay turn mismatch: expected ${expected.turnNumber}, received ${observation.turnNumber}`);
  }
  const legalActions = environment.getLegalActions(actorTeamId);
  return { actorTeamId, observation, legalActions };
}

export async function runHeuristicImitationEpisode(input: {
  episodeId: string;
  seed: number;
  maxTurns?: number;
  onRecord: (record: RlImitationReplayRecord) => void | Promise<void>;
}): Promise<RlImitationEpisodeResult> {
  const environment = new RlEnvironment();
  const firstObservation = environment.reset(input.seed, 4);
  const policy = createHeuristicCpuPolicy();
  policy.setDecisionDiagnosticsEnabled(true);
  const maxTurns = input.maxTurns ?? 300;
  const header: RlImitationEpisodeHeader = {
    type: "episode_start",
    schemaVersion: RL_IMITATION_REPLAY_SCHEMA_VERSION,
    episodeId: input.episodeId,
    seed: input.seed,
    participantCount: 4,
    mapId: firstObservation.config.mapId,
    gameConfig: firstObservation.config,
    maxTurns,
  };
  await input.onRecord(header);

  let decisionCount = 0;
  const sampleCountByTeam: Record<string, number> = Object.fromEntries(
    firstObservation.teams.filter((team) => !team.isNeutral).map((team) => [team.id, 0]),
  );
  let lastTurn = firstObservation.turnNumber;

  while (!environment.isTerminal()) {
    const teamId = environment.getCurrentActorTeamId();
    if (!teamId) throw new Error("Heuristic imitation collector reached a decision without an actor");
    const observation = environment.getObservationForEncoding(teamId);
    lastTurn = observation.turnNumber;
    if (observation.turnNumber > maxTurns) break;
    const legalActions = environment.getLegalActions(teamId);
    if (!legalActions.length) throw new Error(`No legal actions for ${teamId}`);

    environment.stepWithPolicy(policy);
    const selectedActionKey = policy.getLastDecisionDiagnostics()?.selectedActionKey;
    if (!selectedActionKey) throw new Error(`Heuristic policy did not report its selected action for ${teamId}`);
    if (!legalActions.some((action) => action.actionKey === selectedActionKey)) {
      throw new Error(`Heuristic selected action outside legal actions: ${selectedActionKey}`);
    }
    await input.onRecord({
      type: "decision",
      episodeId: input.episodeId,
      teamId,
      selectedActionKey,
      turnNumber: observation.turnNumber,
    });
    decisionCount += 1;
    sampleCountByTeam[teamId] = (sampleCountByTeam[teamId] ?? 0) + 1;
  }

  const result = environment.getResult();
  const end: RlImitationEpisodeEnd = {
    type: "episode_end",
    episodeId: input.episodeId,
    terminal: result.terminal,
    winnerTeamId: result.winnerTeamId,
    loserTeamIds: result.loserTeamIds,
    endReason: result.terminal ? result.endReason : "max_turns",
    endTurn: lastTurn,
    decisionCount,
    finalStateHash: environment.getStateHash(),
  };
  await input.onRecord(end);
  return { ...end, seed: input.seed, sampleCountByTeam };
}

export async function replayHeuristicImitationEpisode(input: {
  episode: RlImitationEpisode;
  onFeatureSpec?: (featureSpec: RlFeatureSpec) => void | Promise<void>;
  onEncodedDecision?: (decision: RlReplayEncodedDecision) => void | Promise<void>;
  rngSidecar?: RlReplayRngSidecar;
  timing?: { directReplayMs: number };
}): Promise<RlImitationEpisodeEnd> {
  const { header, decisions, end: expectedEnd } = input.episode;
  if (header.schemaVersion !== RL_IMITATION_REPLAY_SCHEMA_VERSION) {
    throw new Error(`Unsupported replay schemaVersion ${header.schemaVersion}`);
  }
  const environment = new RlEnvironment();
  const initial = environment.reset(header.seed, header.participantCount);
  if (initial.config.mapId !== header.mapId || JSON.stringify(initial.config) !== JSON.stringify(header.gameConfig)) {
    throw new Error("Replay initial game settings do not match the saved episode");
  }
  await input.onFeatureSpec?.(createRlFeatureSpec(initial));
  if (input.rngSidecar) validateRlReplayRngSidecar(input.episode, input.rngSidecar);
  const policy = input.rngSidecar ? undefined : createHeuristicCpuPolicy();
  policy?.setDecisionDiagnosticsEnabled(true);
  const observationEncoderCache = createRlObservationEncoderCache();

  let directReplayMs = 0;
  for (const [decisionIndex, record] of decisions.entries()) {
    const decisionStarted = performance.now();
    const { observation, legalActions } = currentDecision(environment, record);
    const selectedActionIndex = legalActions.findIndex((action) => action.actionKey === record.selectedActionKey);
    if (selectedActionIndex < 0) {
      throw new Error(`Replay action is not legal at decision ${record.selectedActionKey}`);
    }
    let callbackMs = 0;
    if (input.onEncodedDecision) {
      const callbackStarted = performance.now();
      await input.onEncodedDecision({
        record,
        encodedObservation: encodeRlObservation(observation, observationEncoderCache),
        encodedLegalActions: encodeRlLegalActions(observation, legalActions),
        selectedActionIndex,
      });
      callbackMs = performance.now() - callbackStarted;
    }
    if (input.rngSidecar) {
      environment.stepReplayAction(record.selectedActionKey, input.rngSidecar.rngStatesAfterPolicy[decisionIndex]);
    } else {
      environment.stepWithPolicy(policy!);
      const reproducedKey = policy!.getLastDecisionDiagnostics()?.selectedActionKey;
      if (reproducedKey !== record.selectedActionKey) {
        throw new Error(`Replay Heuristic selection mismatch: expected ${record.selectedActionKey}, received ${reproducedKey ?? "none"}`);
      }
    }
    directReplayMs += Math.max(0, performance.now() - decisionStarted - callbackMs);
  }

  const result = environment.getResult();
  const replayEndTurn = environment.getObservation(initial.observingTeamId).turnNumber;
  const actual: RlImitationEpisodeEnd = {
    type: "episode_end",
    episodeId: header.episodeId,
    terminal: result.terminal,
    winnerTeamId: result.winnerTeamId,
    loserTeamIds: result.loserTeamIds,
    endReason: result.terminal ? result.endReason : "max_turns",
    endTurn: replayEndTurn,
    decisionCount: decisions.length,
    finalStateHash: environment.getStateHash(),
  };
  if (
    actual.finalStateHash !== expectedEnd.finalStateHash
    || actual.terminal !== expectedEnd.terminal
    || actual.endReason !== expectedEnd.endReason
    || actual.winnerTeamId !== expectedEnd.winnerTeamId
    || actual.endTurn !== expectedEnd.endTurn
    || actual.decisionCount !== expectedEnd.decisionCount
    || JSON.stringify(actual.loserTeamIds) !== JSON.stringify(expectedEnd.loserTeamIds)
  ) {
    throw new Error(`Replay final result mismatch for ${header.episodeId}`);
  }
  if (input.timing) input.timing.directReplayMs = directReplayMs;
  return actual;
}

export async function generateRlReplayRngSidecar(episode: RlImitationEpisode): Promise<RlReplayRngSidecar> {
  const { header, decisions, end: expectedEnd } = episode;
  if (header.schemaVersion !== RL_IMITATION_REPLAY_SCHEMA_VERSION) {
    throw new Error(`Unsupported replay schemaVersion ${header.schemaVersion}`);
  }
  const environment = new RlEnvironment();
  const initial = environment.reset(header.seed, header.participantCount);
  if (initial.config.mapId !== header.mapId || JSON.stringify(initial.config) !== JSON.stringify(header.gameConfig)) {
    throw new Error("Replay initial game settings do not match the saved episode");
  }
  const policy = createHeuristicCpuPolicy();
  policy.setDecisionDiagnosticsEnabled(true);
  const rngStatesAfterPolicy: number[] = [];
  for (const record of decisions) {
    const { legalActions } = currentDecision(environment, record);
    if (!legalActions.some((action) => action.actionKey === record.selectedActionKey)) {
      throw new Error(`Replay action is not legal at decision ${record.selectedActionKey}`);
    }
    const applied = environment.stepWithPolicyForReplay(policy);
    if (applied.actionKey !== record.selectedActionKey) {
      throw new Error(`Replay Heuristic selection mismatch: expected ${record.selectedActionKey}, received ${applied.actionKey}`);
    }
    rngStatesAfterPolicy.push(applied.rngStateAfterPolicy);
  }
  const result = environment.getResult();
  const actual: RlImitationEpisodeEnd = {
    type: "episode_end",
    episodeId: header.episodeId,
    terminal: result.terminal,
    winnerTeamId: result.winnerTeamId,
    loserTeamIds: result.loserTeamIds,
    endReason: result.terminal ? result.endReason : "max_turns",
    endTurn: environment.getObservation(initial.observingTeamId).turnNumber,
    decisionCount: decisions.length,
    finalStateHash: environment.getStateHash(),
  };
  if (
    actual.finalStateHash !== expectedEnd.finalStateHash
    || actual.terminal !== expectedEnd.terminal
    || actual.endReason !== expectedEnd.endReason
    || actual.winnerTeamId !== expectedEnd.winnerTeamId
    || actual.endTurn !== expectedEnd.endTurn
    || actual.decisionCount !== expectedEnd.decisionCount
    || JSON.stringify(actual.loserTeamIds) !== JSON.stringify(expectedEnd.loserTeamIds)
  ) {
    throw new Error(`Replay final result mismatch while generating RNG sidecar for ${header.episodeId}`);
  }
  return {
    schemaVersion: RL_REPLAY_RNG_SIDECAR_SCHEMA_VERSION,
    replayIdentity: getRlReplayIdentity(episode),
    episodeId: header.episodeId,
    seed: header.seed,
    decisionCount: decisions.length,
    rngStatesAfterPolicy,
  };
}

export async function replayRlImitationEpisodePrefix(input: {
  episode: RlImitationEpisode;
  rngSidecar: RlReplayRngSidecar;
  maxDecisions?: number;
  onEncodedDecision: (decision: RlReplayEncodedDecision) => void | Promise<void>;
  onDecisionCompleted?: (profile: RlReplayPrefixProfile) => void | Promise<void>;
}): Promise<{ decisionCount: number; profile: RlReplayPrefixProfile }> {
  validateRlReplayRngSidecar(input.episode, input.rngSidecar);
  const environment = new RlEnvironment();
  const initial = environment.reset(input.episode.header.seed, input.episode.header.participantCount);
  if (initial.config.mapId !== input.episode.header.mapId || JSON.stringify(initial.config) !== JSON.stringify(input.episode.header.gameConfig)) {
    throw new Error("Replay initial game settings do not match the saved episode");
  }
  const profile: RlReplayPrefixProfile = {
    getObservationMs: 0,
    getLegalActionsMs: 0,
    encodeObservationMs: 0,
    encodeLegalActionsMs: 0,
    stepReplayActionMs: 0,
  };
  const observationEncoderCache = createRlObservationEncoderCache();
  const limit = Math.min(input.maxDecisions ?? input.episode.decisions.length, input.episode.decisions.length);
  for (let index = 0; index < limit; index += 1) {
    const record = input.episode.decisions[index];
    const actorTeamId = environment.getCurrentActorTeamId();
    if (actorTeamId !== record.teamId) throw new Error(`Replay actor mismatch: expected ${record.teamId}, received ${actorTeamId ?? "none"}`);
    let started = performance.now();
    const observation = environment.getObservationForEncoding(actorTeamId);
    profile.getObservationMs += performance.now() - started;
    if (observation.turnNumber !== record.turnNumber) throw new Error(`Replay turn mismatch: expected ${record.turnNumber}, received ${observation.turnNumber}`);
    started = performance.now();
    const legalActions = environment.getLegalActions(actorTeamId);
    profile.getLegalActionsMs += performance.now() - started;
    const selectedActionIndex = legalActions.findIndex((action) => action.actionKey === record.selectedActionKey);
    if (selectedActionIndex < 0) throw new Error(`Replay action is not legal at decision ${record.selectedActionKey}`);
    started = performance.now();
    const encodedObservation = encodeRlObservation(observation, observationEncoderCache);
    profile.encodeObservationMs += performance.now() - started;
    started = performance.now();
    const encodedLegalActions = encodeRlLegalActions(observation, legalActions);
    profile.encodeLegalActionsMs += performance.now() - started;
    await input.onEncodedDecision({ record, encodedObservation, encodedLegalActions, selectedActionIndex });
    started = performance.now();
    environment.stepReplayAction(record.selectedActionKey, input.rngSidecar.rngStatesAfterPolicy[index]);
    profile.stepReplayActionMs += performance.now() - started;
    await input.onDecisionCompleted?.({ ...profile });
  }
  return { decisionCount: limit, profile };
}

export function parseRlImitationReplayRecords(records: readonly RlImitationReplayRecord[]): RlImitationEpisode[] {
  const episodes: RlImitationEpisode[] = [];
  let current: { header: RlImitationEpisodeHeader; decisions: RlImitationDecisionRecord[] } | undefined;
  for (const record of records) {
    if (record.type === "episode_start") {
      if (current) throw new Error(`Episode ${current.header.episodeId} has no end record`);
      current = { header: record, decisions: [] };
    } else if (record.type === "decision") {
      if (!current || record.episodeId !== current.header.episodeId) throw new Error(`Decision outside matching episode: ${record.episodeId}`);
      current.decisions.push(record);
    } else {
      if (!current || record.episodeId !== current.header.episodeId) throw new Error(`End outside matching episode: ${record.episodeId}`);
      episodes.push({ ...current, end: record });
      current = undefined;
    }
  }
  if (current) throw new Error(`Episode ${current.header.episodeId} has no end record`);
  return episodes;
}
