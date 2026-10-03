import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { encodeRlLegalActionsV2, type EncodedLegalActionsV2 } from "./rlActionEncoder";
import { calculateBattleAdvantage } from "../engine/battleAdvantage";
import type { GameResultReason, GameState } from "../types";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2, type RlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
  type EncodedObservation,
} from "./rlObservationEncoder";
import { adjudicatePpoTimeLimit } from "./rlPpoAdjudication";
import { DEFAULT_PPO_HYPERPARAMETERS } from "./rlPpoSelfPlay";
import { createPpoTurnDiagnostics, type PpoTurnDiagnostics } from "./rlPpoTurnDiagnostics";
import { PythonPpoClient, type PpoHyperparameters } from "./pythonPpoClient";
import type { RlTorchDevice } from "./rlTorchDevice";

export const UPDATE2_CHECKPOINT_SHA256 =
  "5caa54ac43386e2623a7c2eb6fdbd74e272277963e5030e157f20ffdcce63f72";
export const UPDATE3_CHECKPOINT_SHA256 =
  "c7942448772ce4afc3c3e39ff49e5d8b9c27106735e67a321bcc3aeb961060b1";

export type PpoEvaluationCheckpointLabel = "update2" | "update3";

export type PpoEvaluationSeatAssignment = {
  seed: number;
  rotationIndex: number;
  update3TeamId: string;
  checkpointByTeam: Record<string, PpoEvaluationCheckpointLabel>;
};

export type PpoEvaluationPolicyClient = {
  start(input: {
    seed: number;
    featureSpec: RlFeatureSpecV2;
    hyperparameters: PpoHyperparameters;
    initialCheckpoint: string;
    evaluationCheckpoint: string;
  }): Promise<unknown>;
  act(
    observation: EncodedObservation,
    legalActions: EncodedLegalActionsV2,
  ): Promise<{ actionKey: string }>;
  close(): Promise<void>;
};

export type PpoEvaluationPolicyClientFactory = (input: {
  teamId: string;
  checkpointLabel: PpoEvaluationCheckpointLabel;
  checkpointPath: string;
  seed: number;
}) => PpoEvaluationPolicyClient;

type PpoEvaluationEnvironment = Pick<
  RlEnvironmentV2,
  | "reset"
  | "isTerminal"
  | "getCurrentActorTeamId"
  | "getObservationForEncoding"
  | "getLegalActionsForEncoding"
  | "stepWithoutObservation"
  | "getProgressHash"
  | "getResult"
  | "getStateHash"
  | "getStateForValidation"
>;

export type PpoEvaluationTeamResult = {
  teamId: string;
  checkpoint: PpoEvaluationCheckpointLabel;
  status: GameState["teams"][number]["status"];
  ownedBaseCount: number;
  livingKingHp: number;
  totalLivingHp: number;
  livingUnitCount: number;
  battleAdvantageRaw: number;
  battleAdvantageShare: number;
  rank: number;
};

export type PpoEvaluationMatchResult = {
  seed: number;
  rotationIndex: number;
  update3TeamId: string;
  checkpointByTeam: Record<string, PpoEvaluationCheckpointLabel>;
  winnerTeamId: string | null;
  resultReason: GameResultReason;
  finalStateHash: string;
  decisionCount: number;
  turnDiagnostics: PpoTurnDiagnostics;
  teams: PpoEvaluationTeamResult[];
};

export type PpoEvaluationAggregate = {
  matchCount: number;
  update3WinCount: number;
  update3AverageRank: number | null;
  update3RankDistribution: Record<string, number>;
  update3AverageFinalMetrics: {
    ownedBaseCount: number | null;
    livingKingHp: number | null;
    totalLivingHp: number | null;
    livingUnitCount: number | null;
    battleAdvantageRaw: number | null;
    battleAdvantageShare: number | null;
  };
  averageMatchTurn: number | null;
  averageFirstDefeatTurn: number | null;
  averageFinalDuelEntryTurn: number | null;
  averageFinalDuelDurationTurns: number | null;
  resultReasonCounts: Record<string, number>;
};

export type PpoCheckpointEvaluationResult = {
  checkpoints: {
    update2: { path: string; sha256: string };
    update3: { path: string; sha256: string };
  };
  seedStart: number;
  seedCount: number;
  matches: PpoEvaluationMatchResult[];
  aggregate: PpoEvaluationAggregate;
};

export function createPpoEvaluationSeatAssignments(input: {
  seedStart: number;
  seedCount: number;
  teamIds?: readonly string[];
}): PpoEvaluationSeatAssignment[] {
  const teamIds = input.teamIds ?? ["team-1", "team-2", "team-3", "team-4"];
  if (!Number.isInteger(input.seedStart) || input.seedStart < 0) {
    throw new Error("PPO evaluation seedStart must be a non-negative integer");
  }
  if (!Number.isInteger(input.seedCount) || input.seedCount <= 0) {
    throw new Error("PPO evaluation seedCount must be a positive integer");
  }
  if (teamIds.length !== 4 || new Set(teamIds).size !== 4) {
    throw new Error("PPO evaluation requires four unique team IDs");
  }

  return Array.from({ length: input.seedCount }, (_, seedOffset) =>
    teamIds.map((update3TeamId, rotationIndex) => ({
      seed: input.seedStart + seedOffset,
      rotationIndex,
      update3TeamId,
      checkpointByTeam: Object.fromEntries(
        teamIds.map((teamId) => [
          teamId,
          (teamId === update3TeamId ? "update3" : "update2") as PpoEvaluationCheckpointLabel,
        ]),
      ),
    })),
  ).flat();
}

function average(values: readonly number[]) {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

export function aggregatePpoCheckpointEvaluation(
  matches: readonly PpoEvaluationMatchResult[],
): PpoEvaluationAggregate {
  const update3Teams = matches.map((match) => {
    const team = match.teams.find((candidate) =>
      candidate.teamId === match.update3TeamId,
    );
    if (!team) {
      throw new Error(`Missing update3 team result: ${match.update3TeamId}`);
    }
    return team;
  });
  const update3RankDistribution: Record<string, number> = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
  };
  for (const team of update3Teams) {
    const rank = String(team.rank);
    update3RankDistribution[rank] = (update3RankDistribution[rank] ?? 0) + 1;
  }
  const firstDefeatTurns = matches.flatMap((match) => {
    const turns = Object.values(match.turnDiagnostics.defeatedTeamTurns);
    return turns.length ? [Math.min(...turns)] : [];
  });
  const finalDuels = matches.flatMap((match) =>
    match.turnDiagnostics.finalDuel ? [match.turnDiagnostics.finalDuel] : [],
  );
  const resultReasonCounts: Record<string, number> = {};
  for (const match of matches) {
    resultReasonCounts[match.resultReason] =
      (resultReasonCounts[match.resultReason] ?? 0) + 1;
  }

  return {
    matchCount: matches.length,
    update3WinCount: matches.filter((match) =>
      match.winnerTeamId === match.update3TeamId,
    ).length,
    update3AverageRank: average(update3Teams.map((team) => team.rank)),
    update3RankDistribution,
    update3AverageFinalMetrics: {
      ownedBaseCount: average(update3Teams.map((team) => team.ownedBaseCount)),
      livingKingHp: average(update3Teams.map((team) => team.livingKingHp)),
      totalLivingHp: average(update3Teams.map((team) => team.totalLivingHp)),
      livingUnitCount: average(update3Teams.map((team) => team.livingUnitCount)),
      battleAdvantageRaw: average(update3Teams.map((team) => team.battleAdvantageRaw)),
      battleAdvantageShare: average(update3Teams.map((team) => team.battleAdvantageShare)),
    },
    averageMatchTurn: average(
      matches.map((match) => match.turnDiagnostics.finalStateTurnNumber),
    ),
    averageFirstDefeatTurn: average(firstDefeatTurns),
    averageFinalDuelEntryTurn: average(
      finalDuels.map((duel) => duel.entryTurn),
    ),
    averageFinalDuelDurationTurns: average(
      finalDuels.flatMap((duel) =>
        duel.evaluatedTurns === undefined ? [] : [duel.evaluatedTurns],
      ),
    ),
    resultReasonCounts,
  };
}

export function createPythonPpoEvaluationClientFactory(options: {
  python?: string;
  device?: RlTorchDevice;
} = {}): PpoEvaluationPolicyClientFactory {
  return () => new PythonPpoClient({
    command: options.python ?? "python",
    device: options.device ?? "auto",
  });
}

export async function runPpoCheckpointEvaluationMatch(input: {
  assignment: PpoEvaluationSeatAssignment;
  checkpointPaths: Record<PpoEvaluationCheckpointLabel, string>;
  clientFactory: PpoEvaluationPolicyClientFactory;
  maxTurns?: number;
  maxDecisions?: number;
  environmentFactory?: () => PpoEvaluationEnvironment;
}): Promise<PpoEvaluationMatchResult> {
  const environment = input.environmentFactory?.() ?? new RlEnvironmentV2();
  const firstObservation = environment.reset(input.assignment.seed, 4);
  const featureSpec = createRlFeatureSpecV2(firstObservation);
  const teamIds = environment.getStateForValidation().teams
    .filter((team) => !team.isNeutral && team.status !== "neutral")
    .map((team) => team.id);
  if (teamIds.length !== 4) {
    throw new Error(`PPO checkpoint evaluation expected four teams, got ${teamIds.length}`);
  }
  for (const teamId of teamIds) {
    if (!input.assignment.checkpointByTeam[teamId]) {
      throw new Error(`Missing PPO evaluation checkpoint assignment for ${teamId}`);
    }
  }
  const update3TeamIds = teamIds.filter((teamId) =>
    input.assignment.checkpointByTeam[teamId] === "update3",
  );
  if (
    update3TeamIds.length !== 1
    || update3TeamIds[0] !== input.assignment.update3TeamId
  ) {
    throw new Error("PPO evaluation requires exactly one matching update3 seat");
  }

  const clients = new Map<string, PpoEvaluationPolicyClient>();
  try {
    for (const teamId of teamIds) {
      const checkpointLabel = input.assignment.checkpointByTeam[teamId];
      const checkpointPath = input.checkpointPaths[checkpointLabel];
      const client = input.clientFactory({
        teamId,
        checkpointLabel,
        checkpointPath,
        seed: input.assignment.seed,
      });
      clients.set(teamId, client);
      await client.start({
        seed: input.assignment.seed,
        featureSpec,
        hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
        initialCheckpoint: checkpointPath,
        evaluationCheckpoint: checkpointPath,
      });
    }

    const encoderCache = createRlObservationEncoderCache();
    let decisionCount = 0;
    while (!environment.isTerminal()) {
      const actorTeamId = environment.getCurrentActorTeamId();
      if (!actorTeamId) throw new Error("PPO evaluation has no current actor");
      const observation = environment.getObservationForEncoding(actorTeamId);
      if (observation.turnNumber > (input.maxTurns ?? 1_000)) {
        throw new Error("PPO evaluation exceeded maxTurns without a game terminal");
      }
      if (decisionCount >= (input.maxDecisions ?? 100_000)) {
        throw new Error("PPO evaluation exceeded maxDecisions without a game terminal");
      }
      const legalActions = environment.getLegalActionsForEncoding(actorTeamId);
      if (!legalActions.length) {
        throw new Error(`PPO evaluation has no legal actions for ${actorTeamId}`);
      }
      const client = clients.get(actorTeamId);
      if (!client) throw new Error(`Missing PPO evaluation client for ${actorTeamId}`);
      const selected = await client.act(
        encodeRlObservationV2(observation, encoderCache),
        encodeRlLegalActionsV2(observation, legalActions),
      );
      const before = environment.getProgressHash();
      environment.stepWithoutObservation(selected.actionKey);
      decisionCount += 1;
      if (environment.getProgressHash() === before) {
        throw new Error(`PPO evaluation phase stall at decision ${decisionCount}`);
      }
    }

    const result = environment.getResult();
    if (!result.terminal || result.endReason !== "victory" || !result.resultReason) {
      throw new Error(`PPO evaluation ended without a game terminal: ${result.endReason}`);
    }
    const state = environment.getStateForValidation();
    const turnDiagnostics = createPpoTurnDiagnostics(state);
    const adjudicationByTeam = new Map(
      adjudicatePpoTimeLimit(state).map((team) => [team.teamId, team]),
    );
    const advantageByTeam = new Map(
      calculateBattleAdvantage(state).map((team) => [team.teamId, team]),
    );
    const teams = teamIds.map((teamId): PpoEvaluationTeamResult => {
      const adjudication = adjudicationByTeam.get(teamId);
      const advantage = advantageByTeam.get(teamId);
      if (!adjudication || !advantage) {
        throw new Error(`Missing final PPO evaluation metrics for ${teamId}`);
      }
      return {
        teamId,
        checkpoint: input.assignment.checkpointByTeam[teamId],
        status: adjudication.status,
        ownedBaseCount: adjudication.ownedBaseCount,
        livingKingHp: adjudication.livingKingHp,
        totalLivingHp: adjudication.totalLivingHp,
        livingUnitCount: adjudication.livingUnitCount,
        battleAdvantageRaw: advantage.battleAdvantageRaw,
        battleAdvantageShare: advantage.battleAdvantageShare,
        rank: adjudication.rank,
      };
    });

    return {
      seed: input.assignment.seed,
      rotationIndex: input.assignment.rotationIndex,
      update3TeamId: input.assignment.update3TeamId,
      checkpointByTeam: { ...input.assignment.checkpointByTeam },
      winnerTeamId: result.winnerTeamId ?? null,
      resultReason: result.resultReason,
      finalStateHash: environment.getStateHash(),
      decisionCount,
      turnDiagnostics,
      teams,
    };
  } finally {
    await Promise.all([...clients.values()].map((client) => client.close()));
  }
}

export async function runPpoCheckpointEvaluation(input: {
  update2Checkpoint: string;
  update3Checkpoint: string;
  update2Sha256: string;
  update3Sha256: string;
  seedStart: number;
  seedCount: number;
  maxTurns?: number;
  maxDecisions?: number;
  clientFactory: PpoEvaluationPolicyClientFactory;
  environmentFactory?: () => PpoEvaluationEnvironment;
}): Promise<PpoCheckpointEvaluationResult> {
  const assignments = createPpoEvaluationSeatAssignments(input);
  const matches: PpoEvaluationMatchResult[] = [];
  for (const assignment of assignments) {
    matches.push(await runPpoCheckpointEvaluationMatch({
      assignment,
      checkpointPaths: {
        update2: input.update2Checkpoint,
        update3: input.update3Checkpoint,
      },
      clientFactory: input.clientFactory,
      maxTurns: input.maxTurns,
      maxDecisions: input.maxDecisions,
      environmentFactory: input.environmentFactory,
    }));
  }
  return {
    checkpoints: {
      update2: { path: input.update2Checkpoint, sha256: input.update2Sha256 },
      update3: { path: input.update3Checkpoint, sha256: input.update3Sha256 },
    },
    seedStart: input.seedStart,
    seedCount: input.seedCount,
    matches,
    aggregate: aggregatePpoCheckpointEvaluation(matches),
  };
}

export async function calculateFileSha256(path: string) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

export async function verifyPpoEvaluationCheckpoint(
  path: string,
  expectedSha256: string,
) {
  if (!/^[a-f0-9]{64}$/i.test(expectedSha256)) {
    throw new Error("Expected checkpoint SHA256 must contain 64 hexadecimal characters");
  }
  const actualSha256 = await calculateFileSha256(path);
  if (actualSha256 !== expectedSha256.toLowerCase()) {
    throw new Error(
      `PPO evaluation checkpoint SHA256 mismatch for ${path}: `
      + `expected=${expectedSha256.toLowerCase()} actual=${actualSha256}`,
    );
  }
  return actualSha256;
}
