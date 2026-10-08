import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { calculateBattleAdvantage } from "../engine/battleAdvantage";
import { createHeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "./rlObservationEncoder";
import { adjudicatePpoTimeLimit } from "./rlPpoAdjudication";
import { DEFAULT_PPO_HYPERPARAMETERS } from "./rlPpoSelfPlay";
import { createPpoTurnDiagnostics } from "./rlPpoTurnDiagnostics";
import { PythonPpoClient } from "./pythonPpoClient";
import { parseRlTorchDevice } from "./rlTorchDevice";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const required = (name: string) => {
  const result = value(name);
  if (!result) throw new Error(`${name} is required`);
  return result;
};
const integer = (name: string, fallback: number, minimum: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return parsed;
};

async function sha256(path: string) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

function logCombination(n: number, k: number) {
  const m = Math.min(k, n - k);
  let total = 0;
  for (let i = 1; i <= m; i += 1) {
    total += Math.log(n - m + i) - Math.log(i);
  }
  return total;
}

function binomialUpperTail(k: number, n: number, p: number) {
  let total = 0;
  for (let x = k; x <= n; x += 1) {
    total += Math.exp(
      logCombination(n, x)
      + x * Math.log(p)
      + (n - x) * Math.log(1 - p),
    );
  }
  return Math.min(1, total);
}

function wilsonInterval(successes: number, n: number, z = 1.959963984540054) {
  if (!n) return { low: null, high: null };
  const p = successes / n;
  const z2 = z * z;
  const denominator = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denominator;
  const radius = z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator;
  return { low: Math.max(0, center - radius), high: Math.min(1, center + radius) };
}

const checkpoint = required("--checkpoint");
const expectedSha = required("--sha256").toLowerCase();
const actualSha = await sha256(checkpoint);
if (actualSha !== expectedSha) {
  throw new Error(`checkpoint SHA mismatch expected=${expectedSha} actual=${actualSha}`);
}

const seedStart = integer("--seed-start", 1000, 0);
const seedCount = integer("--seed-count", 25, 1);
const maxTurns = integer("--max-turns", 1000, 1);
const maxDecisions = integer("--max-decisions", 50000, 1);
const device = parseRlTorchDevice(value("--device") ?? "auto");
const teamIds = ["team-1", "team-2", "team-3", "team-4"] as const;

type MatchResult = {
  seed: number;
  rotationIndex: number;
  ppoTeamId: string;
  winnerTeamId: string | null;
  ppoWon: boolean;
  ppoRank: number;
  decisionCount: number;
  finalTurnNumber: number;
  resultReason: string | null;
  ppoFinalMetrics: {
    ownedBaseCount: number;
    livingKingHp: number;
    totalLivingHp: number;
    livingUnitCount: number;
    battleAdvantageRaw: number;
    battleAdvantageShare: number;
  };
};

const matches: MatchResult[] = [];

for (let seedOffset = 0; seedOffset < seedCount; seedOffset += 1) {
  const seed = seedStart + seedOffset;
  for (let rotationIndex = 0; rotationIndex < teamIds.length; rotationIndex += 1) {
    const ppoTeamId = teamIds[rotationIndex];
    const environment = new RlEnvironmentV2();
    const firstObservation = environment.reset(seed, 4);
    const featureSpec = createRlFeatureSpecV2(firstObservation);
    const client = new PythonPpoClient({ device });
    const heuristic = createHeuristicCpuPolicy();
    const encoderCache = createRlObservationEncoderCache();
    let decisionCount = 0;

    try {
      await client.start({
        seed,
        featureSpec,
        hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
        initialCheckpoint: checkpoint,
        evaluationCheckpoint: checkpoint,
      });

      while (!environment.isTerminal()) {
        const actorTeamId = environment.getCurrentActorTeamId();
        if (!actorTeamId) throw new Error("mixed evaluation has no current actor");
        const observation = environment.getObservationForEncoding(actorTeamId);
        if (observation.turnNumber > maxTurns) {
          throw new Error(`mixed evaluation exceeded maxTurns seed=${seed} seat=${ppoTeamId}`);
        }
        if (decisionCount >= maxDecisions) {
          throw new Error(`mixed evaluation exceeded maxDecisions seed=${seed} seat=${ppoTeamId}`);
        }

        const before = environment.getProgressHash();
        if (actorTeamId === ppoTeamId) {
          const legalActions = environment.getLegalActionsForEncoding(actorTeamId);
          if (!legalActions.length) {
            throw new Error(`mixed evaluation has no legal PPO actions for ${actorTeamId}`);
          }
          const selected = await client.act(
            encodeRlObservationV2(observation, encoderCache),
            encodeRlLegalActionsV2(observation, legalActions),
          );
          environment.stepWithoutObservation(selected.actionKey);
        } else {
          environment.stepWithPolicyForReplay(heuristic);
        }
        decisionCount += 1;
        if (environment.getProgressHash() === before) {
          throw new Error(`mixed evaluation phase stall at decision ${decisionCount}`);
        }
      }

      const result = environment.getResult();
      if (!result.terminal || result.endReason !== "victory") {
        throw new Error(`mixed evaluation ended without victory: ${result.endReason}`);
      }
      const state = environment.getStateForValidation();
      const ranks = new Map(
        adjudicatePpoTimeLimit(state).map((team) => [team.teamId, team]),
      );
      const advantages = new Map(
        calculateBattleAdvantage(state).map((team) => [team.teamId, team]),
      );
      const ppoRank = ranks.get(ppoTeamId);
      const advantage = advantages.get(ppoTeamId);
      if (!ppoRank || !advantage) throw new Error("missing PPO final metrics");
      const turnDiagnostics = createPpoTurnDiagnostics(state);
      matches.push({
        seed,
        rotationIndex,
        ppoTeamId,
        winnerTeamId: result.winnerTeamId ?? null,
        ppoWon: result.winnerTeamId === ppoTeamId,
        ppoRank: ppoRank.rank,
        decisionCount,
        finalTurnNumber: turnDiagnostics.finalStateTurnNumber,
        resultReason: result.resultReason ?? null,
        ppoFinalMetrics: {
          ownedBaseCount: ppoRank.ownedBaseCount,
          livingKingHp: ppoRank.livingKingHp,
          totalLivingHp: ppoRank.totalLivingHp,
          livingUnitCount: ppoRank.livingUnitCount,
          battleAdvantageRaw: advantage.battleAdvantageRaw,
          battleAdvantageShare: advantage.battleAdvantageShare,
        },
      });
      process.stderr.write(
        `[ppo-vs-heuristic] completed ${matches.length}/${seedCount * 4} seed=${seed} seat=${ppoTeamId} win=${result.winnerTeamId === ppoTeamId} turn=${turnDiagnostics.finalStateTurnNumber}\n`,
      );
    } finally {
      await client.close();
    }
  }
}

const winCount = matches.filter((match) => match.ppoWon).length;
const matchCount = matches.length;
const average = (values: readonly number[]) =>
  values.reduce((sum, item) => sum + item, 0) / values.length;
const rankDistribution = Object.fromEntries(
  [1, 2, 3, 4].map((rank) => [
    String(rank),
    matches.filter((match) => match.ppoRank === rank).length,
  ]),
);
const pValue = binomialUpperTail(winCount, matchCount, 0.25);
const confidence95 = wilsonInterval(winCount, matchCount);

process.stdout.write(JSON.stringify({
  probe: "ppo_v10_vs_heuristic_1v3",
  checkpoint: {
    path: checkpoint,
    sha256: actualSha,
  },
  design: {
    seedStart,
    seedCount,
    seatRotationsPerSeed: 4,
    matchCount,
    ppoSeatsPerMatch: 1,
    heuristicSeatsPerMatch: 3,
    nullWinProbability: 0.25,
    significanceAlpha: 0.05,
    maxTurns,
    maxDecisions,
  },
  aggregate: {
    winCount,
    winRate: winCount / matchCount,
    averageRank: average(matches.map((match) => match.ppoRank)),
    rankDistribution,
    averageFinalTurn: average(matches.map((match) => match.finalTurnNumber)),
    averageDecisionCount: average(matches.map((match) => match.decisionCount)),
    exactBinomialOneSidedPValue: pValue,
    winRateWilson95: confidence95,
    significantAboveChanceAt05: pValue < 0.05,
  },
  matches,
}, null, 2));
