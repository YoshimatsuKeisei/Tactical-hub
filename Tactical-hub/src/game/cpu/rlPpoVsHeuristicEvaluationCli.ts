import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
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
const outputPath = value("--output");
const resumePath = value("--resume");
const actualSha = await sha256(checkpoint);
if (actualSha !== expectedSha) {
  throw new Error(`checkpoint SHA mismatch expected=${expectedSha} actual=${actualSha}`);
}

const seedStart = integer("--seed-start", 1000, 0);
const seedCount = integer("--seed-count", 40, 1);
const maxTurns = integer("--max-turns", 1000, 1);
const maxDecisions = integer("--max-decisions", 100000, 1);
const device = parseRlTorchDevice(value("--device") ?? "auto");
const teamIds = ["team-1", "team-2", "team-3", "team-4"] as const;

type MatchStatus = "victory" | "limit_reached";
type MatchResult = {
  seed: number;
  rotationIndex: number;
  ppoTeamId: string;
  status: MatchStatus;
  limitReason: "max_turns" | "max_decisions" | null;
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

type PersistedResult = {
  probe: "ppo_v10_vs_heuristic_1v3";
  allExact: boolean;
  checkpoint: { path: string; sha256: string };
  design: {
    seedStart: number;
    seedCount: number;
    seatAssignment: string;
    independentSeedPerMatch: true;
    matchCount: number;
    ppoSeatsPerMatch: 1;
    heuristicSeatsPerMatch: 3;
    nullWinProbability: 0.25;
    significanceAlpha: 0.05;
    maxTurns: number;
    maxDecisions: number;
    limitHandling: string;
  };
  progress: {
    completedMatches: number;
    remainingMatches: number;
  };
  aggregate: ReturnType<typeof aggregate>;
  matches: MatchResult[];
};

function average(values: readonly number[]) {
  return values.length
    ? values.reduce((sum, item) => sum + item, 0) / values.length
    : null;
}

function aggregate(matches: readonly MatchResult[], plannedMatchCount: number) {
  const winCount = matches.filter((match) => match.ppoWon).length;
  const limitReachedCount = matches.filter((match) => match.status === "limit_reached").length;
  const naturalVictoryCount = matches.length - limitReachedCount;
  const rankDistribution = Object.fromEntries(
    [1, 2, 3, 4].map((rank) => [
      String(rank),
      matches.filter((match) => match.ppoRank === rank).length,
    ]),
  );
  // Conservative primary test: a limit-reached match remains in the denominator
  // and never counts as a PPO win.
  const pValue = matches.length
    ? binomialUpperTail(winCount, matches.length, 0.25)
    : null;
  const confidence95 = matches.length
    ? wilsonInterval(winCount, matches.length)
    : { low: null, high: null };
  return {
    plannedMatchCount,
    completedMatchCount: matches.length,
    naturalVictoryCount,
    limitReachedCount,
    limitReachedRate: matches.length ? limitReachedCount / matches.length : 0,
    winCount,
    winRateConservative: matches.length ? winCount / matches.length : 0,
    averageRank: average(matches.map((match) => match.ppoRank)),
    rankDistribution,
    averageFinalTurn: average(matches.map((match) => match.finalTurnNumber)),
    averageDecisionCount: average(matches.map((match) => match.decisionCount)),
    exactBinomialOneSidedPValueConservative: pValue,
    winRateWilson95Conservative: confidence95,
    significantAboveChanceAt05Conservative: pValue !== null && pValue < 0.05,
  };
}

function buildResult(matches: MatchResult[]): PersistedResult {
  return {
    probe: "ppo_v10_vs_heuristic_1v3",
    allExact: true,
    checkpoint: { path: checkpoint, sha256: actualSha },
    design: {
      seedStart,
      seedCount,
      seatAssignment: "balanced_cycle_team1_team2_team3_team4",
      independentSeedPerMatch: true,
      matchCount: seedCount,
      ppoSeatsPerMatch: 1,
      heuristicSeatsPerMatch: 3,
      nullWinProbability: 0.25,
      significanceAlpha: 0.05,
      maxTurns,
      maxDecisions,
      limitHandling: "continue; limit-reached counts as non-win in the primary conservative win-rate test",
    },
    progress: {
      completedMatches: matches.length,
      remainingMatches: Math.max(0, seedCount - matches.length),
    },
    aggregate: aggregate(matches, seedCount),
    matches,
  };
}

function persist(matches: MatchResult[]) {
  if (!outputPath) return;
  const tempPath = `${outputPath}.tmp`;
  writeFileSync(tempPath, JSON.stringify(buildResult(matches), null, 2), "utf8");
  renameSync(tempPath, outputPath);
}

const matches: MatchResult[] = [];
if (resumePath && existsSync(resumePath)) {
  const resumed = JSON.parse(readFileSync(resumePath, "utf8")) as Partial<PersistedResult>;
  if (resumed.probe !== "ppo_v10_vs_heuristic_1v3") {
    throw new Error("resume file has unexpected probe");
  }
  if (resumed.checkpoint?.sha256 !== actualSha) {
    throw new Error("resume file checkpoint SHA mismatch");
  }
  if (
    resumed.design?.seedStart !== seedStart
    || resumed.design?.seedCount !== seedCount
    || resumed.design?.maxTurns !== maxTurns
    || resumed.design?.maxDecisions !== maxDecisions
  ) {
    throw new Error("resume file evaluation design mismatch");
  }
  for (const match of resumed.matches ?? []) matches.push(match);
}

for (let seedOffset = 0; seedOffset < seedCount; seedOffset += 1) {
  const seed = seedStart + seedOffset;
  const rotationIndex = seedOffset % teamIds.length;
  const ppoTeamId = teamIds[rotationIndex];
  if (matches.some((match) => match.seed === seed && match.ppoTeamId === ppoTeamId)) {
    continue;
  }

  const environment = new RlEnvironmentV2();
  const firstObservation = environment.reset(seed, 4);
  const featureSpec = createRlFeatureSpecV2(firstObservation);
  const client = new PythonPpoClient({ device });
  const heuristic = createHeuristicCpuPolicy();
  const encoderCache = createRlObservationEncoderCache();
  let decisionCount = 0;
  let limitReason: MatchResult["limitReason"] = null;

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
        limitReason = "max_turns";
        break;
      }
      if (decisionCount >= maxDecisions) {
        limitReason = "max_decisions";
        break;
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
    if (!limitReason && (!result.terminal || result.endReason !== "victory")) {
      throw new Error(`mixed evaluation ended unexpectedly: ${result.endReason}`);
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
    const status: MatchStatus = limitReason ? "limit_reached" : "victory";
    const winnerTeamId = status === "victory" ? (result.winnerTeamId ?? null) : null;
    const match: MatchResult = {
      seed,
      rotationIndex,
      ppoTeamId,
      status,
      limitReason,
      winnerTeamId,
      ppoWon: status === "victory" && winnerTeamId === ppoTeamId,
      ppoRank: ppoRank.rank,
      decisionCount,
      finalTurnNumber: turnDiagnostics.finalStateTurnNumber,
      resultReason: status === "victory" ? (result.resultReason ?? null) : null,
      ppoFinalMetrics: {
        ownedBaseCount: ppoRank.ownedBaseCount,
        livingKingHp: ppoRank.livingKingHp,
        totalLivingHp: ppoRank.totalLivingHp,
        livingUnitCount: ppoRank.livingUnitCount,
        battleAdvantageRaw: advantage.battleAdvantageRaw,
        battleAdvantageShare: advantage.battleAdvantageShare,
      },
    };
    matches.push(match);
    persist(matches);
    process.stderr.write(
      `[ppo-vs-heuristic] completed ${matches.length}/${seedCount} seed=${seed} seat=${ppoTeamId} status=${status} limit=${limitReason ?? "none"} win=${match.ppoWon} rank=${match.ppoRank} turn=${match.finalTurnNumber} decisions=${decisionCount}\n`,
    );
  } finally {
    await client.close();
  }
}

const finalResult = buildResult(matches);
persist(matches);
process.stdout.write(JSON.stringify(finalResult, null, 2));
