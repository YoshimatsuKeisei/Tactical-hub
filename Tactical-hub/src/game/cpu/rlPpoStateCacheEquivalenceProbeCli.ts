import { PythonPpoClient } from "./pythonPpoClient";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "./rlObservationEncoder";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const positiveInteger = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be positive`);
  }
  return parsed;
};

const resume = value("--resume");
if (!resume) throw new Error("Specify --resume PPO_CHECKPOINT");

const decisions = positiveInteger("--decisions", 5_000);
const gameSeed = positiveInteger("--game-seed", 9);
const trainerSeed = positiveInteger("--trainer-seed", 7);
const python = value("--python") ?? "python";

const hyperparameters = {
  learningRate: 3e-4,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipEpsilon: 0.2,
  valueCoefficient: 0.5,
  entropyCoefficient: 0.01,
  maxGradientNorm: 0.5,
};

const makeEnvironment = () => {
  const environment = new RlEnvironmentV2(
    undefined,
    true,
    { cpuStep: { rlInPlacePhaseTransitions: true } },
  );
  environment.reset(gameSeed, 4);
  return environment;
};

const baselineEnvironment = makeEnvironment();
const candidateEnvironment = makeEnvironment();
const baselineCache = createRlObservationEncoderCache();
const candidateCache = createRlObservationEncoderCache();

const firstActor = baselineEnvironment.getCurrentActorTeamId();
if (!firstActor) throw new Error("Baseline environment has no actor");

const featureSpec = createRlFeatureSpecV2(
  baselineEnvironment.getObservationForEncoding(firstActor),
);

const makeClient = (prepareMode: string) => new PythonPpoClient({
  command: python,
  cwd: process.cwd(),
  device: "cuda",
  env: {
    PPO_PACKED_PREPARE_MODE: prepareMode,
  },
});

const baselineClient = makeClient(
  "grouped_h2d_skip_empty_manual_categorical",
);
const candidateClient = makeClient(
  "grouped_h2d_skip_empty_manual_categorical_state_cache",
);

const startClient = (client: PythonPpoClient) => client.start({
  seed: trainerSeed,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});

const baselineReady = await startClient(baselineClient);
const candidateReady = await startClient(candidateClient);

if (
  baselineReady.selectedDevice !== "cuda"
  || candidateReady.selectedDevice !== "cuda"
) {
  throw new Error("CUDA is required");
}
if (
  baselineReady.updateCount !== candidateReady.updateCount
  || baselineReady.episodeCount !== candidateReady.episodeCount
) {
  throw new Error("Baseline/candidate checkpoint counters differ");
}

const encode = (
  environment: RlEnvironmentV2,
  cache: ReturnType<typeof createRlObservationEncoderCache>,
  decisionIndex: number,
) => {
  if (environment.isTerminal()) {
    throw new Error(`Environment terminated before decision ${decisionIndex}`);
  }
  const actor = environment.getCurrentActorTeamId();
  if (!actor) throw new Error(`No actor at decision ${decisionIndex}`);
  const observation = environment.getObservationForEncoding(actor);
  const legal = environment.getLegalActionsForEncoding(actor);
  if (!legal.length) throw new Error(`No legal actions at decision ${decisionIndex}`);
  return {
    observation: encodeRlObservationV2(observation, cache),
    legalActions: encodeRlLegalActionsV2(observation, legal),
  };
};

let baselineActMs = 0;
let candidateActMs = 0;
let baselineEncodeStepMs = 0;
let candidateEncodeStepMs = 0;
let comparedDecisions = 0;

let actionMismatchCount = 0;
let logProbabilityMismatchCount = 0;
let valueMismatchCount = 0;
let stateHashMismatchCount = 0;
let maxAbsLogProbabilityDiff = 0;
let maxAbsValueDiff = 0;
let firstMismatch: Record<string, unknown> | undefined;

let baselinePostDiagnostics:
  Awaited<ReturnType<PythonPpoClient["diagnostics"]>>
  | undefined;
let candidatePostDiagnostics:
  Awaited<ReturnType<PythonPpoClient["diagnostics"]>>
  | undefined;

try {
  for (let decisionIndex = 0; decisionIndex < decisions; decisionIndex += 1) {
    let started = performance.now();
    const baselineSample = encode(
      baselineEnvironment,
      baselineCache,
      decisionIndex,
    );
    baselineEncodeStepMs += performance.now() - started;

    started = performance.now();
    const baselineAction = await baselineClient.act(
      baselineSample.observation,
      baselineSample.legalActions,
    );
    baselineActMs += performance.now() - started;

    started = performance.now();
    const candidateSample = encode(
      candidateEnvironment,
      candidateCache,
      decisionIndex,
    );
    candidateEncodeStepMs += performance.now() - started;

    started = performance.now();
    const candidateAction = await candidateClient.act(
      candidateSample.observation,
      candidateSample.legalActions,
    );
    candidateActMs += performance.now() - started;

    const logDiff = Math.abs(
      baselineAction.logProbability - candidateAction.logProbability,
    );
    const valueDiff = Math.abs(
      baselineAction.value - candidateAction.value,
    );
    maxAbsLogProbabilityDiff = Math.max(
      maxAbsLogProbabilityDiff,
      logDiff,
    );
    maxAbsValueDiff = Math.max(
      maxAbsValueDiff,
      valueDiff,
    );

    const actionMismatch =
      baselineAction.actionIndex !== candidateAction.actionIndex
      || baselineAction.actionKey !== candidateAction.actionKey;
    const logMismatch =
      baselineAction.logProbability !== candidateAction.logProbability;
    const valueMismatch =
      baselineAction.value !== candidateAction.value;

    if (actionMismatch) actionMismatchCount += 1;
    if (logMismatch) logProbabilityMismatchCount += 1;
    if (valueMismatch) valueMismatchCount += 1;

    if (
      !firstMismatch
      && (actionMismatch || logMismatch || valueMismatch)
    ) {
      firstMismatch = {
        decisionIndex,
        reason: "action_output",
        baselineAction,
        candidateAction,
        logDiff,
        valueDiff,
      };
    }

    started = performance.now();
    baselineEnvironment.stepWithoutObservation(
      baselineAction.actionKey,
    );
    baselineEncodeStepMs += performance.now() - started;

    started = performance.now();
    candidateEnvironment.stepWithoutObservation(
      candidateAction.actionKey,
    );
    candidateEncodeStepMs += performance.now() - started;

    const baselineHash = baselineEnvironment.getStateHash();
    const candidateHash = candidateEnvironment.getStateHash();
    if (baselineHash !== candidateHash) {
      stateHashMismatchCount += 1;
      if (!firstMismatch) {
        firstMismatch = {
          decisionIndex,
          reason: "state_hash",
          baselineHash,
          candidateHash,
        };
      }
    }

    comparedDecisions += 1;

    if (firstMismatch) break;
  }

  baselinePostDiagnostics =
    await baselineClient.diagnostics();
  candidatePostDiagnostics =
    await candidateClient.diagnostics();
} finally {
  await baselineClient.close();
  await candidateClient.close();
}

const baselineFinalHash = baselineEnvironment.getStateHash();
const candidateFinalHash = candidateEnvironment.getStateHash();

const exactPostRng = Boolean(
  baselinePostDiagnostics
  && candidatePostDiagnostics
  && baselinePostDiagnostics.rngHash
    === candidatePostDiagnostics.rngHash,
);

const exactPostParameters = Boolean(
  baselinePostDiagnostics
  && candidatePostDiagnostics
  && baselinePostDiagnostics.parameterHash
    === candidatePostDiagnostics.parameterHash
  && baselinePostDiagnostics.optimizerHash
    === candidatePostDiagnostics.optimizerHash,
);

const exactFinalHash = baselineFinalHash === candidateFinalHash;

const status = (
  comparedDecisions === decisions
  && actionMismatchCount === 0
  && logProbabilityMismatchCount === 0
  && valueMismatchCount === 0
  && stateHashMismatchCount === 0
  && exactFinalHash
  && exactPostRng
  && exactPostParameters
) ? "passed" : "mismatch";

const actSpeedup = baselineActMs / candidateActMs;
const totalSpeedup =
  (baselineActMs + baselineEncodeStepMs)
  / (candidateActMs + candidateEncodeStepMs);

console.log(JSON.stringify({
  probe: "ppo_state_branch_cache_equivalence",
  status,
  trainerSeed,
  gameSeed,
  requestedDecisions: decisions,
  comparedDecisions,
  checkpoint: {
    updateCount: baselineReady.updateCount,
    episodeCount: baselineReady.episodeCount,
    exactPostRunRngHash: exactPostRng,
    exactPostRunParameterAndOptimizerHash: exactPostParameters,
  },
  equivalence: {
    actionMismatchCount,
    logProbabilityMismatchCount,
    valueMismatchCount,
    stateHashMismatchCount,
    maxAbsLogProbabilityDiff,
    maxAbsValueDiff,
    exactFinalHash,
    baselineFinalHash,
    candidateFinalHash,
    firstMismatch: firstMismatch ?? null,
  },
  timings: {
    baselineActMs: Number(baselineActMs.toFixed(3)),
    candidateActMs: Number(candidateActMs.toFixed(3)),
    actSpeedup: Number(actSpeedup.toFixed(3)),
    actReductionPercent: Number(
      ((1 - candidateActMs / baselineActMs) * 100).toFixed(2),
    ),
    baselineEncodeStepMs: Number(baselineEncodeStepMs.toFixed(3)),
    candidateEncodeStepMs: Number(candidateEncodeStepMs.toFixed(3)),
    baselineTotalMs: Number(
      (baselineActMs + baselineEncodeStepMs).toFixed(3),
    ),
    candidateTotalMs: Number(
      (candidateActMs + candidateEncodeStepMs).toFixed(3),
    ),
    totalSpeedup: Number(totalSpeedup.toFixed(3)),
  },
  gate: {
    strictEquivalence: status === "passed",
    actSpeedupAtLeast1_5x: actSpeedup >= 1.5,
    totalSpeedupAtLeast1_25x: totalSpeedup >= 1.25,
  },
  limits: [
    "Single-environment sequential batch-1 comparison only.",
    "Both sides use the same seed, policy checkpoint, movement path, and in-place phase path.",
    "Candidate reuses only pooled state branches whose packed input bytes and shapes exactly match the immediately cached fingerprint.",
    "Candidate cache is opt-in and is cleared before PPO updates and checkpoint/model reloads.",
    "No GAE, replay, optimizer update, or checkpoint save is performed.",
  ],
}, null, 2));

if (status !== "passed") process.exitCode = 2;
