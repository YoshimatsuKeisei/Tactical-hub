import { performance } from "node:perf_hooks";
import { encodeRlLegalActionsV2, type EncodedLegalActionsV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
  type EncodedObservation,
} from "./rlObservationEncoder";
import { PythonPpoClient } from "./pythonPpoClient";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const positiveInteger = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be positive`);
  return parsed;
};
const resume = value("--resume");
if (!resume) throw new Error("Specify --resume PPO_CHECKPOINT");
const decisions = positiveInteger("--decisions", 1000);
const warmup = positiveInteger("--warmup", 50);
const trainerSeed = positiveInteger("--trainer-seed", 7);
const gameSeed = positiveInteger("--game-seed", 9);
if (warmup >= decisions) throw new Error("--warmup must be smaller than --decisions");
const hyperparameters = {
  learningRate: 3e-4,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipEpsilon: 0.2,
  valueCoefficient: 0.5,
  entropyCoefficient: 0.01,
  maxGradientNorm: 0.5,
};

type TraceSample = {
  observation: EncodedObservation;
  legalActions: EncodedLegalActionsV2;
  actor: string;
  turnNumber: number;
  phase: string;
  legalCount: number;
  expected: {
    actionIndex: number;
    actionKey: string;
    logProbability: number;
    value: number;
  };
};

const summarize = (values: number[]) => {
  if (!values.length) throw new Error("No timing samples");
  const sorted = [...values].sort((a, b) => a - b);
  const percentile = (fraction: number) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
  const total = values.reduce((sum, current) => sum + current, 0);
  return {
    count: values.length,
    totalMs: Number(total.toFixed(3)),
    meanMs: Number((total / values.length).toFixed(4)),
    p50Ms: Number(percentile(0.50).toFixed(4)),
    p95Ms: Number(percentile(0.95).toFixed(4)),
    minMs: Number(sorted[0].toFixed(4)),
    maxMs: Number(sorted[sorted.length - 1].toFixed(4)),
  };
};

const python = value("--python") ?? "python";
const makeClient = (mode: "default" | "grouped_h2d") => new PythonPpoClient({
  command: python,
  cwd: process.cwd(),
  device: "cuda",
  env: mode === "grouped_h2d" ? { PPO_PACKED_PREPARE_MODE: mode } : undefined,
});

const environment = new RlEnvironmentV2(undefined, true);
const initialObservation = environment.reset(gameSeed, 4);
const featureSpec = createRlFeatureSpecV2(initialObservation);
const cache = createRlObservationEncoderCache();
const oracle = makeClient("default");
const oracleStartupStarted = performance.now();
const oracleReady = await oracle.start({
  seed: trainerSeed,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});
const oracleStartupMs = performance.now() - oracleStartupStarted;
if (oracleReady.selectedDevice !== "cuda") throw new Error("CUDA is required");

const trace: TraceSample[] = [];
const encodeObservationMs: number[] = [];
const encodeActionsMs: number[] = [];
const actOracleMs: number[] = [];
const gameStepMs: number[] = [];
const nodeCombinedMs: number[] = [];
const legalCounts: number[] = [];
const phaseCounts = new Map<string, number>();

try {
  for (let index = 0; index < decisions; index += 1) {
    if (environment.isTerminal()) throw new Error(`Environment terminated at decision ${index}`);
    const actor = environment.getCurrentActorTeamId();
    if (!actor) throw new Error(`No actor at decision ${index}`);
    const observation = environment.getObservationForEncoding(actor);
    const legal = environment.getLegalActionsForEncoding(actor);
    if (!legal.length) throw new Error(`No legal action at decision ${index}`);

    let started = performance.now();
    const encodedObservation = encodeRlObservationV2(observation, cache);
    const observationElapsed = performance.now() - started;
    started = performance.now();
    const encodedActions = encodeRlLegalActionsV2(observation, legal);
    const actionsElapsed = performance.now() - started;
    started = performance.now();
    const selected = await oracle.act(encodedObservation, encodedActions);
    const actElapsed = performance.now() - started;
    const before = environment.getProgressHash();
    started = performance.now();
    environment.stepWithoutObservation(selected.actionKey);
    const stepElapsed = performance.now() - started;
    if (environment.getProgressHash() === before) throw new Error(`Phase stall at decision ${index}`);

    trace.push({
      observation: encodedObservation,
      legalActions: encodedActions,
      actor,
      turnNumber: observation.turnNumber,
      phase: observation.phase,
      legalCount: legal.length,
      expected: {
        actionIndex: selected.actionIndex,
        actionKey: selected.actionKey,
        logProbability: selected.logProbability,
        value: selected.value,
      },
    });
    legalCounts.push(legal.length);
    phaseCounts.set(observation.phase, (phaseCounts.get(observation.phase) ?? 0) + 1);
    if (index >= warmup) {
      encodeObservationMs.push(observationElapsed);
      encodeActionsMs.push(actionsElapsed);
      actOracleMs.push(actElapsed);
      gameStepMs.push(stepElapsed);
      nodeCombinedMs.push(observationElapsed + actionsElapsed + stepElapsed);
    }
  }
} finally {
  await oracle.close();
}
const finalStateHash = environment.getStateHash();
const grouped = makeClient("grouped_h2d");
const groupedStartupStarted = performance.now();
const groupedReady = await grouped.start({
  seed: trainerSeed,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});
const groupedStartupMs = performance.now() - groupedStartupStarted;
if (groupedReady.selectedDevice !== "cuda") throw new Error("CUDA is required");

const actGroupedMs: number[] = [];
let actionMismatchCount = 0;
let logProbabilityMismatchCount = 0;
let valueMismatchCount = 0;
let maxAbsLogProbabilityDiff = 0;
let maxAbsValueDiff = 0;
try {
  for (let index = 0; index < trace.length; index += 1) {
    const sample = trace[index];
    const started = performance.now();
    const selected = await grouped.act(sample.observation, sample.legalActions);
    const elapsed = performance.now() - started;
    if (selected.actionIndex !== sample.expected.actionIndex || selected.actionKey !== sample.expected.actionKey) {
      actionMismatchCount += 1;
    }
    if (selected.logProbability !== sample.expected.logProbability) logProbabilityMismatchCount += 1;
    if (selected.value !== sample.expected.value) valueMismatchCount += 1;
    maxAbsLogProbabilityDiff = Math.max(
      maxAbsLogProbabilityDiff,
      Math.abs(selected.logProbability - sample.expected.logProbability),
    );
    maxAbsValueDiff = Math.max(maxAbsValueDiff, Math.abs(selected.value - sample.expected.value));
    if (index >= warmup) actGroupedMs.push(elapsed);
  }
} finally {
  await grouped.close();
}

const oracleAct = summarize(actOracleMs);
const groupedAct = summarize(actGroupedMs);
const nodeEncodeStep = summarize(nodeCombinedMs);
const result = {
  probe: "ppo_packed_prepare_grouped_h2d",
  status: actionMismatchCount === 0 && logProbabilityMismatchCount === 0 && valueMismatchCount === 0
    ? "passed"
    : "mismatch",
  trainerSeed,
  gameSeed,
  decisions,
  warmupExcluded: warmup,
  finalStateHash,
  exactSameEncodedInputs: true,
  actualPolicyGeneratedTrace: true,
  modelUpdate: false,
  equivalence: {
    actionMismatchCount,
    logProbabilityMismatchCount,
    valueMismatchCount,
    maxAbsLogProbabilityDiff,
    maxAbsValueDiff,
  },
  startup: {
    oracleMs: Number(oracleStartupMs.toFixed(3)),
    groupedH2dMs: Number(groupedStartupMs.toFixed(3)),
  },
  timings: {
    oracleAct,
    groupedH2dAct: groupedAct,
    groupedH2dSpeedup: Number((oracleAct.meanMs / groupedAct.meanMs).toFixed(3)),
    encodeObservation: summarize(encodeObservationMs),
    encodeActions: summarize(encodeActionsMs),
    gameStep: summarize(gameStepMs),
    nodeEncodePlusStep: nodeEncodeStep,
    groupedSequentialMeanMs: Number((groupedAct.meanMs + nodeEncodeStep.meanMs).toFixed(4)),
  },
  traceShape: {
    legalActions: {
      min: Math.min(...legalCounts),
      max: Math.max(...legalCounts),
      mean: Number((legalCounts.reduce((sum, count) => sum + count, 0) / legalCounts.length).toFixed(3)),
    },
    phaseCounts: Object.fromEntries(phaseCounts),
  },
  gates: {
    actTransportPrepareForwardAtMost0_5ms: groupedAct.meanMs <= 0.5,
    nodeEncodePlusStepAtMost0_3ms: nodeEncodeStep.meanMs <= 0.3,
    sequentialAtMost1_12ms: groupedAct.meanMs + nodeEncodeStep.meanMs <= 1.12,
  },
  limits: [
    "Oracle trace is generated sequentially by the current policy from the verified resume checkpoint.",
    "Grouped H2D is replayed on the exact same encoded input sequence with the same restored Torch RNG state.",
    "Act timing includes Node packing, stdin/stdout IPC, Python prepare, CUDA inference, sampling and host scalar return.",
    "Node encode+step timing is measured only while generating the oracle trace.",
    "No PPO update or checkpoint save is performed.",
  ],
};
console.log(JSON.stringify(result, null, 2));
if (result.status !== "passed") process.exitCode = 2;
