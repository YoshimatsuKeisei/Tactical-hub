import { performance } from "node:perf_hooks";
import { PythonPpoClient } from "./pythonPpoClient";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "./rlObservationEncoder";
import { encodeRlLegalActionsV2, type EncodedLegalActionsV2 } from "./rlActionEncoder";
import type { EncodedObservation } from "./rlObservationEncoder";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const positiveInteger = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
};

const resume = value("--resume");
if (!resume) throw new Error("Specify --resume PPO_CHECKPOINT");
const checkpoint = resume;
const decisions = positiveInteger("--decisions", 200);
const warmup = positiveInteger("--warmup", 20);
const seed = positiveInteger("--seed", 8);
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

type ProbeSample = {
  observation: EncodedObservation;
  legalActions: EncodedLegalActionsV2;
};

type ProbeOutput = {
  actionIndex: number;
  logProbability: number;
  value: number;
};

const environment = new RlEnvironmentV2();
const initialObservation = environment.reset(seed, 4);
const featureSpec = createRlFeatureSpecV2(initialObservation);
const cache = createRlObservationEncoderCache();
const samples: ProbeSample[] = [];
for (let index = 0; index < decisions; index += 1) {
  if (environment.isTerminal()) throw new Error(`Fixture terminated at decision ${index}`);
  const actor = environment.getCurrentActorTeamId();
  if (!actor) throw new Error(`Fixture has no actor at decision ${index}`);
  const observation = environment.getObservationForEncoding(actor);
  const legal = environment.getLegalActionsForEncoding(actor);
  if (!legal.length) throw new Error(`Fixture has no legal action at decision ${index}`);
  samples.push({
    observation: encodeRlObservationV2(observation, cache),
    legalActions: encodeRlLegalActionsV2(observation, legal),
  });
  environment.stepWithoutObservation(legal[0].actionKey);
}

async function benchmark(label: string, compileMode?: "reduce-overhead") {
  if (compileMode) process.env.PPO_ACT_COMPILE_MODE = compileMode;
  else delete process.env.PPO_ACT_COMPILE_MODE;

  const client = new PythonPpoClient({
    command: value("--python") ?? "python",
    cwd: process.cwd(),
    device: "cuda",
  });
  const ready = await client.start({
    seed: 7,
    featureSpec,
    hyperparameters,
    initialCheckpoint: checkpoint,
    resume: checkpoint,
  });
  if (ready.selectedDevice !== "cuda") throw new Error("Compile probe requires CUDA");

  let totalMs = 0;
  let timed = 0;
  const outputs: ProbeOutput[] = [];
  try {
    for (let index = 0; index < samples.length; index += 1) {
      const sample = samples[index];
      const started = performance.now();
      const selected = await client.act(sample.observation, sample.legalActions);
      const elapsed = performance.now() - started;
      outputs.push({
        actionIndex: selected.actionIndex,
        logProbability: selected.logProbability,
        value: selected.value,
      });
      if (index >= warmup) {
        totalMs += elapsed;
        timed += 1;
      }
    }
  } finally {
    await client.close();
    delete process.env.PPO_ACT_COMPILE_MODE;
  }

  return {
    label,
    compileMode: compileMode ?? "eager",
    timedDecisions: timed,
    totalRoundTripMs: Number(totalMs.toFixed(3)),
    msPerDecision: Number((totalMs / timed).toFixed(4)),
    decisionsPerSecond: Number((timed / (totalMs / 1000)).toFixed(3)),
    outputs,
  };
}

const eager = await benchmark("cuda_eager");
const compiled = await benchmark("cuda_reduce_overhead", "reduce-overhead");

let actionIndexMismatchCount = 0;
let maxAbsLogProbabilityDiff = 0;
let maxAbsValueDiff = 0;
for (let index = 0; index < decisions; index += 1) {
  const left = eager.outputs[index];
  const right = compiled.outputs[index];
  if (left.actionIndex !== right.actionIndex) actionIndexMismatchCount += 1;
  maxAbsLogProbabilityDiff = Math.max(
    maxAbsLogProbabilityDiff,
    Math.abs(left.logProbability - right.logProbability),
  );
  maxAbsValueDiff = Math.max(
    maxAbsValueDiff,
    Math.abs(left.value - right.value),
  );
}

const tolerance = 3e-4;
const equivalentWithinTolerance =
  actionIndexMismatchCount === 0
  && maxAbsLogProbabilityDiff <= tolerance
  && maxAbsValueDiff <= tolerance;
const eagerRow = { ...eager };
const compiledRow = { ...compiled };
delete (eagerRow as Partial<typeof eager>).outputs;
delete (compiledRow as Partial<typeof compiled>).outputs;

console.log(JSON.stringify({
  probe: "ppo_packed_act_torch_compile",
  status: "passed",
  fixtureSeed: seed,
  fixtureFinalStateHash: environment.getStateHash(),
  decisions,
  warmupExcluded: warmup,
  modelUpdate: false,
  exactSameEncodedInputs: true,
  productionPackedActPath: true,
  compileScope: "action Forward only; PPO update remains eager",
  rows: [eagerRow, compiledRow],
  eagerToCompiledSpeedup: Number(
    (eager.msPerDecision / compiled.msPerDecision).toFixed(3),
  ),
  actionIndexMismatchCount,
  maxAbsLogProbabilityDiff: Number(maxAbsLogProbabilityDiff.toExponential(6)),
  maxAbsValueDiff: Number(maxAbsValueDiff.toExponential(6)),
  tolerance,
  equivalentWithinTolerance,
  limits: [
    "Measures live production batch=1 packedAct round-trip on identical pre-encoded inputs.",
    "Compilation warmup is excluded from timing but still contributes to notebook wall time.",
    "Excludes observation/action encoding, game step, replay, PPO update, and checkpoint save.",
  ].join(" "),
}, null, 2));
