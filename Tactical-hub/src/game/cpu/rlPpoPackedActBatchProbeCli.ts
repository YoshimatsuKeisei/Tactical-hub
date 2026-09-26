import { performance } from "node:perf_hooks";
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
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`);
  return parsed;
};
const resume = value("--resume");
if (!resume) throw new Error("Specify --resume PPO_CHECKPOINT");
const decisions = positiveInteger("--decisions", 50);
const warmup = positiveInteger("--warmup", 5);
if (warmup >= decisions) throw new Error("--warmup must be smaller than --decisions");
const seed = positiveInteger("--seed", 8);
const batchSizes = (value("--batch-sizes") ?? "1,2,4,8,16")
  .split(",")
  .map((item) => Number(item.trim()))
  .filter(Number.isFinite);
if (
  !batchSizes.length
  || batchSizes.some((size) => !Number.isInteger(size) || size < 1 || size > 16)
  || batchSizes.some((size, index) => index > 0 && size <= batchSizes[index - 1])
) throw new Error("--batch-sizes must be unique ascending integers in 1..16");
if (batchSizes[0] !== 1) throw new Error("Batch size 1 must be included as the baseline");

const hyperparameters = {
  learningRate: 3e-4,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipEpsilon: 0.2,
  valueCoefficient: 0.5,
  entropyCoefficient: 0.01,
  maxGradientNorm: 0.5,
};

const rows: Array<Record<string, unknown>> = [];
let baselineMsPerDecision = 0;

for (const batchSize of batchSizes) {
  const environments = Array.from({ length: batchSize }, (_, index) => {
    const environment = new RlEnvironmentV2();
    const initialObservation = environment.reset(seed + index, 4);
    return {
      environment,
      cache: createRlObservationEncoderCache(),
      initialObservation,
    };
  });
  const featureSpec = createRlFeatureSpecV2(environments[0].initialObservation);
  const client = new PythonPpoClient({
    command: value("--python") ?? "python",
    cwd: process.cwd(),
    device: "cuda",
  });
  const ready = await client.start({
    seed: 7,
    featureSpec,
    hyperparameters,
    initialCheckpoint: resume,
    resume,
  });
  if (ready.selectedDevice !== "cuda") throw new Error("Batch probe requires CUDA");

  let encodeMs = 0;
  let roundTripMs = 0;
  let stepMs = 0;
  let maxLegalActions = 0;

  try {
    for (let step = 0; step < decisions; step += 1) {
      const encodeStarted = performance.now();
      const samples = environments.map(({ environment, cache }, environmentIndex) => {
        if (environment.isTerminal()) throw new Error(`Environment ${environmentIndex} terminated at step ${step}`);
        const teamId = environment.getCurrentActorTeamId();
        if (!teamId) throw new Error(`Environment ${environmentIndex} has no actor at step ${step}`);
        const observation = environment.getObservationForEncoding(teamId);
        const legal = environment.getLegalActionsForEncoding(teamId);
        if (!legal.length) throw new Error(`Environment ${environmentIndex} has no legal actions at step ${step}`);
        maxLegalActions = Math.max(maxLegalActions, legal.length);
        return {
          observation: encodeRlObservationV2(observation, cache),
          legalActions: encodeRlLegalActionsV2(observation, legal),
        };
      });
      const encodedElapsed = performance.now() - encodeStarted;

      const roundTripStarted = performance.now();
      const selected = batchSize === 1
        ? [await client.act(samples[0].observation, samples[0].legalActions)]
        : await client.actBatch(samples);
      const roundTripElapsed = performance.now() - roundTripStarted;

      const stepStarted = performance.now();
      selected.forEach((action, index) => {
        environments[index].environment.stepWithoutObservation(action.actionKey);
      });
      const steppedElapsed = performance.now() - stepStarted;

      if (step >= warmup) {
        encodeMs += encodedElapsed;
        roundTripMs += roundTripElapsed;
        stepMs += steppedElapsed;
      }
    }
  } finally {
    await client.close();
  }

  const timedSteps = decisions - warmup;
  const timedDecisions = timedSteps * batchSize;
  const totalMs = encodeMs + roundTripMs + stepMs;
  const msPerDecision = totalMs / timedDecisions;
  if (batchSize === 1) baselineMsPerDecision = msPerDecision;

  rows.push({
    batchSize,
    timedSteps,
    timedDecisions,
    encodeMs: Number(encodeMs.toFixed(3)),
    roundTripMs: Number(roundTripMs.toFixed(3)),
    stepMs: Number(stepMs.toFixed(3)),
    totalMs: Number(totalMs.toFixed(3)),
    msPerDecision: Number(msPerDecision.toFixed(4)),
    throughputVsBatch1: Number((baselineMsPerDecision / msPerDecision).toFixed(3)),
    maxLegalActions,
    finalStateHashes: environments.map(({ environment }) => environment.getStateHash()),
  });
}

console.log(JSON.stringify({
  probe: "ppo_packed_act_batch_end_to_end",
  status: "passed",
  device: "cuda",
  decisionsPerEnvironment: decisions,
  warmupStepsExcluded: warmup,
  checkpointUpdateCount: 2,
  checkpointEpisodeCount: 2,
  modelUpdate: false,
  baselineUsesProductionPackedAct: true,
  batchPathUsesProductionPackingAndPipe: true,
  rows,
  limits: [
    "Includes Node observation/legal-action encoding, production packed-v1 packing,",
    "live stdin/stdout IPC, Python packed decode/tensor preparation, GPU Forward,",
    "sampling/log-prob/value host conversion, response handling, and game step.",
    "Excludes trajectory storage, adjudication, replay, PPO update, checkpoint save,",
    "and scheduler effects from games terminating at different times.",
  ].join(" "),
}, null, 2));
