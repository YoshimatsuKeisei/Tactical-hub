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
import type { RlTorchDevice } from "./rlTorchDevice";

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

const environment = new RlEnvironmentV2();
const initialObservation = environment.reset(seed, 4);
const featureSpec = createRlFeatureSpecV2(initialObservation);
const cache = createRlObservationEncoderCache();
const samples: ProbeSample[] = [];
for (let index = 0; index < decisions; index += 1) {
  if (environment.isTerminal()) {
    throw new Error(`Fixture environment terminated at decision ${index}`);
  }
  const actor = environment.getCurrentActorTeamId();
  if (!actor) throw new Error(`Fixture has no actor at decision ${index}`);
  const observation = environment.getObservationForEncoding(actor);
  const legal = environment.getLegalActionsForEncoding(actor);
  if (!legal.length) throw new Error(`Fixture has no legal action at decision ${index}`);
  samples.push({
    observation: encodeRlObservationV2(observation, cache),
    legalActions: encodeRlLegalActionsV2(observation, legal),
  });
  // Deterministic fixture advancement only. Model outputs are deliberately ignored.
  environment.stepWithoutObservation(legal[0].actionKey);
}

async function benchmark(device: RlTorchDevice) {
  const client = new PythonPpoClient({
    command: value("--python") ?? "python",
    cwd: process.cwd(),
    device,
  });
  const ready = await client.start({
    seed: 7,
    featureSpec,
    hyperparameters,
    initialCheckpoint: checkpoint,
    resume: checkpoint,
  });
  if (ready.selectedDevice !== device) {
    throw new Error(`Requested ${device}, selected ${ready.selectedDevice}`);
  }

  let totalMs = 0;
  let timed = 0;
  let maxLegalActions = 0;
  try {
    for (let index = 0; index < samples.length; index += 1) {
      const sample = samples[index];
      maxLegalActions = Math.max(maxLegalActions, sample.legalActions.actionKeys.length);
      const started = performance.now();
      const selected = await client.act(sample.observation, sample.legalActions);
      const elapsed = performance.now() - started;
      if (!Number.isFinite(selected.logProbability) || !Number.isFinite(selected.value)) {
        throw new Error(`${device} returned non-finite output at decision ${index}`);
      }
      if (index >= warmup) {
        totalMs += elapsed;
        timed += 1;
      }
    }
  } finally {
    await client.close();
  }

  return {
    device,
    timedDecisions: timed,
    totalRoundTripMs: Number(totalMs.toFixed(3)),
    msPerDecision: Number((totalMs / timed).toFixed(4)),
    decisionsPerSecond: Number((timed / (totalMs / 1000)).toFixed(3)),
    maxLegalActions,
  };
}

const cpu = await benchmark("cpu");
const cuda = await benchmark("cuda");
const faster = cpu.msPerDecision <= cuda.msPerDecision ? cpu : cuda;
const slower = faster.device === "cpu" ? cuda : cpu;

console.log(JSON.stringify({
  probe: "ppo_packed_act_cpu_vs_cuda",
  status: "passed",
  fixtureSeed: seed,
  fixtureFinalStateHash: environment.getStateHash(),
  decisions,
  warmupExcluded: warmup,
  modelUpdate: false,
  exactSameEncodedInputs: true,
  productionPackedActPath: true,
  rows: [cpu, cuda],
  fasterDevice: faster.device,
  speedupOfFasterDevice: Number((slower.msPerDecision / faster.msPerDecision).toFixed(3)),
  limits: [
    "Measures production batch=1 packedAct round-trip only on identical pre-encoded inputs.",
    "Includes Node packing, live stdin/stdout IPC, Python packed decode/tensor preparation,",
    "model Forward, sampling/log-probability/value host conversion, and response handling.",
    "Excludes observation/action encoding, game stepping, replay, PPO update, and checkpoint save.",
  ].join(" "),
}, null, 2));
