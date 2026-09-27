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
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be positive`);
  return parsed;
};

const resume = value("--resume");
if (!resume) throw new Error("Specify --resume PPO_CHECKPOINT");
const environmentCount = positiveInteger("--environments", 8);
const rounds = positiveInteger("--rounds", 256);
const firstGameSeed = positiveInteger("--first-game-seed", 9);
const trainerSeed = positiveInteger("--trainer-seed", 7);
if (environmentCount !== 8) throw new Error("This matched-bulk gate intentionally supports exactly 8 environments");

const hyperparameters = {
  learningRate: 3e-4,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipEpsilon: 0.2,
  valueCoefficient: 0.5,
  entropyCoefficient: 0.01,
  maxGradientNorm: 0.5,
};

type EnvSlot = {
  environment: RlEnvironmentV2;
  cache: ReturnType<typeof createRlObservationEncoderCache>;
  seed: number;
};

const makeSlots = () => Array.from({ length: environmentCount }, (_, index): EnvSlot => {
  const environment = new RlEnvironmentV2(undefined, true);
  const seed = firstGameSeed + index;
  environment.reset(seed, 4);
  return { environment, cache: createRlObservationEncoderCache(), seed };
});

const serialSlots = makeSlots();
const batchSlots = makeSlots();
const featureSpec = createRlFeatureSpecV2(serialSlots[0].environment.getObservationForEncoding(
  serialSlots[0].environment.getCurrentActorTeamId()!,
));

const makeClient = () => new PythonPpoClient({
  command: value("--python") ?? "python",
  cwd: process.cwd(),
  device: "cuda",
});
const serialClient = makeClient();
const batchClient = makeClient();

const startClient = async (client: PythonPpoClient) => client.start({
  seed: trainerSeed,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});
const serialReady = await startClient(serialClient);
const batchReady = await startClient(batchClient);
if (serialReady.selectedDevice !== "cuda" || batchReady.selectedDevice !== "cuda") {
  throw new Error("CUDA is required");
}
if (
  serialReady.updateCount !== batchReady.updateCount
  || serialReady.episodeCount !== batchReady.episodeCount
) {
  throw new Error("Serial/batch checkpoints disagree on counters");
}

let actionMismatchCount = 0;
let logProbabilityMismatchCount = 0;
let valueMismatchCount = 0;
let firstMismatch: Record<string, unknown> | undefined;
let maxAbsLogProbabilityDiff = 0;
let maxAbsValueDiff = 0;
let comparedDecisions = 0;
let serialActMs = 0;
let batchActMs = 0;
let serialEncodeStepMs = 0;
let batchEncodeStepMs = 0;
let serialPostDiagnostics: Awaited<ReturnType<PythonPpoClient["diagnostics"]>> | undefined;
let batchPostDiagnostics: Awaited<ReturnType<PythonPpoClient["diagnostics"]>> | undefined;

const encodeSlot = (slot: EnvSlot, slotIndex: number, round: number) => {
  if (slot.environment.isTerminal()) {
    throw new Error(`Environment ${slotIndex} terminated before round ${round}; use a shorter probe or full episode scheduler`);
  }
  const actor = slot.environment.getCurrentActorTeamId();
  if (!actor) throw new Error(`Environment ${slotIndex} has no actor at round ${round}`);
  const observation = slot.environment.getObservationForEncoding(actor);
  const legal = slot.environment.getLegalActionsForEncoding(actor);
  if (!legal.length) throw new Error(`Environment ${slotIndex} has no legal action at round ${round}`);
  return {
    observation: encodeRlObservationV2(observation, slot.cache),
    legalActions: encodeRlLegalActionsV2(observation, legal),
  };
};

try {
  for (let round = 0; round < rounds; round += 1) {
    let started = performance.now();
    const serialSamples = serialSlots.map((slot, index) => encodeSlot(slot, index, round));
    serialEncodeStepMs += performance.now() - started;

    const serialActions = [];
    started = performance.now();
    for (let index = 0; index < serialSamples.length; index += 1) {
      serialActions.push(await serialClient.act(
        serialSamples[index].observation,
        serialSamples[index].legalActions,
      ));
    }
    serialActMs += performance.now() - started;

    started = performance.now();
    serialActions.forEach((selected, index) => {
      serialSlots[index].environment.stepWithoutObservation(selected.actionKey);
    });
    serialEncodeStepMs += performance.now() - started;

    started = performance.now();
    const batchSamples = batchSlots.map((slot, index) => encodeSlot(slot, index, round));
    batchEncodeStepMs += performance.now() - started;

    started = performance.now();
    const batchActions = await batchClient.actBatch(batchSamples);
    batchActMs += performance.now() - started;

    started = performance.now();
    batchActions.forEach((selected, index) => {
      batchSlots[index].environment.stepWithoutObservation(selected.actionKey);
    });
    batchEncodeStepMs += performance.now() - started;

    for (let index = 0; index < environmentCount; index += 1) {
      const serial = serialActions[index];
      const batch = batchActions[index];
      const logDiff = Math.abs(serial.logProbability - batch.logProbability);
      const valueDiff = Math.abs(serial.value - batch.value);
      maxAbsLogProbabilityDiff = Math.max(maxAbsLogProbabilityDiff, logDiff);
      maxAbsValueDiff = Math.max(maxAbsValueDiff, valueDiff);

      const actionMismatch = serial.actionIndex !== batch.actionIndex || serial.actionKey !== batch.actionKey;
      const logMismatch = serial.logProbability !== batch.logProbability;
      const valueMismatch = serial.value !== batch.value;
      if (actionMismatch) actionMismatchCount += 1;
      if (logMismatch) logProbabilityMismatchCount += 1;
      if (valueMismatch) valueMismatchCount += 1;
      comparedDecisions += 1;

      if (!firstMismatch && (actionMismatch || logMismatch || valueMismatch)) {
        firstMismatch = {
          round,
          environmentIndex: index,
          seed: firstGameSeed + index,
          serial,
          batch,
          logDiff,
          valueDiff,
        };
      }
    }

    const serialHashes = serialSlots.map(({ environment }) => environment.getStateHash());
    const batchHashes = batchSlots.map(({ environment }) => environment.getStateHash());
    if (!firstMismatch) {
      const mismatchIndex = serialHashes.findIndex((hash, index) => hash !== batchHashes[index]);
      if (mismatchIndex >= 0) {
        firstMismatch = {
          round,
          environmentIndex: mismatchIndex,
          seed: firstGameSeed + mismatchIndex,
          reason: "stateHash",
          serialHash: serialHashes[mismatchIndex],
          batchHash: batchHashes[mismatchIndex],
        };
      }
    }
  }
  serialPostDiagnostics = await serialClient.diagnostics();
  batchPostDiagnostics = await batchClient.diagnostics();
} finally {
  await serialClient.close();
  await batchClient.close();
}
const serialFinalHashes = serialSlots.map(({ environment }) => environment.getStateHash());
const batchFinalHashes = batchSlots.map(({ environment }) => environment.getStateHash());
const exactFinalHashes = serialFinalHashes.every((hash, index) => hash === batchFinalHashes[index]);
const exactPostRng = Boolean(
  serialPostDiagnostics
  && batchPostDiagnostics
  && serialPostDiagnostics.rngHash === batchPostDiagnostics.rngHash
);
const exactPostParameters = Boolean(
  serialPostDiagnostics
  && batchPostDiagnostics
  && serialPostDiagnostics.parameterHash === batchPostDiagnostics.parameterHash
  && serialPostDiagnostics.optimizerHash === batchPostDiagnostics.optimizerHash
);
const status = (
  actionMismatchCount === 0
  && logProbabilityMismatchCount === 0
  && valueMismatchCount === 0
  && exactFinalHashes
  && exactPostRng
  && exactPostParameters
) ? "passed" : "mismatch";

console.log(JSON.stringify({
  probe: "ppo_matched_bulk_equivalence_1_vs_8",
  status,
  trainerSeed,
  firstGameSeed,
  seeds: Array.from({ length: environmentCount }, (_, index) => firstGameSeed + index),
  environmentCount,
  rounds,
  comparedDecisions,
  checkpoint: {
    updateCount: serialReady.updateCount,
    episodeCount: serialReady.episodeCount,
    exactPostRunRngHash: exactPostRng,
    exactPostRunParameterAndOptimizerHash: exactPostParameters,
    serialPostRngHash: serialPostDiagnostics?.rngHash ?? null,
    batchPostRngHash: batchPostDiagnostics?.rngHash ?? null,
  },
  equivalence: {
    actionMismatchCount,
    logProbabilityMismatchCount,
    valueMismatchCount,
    maxAbsLogProbabilityDiff,
    maxAbsValueDiff,
    exactFinalHashes,
    firstMismatch: firstMismatch ?? null,
  },
  timings: {
    serialActMs: Number(serialActMs.toFixed(3)),
    batchActMs: Number(batchActMs.toFixed(3)),
    actThroughputRatio: Number((serialActMs / batchActMs).toFixed(3)),
    serialEncodeStepMs: Number(serialEncodeStepMs.toFixed(3)),
    batchEncodeStepMs: Number(batchEncodeStepMs.toFixed(3)),
    serialTotalMs: Number((serialActMs + serialEncodeStepMs).toFixed(3)),
    batchTotalMs: Number((batchActMs + batchEncodeStepMs).toFixed(3)),
    totalThroughputRatio: Number(
      ((serialActMs + serialEncodeStepMs) / (batchActMs + batchEncodeStepMs)).toFixed(3),
    ),
  },
  finalStateHashes: {
    serial: serialFinalHashes,
    batch: batchFinalHashes,
  },
  limits: [
    "This is a pre-training equivalence/throughput feasibility gate.",
    "It uses the same 8 game seeds and round-robin decision order for serial and batched inference.",
    "It does not perform GAE, replay, optimizer update, checkpoint save, or episode completion.",
    "A mismatch means the current packed actBatch path is not Phase 12B-1 equivalent and must not be used for matched bulk training as-is.",
  ],
}, null, 2));

if (status !== "passed") process.exitCode = 2;
