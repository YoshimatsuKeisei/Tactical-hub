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

const environmentCount = positiveInteger("--environments", 8);
const rounds = positiveInteger("--rounds", 128);
const firstGameSeed = positiveInteger("--first-game-seed", 9);
const trainerSeed = positiveInteger("--trainer-seed", 7);

if (environmentCount !== 8) {
  throw new Error("This strict stream gate intentionally supports exactly 8 environments");
}

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
  const environment = new RlEnvironmentV2(
    undefined,
    true,
    { cpuStep: { rlInPlacePhaseTransitions: true } },
  );
  const seed = firstGameSeed + index;
  environment.reset(seed, 4);
  return {
    environment,
    cache: createRlObservationEncoderCache(),
    seed,
  };
});

const serialSlots = makeSlots();
const streamSlots = makeSlots();

const firstActor = serialSlots[0].environment.getCurrentActorTeamId();
if (!firstActor) throw new Error("First serial environment has no actor");

const featureSpec = createRlFeatureSpecV2(
  serialSlots[0].environment.getObservationForEncoding(firstActor),
);

const makeClient = () => new PythonPpoClient({
  command: value("--python") ?? "python",
  cwd: process.cwd(),
  device: "cuda",
  env: {
    PPO_PACKED_PREPARE_MODE:
      "grouped_h2d_skip_empty_manual_categorical",
  },
});

const serialClient = makeClient();
const streamClient = makeClient();

const startClient = async (client: PythonPpoClient) => client.start({
  seed: trainerSeed,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});

const serialReady = await startClient(serialClient);
const streamReady = await startClient(streamClient);

if (
  serialReady.selectedDevice !== "cuda"
  || streamReady.selectedDevice !== "cuda"
) {
  throw new Error("CUDA is required");
}

if (
  serialReady.updateCount !== streamReady.updateCount
  || serialReady.episodeCount !== streamReady.episodeCount
) {
  throw new Error("Serial/stream checkpoints disagree on counters");
}

const encodeSlot = (
  slot: EnvSlot,
  slotIndex: number,
  round: number,
) => {
  if (slot.environment.isTerminal()) {
    throw new Error(
      `Environment ${slotIndex} terminated before round ${round}`,
    );
  }
  const actor = slot.environment.getCurrentActorTeamId();
  if (!actor) {
    throw new Error(
      `Environment ${slotIndex} has no actor at round ${round}`,
    );
  }
  const observation =
    slot.environment.getObservationForEncoding(actor);
  const legal =
    slot.environment.getLegalActionsForEncoding(actor);
  if (!legal.length) {
    throw new Error(
      `Environment ${slotIndex} has no legal action at round ${round}`,
    );
  }
  return {
    observation: encodeRlObservationV2(
      observation,
      slot.cache,
    ),
    legalActions: encodeRlLegalActionsV2(
      observation,
      legal,
    ),
  };
};

let actionMismatchCount = 0;
let logProbabilityMismatchCount = 0;
let valueMismatchCount = 0;
let firstMismatch: Record<string, unknown> | undefined;
let maxAbsLogProbabilityDiff = 0;
let maxAbsValueDiff = 0;
let comparedDecisions = 0;

let serialActMs = 0;
let streamActMs = 0;
let serialEncodeStepMs = 0;
let streamEncodeStepMs = 0;

let serialPostDiagnostics:
  Awaited<ReturnType<PythonPpoClient["diagnostics"]>>
  | undefined;
let streamPostDiagnostics:
  Awaited<ReturnType<PythonPpoClient["diagnostics"]>>
  | undefined;

try {
  for (let round = 0; round < rounds; round += 1) {
    let started = performance.now();

    const serialSamples = serialSlots.map(
      (slot, index) => encodeSlot(slot, index, round),
    );

    serialEncodeStepMs += performance.now() - started;

    const serialActions = [];
    started = performance.now();

    for (let index = 0; index < serialSamples.length; index += 1) {
      serialActions.push(
        await serialClient.act(
          serialSamples[index].observation,
          serialSamples[index].legalActions,
        ),
      );
    }

    serialActMs += performance.now() - started;

    started = performance.now();

    serialActions.forEach((selected, index) => {
      serialSlots[index].environment.stepWithoutObservation(
        selected.actionKey,
      );
    });

    serialEncodeStepMs += performance.now() - started;

    started = performance.now();

    const streamSamples = streamSlots.map(
      (slot, index) => encodeSlot(slot, index, round),
    );

    streamEncodeStepMs += performance.now() - started;

    started = performance.now();

    const streamActions =
      await streamClient.actStreamBatch(streamSamples);

    streamActMs += performance.now() - started;

    started = performance.now();

    streamActions.forEach((selected, index) => {
      streamSlots[index].environment.stepWithoutObservation(
        selected.actionKey,
      );
    });

    streamEncodeStepMs += performance.now() - started;

    for (let index = 0; index < environmentCount; index += 1) {
      const serial = serialActions[index];
      const candidate = streamActions[index];

      const logDiff = Math.abs(
        serial.logProbability - candidate.logProbability,
      );
      const valueDiff = Math.abs(
        serial.value - candidate.value,
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
        serial.actionIndex !== candidate.actionIndex
        || serial.actionKey !== candidate.actionKey;

      const logMismatch =
        serial.logProbability
        !== candidate.logProbability;

      const valueMismatch =
        serial.value !== candidate.value;

      if (actionMismatch) actionMismatchCount += 1;
      if (logMismatch) logProbabilityMismatchCount += 1;
      if (valueMismatch) valueMismatchCount += 1;

      comparedDecisions += 1;

      if (
        !firstMismatch
        && (actionMismatch || logMismatch || valueMismatch)
      ) {
        firstMismatch = {
          round,
          environmentIndex: index,
          seed: firstGameSeed + index,
          serial,
          candidate,
          logDiff,
          valueDiff,
        };
      }
    }

    const serialHashes = serialSlots.map(
      ({ environment }) => environment.getStateHash(),
    );
    const streamHashes = streamSlots.map(
      ({ environment }) => environment.getStateHash(),
    );

    if (!firstMismatch) {
      const mismatchIndex = serialHashes.findIndex(
        (hash, index) => hash !== streamHashes[index],
      );
      if (mismatchIndex >= 0) {
        firstMismatch = {
          round,
          environmentIndex: mismatchIndex,
          seed: firstGameSeed + mismatchIndex,
          reason: "stateHash",
          serialHash: serialHashes[mismatchIndex],
          candidateHash: streamHashes[mismatchIndex],
        };
      }
    }
  }

  serialPostDiagnostics =
    await serialClient.diagnostics();
  streamPostDiagnostics =
    await streamClient.diagnostics();
} finally {
  await serialClient.close();
  await streamClient.close();
}

const serialFinalHashes = serialSlots.map(
  ({ environment }) => environment.getStateHash(),
);
const streamFinalHashes = streamSlots.map(
  ({ environment }) => environment.getStateHash(),
);

const exactFinalHashes = serialFinalHashes.every(
  (hash, index) => hash === streamFinalHashes[index],
);

const exactPostRng = Boolean(
  serialPostDiagnostics
  && streamPostDiagnostics
  && serialPostDiagnostics.rngHash
    === streamPostDiagnostics.rngHash,
);

const exactPostParameters = Boolean(
  serialPostDiagnostics
  && streamPostDiagnostics
  && serialPostDiagnostics.parameterHash
    === streamPostDiagnostics.parameterHash
  && serialPostDiagnostics.optimizerHash
    === streamPostDiagnostics.optimizerHash,
);

const status = (
  actionMismatchCount === 0
  && logProbabilityMismatchCount === 0
  && valueMismatchCount === 0
  && exactFinalHashes
  && exactPostRng
  && exactPostParameters
)
  ? "passed"
  : "mismatch";

console.log(JSON.stringify({
  probe: "ppo_matched_stream_equivalence_1_vs_8",
  status,
  trainerSeed,
  firstGameSeed,
  seeds: Array.from(
    { length: environmentCount },
    (_, index) => firstGameSeed + index,
  ),
  environmentCount,
  rounds,
  comparedDecisions,

  checkpoint: {
    updateCount: serialReady.updateCount,
    episodeCount: serialReady.episodeCount,
    exactPostRunRngHash: exactPostRng,
    exactPostRunParameterAndOptimizerHash:
      exactPostParameters,
    serialPostRngHash:
      serialPostDiagnostics?.rngHash ?? null,
    streamPostRngHash:
      streamPostDiagnostics?.rngHash ?? null,
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
    serialActMs:
      Number(serialActMs.toFixed(3)),
    streamActMs:
      Number(streamActMs.toFixed(3)),
    actThroughputRatio:
      Number((serialActMs / streamActMs).toFixed(3)),

    serialEncodeStepMs:
      Number(serialEncodeStepMs.toFixed(3)),
    streamEncodeStepMs:
      Number(streamEncodeStepMs.toFixed(3)),

    serialTotalMs:
      Number(
        (serialActMs + serialEncodeStepMs).toFixed(3),
      ),
    streamTotalMs:
      Number(
        (streamActMs + streamEncodeStepMs).toFixed(3),
      ),

    totalThroughputRatio:
      Number(
        (
          (serialActMs + serialEncodeStepMs)
          / (streamActMs + streamEncodeStepMs)
        ).toFixed(3),
      ),
  },

  finalStateHashes: {
    serial: serialFinalHashes,
    stream: streamFinalHashes,
  },

  gate: {
    strictEquivalence: status === "passed",
    actThroughputAtLeast4x:
      serialActMs / streamActMs >= 4,
    totalThroughputAtLeast4x:
      (
        (serialActMs + serialEncodeStepMs)
        / (streamActMs + streamEncodeStepMs)
      ) >= 4,
  },

  limits: [
    "Pre-training strict-equivalence and throughput gate only.",
    "Same eight seeds and same round-robin environment order.",
    "Each candidate forward stays batch-size one with its original tensor shapes.",
    "Only CUDA forwards are overlapped; manual categorical sampling remains serial in original order.",
    "No GAE, replay, optimizer update, checkpoint save, or episode completion.",
  ],
}, null, 2));

if (status !== "passed") process.exitCode = 2;
