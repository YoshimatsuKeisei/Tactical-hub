import { performance } from "node:perf_hooks";
import { PythonPpoClient } from "./pythonPpoClient";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "./rlObservationEncoder";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import { packPpoActBatchInput } from "./rlPpoPackedBatch";

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
const rounds = positiveInteger("--rounds", 100);
const warmup = positiveInteger("--warmup", 10);
if (warmup >= rounds) throw new Error("--warmup must be smaller than --rounds");

const seeds = Array.from({ length: 8 }, (_, index) => 9 + index);
const hyperparameters = {
  learningRate: 3e-4,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipEpsilon: 0.2,
  valueCoefficient: 0.5,
  entropyCoefficient: 0.01,
  maxGradientNorm: 0.5,
};
const makeEnvironments = () => seeds.map((seed) => {
  const environment = new RlEnvironmentV2();
  const initialObservation = environment.reset(seed, 4);
  return {
    environment,
    cache: createRlObservationEncoderCache(),
    initialObservation,
  };
});

const denseEnvironments = makeEnvironments();
const sparseEnvironments = makeEnvironments();
const featureSpec = createRlFeatureSpecV2(
  denseEnvironments[0].initialObservation,
);

const makeClient = (sparse: boolean) => new PythonPpoClient({
  command: value("--python") ?? "python",
  cwd: process.cwd(),
  device: "cuda",
  compactPaddedRows: true,
  env: {
    PPO_PACKED_PREPARE_MODE: "fast_batch_v2",
    ...(sparse ? { PPO_SPARSE_ACTION_TRANSPORT: "1" } : {}),
  },
});

const denseClient = makeClient(false);
const sparseClient = makeClient(true);

const collectSamples = (
  entries: ReturnType<typeof makeEnvironments>,
  round: number,
) => entries.map(({ environment, cache }, environmentIndex) => {
  if (environment.isTerminal()) {
    throw new Error(
      `Environment ${environmentIndex} terminated at round ${round}`,
    );
  }
  const teamId = environment.getCurrentActorTeamId();
  if (!teamId) {
    throw new Error(
      `Environment ${environmentIndex} has no actor at round ${round}`,
    );
  }
  const observation = environment.getObservationForEncoding(teamId);
  const legal = environment.getLegalActionsForEncoding(teamId);
  if (!legal.length) {
    throw new Error(
      `Environment ${environmentIndex} has no legal actions at round ${round}`,
    );
  }
  return {
    observation: encodeRlObservationV2(observation, cache),
    legalActions: encodeRlLegalActionsV2(observation, legal),
  };
});
const denseReady = await denseClient.start({
  seed: 7,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});
const sparseReady = await sparseClient.start({
  seed: 7,
  featureSpec,
  hyperparameters,
  initialCheckpoint: resume,
  resume,
});
if (denseReady.selectedDevice !== "cuda" || sparseReady.selectedDevice !== "cuda") {
  throw new Error("Sparse Action transport probe requires CUDA");
}

let denseRoundTripMs = 0;
let sparseRoundTripMs = 0;
let comparedDecisions = 0;
let firstWireAudit: Record<string, unknown> | undefined;
const mismatches: Array<Record<string, unknown>> = [];

try {
  for (let round = 0; round < rounds; round += 1) {
    const denseHashesBefore = denseEnvironments.map(
      ({ environment }) => environment.getStateHash(),
    );
    const sparseHashesBefore = sparseEnvironments.map(
      ({ environment }) => environment.getStateHash(),
    );
    if (JSON.stringify(denseHashesBefore) !== JSON.stringify(sparseHashesBefore)) {
      throw new Error(`State mismatch before round ${round}`);
    }

    const denseSamples = collectSamples(denseEnvironments, round);
    const sparseSamples = collectSamples(sparseEnvironments, round);
    for (let index = 0; index < denseSamples.length; index += 1) {
      if (
        JSON.stringify(denseSamples[index].legalActions.actionKeys)
        !== JSON.stringify(sparseSamples[index].legalActions.actionKeys)
      ) {
        throw new Error(`Legal-action mismatch at round ${round}, env ${index}`);
      }
    }

    if (round === 0) {
      const densePacked = packPpoActBatchInput(
        denseSamples.map(({ observation, legalActions }) => ({
          observation,
          actions: legalActions.actions,
        })),
        featureSpec,
        { compactMaskedPrefixes: true },
      );
      const sparsePacked = packPpoActBatchInput(
        denseSamples.map(({ observation, legalActions }) => ({
          observation,
          actions: legalActions.actions,
        })),
        featureSpec,
        { compactMaskedPrefixes: true, sparseActions: true },
      );
      const denseActionBytes = densePacked.tensors.find(
        (tensor) => tensor.name === "actions",
      )?.byteLength ?? 0;
      const sparseActionBytes = sparsePacked.tensors
        .filter((tensor) =>
          tensor.name === "actionSparseIndices"
          || tensor.name === "actionSparseValues"
        )
        .reduce((sum, tensor) => sum + tensor.byteLength, 0);
      firstWireAudit = {
        densePayloadBytes: densePacked.payload.byteLength,
        sparsePayloadBytes: sparsePacked.payload.byteLength,
        payloadReduction:
          1 - sparsePacked.payload.byteLength / densePacked.payload.byteLength,
        denseActionBytes,
        sparseActionBytes,
        actionReduction: 1 - sparseActionBytes / denseActionBytes,
        sparseActionShape: sparsePacked.actionSparseShape,
      };
    }

    const denseStarted = performance.now();
    const denseActions = await denseClient.actBatch(denseSamples);
    const denseElapsed = performance.now() - denseStarted;

    const sparseStarted = performance.now();
    const sparseActions = await sparseClient.actBatch(sparseSamples);
    const sparseElapsed = performance.now() - sparseStarted;

    for (let index = 0; index < denseActions.length; index += 1) {
      const dense = denseActions[index];
      const sparse = sparseActions[index];
      const exact = (
        dense.actionIndex === sparse.actionIndex
        && Object.is(dense.logProbability, sparse.logProbability)
        && Object.is(dense.value, sparse.value)
        && JSON.stringify(dense.actionKey) === JSON.stringify(sparse.actionKey)
      );
      if (!exact) {
        mismatches.push({
          round,
          environmentIndex: index,
          dense,
          sparse,
        });
      }
    }
    if (mismatches.length) break;

    denseActions.forEach((action, index) => {
      denseEnvironments[index].environment.stepWithoutObservation(
        action.actionKey,
      );
    });
    sparseActions.forEach((action, index) => {
      sparseEnvironments[index].environment.stepWithoutObservation(
        action.actionKey,
      );
    });

    const denseHashesAfter = denseEnvironments.map(
      ({ environment }) => environment.getStateHash(),
    );
    const sparseHashesAfter = sparseEnvironments.map(
      ({ environment }) => environment.getStateHash(),
    );
    if (JSON.stringify(denseHashesAfter) !== JSON.stringify(sparseHashesAfter)) {
      mismatches.push({
        round,
        kind: "stateHash",
        denseHashesAfter,
        sparseHashesAfter,
      });
      break;
    }

    if (round >= warmup) {
      denseRoundTripMs += denseElapsed;
      sparseRoundTripMs += sparseElapsed;
      comparedDecisions += denseActions.length;
    }
  }
  const denseDiagnostics = await denseClient.diagnostics();
  const sparseDiagnostics = await sparseClient.diagnostics();
  const diagnosticKeys = [
    "parameterHash",
    "optimizerHash",
    "rngHash",
    "gradientHash",
  ] as const;
  const diagnosticsExact = diagnosticKeys.every(
    (key) => denseDiagnostics[key] === sparseDiagnostics[key],
  );

  const denseFinalHashes = denseEnvironments.map(
    ({ environment }) => environment.getStateHash(),
  );
  const sparseFinalHashes = sparseEnvironments.map(
    ({ environment }) => environment.getStateHash(),
  );
  const finalStateHashesExact =
    JSON.stringify(denseFinalHashes) === JSON.stringify(sparseFinalHashes);
  const allExact =
    mismatches.length === 0
    && diagnosticsExact
    && finalStateHashesExact;

  console.log(JSON.stringify({
    probe: "ppo_sparse_action_transport_equivalence",
    status: allExact ? "passed" : "failed",
    base: "fast_batch_v5_compact_rows",
    sparseTransportOnly: true,
    directSparseEncoderIntegrated: false,
    retentionChanged: false,
    modelUpdate: false,
    environmentCount: 8,
    seeds,
    rounds,
    warmupRoundsExcludedFromTiming: warmup,
    timedDecisionsPerPath: comparedDecisions,
    allExact,
    mismatchCount: mismatches.length,
    firstMismatch: mismatches[0] ?? null,
    diagnosticsExact,
    diagnostics: {
      dense: denseDiagnostics,
      sparse: sparseDiagnostics,
    },
    finalStateHashesExact,
    finalStateHashes: denseFinalHashes,
    wire: firstWireAudit,
    timing: {
      denseRoundTripMs: Number(denseRoundTripMs.toFixed(3)),
      sparseRoundTripMs: Number(sparseRoundTripMs.toFixed(3)),
      denseMsPerDecision: Number(
        (denseRoundTripMs / comparedDecisions).toFixed(6),
      ),
      sparseMsPerDecision: Number(
        (sparseRoundTripMs / comparedDecisions).toFixed(6),
      ),
      sparseVsDenseSpeedup: Number(
        (denseRoundTripMs / sparseRoundTripMs).toFixed(4),
      ),
    },
  }, null, 2));

  if (!allExact) process.exitCode = 1;
} finally {
  await denseClient.close();
  await sparseClient.close();
}
