import { encodeRlLegalActionsSparseV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationCompactV2,
} from "./rlObservationEncoder";
import { packPpoActBatchInput } from "./rlPpoPackedBatch";
import {
  DEFAULT_PPO_HYPERPARAMETERS,
} from "./rlPpoSelfPlay";
import { PpoRolloutWorkerV7Pool } from "./rlPpoRolloutWorkerV7Pool";
import {
  combinePackedSingleSampleBatchesV7,
  fromTransferablePackedBcBatch,
} from "./rlPpoWorkerPackedV7";

function createEnvironment() {
  return new RlEnvironmentV2(
    undefined,
    true,
    {
      cpuStep: {
        rlInPlacePhaseTransitions: true,
        rlPrevalidatedMovement: true,
        rlInPlaceProduction: true,
      },
    },
  );
}

const probe = createEnvironment();
const initial = probe.reset(7, 4);
const featureSpec = createRlFeatureSpecV2(initial);
const firstGameSeed = 9;

const serial = Array.from({ length: 8 }, (_, environmentIndex) => {
  const environment = createEnvironment();
  environment.reset(firstGameSeed + environmentIndex, 4);
  return {
    environmentIndex,
    environment,
    encoderCache: createRlObservationEncoderCache(),
  };
});

const pool = await PpoRolloutWorkerV7Pool.create({
  workerCount: 4,
  environmentCount: 8,
  firstGameSeed,
  featureSpec,
  hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
  safetyMaxTurns: 1_000,
  safetyMaxActions: 100,
});

function serialPrepare() {
  return serial.map((slot) => {
    const actor = slot.environment.getCurrentActorTeamId();
    if (!actor) throw new Error("V7D worker preflight missing serial actor");
    const observation = slot.environment.getObservationForEncoding(actor);
    const legal = slot.environment.getLegalActionsForEncoding(actor);
    if (!legal.length) {
      throw new Error("V7D worker preflight missing serial legal action");
    }
    return {
      environmentIndex: slot.environmentIndex,
      progressHash: slot.environment.getProgressHash(),
      observation: encodeRlObservationCompactV2(
        observation,
        slot.encoderCache,
      ),
      legalActions: encodeRlLegalActionsSparseV2(
        observation,
        legal,
      ),
    };
  });
}

function directPack(
  samples: ReturnType<typeof serialPrepare>,
) {
  return packPpoActBatchInput(
    samples.map((sample) => ({
      observation: sample.observation,
      sparseActions: sample.legalActions.sparseActions,
    })),
    featureSpec,
    {
      compactMaskedPrefixes: true,
      sparseActions: true,
    },
  );
}

function assertPackedEqual(
  round: number,
  workerSamples: Awaited<ReturnType<typeof pool.prepare>>["samples"],
  serialSamples: ReturnType<typeof serialPrepare>,
) {
  if (workerSamples.length !== serialSamples.length) {
    throw new Error(
      `round ${round} sample count mismatch: ${workerSamples.length} != ${serialSamples.length}`,
    );
  }

  for (let index = 0; index < serialSamples.length; index += 1) {
    const worker = workerSamples[index];
    const expected = serialSamples[index];
    if (worker.environmentIndex !== expected.environmentIndex) {
      throw new Error(
        `round ${round} environment order mismatch at ${index}`,
      );
    }
    if (worker.progressHash !== expected.progressHash) {
      throw new Error(
        `round ${round} progress hash mismatch env=${worker.environmentIndex}`,
      );
    }
    if (
      JSON.stringify(worker.actionKeys)
      !== JSON.stringify(expected.legalActions.actionKeys)
    ) {
      throw new Error(
        `round ${round} action-key mismatch env=${worker.environmentIndex}`,
      );
    }
  }

  const direct = directPack(serialSamples);
  const combined = combinePackedSingleSampleBatchesV7(
    workerSamples.map(
      (sample) => fromTransferablePackedBcBatch(sample.packed),
    ),
  );

  if (JSON.stringify(combined.tensors) !== JSON.stringify(direct.tensors)) {
    throw new Error(`round ${round} packed tensor descriptors differ`);
  }
  if (
    JSON.stringify(combined.rowCompaction)
    !== JSON.stringify(direct.rowCompaction)
  ) {
    throw new Error(`round ${round} rowCompaction differs`);
  }
  if (
    JSON.stringify(combined.actionSparseShape)
    !== JSON.stringify(direct.actionSparseShape)
  ) {
    throw new Error(`round ${round} actionSparseShape differs`);
  }
  if (!combined.payload.equals(direct.payload)) {
    throw new Error(`round ${round} packed payload differs`);
  }
}

try {
  let prepareBarrierMs = 0;
  let applyBarrierMs = 0;

  for (let round = 0; round < 2; round += 1) {
    const expected = serialPrepare();
    const prepared = await pool.prepare(round);
    prepareBarrierMs += prepared.barrierMs;
    assertPackedEqual(round, prepared.samples, expected);

    const actions = prepared.samples.map((sample) => ({
      environmentIndex: sample.environmentIndex,
      actionIndex: 0,
      actionKey: sample.actionKeys[0],
      logProbability: 0,
      value: 0,
    }));

    for (const action of actions) {
      serial[action.environmentIndex].environment.stepWithoutObservation(
        action.actionKey,
      );
    }

    const applied = await pool.apply(round, actions);
    applyBarrierMs += applied.barrierMs;
    if (
      applied.finalized.some((item) =>
        item.summary.outcomeKind === "abnormal_truncated"
      )
    ) {
      throw new Error(
        `round ${round} unexpectedly finalized as abnormal_truncated`,
      );
    }
  }

  console.log(JSON.stringify({
    preflight: "passed",
    workerCount: pool.workerCount,
    environmentCount: 8,
    rounds: 2,
    exactPackedBatch: true,
    prepareBarrierMs,
    applyBarrierMs,
  }, null, 2));
} finally {
  await pool.close();
}
