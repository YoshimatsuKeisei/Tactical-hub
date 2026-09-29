import { encodeRlLegalActionsSparseV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationCompactV2,
} from "./rlObservationEncoder";
import {
  DEFAULT_PPO_HYPERPARAMETERS,
} from "./rlPpoSelfPlay";
import { PpoRolloutWorkerV7Pool } from "./rlPpoRolloutWorkerV7Pool";

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
    if (!actor) throw new Error("V7 worker preflight missing serial actor");
    const observation = slot.environment.getObservationForEncoding(actor);
    const legal = slot.environment.getLegalActionsForEncoding(actor);
    if (!legal.length) {
      throw new Error("V7 worker preflight missing serial legal action");
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

function assertPreparedEqual(
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
      throw new Error(`round ${round} environment order mismatch at ${index}`);
    }
    if (worker.progressHash !== expected.progressHash) {
      throw new Error(
        `round ${round} progress hash mismatch env=${worker.environmentIndex}`,
      );
    }
    if (
      JSON.stringify(worker.observation)
      !== JSON.stringify(expected.observation)
    ) {
      throw new Error(
        `round ${round} observation mismatch env=${worker.environmentIndex}`,
      );
    }
    if (
      JSON.stringify(worker.legalActions)
      !== JSON.stringify(expected.legalActions)
    ) {
      throw new Error(
        `round ${round} legal-action mismatch env=${worker.environmentIndex}`,
      );
    }
  }
}

try {
  let prepareBarrierMs = 0;
  let applyBarrierMs = 0;

  for (let round = 0; round < 2; round += 1) {
    const expected = serialPrepare();
    const prepared = await pool.prepare(round);
    prepareBarrierMs += prepared.barrierMs;
    assertPreparedEqual(round, prepared.samples, expected);

    const actions = prepared.samples.map((sample) => ({
      environmentIndex: sample.environmentIndex,
      actionIndex: 0,
      actionKey: sample.legalActions.actionKeys[0],
      logProbability: 0,
      value: 0,
    }));

    for (const action of actions) {
      const slot = serial[action.environmentIndex];
      slot.environment.stepWithoutObservation(action.actionKey);
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

    for (const slot of serial) {
      const workerNext = round === 0
        ? undefined
        : slot.environment.getProgressHash();
      if (workerNext !== undefined && !workerNext) {
        throw new Error("serial progress hash unexpectedly empty");
      }
    }
  }

  console.log(JSON.stringify({
    preflight: "passed",
    workerCount: pool.workerCount,
    environmentCount: 8,
    rounds: 2,
    exactEncodedSamples: true,
    prepareBarrierMs,
    applyBarrierMs,
  }, null, 2));
} finally {
  await pool.close();
}
