import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  DEFAULT_PPO_HYPERPARAMETERS,
} from "./rlPpoSelfPlay";
import {
  combinePackedSingleSampleBatches,
  type PackedBcBatch,
} from "./rlBcPackedBatch";
import { PpoRolloutWorkerPool } from "./rlPpoRolloutWorkerPool";

const probe = new RlEnvironmentV2(
  undefined,
  true,
  { cpuStep: { rlInPlacePhaseTransitions: true } },
);
const initial = probe.reset(7, 4);
const featureSpec = createRlFeatureSpecV2(initial);

const pool = await PpoRolloutWorkerPool.create({
  workerCount: 4,
  environmentCount: 8,
  firstGameSeed: 9,
  featureSpec,
  hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
  safetyMaxTurns: 1_000,
  safetyMaxActions: 100,
});

try {
  const first = await pool.prepare(0);
  if (first.samples.length !== 8) {
    throw new Error(
      `Expected 8 first-round samples, received ${first.samples.length}`,
    );
  }

  const combine = (
    samples: typeof first.samples,
  ) => combinePackedSingleSampleBatches(
    samples.map((sample): PackedBcBatch => ({
      batchSize: 1,
      tensors: sample.packed.tensors,
      payload: Buffer.from(sample.packed.payload),
    })),
  );

  const firstPacked = combine(first.samples);
  if (firstPacked.batchSize !== 8) {
    throw new Error("First worker-packed batch size mismatch");
  }

  const firstApplied = await pool.apply(
    0,
    first.samples.map((sample) => ({
      environmentIndex: sample.environmentIndex,
      actionIndex: 0,
      actionKey: sample.actionKeys[0],
      logProbability: 0,
      value: 0,
    })),
  );

  const second = await pool.prepare(1);
  if (second.samples.length !== 8) {
    throw new Error(
      `Expected 8 second-round samples, received ${second.samples.length}`,
    );
  }

  const secondPacked = combine(second.samples);
  if (secondPacked.batchSize !== 8) {
    throw new Error("Second worker-packed batch size mismatch");
  }

  const secondApplied = await pool.apply(
    1,
    second.samples.map((sample) => ({
      environmentIndex: sample.environmentIndex,
      actionIndex: 0,
      actionKey: sample.actionKeys[0],
      logProbability: 0,
      value: 0,
    })),
  );

  console.log(JSON.stringify({
    status: "passed",
    workerCount: pool.workerCount,
    firstRoundSamples: first.samples.length,
    secondRoundSamples: second.samples.length,
    firstPackedBytes: firstPacked.payload.byteLength,
    secondPackedBytes: secondPacked.payload.byteLength,
    finalizedDuringProbe:
      first.finalized.length
      + firstApplied.finalized.length
      + second.finalized.length
      + secondApplied.finalized.length,
    prepareBarrierMs:
      first.barrierMs + second.barrierMs,
    applyBarrierMs:
      firstApplied.barrierMs + secondApplied.barrierMs,
  }, null, 2));
} finally {
  await pool.close();
}
