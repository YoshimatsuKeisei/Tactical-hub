import assert from "node:assert/strict";
import { encodeRlLegalActionsSparseV2 } from "./rlActionEncoder";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import { createRlObservationEncoderCache, encodeRlObservationCompactPackedMapV2 } from "./rlObservationEncoder";
import { DEFAULT_PPO_HYPERPARAMETERS } from "./rlPpoSelfPlay";
import { packPpoActBatchInput } from "./rlPpoPackedBatch";
import { PpoRolloutWorkerV7Pool } from "./rlPpoRolloutWorkerV7Pool";
import { combinePackedWorkerBatchesV7, fromTransferablePackedBcBatch } from "./rlPpoWorkerPackedV7";

function createEnvironment() {
  return new RlEnvironmentV2(undefined, true, {
    cpuStep: {
      rlInPlacePhaseTransitions: true,
      rlPrevalidatedMovement: true,
      rlInPlaceProduction: true,
    },
  });
}
const probe = createEnvironment();
const first = probe.reset(7, 4);
const featureSpec = createRlFeatureSpecV2(first);
const firstGameSeed = 9;
const environmentCount = 8;
const workerCount = 4;
const direct = Array.from({ length: environmentCount }, (_, environmentIndex) => {
  const environment = createEnvironment();
  environment.reset(firstGameSeed + environmentIndex, 4);
  return { environmentIndex, environment, encoderCache: createRlObservationEncoderCache() };
});
const pool = await PpoRolloutWorkerV7Pool.create({
  workerCount,
  environmentCount,
  firstGameSeed,
  featureSpec,
  hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
  safetyMaxTurns: 1000,
  safetyMaxActions: 20,
});
try {
  for (let round = 0; round < 2; round += 1) {
    const prepared = await pool.prepare(round);
    assert.deepEqual(
      prepared.groups.flatMap((group) => group.environmentIndices),
      direct.map((entry) => entry.environmentIndex),
    );
    const samples: Parameters<typeof packPpoActBatchInput>[0] = [];
    const actionKeys: string[][] = [];
    for (const entry of direct) {
      const actor = entry.environment.getCurrentActorTeamId();
      assert.ok(actor);
      const observation = entry.environment.getObservationForEncoding(actor);
      const legal = entry.environment.getLegalActionsForEncoding(actor);
      assert.ok(legal.length > 0);
      const encodedObservation = encodeRlObservationCompactPackedMapV2(observation, entry.encoderCache);
      const encodedActions = encodeRlLegalActionsSparseV2(observation, legal);
      samples.push({ observation: encodedObservation, sparseActions: encodedActions.sparseActions });
      actionKeys.push(encodedActions.actionKeys);
    }
    const directPacked = packPpoActBatchInput(samples, featureSpec, {
      compactMaskedPrefixes: true,
      sparseActions: true,
    });
    const workerPacked = combinePackedWorkerBatchesV7(
      prepared.groups.map((group) => fromTransferablePackedBcBatch(group.packed)),
    );
    assert.equal(workerPacked.batchSize, directPacked.batchSize);
    assert.deepEqual(workerPacked.tensors, directPacked.tensors);
    assert.deepEqual(workerPacked.rowCompaction, directPacked.rowCompaction);
    assert.deepEqual(workerPacked.actionSparseShape, directPacked.actionSparseShape);
    assert.equal(Buffer.compare(workerPacked.payload, directPacked.payload), 0,
      "round " + round + " packed payload mismatch");
    assert.deepEqual(prepared.groups.flatMap((group) => group.actionKeys), actionKeys);

    const actions = actionKeys.map((keys, environmentIndex) => {
      assert.ok(keys.length > 0);
      return { environmentIndex, actionIndex: 0, actionKey: keys[0], logProbability: 0, value: 0 };
    });
    await pool.apply(round, actions);
    direct.forEach((entry, environmentIndex) => {
      entry.environment.stepWithoutObservation(actionKeys[environmentIndex][0]);
    });
  }
  console.log(JSON.stringify({
    preflight: "passed",
    workerCount,
    environmentCount,
    rounds: 2,
    groupedPackedExact: true,
  }));
} finally {
  await pool.close();
}
