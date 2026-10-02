import assert from "node:assert/strict";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import { DEFAULT_PPO_HYPERPARAMETERS } from "./rlPpoSelfPlay";
import type { PpoRolloutWorkerV7PreparedGroup } from "./rlPpoRolloutWorkerV7Messages";
import { PpoRolloutWorkerV7Pool } from "./rlPpoRolloutWorkerV7Pool";

const initialSeed = 19;
const recycleSeedStride = 7;
const probe = new RlEnvironmentV2(undefined, true, {
  cpuStep: {
    rlInPlacePhaseTransitions: true,
    rlPrevalidatedMovement: true,
    rlInPlaceProduction: true,
  },
});
const featureSpec = createRlFeatureSpecV2(
  probe.reset(initialSeed, 4),
);

const pool = await PpoRolloutWorkerV7Pool.create({
  workerCount: 1,
  environmentCount: 1,
  firstGameSeed: initialSeed,
  featureSpec,
  hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
  safetyMaxTurns: 1_000,
  safetyMaxActions: 1,
  autoRecycle: true,
  recycleSeedStride,
});

const actionFor = (group: PpoRolloutWorkerV7PreparedGroup) => {
  assert.deepEqual(group.environmentIndices, [0]);
  assert.equal(group.decisionIndices[0], 0);
  const actionKey = group.actionKeys[0]?.[0];
  assert.ok(actionKey);
  return [{
    environmentIndex: 0,
    actionIndex: 0,
    actionKey,
    logProbability: 0,
    value: 0,
  }];
};

try {
  const initial = await pool.prepareWorker(0, 0);
  assert.ok(initial.group);

  const recycled = await pool.advanceWorker(
    0,
    0,
    actionFor(initial.group),
  );
  assert.equal(recycled.finalized.length, 1);
  assert.equal(recycled.finalized[0].summary.seed, initialSeed);
  assert.ok(recycled.group);
  const diagnostics = await pool.getWorkerDiagnostics(0);
  assert.deepEqual(diagnostics.map((entry) => ({
    environmentIndex: entry.environmentIndex,
    currentEpisodeSeed: entry.currentEpisodeSeed,
    generation: entry.generation,
    episodeDecisionCount: entry.episodeDecisionCount,
  })), [{
    environmentIndex: 0,
    currentEpisodeSeed: initialSeed + recycleSeedStride,
    generation: 1,
    episodeDecisionCount: 0,
  }]);

  await pool.setWorkerAutoRecycle(0, false);

  const drained = await pool.advanceWorker(
    0,
    1,
    actionFor(recycled.group),
  );
  assert.equal(drained.finalized.length, 1);
  assert.equal(
    drained.finalized[0].summary.seed,
    initialSeed + recycleSeedStride,
  );
  assert.equal(drained.group, undefined);

  const finished = await pool.prepareWorker(0, 2);
  assert.equal(finished.finalized.length, 0);
  assert.equal(finished.group, undefined);

  console.log(JSON.stringify({
    preflight: "passed",
    initialSeed,
    recycleSeedStride,
    recycledSeed: initialSeed + recycleSeedStride,
    disabledGenerationStayedFinished: true,
    diagnosticsRequestVerified: true,
  }));
} finally {
  await pool.close();
}
