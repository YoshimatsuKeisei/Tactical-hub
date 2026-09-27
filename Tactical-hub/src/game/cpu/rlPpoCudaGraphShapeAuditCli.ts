import { performance } from "node:perf_hooks";
import { PythonPpoClient } from "./pythonPpoClient";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
  type EncodedObservation,
} from "./rlObservationEncoder";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";

const STRATEGIC_NAMES = [
  "siegeStates",
  "kingCampaignStates",
  "rewardPlacementRequests",
  "strategistCooldowns",
  "teleportCooldowns",
  "productionIntents",
  "movementIntents",
  "attackIntents",
  "strategistActionIntents",
  "teleportIntents",
] as const;

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const positiveInteger = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`);
  return parsed;
};

const resume = value("--resume");
if (!resume) throw new Error("Specify --resume PPO_CHECKPOINT");

const trainerSeed = positiveInteger("--trainer-seed", 7);
const gameSeed = positiveInteger("--game-seed", 9);
const decisions = positiveInteger("--decisions", 50_000);
const progressEvery = positiveInteger("--progress-every", 5_000);
const python = value("--python") ?? "python";

type ShapeRecord = {
  actionCount: number;
  strategicCounts: number[];
  nonemptyBits: string;
};

const nextPowerOfTwo = (value: number) => {
  if (value <= 0) return 0;
  let bucket = 1;
  while (bucket < value) bucket *= 2;
  return bucket;
};

const exactSignature = (record: ShapeRecord) =>
  `a=${record.actionCount}|s=${record.strategicCounts.join(",")}|n=${record.nonemptyBits}`;

const bucketSignature = (record: ShapeRecord) =>
  `a=${nextPowerOfTwo(record.actionCount)}|s=${record.strategicCounts.map(nextPowerOfTwo).join(",")}|n=${record.nonemptyBits}`;

const increment = (map: Map<string, number>, key: string) => {
  map.set(key, (map.get(key) ?? 0) + 1);
};

const summarize = (map: Map<string, number>, total: number, topLimit = 20) => {
  const entries = [...map.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
  const coverage = (k: number) =>
    total ? entries.slice(0, k).reduce((sum, [, count]) => sum + count, 0) / total : 0;
  return {
    uniqueCount: entries.length,
    top1Coverage: coverage(1),
    top2Coverage: coverage(2),
    top4Coverage: coverage(4),
    top8Coverage: coverage(8),
    top16Coverage: coverage(16),
    top: entries.slice(0, topLimit).map(([signature, count]) => ({
      signature,
      count,
      coverage: total ? count / total : 0,
    })),
  };
};

const phaseCounts = new Map<string, number>();
const actionCounts = new Map<string, number>();
const exactShapes = new Map<string, number>();
const bucketShapes = new Map<string, number>();
const windows = new Map<number, {
  total: number;
  exact: Map<string, number>;
  bucket: Map<string, number>;
}>();

const environment = new RlEnvironmentV2(undefined, true);
const firstObservation = environment.reset(gameSeed, 4);
const featureSpec = createRlFeatureSpecV2(firstObservation);
const encoderCache = createRlObservationEncoderCache();

const client = new PythonPpoClient({
  command: python,
  cwd: process.cwd(),
  device: "cuda",
  env: {
    PPO_PACKED_PREPARE_MODE: "grouped_h2d_skip_empty_manual_categorical",
  },
});

const ready = await client.start({
  seed: trainerSeed,
  featureSpec,
  hyperparameters: {
    learningRate: 3e-4,
    gamma: 0.99,
    gaeLambda: 0.95,
    clipEpsilon: 0.2,
    valueCoefficient: 0.5,
    entropyCoefficient: 0.01,
    maxGradientNorm: 0.5,
  },
  initialCheckpoint: resume,
  resume,
});

if (ready.selectedDevice !== "cuda") throw new Error("CUDA is required");

const shapeRecord = (observation: EncodedObservation, actionCount: number): ShapeRecord => {
  const strategicCounts = STRATEGIC_NAMES.map((name) => observation.strategicState[name].length);
  const nonemptyBits = [
    observation.teamMask.some(Boolean),
    observation.unitMask.some(Boolean),
    observation.baseMask.some(Boolean),
    observation.constructionMask.some(Boolean),
    observation.map.length > 0 && observation.map.some((row) => row.length > 0),
    ...STRATEGIC_NAMES.map((name) => observation.strategicState[name].length > 0),
  ].map(Number).join("");
  return { actionCount, strategicCounts, nonemptyBits };
};

let completedDecisions = 0;
const started = performance.now();

try {
  while (!environment.isTerminal() && completedDecisions < decisions) {
    const actor = environment.getCurrentActorTeamId();
    if (!actor) throw new Error(`No actor at decision ${completedDecisions}`);
    const observation = environment.getObservationForEncoding(actor);
    const legal = environment.getLegalActionsForEncoding(actor);
    if (!legal.length) throw new Error(`No legal actions at decision ${completedDecisions}`);

    const encodedObservation = encodeRlObservationV2(observation, encoderCache);
    const encodedActions = encodeRlLegalActionsV2(observation, legal);
    const record = shapeRecord(encodedObservation, encodedActions.actions.length);
    const exact = exactSignature(record);
    const bucket = bucketSignature(record);

    increment(exactShapes, exact);
    increment(bucketShapes, bucket);
    increment(actionCounts, String(record.actionCount));
    increment(phaseCounts, observation.phase);

    const windowIndex = Math.floor(completedDecisions / progressEvery);
    let window = windows.get(windowIndex);
    if (!window) {
      window = { total: 0, exact: new Map(), bucket: new Map() };
      windows.set(windowIndex, window);
    }
    window.total += 1;
    increment(window.exact, exact);
    increment(window.bucket, bucket);

    const selected = await client.act(encodedObservation, encodedActions);
    environment.stepWithoutObservation(selected.actionKey);
    completedDecisions += 1;

    if (completedDecisions % progressEvery === 0 || completedDecisions === decisions) {
      process.stderr.write(`[CUDA graph shape audit] ${JSON.stringify({
        decisions: completedDecisions,
        exactShapeCount: exactShapes.size,
        bucketShapeCount: bucketShapes.size,
        elapsedMs: Math.round((performance.now() - started) * 100) / 100,
      })}\n`);
    }
  }
} finally {
  await client.close();
}

const elapsedMs = performance.now() - started;
const result = environment.getResult();
const finalStateHash = environment.getStateHash();
const actionDistribution = [...actionCounts.entries()]
  .map(([count, occurrences]) => ({ actionCount: Number(count), occurrences }))
  .sort((left, right) => left.actionCount - right.actionCount);

const windowSummaries = [...windows.entries()]
  .sort(([left], [right]) => left - right)
  .map(([index, window]) => ({
    startDecision: index * progressEvery,
    endDecisionExclusive: index * progressEvery + window.total,
    total: window.total,
    exact: summarize(window.exact, window.total, 8),
    powerOfTwoBucket: summarize(window.bucket, window.total, 8),
  }));

const exactSummary = summarize(exactShapes, completedDecisions);
const bucketSummary = summarize(bucketShapes, completedDecisions);

console.log(JSON.stringify({
  probe: "ppo_cuda_graph_shape_audit",
  status: completedDecisions === decisions ? "passed" : "ended_early",
  trainerSeed,
  gameSeed,
  requestedDecisions: decisions,
  completedDecisions,
  elapsedMs: Math.round(elapsedMs * 100) / 100,
  finalStateHash,
  environmentResult: result,
  phaseCounts: Object.fromEntries([...phaseCounts.entries()].sort()),
  legalActionDistribution: actionDistribution,
  exactShapeSignatures: exactSummary,
  powerOfTwoBucketSignatures: bucketSummary,
  windows: windowSummaries,
  feasibility: {
    top4BucketCoverage: bucketSummary.top4Coverage,
    top8BucketCoverage: bucketSummary.top8Coverage,
    top16BucketCoverage: bucketSummary.top16Coverage,
    suggestedGate: {
      top8CoverageAtLeast0_98: bucketSummary.top8Coverage >= 0.98,
      uniqueBucketCountAtMost16: bucketSummary.uniqueCount <= 16,
    },
  },
  notes: [
    "Batch size is fixed at one; units, bases, constructions and map are already padded by the observation encoder.",
    "Exact signatures include legal-action rows, all ten strategic-table row counts, and skip-empty nonempty branches.",
    "Power-of-two signatures estimate a fixed-shape bucket scheme that pads only variable action/strategic row counts upward.",
    "This probe performs real policy rollout only. It does not replay, update PPO, or save a checkpoint.",
    "The feasibility gate is a proposed engineering gate, not proof that CUDA Graph will achieve a particular speedup.",
  ],
}, null, 2));

if (completedDecisions !== decisions) process.exitCode = 2;
