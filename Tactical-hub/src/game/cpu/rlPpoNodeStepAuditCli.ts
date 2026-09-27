import { performance } from "node:perf_hooks";
import { PythonPpoClient } from "./pythonPpoClient";
import {
  RlEnvironmentV2,
  type RlEnvironmentInstrumentation,
} from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "./rlObservationEncoder";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import type { CpuDecision } from "./types";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
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

const trainerSeed = positiveInteger("--trainer-seed", 7);
const gameSeed = positiveInteger("--game-seed", 9);
const decisions = positiveInteger("--decisions", 10_000);
const windowSize = positiveInteger("--window-size", 1_000);
const python = value("--python") ?? "python";
const inPlacePhases = args.includes("--in-place-phases");

type Metric = {
  count: number;
  totalMs: number;
  maxMs: number;
};

type Bucket = {
  decisions: number;
  step: Metric;
  runtimeClone: Metric;
  policy: Metric;
  apply: Metric;
  log: Metric;
  enumerate: Metric;
  applyByKind: Map<string, Metric>;
  enumerateByPhase: Map<string, Metric>;
};

const metric = (): Metric => ({ count: 0, totalMs: 0, maxMs: 0 });
const bucket = (): Bucket => ({
  decisions: 0,
  step: metric(),
  runtimeClone: metric(),
  policy: metric(),
  apply: metric(),
  log: metric(),
  enumerate: metric(),
  applyByKind: new Map(),
  enumerateByPhase: new Map(),
});

const add = (target: Metric, milliseconds: number) => {
  target.count += 1;
  target.totalMs += milliseconds;
  target.maxMs = Math.max(target.maxMs, milliseconds);
};

const addNamed = (map: Map<string, Metric>, key: string, milliseconds: number) => {
  let target = map.get(key);
  if (!target) {
    target = metric();
    map.set(key, target);
  }
  add(target, milliseconds);
};

let overall = bucket();
let currentWindow = bucket();

const both = (selector: (value: Bucket) => Metric, milliseconds: number) => {
  add(selector(overall), milliseconds);
  add(selector(currentWindow), milliseconds);
};

const instrumentation: RlEnvironmentInstrumentation = {
  cpuStep: {
    rlInPlacePhaseTransitions: inPlacePhases,
    onRuntimeClone(milliseconds) {
      both((entry) => entry.runtimeClone, milliseconds);
    },
    onPolicy(milliseconds) {
      both((entry) => entry.policy, milliseconds);
    },
    onApply(milliseconds, decision: CpuDecision, phaseBefore) {
      both((entry) => entry.apply, milliseconds);
      addNamed(overall.applyByKind, decision.kind, milliseconds);
      addNamed(currentWindow.applyByKind, decision.kind, milliseconds);
      // Preserve phase in the key for expensive resolve paths.
      addNamed(overall.applyByKind, `${decision.kind}@${phaseBefore}`, milliseconds);
      addNamed(currentWindow.applyByKind, `${decision.kind}@${phaseBefore}`, milliseconds);
    },
    onLog(milliseconds) {
      both((entry) => entry.log, milliseconds);
    },
  },
  onEnumerate(milliseconds, phase) {
    both((entry) => entry.enumerate, milliseconds);
    addNamed(overall.enumerateByPhase, phase, milliseconds);
    addNamed(currentWindow.enumerateByPhase, phase, milliseconds);
  },
};

const serializeMetric = (value: Metric) => ({
  count: value.count,
  totalMs: Number(value.totalMs.toFixed(3)),
  meanMs: value.count ? Number((value.totalMs / value.count).toFixed(6)) : 0,
  maxMs: Number(value.maxMs.toFixed(6)),
});

const serializeNamed = (values: Map<string, Metric>) =>
  Object.fromEntries(
    [...values.entries()]
      .sort((left, right) => right[1].totalMs - left[1].totalMs || left[0].localeCompare(right[0]))
      .map(([key, value]) => [key, serializeMetric(value)]),
  );

const serializeBucket = (value: Bucket) => {
  const accounted =
    value.runtimeClone.totalMs
    + value.policy.totalMs
    + value.apply.totalMs
    + value.log.totalMs
    + value.enumerate.totalMs;
  const unaccounted = Math.max(0, value.step.totalMs - accounted);
  return {
    decisions: value.decisions,
    step: serializeMetric(value.step),
    runtimeClone: serializeMetric(value.runtimeClone),
    policy: serializeMetric(value.policy),
    apply: serializeMetric(value.apply),
    log: serializeMetric(value.log),
    enumerate: serializeMetric(value.enumerate),
    accountedMs: Number(accounted.toFixed(3)),
    unaccountedMs: Number(unaccounted.toFixed(3)),
    accountedPercent: value.step.totalMs
      ? Number(((accounted / value.step.totalMs) * 100).toFixed(2))
      : 0,
    applyByKind: serializeNamed(value.applyByKind),
    enumerateByPhase: serializeNamed(value.enumerateByPhase),
  };
};

const environment = new RlEnvironmentV2(undefined, true, instrumentation);
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

// Exclude reset()/startup automatic work from rollout step accounting.
overall = bucket();
currentWindow = bucket();

const windows: Array<Record<string, unknown>> = [];
const wallStarted = performance.now();
let completedDecisions = 0;

try {
  while (!environment.isTerminal() && completedDecisions < decisions) {
    const actor = environment.getCurrentActorTeamId();
    if (!actor) throw new Error(`No actor at decision ${completedDecisions}`);

    const observation = environment.getObservationForEncoding(actor);
    const legal = environment.getLegalActionsForEncoding(actor);
    if (!legal.length) throw new Error(`No legal actions at decision ${completedDecisions}`);

    const encodedObservation = encodeRlObservationV2(observation, encoderCache);
    const encodedActions = encodeRlLegalActionsV2(observation, legal);
    const selected = await client.act(encodedObservation, encodedActions);

    const stepStarted = performance.now();
    environment.stepWithoutObservation(selected.actionKey);
    const stepMs = performance.now() - stepStarted;
    both((entry) => entry.step, stepMs);

    completedDecisions += 1;
    overall.decisions += 1;
    currentWindow.decisions += 1;

    if (
      currentWindow.decisions === windowSize
      || completedDecisions === decisions
      || environment.isTerminal()
    ) {
      const serialized = serializeBucket(currentWindow);
      windows.push({
        startDecision: completedDecisions - currentWindow.decisions,
        endDecisionExclusive: completedDecisions,
        ...serialized,
      });
      process.stderr.write(
        `[Node step audit] ${JSON.stringify({
          completedDecisions,
          stepMeanMs: serialized.step.meanMs,
          runtimeCloneTotalMs: serialized.runtimeClone.totalMs,
          applyTotalMs: serialized.apply.totalMs,
          enumerateTotalMs: serialized.enumerate.totalMs,
          unaccountedMs: serialized.unaccountedMs,
        })}\n`,
      );
      currentWindow = bucket();
    }
  }
} finally {
  await client.close();
}

const wallMs = performance.now() - wallStarted;
const result = environment.getResult();
const finalStateHash = environment.getStateHash();

console.log(JSON.stringify({
  probe: "ppo_node_step_audit",
  status: completedDecisions === decisions ? "passed" : "ended_early",
  trainerSeed,
  gameSeed,
  requestedDecisions: decisions,
  completedDecisions,
  inPlacePhases,
  wallMs: Number(wallMs.toFixed(3)),
  finalStateHash,
  environmentResult: result,
  overall: serializeBucket(overall),
  windows,
  notes: [
    "The rollout uses the current integrated policy path: grouped H2D + skip-empty + manual categorical.",
    "RL in-place movement is enabled to match the current 50k baseline condition.",
    inPlacePhases
      ? "RL-only submit_movement and submit_strategist GameState in-place paths are enabled."
      : "RL-only submit_movement and submit_strategist GameState in-place paths are disabled.",
    "runtimeClone measures structuredClone(sourceRuntime) inside advanceCpuOneStep.",
    "apply measures engine action application excluding runtime clone, policy callback, and logging.",
    "enumerate measures enumerateRlDecisionsV2 calls inside advanceAutomatic.",
    "step measures the complete stepWithoutObservation call; unaccounted is step minus the instrumented subcomponents.",
    "This probe does not replay, update PPO, or save checkpoints.",
  ],
}, null, 2));

if (completedDecisions !== decisions) process.exitCode = 2;
