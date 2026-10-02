import { runPpoAsyncRolloutProbeV8 } from "./rlPpoAsyncRolloutProbeV8";
import { PythonPpoClient } from "./pythonPpoClient";

const args = process.argv.slice(2);
if (args.includes("--profile")) process.env.PPO_PROFILE = "1";
if (args.includes("--node-profile")) {
  process.env.PPO_NODE_PROFILE = "1";
}

const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};

const numeric = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be positive`);
  }
  return parsed;
};

const positiveInteger = (
  name: string,
  fallback: number,
) => {
  const parsed = numeric(name, fallback);
  if (!Number.isInteger(parsed)) {
    throw new Error(`${name} must be an integer`);
  }
  return parsed;
};

const python = value("--python") ?? "python";
const naturalRecycleDiagnostic = args.includes(
  "--natural-recycle-diagnostic",
);

const result = await runPpoAsyncRolloutProbeV8({
  seed: positiveInteger("--seed", 7),
  environmentCount:
    positiveInteger("--environments", 8),
  rolloutWorkerCount:
    positiveInteger("--rollout-workers", 4),
  continuousRecycle:
    args.includes("--continuous-recycle")
    || naturalRecycleDiagnostic,
  naturalRecycleProbe: args.includes("--natural-recycle-probe"),
  naturalRecycleDiagnostic,
  targetDecisions: value("--target-decisions") === undefined
    ? undefined
    : positiveInteger("--target-decisions", 1),
  initialCheckpoint:
    value("--initial-checkpoint")
    ?? "rl-checkpoints/bc-v2-init.pt",
  resume: value("--resume"),
  safetyMaxTurns:
    positiveInteger("--safety-max-turns", 1_000),
  safetyMaxActions:
    positiveInteger("--safety-max-actions", 100_000),
  memoryLogInterval:
    positiveInteger("--memory-log-interval", 5_000),
  hyperparameters: {
    learningRate: numeric("--learning-rate", 3e-4),
    gamma: numeric("--gamma", 0.99),
    gaeLambda: numeric("--gae-lambda", 0.95),
    clipEpsilon: numeric("--clip-epsilon", 0.2),
    valueCoefficient: numeric("--value-coef", 0.5),
    entropyCoefficient: numeric("--entropy-coef", 0.01),
    maxGradientNorm: numeric("--max-grad-norm", 0.5),
  },
  client: new PythonPpoClient({
    command: python,
    device: "cuda",
    compactPaddedRows: true,
    env: {
      PPO_PACKED_PREPARE_MODE: "fast_batch_v2",
      PPO_SPARSE_ACTION_TRANSPORT: "1",
      PPO_RETENTION_STORAGE_MODE: "raw",
      PPO_PERSISTENT_ACT_H2D: "1",
      PPO_ACT_CUDA_GRAPH_HOT: "1",
      PPO_ACT_CUDA_GRAPH_MIN_HITS: "2",
      PPO_ACT_CUDA_GRAPH_MAX_ENTRIES: "32",
      PPO_PERSISTENT_REPLAY_H2D: "1",
      PPO_IMMUTABLE_RAW_RETENTION: "1",
    },
  }),
});

if (naturalRecycleDiagnostic) {
  console.log(
    "ASYNC_V8_NATURAL_TERMINATION_DIAGNOSTIC_RESULT="
    + JSON.stringify(result),
  );
} else {
  console.log(JSON.stringify(result, null, 2));
}
