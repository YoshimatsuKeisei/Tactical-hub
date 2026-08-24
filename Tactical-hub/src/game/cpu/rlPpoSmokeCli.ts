import { runPpoSelfPlaySmoke } from "./rlPpoSelfPlay";
import { PythonPpoClient } from "./pythonPpoClient";
import { parseRlTorchDevice } from "./rlTorchDevice";

const args = process.argv.slice(2);
const value = (name: string) => { const index = args.indexOf(name); return index >= 0 ? args[index + 1] : undefined; };
const numeric = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${name} must be positive`);
  return parsed;
};
const positiveInteger = (name: string, fallback: number) => {
  const parsed = numeric(name, fallback);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
};

const result = await runPpoSelfPlaySmoke({
  seed: positiveInteger("--seed", 1), episodes: positiveInteger("--episodes", 1),
  initialCheckpoint: value("--initial-checkpoint") ?? "rl-checkpoints/bc-v2-init.pt",
  outputCheckpoint: value("--checkpoint") ?? "rl-checkpoints/ppo-latest.pt",
  bestCheckpoint: value("--best-checkpoint") ?? "rl-checkpoints/ppo-best.pt",
  resume: value("--resume"), safetyMaxTurns: positiveInteger("--safety-max-turns", 1_000),
  safetyMaxActions: positiveInteger("--safety-max-actions", 100_000),
  replayChunkSize: positiveInteger("--replay-chunk-size", 8),
  memoryLogInterval: positiveInteger("--memory-log-interval", 500),
  hyperparameters: {
    learningRate: numeric("--learning-rate", 3e-4), gamma: numeric("--gamma", 0.99),
    gaeLambda: numeric("--gae-lambda", 0.95), clipEpsilon: numeric("--clip-epsilon", 0.2),
    valueCoefficient: numeric("--value-coef", 0.5), entropyCoefficient: numeric("--entropy-coef", 0.01),
    maxGradientNorm: numeric("--max-grad-norm", 0.5),
  },
  client: new PythonPpoClient({ command: value("--python") ?? "python", device: parseRlTorchDevice(value("--device") ?? "auto") }),
});
console.log(JSON.stringify(result, null, 2));
