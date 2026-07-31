import { runBehavioralCloning } from "./rlBehavioralCloning";
import { parseEpisodeRange } from "./rlReplayReader";
import { formatRlElapsed, formatRlRate } from "./rlProgress";
import { parseRlTorchDevice } from "./rlTorchDevice";

function stringArg(name: string, fallback: string) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
function numberArg(name: string, fallback: number) {
  const value = Number(stringArg(name, String(fallback)));
  if (!Number.isFinite(value)) throw new Error(`Invalid --${name}`);
  return value;
}
function optionalStringArg(name: string) {
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`--${name} requires a path`);
  return value;
}
function optionalPositiveIntegerArg(name: string) {
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const value = Number(process.argv[index + 1]);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}

const result = await runBehavioralCloning({
  dataPath: stringArg("data", "rl-data/bc100"),
  epochs: numberArg("epochs", 1),
  batchSize: numberArg("batch-size", 16),
  learningRate: numberArg("learning-rate", 1e-4),
  checkpointPath: stringArg("checkpoint", "rl-checkpoints/bc-best.pt"),
  latestCheckpointPath: optionalStringArg("latest-checkpoint"),
  resumePath: optionalStringArg("resume"),
  trainRange: parseEpisodeRange(stringArg("train-range", "1-80")),
  validationRange: parseEpisodeRange(stringArg("validation-range", "81-90")),
  testRange: parseEpisodeRange(stringArg("test-range", "91-100")),
  seed: numberArg("seed", 1),
  pythonCommand: stringArg("python", "python"),
  workerCount: numberArg("workers", 1),
  torchThreads: optionalPositiveIntegerArg("torch-threads"),
  torchInteropThreads: optionalPositiveIntegerArg("torch-interop-threads"),
  device: parseRlTorchDevice(stringArg("device", "auto")),
  onProgress: (progress) => {
    console.error(
      `[BC] phase=${progress.phase} epoch=${progress.epoch}/${progress.totalEpochs}`
      + ` | episode=${progress.episode}/${progress.totalEpisodes}`
      + ` | samples=${progress.processedSamples} batches=${progress.processedBatches}`
      + ` | recent=${formatRlRate(progress.recentSamplesPerSecond)} samples/s`
      + ` | elapsed=${formatRlElapsed(progress.elapsedMs)}`
      + (progress.kind === "heartbeat" ? " | heartbeat" : ""),
    );
  },
  onStatus: (message) => console.error(message),
});
console.log(JSON.stringify(result, null, 2));
