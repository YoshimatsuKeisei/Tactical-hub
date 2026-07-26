import { runRlBcShortProfile } from "./rlBcProfiler";
import { parseRlTorchDevice } from "./rlTorchDevice";

function value(name: string, fallback: string) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
function integer(name: string, fallback: number, allowZero = false) {
  const parsed = Number(value(name, String(fallback)));
  if (!Number.isInteger(parsed) || parsed < (allowZero ? 0 : 1)) throw new Error(`--${name} must be ${allowZero ? "non-negative" : "positive"}`);
  return parsed;
}

const result = await runRlBcShortProfile({
  dataPath: value("data", "rl-data/bc100"),
  samples: integer("samples", 4096),
  warmupSamples: integer("warmup-samples", 256, true),
  batchSize: integer("batch-size", 64),
  workerCount: integer("workers", 1),
  profileEpisodeCount: integer("profile-episodes", 4),
  device: parseRlTorchDevice(value("device", "auto")),
  pythonCommand: value("python", "python"),
});
console.log(JSON.stringify(result, null, 2));
