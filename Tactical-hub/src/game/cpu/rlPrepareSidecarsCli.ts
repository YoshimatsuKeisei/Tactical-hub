import { prepareRlReplaySidecars } from "./rlPrepareSidecars";
import { formatRlElapsed } from "./rlProgress";

function stringArg(name: string, fallback: string) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
function positiveIntegerArg(name: string, fallback: number) {
  const value = Number(stringArg(name, String(fallback)));
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}

try {
  const result = await prepareRlReplaySidecars({
    dataPath: stringArg("data", "rl-data/bc100"),
    workerCount: positiveIntegerArg("workers", 1),
    onProgress: (progress) => {
      console.error(
        `[Sidecar] ${progress.completed}/${progress.total}`
        + ` | generated=${progress.generated} reused=${progress.reused} failed=${progress.failed}`
        + ` | elapsed=${formatRlElapsed(progress.elapsedMs)}`,
      );
    },
  });
  console.log(JSON.stringify(result, null, 2));
  if (result.failedCount > 0) process.exitCode = 1;
} catch (error) {
  console.error(error instanceof Error ? `${error.name}: ${error.message}` : String(error));
  process.exitCode = 1;
}
