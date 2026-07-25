import { runParallelRlImitationCollection } from "./rlImitationParallel";

function numberArg(name: string, fallback: number) {
  const index = process.argv.indexOf(`--${name}`);
  const value = index >= 0 ? Number(process.argv[index + 1]) : fallback;
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}
function stringArg(name: string, fallback: string) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

async function main() {
  const result = await runParallelRlImitationCollection({
    seedStart: numberArg("seed", 1),
    episodeCount: process.argv.includes("--episodes") ? numberArg("episodes", 1) : numberArg("matches", 1),
    maxTurns: numberArg("max-turns", 300),
    outputPath: stringArg("output", "rl-imitation-replay.jsonl"),
    workerCount: numberArg("workers", 1),
  });
  console.log(JSON.stringify(result, null, 2));
  if (result.failedEpisodeCount > 0) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? `${error.name}: ${error.message}` : String(error));
  process.exitCode = 1;
});
