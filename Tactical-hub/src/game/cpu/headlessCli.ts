import { readFileSync, writeFileSync } from "node:fs";
import type { HeadlessMode } from "./headlessSimulation";
import { getAvailableHeadlessParallelism, runParallelHeadlessBatch } from "./headlessParallel";
import { compareHeadlessTraces, type HeadlessTraceEntry } from "./headlessTrace";

function numberArg(name: string, fallback: number) {
  const index = process.argv.indexOf(`--${name}`);
  const value = index >= 0 ? Number(process.argv[index + 1]) : fallback;
  if (!Number.isFinite(value)) throw new Error(`Invalid --${name}`);
  return value;
}
function stringArg(name: string, fallback: string) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
function hasFlag(name: string) { return process.argv.includes(`--${name}`); }

async function main() {
  const compareIndex = process.argv.indexOf("--compare-traces");
  if (compareIndex >= 0) {
    const leftPath = process.argv[compareIndex + 1], rightPath = process.argv[compareIndex + 2];
    if (!leftPath || !rightPath) throw new Error("--compare-traces requires two JSONL paths");
    const readTrace = (path: string) => readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as HeadlessTraceEntry);
    console.log(JSON.stringify(compareHeadlessTraces(readTrace(leftPath), readTrace(rightPath)), null, 2));
    return;
  }
  const participantCount = process.argv.includes("--participants") ? numberArg("participants", 4) : numberArg("players", 4);
  if (![3, 4].includes(participantCount)) throw new Error("--participants/--players must be 3 or 4 (a five-player map is not implemented yet)");
  const mode = stringArg("mode", "debug");
  if (!["debug", "sweep", "training"].includes(mode)) throw new Error("--mode must be debug, sweep, or training");
  const policyName = stringArg("policy", "random");
  if (!["random", "heuristic"].includes(policyName)) throw new Error("--policy must be random or heuristic");
  const workers = numberArg("workers", 1);
  if (!Number.isInteger(workers) || workers <= 0) throw new Error("--workers must be a positive integer");
  const matchCount = numberArg("matches", 1);
  const traceFile = stringArg("trace-file", "");
  if (traceFile && matchCount !== 1) throw new Error("--trace-file currently requires --matches 1");
  if (workers > getAvailableHeadlessParallelism()) console.warn(`Requested ${workers} workers, but Node reports ${getAvailableHeadlessParallelism()} available parallel CPUs.`);
  const batch = await runParallelHeadlessBatch({
    participantCount: participantCount as 3 | 4,
    matchCount,
    seedStart: numberArg("seed", 1),
    maxTurns: numberArg("max-turns", 100),
    maxActions: numberArg("max-actions", 100_000),
    mode: mode as HeadlessMode,
    profile: hasFlag("profile"),
    trace: Boolean(traceFile),
    policyName: policyName as "random" | "heuristic",
    workerCount: workers,
  });
  if (traceFile) writeFileSync(traceFile, `${(batch.matches[0].trace ?? []).map((entry) => JSON.stringify(entry)).join("\n")}\n`, "utf8");
  const printable = { ...batch, matches: batch.matches.map(({ finalState: _finalState, trace: _trace, ...match }) => match) };
  console.log(JSON.stringify(printable, null, 2));
}

main().catch((error) => { console.error(error instanceof Error ? `${error.name}: ${error.message}` : String(error)); process.exitCode = 1; });
