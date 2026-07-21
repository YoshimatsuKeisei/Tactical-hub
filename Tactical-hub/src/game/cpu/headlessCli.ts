import { runHeadlessBatch, type HeadlessMode } from "./headlessSimulation";

declare const process: { argv: string[] };

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

const participantCount = numberArg("players", 4);
if (![3, 4].includes(participantCount)) throw new Error("--players must be 3 or 4 (a five-player map is not implemented yet)");
const mode = stringArg("mode", "debug");
if (!["debug", "sweep", "training"].includes(mode)) throw new Error("--mode must be debug, sweep, or training");
const batch = runHeadlessBatch({
  participantCount: participantCount as 3 | 4,
  matchCount: numberArg("matches", 1),
  seedStart: numberArg("seed", 1),
  maxTurns: numberArg("max-turns", 100),
  maxActions: numberArg("max-actions", 100_000),
  mode: mode as HeadlessMode,
  profile: hasFlag("profile"),
});
const printable = {
  ...batch,
  matches: batch.matches.map(({ finalState: _finalState, ...match }) => match),
};
console.log(JSON.stringify(printable, null, 2));
