import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  parseRlImitationReplayRecords,
  replayHeuristicImitationEpisode,
  type RlImitationReplayRecord,
} from "./rlImitationCollector";

const index = process.argv.indexOf("--input");
if (index < 0 || !process.argv[index + 1]) throw new Error("--input is required");
const inputPath = resolve(process.argv[index + 1]);
const records = readFileSync(inputPath, "utf8")
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line) as RlImitationReplayRecord);
const episodes = parseRlImitationReplayRecords(records);
const results = [];
for (const episode of episodes) results.push(await replayHeuristicImitationEpisode({ episode }));
console.log(JSON.stringify({
  episodeCount: results.length,
  decisionCount: results.reduce((sum, result) => sum + result.decisionCount, 0),
  replayMatched: true,
  results,
  inputPath,
}, null, 2));
