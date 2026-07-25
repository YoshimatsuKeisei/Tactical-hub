import { createReadStream, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";
import type {
  RlImitationDecisionRecord,
  RlImitationEpisode,
  RlImitationEpisodeEnd,
  RlImitationEpisodeHeader,
  RlImitationReplayRecord,
} from "./rlImitationCollector";

export type EpisodeRange = { from: number; to: number };

export function parseEpisodeRange(value: string): EpisodeRange {
  const match = /^(\d+)-(\d+)$/.exec(value);
  if (!match) throw new Error(`Invalid episode range: ${value}`);
  const from = Number(match[1]), to = Number(match[2]);
  if (from <= 0 || to < from) throw new Error(`Invalid episode range: ${value}`);
  return { from, to };
}

function replayFiles(path: string) {
  const resolved = resolve(path);
  if (!statSync(resolved).isDirectory()) return [resolved];
  return readdirSync(resolved)
    .map((name) => join(resolved, name))
    .filter((entry) => statSync(entry).isFile())
    .sort((left, right) => left.localeCompare(right));
}

export async function* readRlImitationEpisodes(
  path: string,
  range: EpisodeRange,
): AsyncGenerator<{ episodeNumber: number; episode: RlImitationEpisode }> {
  let episodeNumber = 0;
  let header: RlImitationEpisodeHeader | undefined;
  let decisions: RlImitationDecisionRecord[] = [];
  for (const file of replayFiles(path)) {
    const lines = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
    for await (const line of lines) {
      if (!line) continue;
      const record = JSON.parse(line) as RlImitationReplayRecord;
      if (record.type === "episode_start") {
        if (header) throw new Error(`Episode ${header.episodeId} has no end record`);
        header = record;
        decisions = [];
      } else if (record.type === "decision") {
        if (!header || record.episodeId !== header.episodeId) throw new Error(`Decision outside matching episode: ${record.episodeId}`);
        decisions.push(record);
      } else {
        if (!header || record.episodeId !== header.episodeId) throw new Error(`End outside matching episode: ${record.episodeId}`);
        episodeNumber += 1;
        if (episodeNumber >= range.from && episodeNumber <= range.to) {
          yield { episodeNumber, episode: { header, decisions, end: record as RlImitationEpisodeEnd } };
        }
        header = undefined;
        decisions = [];
        if (episodeNumber >= range.to) return;
      }
    }
  }
  if (header) throw new Error(`Episode ${header.episodeId} has no end record`);
}
