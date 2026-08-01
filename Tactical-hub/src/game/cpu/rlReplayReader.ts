import { createReadStream, readdirSync, statSync } from "node:fs";
import { open } from "node:fs/promises";
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
    .sort((left, right) => left < right ? -1 : left > right ? 1 : 0);
}

export type RlReplayEpisodeIndex = {
  sourcePath: string;
  episodes: Array<{ episodeNumber: number; episodeId: string; header: RlImitationEpisodeHeader; file: string; startOffset: number; endOffsetExclusive: number }>;
};

export async function createRlReplayEpisodeIndex(path: string, maxEpisodeNumber = Number.MAX_SAFE_INTEGER): Promise<RlReplayEpisodeIndex> {
  const episodes: RlReplayEpisodeIndex["episodes"] = [];
  let episodeStart: { startOffset: number; episodeId: string; header: RlImitationEpisodeHeader } | undefined;
  for (const file of replayFiles(path)) {
    if (episodes.length >= maxEpisodeNumber) break;
    let pending = Buffer.alloc(0);
    let pendingOffset = 0;
    const consumeLine = (raw: Buffer, byteOffset: number, byteEnd: number) => {
      const line = raw.length && raw[raw.length - 1] === 13 ? raw.subarray(0, -1) : raw;
      if (!line.length) return;
      const text = line.toString("utf8");
      if (!text.includes('"type":"episode_start"') && !text.includes('"type":"episode_end"')) return;
      const record = JSON.parse(text) as RlImitationReplayRecord;
      if (record.type === "episode_start") {
        if (episodeStart) throw new Error(`Episode ${episodeStart.episodeId} has no end record`);
        episodeStart = { startOffset: byteOffset, episodeId: record.episodeId, header: record };
      } else if (record.type !== "decision") {
        if (!episodeStart || episodeStart.episodeId !== record.episodeId) throw new Error(`End outside matching episode: ${record.episodeId}`);
        episodes.push({ episodeNumber: episodes.length + 1, episodeId: record.episodeId, header: episodeStart.header, file, startOffset: episodeStart.startOffset, endOffsetExclusive: byteEnd });
        episodeStart = undefined;
      }
    };
    for await (const value of createReadStream(file)) {
      const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value);
      const combined = pending.length ? Buffer.concat([pending, chunk]) : chunk;
      let start = 0;
      for (let index = 0; index < combined.length; index += 1) if (combined[index] === 10) {
        consumeLine(combined.subarray(start, index), pendingOffset + start, pendingOffset + index + 1);
        start = index + 1;
        if (episodes.length >= maxEpisodeNumber) break;
      }
      pending = combined.subarray(start);
      pendingOffset += start;
      if (episodes.length >= maxEpisodeNumber) return { sourcePath: resolve(path), episodes };
    }
    if (pending.length) consumeLine(pending, pendingOffset, pendingOffset + pending.length);
    if (episodeStart && episodes.length < maxEpisodeNumber) throw new Error(`Episode ${episodeStart.episodeId} crosses replay files or has no end record`);
  }
  return { sourcePath: resolve(path), episodes };
}

export async function readRlImitationEpisodeAt(index: RlReplayEpisodeIndex, episodeNumber: number) {
  const entry = index.episodes[episodeNumber - 1];
  if (!entry || entry.episodeNumber !== episodeNumber) throw new Error(`Replay episode ${episodeNumber} is missing`);
  const contents = await readRlImitationEpisodeBytesAt(index, episodeNumber);
  const records = contents.toString("utf8").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as RlImitationReplayRecord);
  const header = records[0] as RlImitationEpisodeHeader;
  const end = records.at(-1) as RlImitationEpisodeEnd;
  const decisions = records.slice(1, -1) as RlImitationDecisionRecord[];
  if (header.type !== "episode_start" || end.type !== "episode_end" || header.episodeId !== entry.episodeId || end.episodeId !== entry.episodeId) throw new Error(`Replay episode ${episodeNumber} index identity mismatch`);
  return { episodeNumber, episode: { header, decisions, end } satisfies RlImitationEpisode };
}

export async function readRlImitationEpisodeBytesAt(index: RlReplayEpisodeIndex, episodeNumber: number) {
  const entry = index.episodes[episodeNumber - 1];
  if (!entry || entry.episodeNumber !== episodeNumber) throw new Error(`Replay episode ${episodeNumber} is missing`);
  const byteLength = entry.endOffsetExclusive - entry.startOffset;
  if (byteLength <= 0) throw new Error(`Replay episode ${episodeNumber} has an invalid byte range`);
  const handle = await open(entry.file, "r");
  try {
    const contents = Buffer.allocUnsafe(byteLength);
    const read = await handle.read(contents, 0, contents.length, entry.startOffset);
    if (read.bytesRead !== contents.length) throw new Error(`Replay episode ${episodeNumber} is truncated`);
    return contents;
  } finally {
    await handle.close();
  }
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
