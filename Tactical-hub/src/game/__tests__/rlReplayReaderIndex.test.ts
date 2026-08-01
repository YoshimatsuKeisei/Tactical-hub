import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createRlReplayEpisodeIndex, readRlImitationEpisodeAt, readRlImitationEpisodeBytesAt } from "../cpu/rlReplayReader";

const episodeLines = (episode: number) => [
  { type: "episode_start", episodeId: `対局-${episode}`, seed: episode, participantCount: 4, mapId: "四人用", gameConfig: {} },
  { type: "decision", episodeId: `対局-${episode}`, teamId: "team-1", turnNumber: 1, selectedActionKey: `移動-${episode}` },
  { type: "episode_end", episodeId: `対局-${episode}`, decisionCount: 1 },
].map((record) => JSON.stringify(record));

function episodeBuffer(episode: number, newline: "\n" | "\r\n", trailingNewline: boolean) {
  return Buffer.from(`${episodeLines(episode).join(newline)}${trailingNewline ? newline : ""}`, "utf8");
}

describe("RL replay byte index", () => {
  for (const [label, newline, finalNewline] of [
    ["LF with final newline", "\n", true],
    ["LF without final newline", "\n", false],
    ["CRLF with final newline", "\r\n", true],
    ["CRLF without final newline", "\r\n", false],
  ] as const) {
    it(`uses exact Buffer ranges for ${label}`, async () => {
      const directory = mkdtempSync(join(tmpdir(), "tactical-hub-replay-index-"));
      try {
        const path = join(directory, "replay.jsonl");
        const expected = [episodeBuffer(1, newline, true), episodeBuffer(2, newline, true), episodeBuffer(3, newline, finalNewline)];
        writeFileSync(path, Buffer.concat(expected));
        const index = await createRlReplayEpisodeIndex(path, 3);
        expect(index.episodes).toHaveLength(3);
        for (let episode = 1; episode <= 3; episode += 1) {
          expect(await readRlImitationEpisodeBytesAt(index, episode)).toEqual(expected[episode - 1]);
          expect((await readRlImitationEpisodeAt(index, episode)).episode.header.episodeId).toBe(`対局-${episode}`);
          const entry = index.episodes[episode - 1];
          expect(entry.endOffsetExclusive - entry.startOffset).toBe(expected[episode - 1].byteLength);
        }
        expect(index.episodes[0].endOffsetExclusive).toBe(index.episodes[1].startOffset);
        expect(index.episodes[1].endOffsetExclusive).toBe(index.episodes[2].startOffset);
        const firstOnly = await createRlReplayEpisodeIndex(path, 1);
        expect(firstOnly.episodes).toHaveLength(1);
        expect(await readRlImitationEpisodeBytesAt(firstOnly, 1)).toEqual(expected[0]);
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });
  }

  it("indexes multiple files without crossing file boundaries", async () => {
    const directory = mkdtempSync(join(tmpdir(), "tactical-hub-replay-index-files-"));
    try {
      const first = [episodeBuffer(1, "\n", true), episodeBuffer(2, "\n", true)];
      const last = episodeBuffer(3, "\r\n", false);
      writeFileSync(join(directory, "part-001.jsonl"), Buffer.concat(first));
      writeFileSync(join(directory, "part-002.jsonl"), last);
      const index = await createRlReplayEpisodeIndex(directory, 3);
      expect(await readRlImitationEpisodeBytesAt(index, 1)).toEqual(first[0]);
      expect(await readRlImitationEpisodeBytesAt(index, 2)).toEqual(first[1]);
      expect(await readRlImitationEpisodeBytesAt(index, 3)).toEqual(last);
      expect(index.episodes[1].file).not.toBe(index.episodes[2].file);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
