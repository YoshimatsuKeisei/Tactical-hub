import { describe, expect, it } from "vitest";
import {
  parseRlImitationReplayRecords,
  replayHeuristicImitationEpisode,
  runHeuristicImitationEpisode,
  type RlImitationReplayRecord,
  type RlReplayEncodedDecision,
} from "../cpu/rlImitationCollector";

function expectFinite(value: unknown): void {
  if (typeof value === "number") {
    expect(Number.isFinite(value)).toBe(true);
  } else if (Array.isArray(value)) {
    for (const entry of value) expectFinite(entry);
  } else if (value && typeof value === "object") {
    for (const entry of Object.values(value)) expectFinite(entry);
  }
}

describe("RL-4A lightweight Heuristic replay", () => {
  it("stores only action records and reproduces encodable legal decisions and the final state", async () => {
    const records: RlImitationReplayRecord[] = [];
    const collected = await runHeuristicImitationEpisode({
      episodeId: "test-episode",
      seed: 404,
      maxTurns: 1,
      onRecord: (record) => {
        records.push(record);
      },
    });
    const [episode] = parseRlImitationReplayRecords(records);
    const encoded: RlReplayEncodedDecision[] = [];
    const replayed = await replayHeuristicImitationEpisode({
      episode,
      onEncodedDecision: (decision) => {
        encoded.push(decision);
      },
    });

    expect(episode.decisions.length).toBeGreaterThan(0);
    expect(Object.values(collected.sampleCountByTeam).every((count) => count > 0)).toBe(true);
    expect(records.every((record) => !("encodedObservation" in record) && !("encodedLegalActions" in record) && !("selectedActionIndex" in record))).toBe(true);
    expect(encoded).toHaveLength(episode.decisions.length);
    for (const decision of encoded) {
      expect(decision.selectedActionIndex).toBeGreaterThanOrEqual(0);
      expect(decision.encodedLegalActions.actionKeys[decision.selectedActionIndex]).toBe(decision.record.selectedActionKey);
      expectFinite(decision.encodedObservation);
      expectFinite(decision.encodedLegalActions.actions);
    }
    expect(replayed.finalStateHash).toBe(collected.finalStateHash);
    expect(replayed.terminal).toBe(collected.terminal);
    expect(replayed.endReason).toBe(collected.endReason);
  });
});
