import { describe, expect, it } from "vitest";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import {
  encodeRlObservationCompactV2,
  encodeRlObservationV2,
} from "../cpu/rlObservationEncoder";
import type { RlObservation } from "../cpu/rlEnvironment";

function historyFixture(allActive = false) {
  const environment = new RlEnvironmentV2(undefined, true);
  environment.reset(313, 4);
  const teamId = environment.getCurrentActorTeamId()!;
  const original = environment.getObservationForEncoding(teamId);
  const max = original.map.tiles.length;
  const tile = original.map.tiles.find((cell) => cell.terrain === "road")!;
  const constructions: RlObservation["constructions"] = Array.from(
    { length: max + 1 },
    (_, i) => ({
      id: `historical-obstacle-${i}`,
      kind: "obstacle",
      ownerTeamId: teamId,
      tiles: [{ x: tile.x, y: tile.y }],
      placedTurn: i,
      active: allActive || i === max,
    }),
  );
  const observation: RlObservation = { ...original, constructions };
  return { observation, max };
}

describe("RL construction history cap regression", () => {
  it("has map-bound construction capacity 484 and reproduces 485-history overflow", () => {
    const { observation, max } = historyFixture();
    expect(max).toBe(484);
    expect(observation.constructions).toHaveLength(max + 1);
    expect(observation.constructions.filter((item) => item.active)).toHaveLength(1);
    const before = structuredClone(observation);
    const selected = encodeRlObservationV2(observation);
    const { constructions, ...other } = observation;
    const expected = encodeRlObservationV2({
      ...other,
      constructions: constructions.slice(1),
    });
    expect(selected).toEqual(expected);
    expect(selected.constructionMask.reduce((sum, flag) => sum + flag, 0)).toBe(max);
    expect(selected.constructions.filter((row) => row[0] === 1)).toHaveLength(1);
    expect(observation).toEqual(before);
  });

  it("preserves exact 484-history encoding without changing contents", () => {
    const { observation, max } = historyFixture();
    const truncated = { ...observation, constructions: observation.constructions.slice(1) };
    const output = encodeRlObservationV2(truncated);
    expect(output.constructions).toHaveLength(max);
    expect(output.constructionMask.every(Boolean)).toBe(true);
    expect(output.constructions.filter((row) => row[0] === 1)).toHaveLength(1);
  });

  it("keeps compact output equal to the reference when history exceeds capacity", () => {
    const { observation, max } = historyFixture();
    const compact = encodeRlObservationCompactV2(observation);
    const expected = encodeRlObservationCompactV2({
      ...observation,
      constructions: observation.constructions.slice(1),
    });
    expect(compact).toEqual(expected);
    expect(compact.constructions).toHaveLength(max);
    expect(compact.rowCompaction?.constructions).toBe(max);
  });

  it("does not silently discard active constructions above the physical capacity", () => {
    const { observation } = historyFixture(true);
    expect(() => encodeRlObservationV2(observation)).toThrow(/constructions|capacity/i);
  });

  it("does not change the fixed V2 feature widths", () => {
    const environment = new RlEnvironmentV2(undefined, true);
    const start = environment.reset(313, 4);
    const spec = createRlFeatureSpecV2(start);
    expect(spec.unitWidth).toBe(58);
    expect(spec.constructionWidth).toBe(22);
    expect(spec.actionFeatureWidth).toBe(3955);
  });
});
