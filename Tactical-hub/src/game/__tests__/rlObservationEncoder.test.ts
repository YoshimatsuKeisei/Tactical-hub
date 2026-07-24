import { describe, expect, it } from "vitest";
import { RlEnvironment } from "../cpu/rlEnvironment";
import { encodeRlObservation, RL_OBSERVATION_ENCODER_VERSION } from "../cpu/rlObservationEncoder";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";

function allNumbers(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (Array.isArray(value)) return value.flatMap(allNumbers);
  if (value && typeof value === "object") return Object.values(value).flatMap(allNumbers);
  return [];
}

describe("RL-1B Observation Encoder", () => {
  it("is deterministic, finite, numeric, and does not mutate its input", () => {
    const environment = new RlEnvironment();
    const observation = environment.reset(100, 4);
    const before = structuredClone(observation);
    const first = encodeRlObservation(observation);
    const second = encodeRlObservation(observation);

    expect(first).toEqual(second);
    expect(first.schemaVersion).toBe(RL_OBSERVATION_ENCODER_VERSION);
    expect(allNumbers(first).every(Number.isFinite)).toBe(true);
    expect(observation).toEqual(before);
  });

  it("omits removed units and aligns zero padding with unitMask", () => {
    const state = createHeadlessInitialState(4);
    const removed = state.units[0];
    if (removed.position.kind === "base") {
      const position = removed.position;
      const base = state.bases.find((entry) => entry.id === position.baseId);
      const slot = base?.slots.find((entry) => entry.id === position.slotId);
      if (slot) slot.unitId = undefined;
    }
    removed.position = { kind: "removed", reason: "defeated" };
    removed.hp = 0;
    const observation = new RlEnvironment().reset(101, 4, state);
    const encoded = encodeRlObservation(observation);
    const livingCount = observation.units.filter((unit) => unit.position.kind !== "removed").length;
    const expectedCapacity = observation.map.tiles.length + observation.bases.reduce((sum, base) => sum + base.slots.length, 0);

    expect(encoded.units).toHaveLength(expectedCapacity);
    expect(encoded.unitMask).toHaveLength(expectedCapacity);
    expect(encoded.unitMask.reduce((sum, value) => sum + value, 0)).toBe(livingCount);
    encoded.units.forEach((row, index) => {
      if (encoded.unitMask[index] === 0) expect(row.every((value) => value === 0)).toBe(true);
    });
  });

  it("uses the observer as team slot zero without encoding raw team ID numbers", () => {
    const state = createHeadlessInitialState(4);
    state.teams.find((team) => team.id === "team-2")!.defeatedUnitCount = 17;
    const environment = new RlEnvironment();
    environment.reset(102, 4, state);
    const teamTwo = encodeRlObservation(environment.getObservation("team-2"));
    const teamFour = encodeRlObservation(environment.getObservation("team-4"));

    expect(teamTwo.teams[0][0]).toBe(1);
    expect(teamTwo.teams[0][7]).toBe(17);
    expect(teamFour.teams[0][0]).toBe(1);
    expect(teamFour.teams[0][7]).not.toBe(17);
    expect(allNumbers(teamTwo).every((value) => typeof value === "number")).toBe(true);
  });

  it("normalizes only present coordinates and keeps categorical encodings stable", () => {
    const observation = new RlEnvironment().reset(103, 4);
    const encoded = encodeRlObservation(observation);
    for (const row of encoded.map.flat()) {
      expect(row[0]).toBeGreaterThanOrEqual(0);
      expect(row[0]).toBeLessThanOrEqual(1);
      expect(row[1]).toBeGreaterThanOrEqual(0);
      expect(row[1]).toBeLessThanOrEqual(1);
      expect(row.slice(2, 8).reduce((sum, value) => sum + value, 0)).toBeLessThanOrEqual(1);
    }
    encoded.units.forEach((row, index) => {
      if (!encoded.unitMask[index]) return;
      expect(row[14]).toBeGreaterThanOrEqual(0);
      expect(row[14]).toBeLessThanOrEqual(1);
      expect(row[15]).toBeGreaterThanOrEqual(0);
      expect(row[15]).toBeLessThanOrEqual(1);
      expect(row.slice(1, 9).reduce((sum, value) => sum + value, 0)).toBe(1);
      expect(row.slice(9, 13).reduce((sum, value) => sum + value, 0)).toBe(1);
    });
  });

  it("encodes only the observing team's already-visible unresolved intents", () => {
    const state = createHeadlessInitialState(4);
    const own = state.units.find((unit) => unit.ownerTeamId === "team-1")!;
    const enemy = state.units.find((unit) => unit.ownerTeamId === "team-2")!;
    state.phase = state.turnState.phase = "attack_input";
    state.turnState.actionIntents = [
      { teamId: "team-1", productionChoices: [], movementIntents: [], attackIntents: [{ teamId: "team-1", attackerUnitId: own.id, pass: true }] },
      { teamId: "team-2", productionChoices: [], movementIntents: [], attackIntents: [{ teamId: "team-2", attackerUnitId: enemy.id, target: { kind: "unit", unitId: own.id }, pass: false }] },
    ];
    const environment = new RlEnvironment();
    environment.reset(104, 4, state);
    const ownObservation = environment.getObservation("team-1");
    const encoded = encodeRlObservation(ownObservation);

    expect(ownObservation.actionIntents.map((intent) => intent.teamId)).toEqual(["team-1"]);
    expect(encoded.strategicState.attackIntents).toHaveLength(1);
    expect(encoded.strategicState.productionIntents).toHaveLength(0);
    expect(encoded.strategicState.movementIntents).toHaveLength(0);
  });
});
