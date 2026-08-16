import { describe, expect, it } from "vitest";
import { mergeHeavyInfantry } from "../engine/heavyInfantry";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit } from "../types";
import { createBcInferenceRequest } from "../cpu/browserBcPolicy";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { encodeRlLegalActions, encodeRlLegalActionsV2 } from "../cpu/rlActionEncoder";
import { buildRlObservation, enumerateRlDecisions, enumerateRlDecisionsV2, RlEnvironment, RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpec, createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import { encodeRlObservation, encodeRlObservationV2 } from "../cpu/rlObservationEncoder";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";

function mergeFixture() {
  const state = createInitialGameState();
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  const base = state.bases.find((candidate) => candidate.id === "home-1")!;
  const removedTeamUnitIds = new Set(state.units.filter((unit) => unit.ownerTeamId === "team-1").map((unit) => unit.id));
  state.units = state.units.filter((unit) => !removedTeamUnitIds.has(unit.id));
  for (const candidate of state.bases) {
    for (const slot of candidate.slots) {
      if (slot.unitId && removedTeamUnitIds.has(slot.unitId)) slot.unitId = undefined;
    }
  }
  const add = (id: string, slotId: string): Unit => {
    const unit: Unit = { id, ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "base", baseId: base.id, slotId }, statuses: [] };
    base.slots.find((slot) => slot.id === slotId)!.unitId = id;
    state.units.push(unit);
    return unit;
  };
  const primary = add("merge-a", "slot_0_1");
  const partner = add("merge-b", "slot_1_1");
  expect(state).toMatchObject({
    phase: "movement_input",
    currentMovementTeamId: "team-1",
    movementCompletedTeamIds: [],
    rewardPlacementRequests: [],
    teleportIntents: [],
  });
  expect(state.teams.find((team) => team.id === "team-1")?.status).toBe("active");
  expect(state.productionCompletedTeamIdsThisTurn).toContain("team-1");
  expect(state.movedUnitIdsThisMovementPhase).not.toContain(primary.id);
  expect(state.movedUnitIdsThisMovementPhase).not.toContain(partner.id);
  return { state, primary, partner };
}

describe("RL schema v1/v2 coexistence", () => {
  it("keeps v1 byte-for-byte formation-blind while v2 adds one heavy unit feature", () => {
    const { state, primary } = mergeFixture();
    primary.hp = 2;
    const normalObservation = buildRlObservation(state, "team-1", "team-1");
    const heavyState = structuredClone(state) as GameState;
    heavyState.units.find((unit) => unit.id === primary.id)!.formation = "heavy";
    const heavyObservation = buildRlObservation(heavyState, "team-1", "team-1");
    const normalV1 = encodeRlObservation(normalObservation);
    const heavyV1 = encodeRlObservation(heavyObservation);
    expect(normalV1).toEqual(heavyV1);
    expect(normalV1.schemaVersion).toBe(1);

    const normalV2 = encodeRlObservationV2(normalObservation);
    const heavyV2 = encodeRlObservationV2(heavyObservation);
    expect(normalV2.schemaVersion).toBe(2);
    expect(normalV2.units[0].length).toBe(normalV1.units[0].length + 1);
    const changedRow = normalV2.units.findIndex((row, index) => row.some((value, feature) => value !== heavyV2.units[index][feature]));
    expect(changedRow).toBeGreaterThanOrEqual(0);
    expect(normalV2.units[changedRow][1]).toBe(0);
    expect(heavyV2.units[changedRow][1]).toBe(1);
  });

  it("keeps v1 Feature Spec and Browser BC explicitly on schema 1 while exposing schema 2 widths", () => {
    const { state } = mergeFixture();
    const observation = buildRlObservation(state, "team-1", "team-1");
    const v1 = createRlFeatureSpec(observation);
    const v2 = createRlFeatureSpecV2(observation);
    expect(v1).toMatchObject({ schemaVersion: 1, observationSchemaVersion: 1, actionSchemaVersion: 1 });
    expect(v2).toMatchObject({ schemaVersion: 2, observationSchemaVersion: 2, actionSchemaVersion: 2 });
    expect(v2.unitWidth).toBe(v1.unitWidth + 1);
    expect(v2.actionFeatureWidth).toBeGreaterThan(v1.actionFeatureWidth);
    const settings: CpuTeamSettings = { "team-1": "bc_cpu", "team-2": "human", "team-3": "human", "team-4": "human" };
    expect(createBcInferenceRequest(state, createCpuRuntime(1), settings)?.request?.featureSpec).toEqual(v1);
  });

  it("enumerates exactly one deterministic merge only in v2 and encodes both infantry references", () => {
    const { state, primary, partner } = mergeFixture();
    const v1Runtime = createCpuRuntime(2);
    const v2Runtime = createCpuRuntime(2);
    const v1 = enumerateRlDecisions(state, v1Runtime, (teamId) => teamId === "team-1");
    const v2 = enumerateRlDecisionsV2(state, v2Runtime, (teamId) => teamId === "team-1");
    expect(v1.some((entry) => entry.decision.kind === "merge_infantry")).toBe(false);
    const merges = v2.filter((entry) => entry.decision.kind === "merge_infantry");
    expect(merges).toHaveLength(1);
    expect(merges[0].action).toMatchObject({
      actionKey: `merge_infantry:team-1:${primary.id}:${partner.id}`,
      actionType: "merge_infantry",
      unitId: primary.id,
      partnerUnitId: partner.id,
    });
    const observation = buildRlObservation(state, "team-1", "team-1");
    expect(encodeRlLegalActionsV2(observation, [merges[0].action]).actions[0].length).toBe(createRlFeatureSpecV2(observation).actionFeatureWidth);
    expect(() => encodeRlLegalActions(observation, v1.map((entry) => entry.action))).not.toThrow();
  });

  it("uses the existing merge engine in v2 step and rejects the stale action afterwards", () => {
    const fixture = mergeFixture();
    const expected = mergeHeavyInfantry(fixture.state, fixture.primary.id, fixture.partner.id);
    const environment = new RlEnvironmentV2();
    environment.reset(3, 4, fixture.state);
    const action = environment.getLegalActions("team-1").find((candidate) => candidate.actionType === "merge_infantry")!;
    expect(action).toBeDefined();
    environment.step(action.actionKey);
    const actual = environment.getObservation("team-1");
    expect(actual.units.find((unit) => unit.id === fixture.primary.id)).toMatchObject(
      expected.units.find((unit) => unit.id === fixture.primary.id)!,
    );
    expect(actual.units.find((unit) => unit.id === fixture.partner.id)).toMatchObject({ hp: 0, position: { kind: "removed", reason: "merged" } });
    expect(actual.bases.find((base) => base.id === "home-1")?.slots.find((slot) => slot.id === "slot_1_1")?.unitId).toBeUndefined();
    expect(actual.movedUnitIdsThisMovementPhase).toEqual(expect.arrayContaining([fixture.primary.id, fixture.partner.id]));
    expect(() => environment.step(action.actionKey)).toThrow(/Illegal or stale/);

    const v1Environment = new RlEnvironment();
    v1Environment.reset(3, 4, fixture.state);
    expect(v1Environment.getLegalActions("team-1").some((candidate) => candidate.actionType === "merge_infantry")).toBe(false);
  });

  it("omits invalid merge pairs and does not add merge decisions to Random or Heuristic policies", () => {
    const { state, partner } = mergeFixture();
    partner.formation = "heavy";
    expect(enumerateRlDecisionsV2(state, createCpuRuntime(4), (teamId) => teamId === "team-1").some((entry) => entry.decision.kind === "merge_infantry")).toBe(false);
    const randomSettings: CpuTeamSettings = { "team-1": "random_cpu", "team-2": "human", "team-3": "human", "team-4": "human" };
    const heuristicSettings: CpuTeamSettings = { ...randomSettings, "team-1": "heuristic_cpu" };
    expect(getRandomCpuDecision(state, createCpuRuntime(4), randomSettings)?.kind).not.toBe("merge_infantry");
    expect(createHeuristicCpuPolicy()(state, createCpuRuntime(4), heuristicSettings)?.kind).not.toBe("merge_infantry");
  });

  it("preserves team-scoped hidden water-ninja visibility", () => {
    const { state } = mergeFixture();
    state.units.push({ id: "hidden", ownerTeamId: "team-2", type: "ninja", hp: 1, position: { kind: "water", x: 4, y: 2 }, statuses: [] });
    expect(buildRlObservation(state, "team-1", "team-1").units.map((unit) => unit.id)).not.toContain("hidden");
  });
});
