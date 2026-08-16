import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { createInitialGameState } from "../initialState";
import { encodeRlLegalActionsV1, encodeRlLegalActionsV2 } from "../cpu/rlActionEncoder";
import { buildRlObservation, enumerateRlDecisionsV1 } from "../cpu/rlEnvironment";
import { encodeRlObservationV1, encodeRlObservationV2 } from "../cpu/rlObservationEncoder";
import { createRlV1ToV2MigrationManifest } from "../cpu/rlSchemaMigration";
import { createCpuRuntime } from "../cpu/types";

function expand(source: number[], targetWidth: number, segments: { sourceStart: number; targetStart: number; width: number }[]) {
  const target = Array(targetWidth).fill(0);
  for (const segment of segments) {
    target.splice(segment.targetStart, segment.width, ...source.slice(segment.sourceStart, segment.sourceStart + segment.width));
  }
  return target;
}

describe("RL schema v1 to v2 migration manifest", () => {
  it("matches the current Phase 10 encoder source of truth", () => {
    const state = createInitialGameState();
    const observation = buildRlObservation(state, "team-1", "team-1");
    const generated = createRlV1ToV2MigrationManifest(observation);
    const stored = JSON.parse(readFileSync(resolve("rl/bc_v1_to_v2_manifest.json"), "utf8"));
    expect(stored).toEqual(generated);
    expect(generated.targetFeatureSpec.unitWidth).toBe(generated.sourceFeatureSpec.unitWidth + 1);
    expect(generated.actionInputColumns.reduce((sum, segment) => sum + segment.width, 0)).toBe(generated.sourceFeatureSpec.actionFeatureWidth);
  });

  it("maps an old-rule observation and non-merge legal actions without changing common feature values", () => {
    const state = createInitialGameState();
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    const observation = buildRlObservation(state, "team-1", "team-1");
    const manifest = createRlV1ToV2MigrationManifest(observation);
    const observationV1 = encodeRlObservationV1(observation);
    const observationV2 = encodeRlObservationV2(observation);
    expect(observationV2.units).toEqual(observationV1.units.map((row) =>
      expand(row, manifest.targetFeatureSpec.unitWidth, manifest.unitInputColumns),
    ));
    const legalActions = enumerateRlDecisionsV1(state, createCpuRuntime(11), (teamId) => teamId === "team-1").map((entry) => entry.action);
    expect(legalActions.some((action) => action.actionType === "merge_infantry")).toBe(false);
    const actionsV1 = encodeRlLegalActionsV1(observation, legalActions);
    const actionsV2 = encodeRlLegalActionsV2(observation, legalActions);
    expect(actionsV2.actionKeys).toEqual(actionsV1.actionKeys);
    expect(actionsV2.actions).toEqual(actionsV1.actions.map((row) =>
      expand(row, manifest.targetFeatureSpec.actionFeatureWidth, manifest.actionInputColumns),
    ));
  });
});
