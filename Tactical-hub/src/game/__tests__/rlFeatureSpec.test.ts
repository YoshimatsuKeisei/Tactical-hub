import { describe, expect, it } from "vitest";
import { RlEnvironment } from "../cpu/rlEnvironment";
import { encodeRlObservation } from "../cpu/rlObservationEncoder";
import { encodeRlLegalActions } from "../cpu/rlActionEncoder";
import { createRlFeatureSpec } from "../cpu/rlFeatureSpec";

describe("RL feature spec", () => {
  it("derives every width from the encoders, including empty strategic tables", () => {
    const env = new RlEnvironment();
    const observation = env.reset(101, 4);
    const encodedObservation = encodeRlObservation(observation);
    const encodedActions = encodeRlLegalActions(
      observation,
      env.getLegalActions(env.getCurrentActorTeamId()!),
    );
    const spec = createRlFeatureSpec(observation);

    expect(spec.globalWidth).toBe(encodedObservation.global.length);
    expect(spec.teamWidth).toBe(encodedObservation.teams[0]?.length);
    expect(spec.unitWidth).toBe(encodedObservation.units[0]?.length);
    expect(spec.mapTileWidth).toBe(encodedObservation.map[0]?.[0]?.length);
    expect(spec.baseWidth).toBe(encodedObservation.bases[0]?.length);
    expect(spec.constructionWidth).toBe(encodedObservation.constructions[0]?.length);
    expect(spec.strategicGlobalWidth).toBe(encodedObservation.strategicState.global.length);
    expect(spec.actionFeatureWidth).toBe(encodedActions.actions[0]?.length);

    for (const width of Object.values(spec.strategicTableRowWidths)) {
      expect(width).toBeGreaterThan(0);
    }
  });
});
