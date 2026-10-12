import { describe, expect, it } from "vitest";
import { RlEnvironment } from "../cpu/rlEnvironment";
import { encodeRlObservation } from "../cpu/rlObservationEncoder";
import { encodeRlLegalActions } from "../cpu/rlActionEncoder";
import { createRlFeatureSpec, createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import { RL_STRATEGIST_ROLES } from "../cpu/rlObservationEncoder";

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

  it("keeps the three legacy role columns and every u40-compatible feature width", () => {
    const v1Environment = new RlEnvironment();
    const v1 = createRlFeatureSpec(v1Environment.reset(101, 4));
    const v2Environment = new RlEnvironment(undefined, 2);
    const v2 = createRlFeatureSpecV2(v2Environment.reset(101, 4));

    expect(RL_STRATEGIST_ROLES).toEqual(["encourage", "builder", "teleporter"]);
    expect(v1).toEqual({
      schemaVersion: 1, observationSchemaVersion: 1, actionSchemaVersion: 1,
      globalWidth: 15, teamWidth: 15, unitWidth: 57, mapTileWidth: 38,
      baseWidth: 87, constructionWidth: 22, strategicGlobalWidth: 36,
      strategicTableRowWidths: {
        siegeStates: 35, kingCampaignStates: 31, rewardPlacementRequests: 67,
        strategistCooldowns: 18, teleportCooldowns: 16, productionIntents: 27,
        movementIntents: 58, attackIntents: 37, strategistActionIntents: 33,
        teleportIntents: 54,
      },
      actionFeatureWidth: 3403,
    });
    expect(v2).toEqual({ ...v1,
      schemaVersion: 2, observationSchemaVersion: 2, actionSchemaVersion: 2,
      unitWidth: 58, actionFeatureWidth: 3955,
    });
  });
});
