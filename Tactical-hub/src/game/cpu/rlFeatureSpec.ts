import type { RlObservation } from "./rlEnvironment";
import { getRlActionFeatureWidth, RL_ACTION_ENCODER_VERSION } from "./rlActionEncoder";
import { getRlObservationFeatureSpec } from "./rlObservationEncoder";

export type RlFeatureSpec = ReturnType<typeof createRlFeatureSpec>;

export function createRlFeatureSpec(observation: RlObservation) {
  const observationSpec = getRlObservationFeatureSpec(observation);
  return {
    schemaVersion: 1,
    observationSchemaVersion: observationSpec.schemaVersion,
    actionSchemaVersion: RL_ACTION_ENCODER_VERSION,
    globalWidth: observationSpec.globalWidth,
    teamWidth: observationSpec.teamWidth,
    unitWidth: observationSpec.unitWidth,
    mapTileWidth: observationSpec.mapTileWidth,
    baseWidth: observationSpec.baseWidth,
    constructionWidth: observationSpec.constructionWidth,
    strategicGlobalWidth: observationSpec.strategicGlobalWidth,
    strategicTableRowWidths: observationSpec.strategicTableRowWidths,
    actionFeatureWidth: getRlActionFeatureWidth(observation),
  };
}
